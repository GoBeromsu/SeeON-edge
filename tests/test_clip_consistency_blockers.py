from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import UUID

import pytest

from worker.pipeline.output.evidence.clip_consistency_database import validate_database
from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox

EVENT_ID = str(UUID(int=(4 << 76) | (2 << 62) | 101))
TIMESTAMP = "2026-08-14T00:00:00.000Z"


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    state_dir = tmp_path / "worker-state"
    clip_store = tmp_path / "clip-store"
    maintenance = tmp_path / "maintenance"
    state_dir.mkdir(mode=0o700)
    (clip_store / "clips" / ".staging").mkdir(parents=True)
    maintenance.mkdir(mode=0o700)
    database = state_dir / "worker-state.sqlite3"
    with EvidenceOutbox.open(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO evidence_events (
                edge_event_id, detected_at, payload_json, state, queued_at,
                next_attempt_at, delivery_state
            ) VALUES (?, ?, '{}', 'READY', 0, 0, 'PENDING')
            """,
            (EVENT_ID, TIMESTAMP),
        )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, local_state) VALUES ('clip-a', 'UNAVAILABLE')"
        )
    manifest = {
        "manifest_schema_version": 2,
        "state": "UNAVAILABLE",
        "clip_id": "clip-a",
        "camera_id": "camera-a",
        "event_refs": [EVENT_ID],
        "clip_start_at": TIMESTAMP,
        "clip_end_at": TIMESTAMP,
        "finalized_at": TIMESTAMP,
        "state_version": 2,
        "reason_code": "NO_FRAMES",
        "legacy_extra": {"allowed": True},
    }
    final = clip_store / "clips" / "clip-a"
    final.mkdir()
    (final / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return database, clip_store, maintenance


def _command(database: Path, clip_store: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).parents[1]
    return subprocess.run(
        [
            sys.executable,
            "scripts/repair_clip_consistency.py",
            "--state-db",
            str(database),
            "--clip-store",
            str(clip_store),
            *extra,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )


def _quiescence(path: Path, database: Path, clip_store: Path) -> None:
    payload = {
        "format_version": 1,
        "state_db": str(database.resolve()),
        "clip_store": str(clip_store.resolve()),
        "stopped_service": "ml-worker",
        "stopped_db_writers": ["event", "config", "fault"],
        "operator_uid": os.getuid(),
        "issued_at": int(time.time()) - 1,
        "expires_at": int(time.time()) + 3599,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


def test_cli_accepts_separate_production_state_and_clip_paths(tmp_path: Path) -> None:
    database, clip_store, _ = _layout(tmp_path)

    completed = _command(database, clip_store)

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["mode"] == "dry-run"


def test_apply_refuses_concurrent_ordinary_database_writer_without_mutation(
    tmp_path: Path,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    quiescence = maintenance / "quiescence.json"
    journal = maintenance / "apply.json"
    _quiescence(quiescence, database, clip_store)
    writer_ready = threading.Event()
    release_writer = threading.Event()

    def writer() -> None:
        with sqlite3.connect(database, isolation_level=None) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE config_current SET saved_at = saved_at")
            writer_ready.set()
            assert release_writer.wait(timeout=3)
            connection.rollback()

    thread = threading.Thread(target=writer)
    thread.start()
    assert writer_ready.wait(timeout=3)
    try:
        completed = _command(
            database,
            clip_store,
            "--apply",
            "--maintenance-root",
            str(maintenance),
            "--journal",
            str(journal),
            "--quiescence-receipt",
            str(quiescence),
        )
    finally:
        release_writer.set()
        thread.join(timeout=3)

    assert completed.returncode == 2
    assert "database_busy" in completed.stderr
    assert not journal.exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM clip_events").fetchone()[0] == 0


def test_ordinary_writer_is_excluded_after_tool_holds_write_boundary(
    tmp_path: Path,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    quiescence = maintenance / "quiescence.json"
    journal = maintenance / "apply.json"
    _quiescence(quiescence, database, clip_store)
    tool_holds_boundary = threading.Event()
    release_tool = threading.Event()
    result: list[object] = []

    def pause_after_boundary(stage: str) -> None:
        if stage == "backup:file_fsynced":
            tool_holds_boundary.set()
            assert release_tool.wait(timeout=3)

    def run_tool() -> None:
        result.append(
            repair_clip_consistency(
                RepairRequest(
                    database,
                    clip_store,
                    apply=True,
                    maintenance_root=maintenance,
                    journal_path=journal,
                    quiescence_receipt=quiescence,
                    fault_hook=pause_after_boundary,
                )
            )
        )

    thread = threading.Thread(target=run_tool)
    thread.start()
    assert tool_holds_boundary.wait(timeout=3)
    try:
        with sqlite3.connect(database, isolation_level=None, timeout=0) as writer:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("BEGIN IMMEDIATE")
    finally:
        release_tool.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert len(result) == 1
    assert journal.is_file()


@pytest.mark.parametrize("duplicate_key", ("event_refs", "clip_id", "sha256"))
def test_authority_manifest_rejects_duplicate_keys(
    tmp_path: Path, duplicate_key: str
) -> None:
    database, clip_store, _ = _layout(tmp_path)
    manifest = clip_store / "clips" / "clip-a" / "manifest.json"
    payload = manifest.read_text(encoding="utf-8").removesuffix("}")
    duplicate_value = {
        "event_refs": json.dumps([EVENT_ID]),
        "clip_id": json.dumps("clip-a"),
        "sha256": json.dumps("0" * 64),
    }[duplicate_key]
    prefix = f'{payload},"sha256":{json.dumps("1" * 64)}' if duplicate_key == "sha256" else payload
    manifest.write_text(
        f'{prefix},"{duplicate_key}":{duplicate_value}}}', encoding="utf-8"
    )

    completed = _command(database, clip_store)

    assert completed.returncode == 2
    assert "final_invalid" in completed.stderr


def test_authority_manifest_rejects_nested_duplicate_keys(tmp_path: Path) -> None:
    database, clip_store, _ = _layout(tmp_path)
    manifest = clip_store / "clips" / "clip-a" / "manifest.json"
    payload = manifest.read_text(encoding="utf-8").removesuffix("}")
    manifest.write_text(
        f'{payload},"nested_legacy":{{"identity":1,"identity":2}}}}',
        encoding="utf-8",
    )

    completed = _command(database, clip_store)

    assert completed.returncode == 2
    assert "final_invalid" in completed.stderr


@pytest.mark.parametrize(
    "drift_sql",
    (
        "DROP INDEX evidence_events_claim_idx",
        "DROP INDEX evidence_clips_publish_idx",
        "DROP INDEX evidence_events_delivery_idx",
    ),
)
def test_schema9_fingerprint_rejects_required_index_drift(
    tmp_path: Path, drift_sql: str
) -> None:
    database, _, _ = _layout(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(drift_sql)
    with sqlite3.connect(database) as connection:
        with pytest.raises(ClipConsistencyError, match="schema_drift"):
            validate_database(connection, now=0)


def test_apply_requires_recoverable_prepared_journal(tmp_path: Path) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    quiescence = maintenance / "quiescence.json"
    journal = maintenance / "apply.json"
    _quiescence(quiescence, database, clip_store)

    completed = _command(
        database,
        clip_store,
        "--apply",
        "--maintenance-root",
        str(maintenance),
        "--journal",
        str(journal),
        "--quiescence-receipt",
        str(quiescence),
    )

    assert completed.returncode == 0
    payload = json.loads(journal.read_text(encoding="utf-8"))
    assert payload["state"] == "DONE"
    assert payload["source_state_sha256"]
    assert payload["plan_sha256"]
    assert payload["backup_receipt_path"]
