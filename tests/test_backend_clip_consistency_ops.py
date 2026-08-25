from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import pytest

from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.ownership import Writer, writer_for_table
from backend.app.edge_db.schema import SCHEMA_VERSION
from backend.app.features.clips import consistency_ops
from backend.app.features.clips.consistency_ops import (
    ClipConsistencyError,
    RepairRequest,
    inspect_finalized_clip,
    repair_clip_consistency,
)


def _layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    state = tmp_path / "state"
    store = tmp_path / "store"
    maintenance = tmp_path / "maintenance"
    state.mkdir(mode=0o700)
    store.mkdir(mode=0o755)
    (store / "clips").mkdir(mode=0o755)
    (store / "clips" / ".staging").mkdir(mode=0o755)
    maintenance.mkdir(mode=0o700)
    database = state / "edge.sqlite3"
    migrate_database(database)
    return database, store, maintenance, maintenance / "quiescence.json"


def _quiescence(path: Path, database: Path, store: Path) -> None:
    now = int(time.time())
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "state_db": str(database.resolve()),
                "clip_store": str(store.resolve()),
                "stopped_service": "ml-worker",
                "stopped_db_writers": ["event", "config", "fault"],
                "operator_uid": os.getuid(),
                "issued_at": now - 1,
                "expires_at": now + 60,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _request(
    database: Path, store: Path, maintenance: Path, receipt: Path, *, apply: bool = False
) -> RepairRequest:
    return RepairRequest(store, maintenance, receipt, apply=apply, database_path=database)


def _ready_clip(store: Path, clip_id: str, *, path: object) -> None:
    directory = store / "clips" / clip_id
    directory.mkdir()
    media = b"verified media"
    (directory / "clip.mp4").write_bytes(media)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 2,
                "state": "READY",
                "clip_id": clip_id,
                "event_refs": ["event:one"],
                "path": path,
                "sha256": sha256(media).hexdigest(),
                "size_bytes": len(media),
                "mime_type": "video/mp4",
                "codec": "h264",
                "duration_ms": 1000,
                "finalized_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("declared_path", [None, "clips/other/clip.mp4"])
def test_finalized_clip_refuses_manifest_that_does_not_name_verified_media(
    tmp_path: Path, declared_path: object
) -> None:
    _, store, _, _ = _layout(tmp_path)
    _ready_clip(store, "clip-one", path=declared_path)

    with pytest.raises(ClipConsistencyError, match="final_invalid"):
        inspect_finalized_clip(store, "clip-one")


def test_apply_refuses_without_quiescence_before_any_mutation(tmp_path: Path) -> None:
    database, store, maintenance, receipt = _layout(tmp_path)
    with pytest.raises(ClipConsistencyError, match="quiescence_invalid"):
        repair_clip_consistency(_request(database, store, maintenance, receipt, apply=True))


def _application_tables(database: Path) -> set[str]:
    with sqlite3.connect(database) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }


def test_dry_run_does_not_mutate_and_invalid_manifest_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, store, maintenance, receipt = _layout(tmp_path)
    monkeypatch.setattr(
        consistency_ops,
        "open_runtime_database",
        lambda path, **_: sqlite3.connect(path, isolation_level=None),
    )
    tables = _application_tables(database)
    assert {"clips", "incidents", "artifacts"} <= tables
    assert tables.isdisjoint({"evidence_events", "evidence_clips", "clip_events"})
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)

    with pytest.raises(ClipConsistencyError, match="schema_drift"):
        repair_clip_consistency(_request(database, store, maintenance, receipt))
    assert not list(maintenance.iterdir())

    bad = store / "clips" / "clip-a"
    bad.mkdir()
    (bad / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ClipConsistencyError, match="final_invalid"):
        inspect_finalized_clip(store, "clip-a")


def test_apply_refuses_after_schema18_retires_evidence_relations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, store, maintenance, receipt = _layout(tmp_path)
    _quiescence(receipt, database, store)
    monkeypatch.setattr(
        consistency_ops,
        "open_runtime_database",
        lambda path, **_: sqlite3.connect(path, isolation_level=None),
    )
    assert writer_for_table("clips") is Writer.API
    assert writer_for_table("incidents") is Writer.API
    assert writer_for_table("artifacts") is Writer.API
    with pytest.raises(ClipConsistencyError, match="schema_drift"):
        repair_clip_consistency(_request(database, store, maintenance, receipt, apply=True))
    assert list(maintenance.glob("clip-consistency-*.json")) == []


def test_apply_writes_no_receipt_when_schema18_retires_evidence_relations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, store, maintenance, receipt = _layout(tmp_path)
    _quiescence(receipt, database, store)
    monkeypatch.setattr(
        consistency_ops,
        "open_runtime_database",
        lambda path, **_: sqlite3.connect(path, isolation_level=None),
    )
    monkeypatch.setattr(consistency_ops, "writer_for_table", lambda _: Writer.API)
    with pytest.raises(ClipConsistencyError, match="schema_drift"):
        repair_clip_consistency(_request(database, store, maintenance, receipt, apply=True))
    with pytest.raises(ClipConsistencyError, match="schema_drift"):
        repair_clip_consistency(_request(database, store, maintenance, receipt, apply=True))
    assert list(maintenance.glob("clip-consistency-*.json")) == []


def test_operator_command_exits_nonzero_on_refusal(tmp_path: Path) -> None:
    command = Path(__file__).parents[1] / "scripts/ops/repair-clip-consistency.py"
    result = subprocess.run(
        [
            sys.executable,
            str(command),
            "--clip-store",
            str(tmp_path / "missing-store"),
            "--maintenance-root",
            str(tmp_path / "missing-maintenance"),
            "--quiescence-receipt",
            str(tmp_path / "missing-receipt"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert '"status": "refused"' in result.stderr
