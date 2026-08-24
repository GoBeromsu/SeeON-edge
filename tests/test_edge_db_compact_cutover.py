from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from compact_cutover_fixtures import TS, cutover_request, sha256

from backend.app.edge_db.compact_cutover import (
    CompactCutoverError,
    CompactCutoverRequest,
    CutoverPhase,
    run_compact_cutover,
)
from backend.app.edge_db.compatibility import MigrationRequiredError
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database


def test_empty_schema17_cutover_is_exact_and_receipted(tmp_path: Path) -> None:
    # Given an immutable 72-table v17 source and byte-identical stopped live clone
    request = cutover_request(tmp_path)
    source_hash = sha256(request.source)

    # When the dedicated compact cutover runs
    result = run_compact_cutover(request)

    # Then source/archive stay exact, every source row has one canonical receipt,
    # and the atomically installed database is the exact schema-18 contract.
    assert sha256(request.source) == source_hash == sha256(request.archive)
    receipt_lines = request.receipt.read_text(encoding="utf-8").splitlines()
    with sqlite3.connect(request.source) as source:
        expected_rows = sum(
            int(source.execute(f'SELECT count(*) FROM "{row[0]}"').fetchone()[0])
            for row in source.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )
    assert len(receipt_lines) == result.source_rows == expected_rows
    assert request.receipt.read_bytes().endswith(b"\n")
    assert all(
        json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        == line
        for line in receipt_lines
    )
    with sqlite3.connect(request.live) as connection:
        tables = tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM pragma_table_list() WHERE schema='main' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        assert tables == (
            "artifacts",
            "audit_events",
            "cameras",
            "clips",
            "credentials",
            "edge_site",
            "incidents",
            "locations",
            "policies",
            "schema_migrations",
        )
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    with pytest.raises(MigrationRequiredError):
        open_runtime_database(request.archive, actor=RuntimeActor.API)


def test_old_sqlite_refuses_before_any_cutover_artifact(tmp_path: Path) -> None:
    # Given a source but an unsafe packaged SQLite version
    request = cutover_request(tmp_path, version=(3, 45, 1))
    before = request.source.read_bytes()

    # When cutover is attempted
    with pytest.raises(Exception, match="below required"):
        run_compact_cutover(request)

    # Then it fails before candidate, archive, or receipt creation.
    assert request.source.read_bytes() == before
    assert not request.archive.exists()
    assert not request.candidate.exists()
    assert not request.receipt.exists()


def test_wrong_source_version_refuses_without_live_replacement(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    with sqlite3.connect(request.source) as connection:
        connection.execute("PRAGMA user_version = 16")
    before = request.live.read_bytes()

    with pytest.raises(CompactCutoverError, match="SOURCE_VERSION"):
        run_compact_cutover(request)

    assert request.live.read_bytes() == before
    assert not request.candidate.exists()


def test_symlink_source_refuses_without_live_replacement(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    real_source = request.source.with_name("real.sqlite3")
    request.source.rename(real_source)
    request.source.symlink_to(real_source)
    before = request.live.read_bytes()

    with pytest.raises(CompactCutoverError, match="SYMLINK"):
        run_compact_cutover(request)

    assert request.live.read_bytes() == before
    assert not request.candidate.exists()


def test_changed_expected_digest_refuses_without_artifacts(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    changed = CompactCutoverRequest(
        source=request.source,
        live=request.live,
        archive=request.archive,
        candidate=request.candidate,
        receipt=request.receipt,
        clip_store=request.clip_store,
        worker_state=request.worker_state,
        expected_source_sha256="0" * 64,
        sqlite_version=request.sqlite_version,
    )

    with pytest.raises(CompactCutoverError, match="SOURCE_CHANGED"):
        run_compact_cutover(changed)

    assert not request.archive.exists()
    assert not request.candidate.exists()


def test_insufficient_space_refuses_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = cutover_request(tmp_path)
    usage = shutil.disk_usage(request.live.parent)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: usage._replace(free=1))

    with pytest.raises(CompactCutoverError, match="INSUFFICIENT_SPACE"):
        run_compact_cutover(request)

    assert not request.archive.exists()
    assert not request.candidate.exists()


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(
            lambda c: c.execute(
                "INSERT INTO evidence_events "
                "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at) "
                "VALUES ('e','2026-08-24T00:00:00Z','{}','STAGED',1,1)"
            ),
            id="event_staged",
        ),
        pytest.param(
            lambda c: c.execute(
                "INSERT INTO evidence_events "
                "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at) "
                "VALUES ('e','2026-08-24T00:00:00Z','{}','READY',1,1)"
            ),
            id="event_ready",
        ),
        pytest.param(
            lambda c: c.execute(
                "INSERT INTO evidence_events "
                "(edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at,"
                "lease_owner,lease_expires_at) VALUES "
                "('e','2026-08-24T00:00:00Z','{}','IN_FLIGHT',1,1,'w',2)"
            ),
            id="event_in_flight",
        ),
        pytest.param(
            lambda c: c.execute(
                "INSERT INTO evidence_clips (clip_id,local_state,publish_state) "
                "VALUES ('c','AWAITING_FINALIZE','WAITING')"
            ),
            id="clip_finalize",
        ),
        pytest.param(
            lambda c: c.execute(
                "INSERT INTO evidence_clips (clip_id,local_state,publish_state) "
                "VALUES ('c','VERIFIED','IN_FLIGHT')"
            ),
            id="clip_publish",
        ),
    ],
)
def test_database_drain_states_refuse_before_candidate(
    tmp_path: Path, seed: Callable[[sqlite3.Connection], None]
) -> None:
    request = cutover_request(tmp_path)
    with sqlite3.connect(request.source) as connection:
        seed(connection)
        connection.commit()
    request.live.write_bytes(request.source.read_bytes())

    with pytest.raises(Exception, match="EDGE_DB_DRAIN_INCOMPLETE"):
        run_compact_cutover(request)

    assert not request.candidate.exists()
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


@pytest.mark.parametrize("subdir", ["delivery-queue", "delivery-queue-dead-letter"])
def test_worker_queue_inventory_refuses_retained_entry(tmp_path: Path, subdir: str) -> None:
    request = cutover_request(tmp_path)
    queue = request.worker_state / subdir
    queue.mkdir()
    (queue / "entry.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CompactCutoverError, match="FILESYSTEM_DRAIN"):
        run_compact_cutover(request)

    assert not request.candidate.exists()


def test_interrupt_before_replace_keeps_live_v17(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    before = request.live.read_bytes()

    def interrupt(phase: CutoverPhase) -> None:
        if phase is CutoverPhase.CANDIDATE_SYNCED:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        run_compact_cutover(request, on_phase=interrupt)

    assert request.live.read_bytes() == before
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)

    resumed = run_compact_cutover(request)
    repeated = run_compact_cutover(request)
    assert resumed.receipt_sha256 == repeated.receipt_sha256
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)


def test_malformed_existing_receipt_is_refused(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    request.receipt.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(CompactCutoverError, match="STALE_RECEIPT"):
        run_compact_cutover(request)

    assert not request.candidate.exists()


def test_manifest_is_rebuilt_with_verified_hashes(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    clip_dir = request.clip_store / "clips" / "clip-1"
    clip_dir.mkdir(parents=True)
    media = clip_dir / "clip.mp4"
    media.write_bytes(b"verified-media")
    manifest = {
        "clip_id": "clip-1",
        "camera_id": "camera-1",
        "event_ref": "event-1",
        "event_type": "fall",
        "started_at": TS,
        "duration_s": 1.25,
        "codec": "h264",
        "path": "clip.mp4",
        "video_available": True,
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )

    run_compact_cutover(request)

    with sqlite3.connect(request.live) as connection:
        row = connection.execute(
            "SELECT manifest_sha256,media_sha256,media_size_bytes FROM clips WHERE clip_id='clip-1'"
        ).fetchone()
    assert row == (
        sha256(clip_dir / "manifest.json"),
        sha256(media),
        len(b"verified-media"),
    )
