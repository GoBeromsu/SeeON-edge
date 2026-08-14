from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from shared.edge_db.compatibility import CURRENT_SCHEMA_RANGE
from shared.edge_db.migrator import migrate_database
from worker.pipeline.output.evidence import clip_consistency_apply as apply_module
from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLock, ClipStoreLockedError
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox
from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode
from worker.pipeline.output.evidence.manifest_models import (
    ReadyClipManifest,
    UnavailableClipManifest,
)

EVENTS = tuple(str(UUID(int=(4 << 76) | (2 << 62) | value)) for value in range(1, 9))
TIMESTAMP = "2026-08-14T00:00:00.000Z"


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    state = tmp_path / "state"
    clips = tmp_path / "clip-store"
    maintenance = tmp_path / "maintenance"
    state.mkdir(mode=0o700)
    (clips / "clips" / ".staging").mkdir(parents=True)
    maintenance.mkdir(mode=0o700)
    database = state / "worker-state.sqlite3"
    with EvidenceOutbox.open(database):
        pass
    return database, clips, maintenance


def _edge_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Central edge.sqlite3 layout used by the newer runtime architecture."""
    state = tmp_path / "edge-state"
    clips = tmp_path / "clip-store"
    maintenance = tmp_path / "maintenance"
    state.mkdir(mode=0o700)
    (clips / "clips" / ".staging").mkdir(parents=True)
    maintenance.mkdir(mode=0o700)
    database = state / "edge.sqlite3"
    migrate_database(database)
    return database, clips, maintenance


def _seed(
    database: Path,
    *,
    clips: tuple[tuple[str, str], ...],
    relations: tuple[tuple[str, str, int], ...] = (),
) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            """
            INSERT INTO evidence_events (
                edge_event_id, detected_at, payload_json, state, queued_at,
                next_attempt_at, delivery_state
            ) VALUES (?, ?, '{}', 'READY', 0, 0, 'PENDING')
            """,
            ((event_id, TIMESTAMP) for event_id in EVENTS),
        )
        connection.executemany(
            "INSERT INTO evidence_clips (clip_id, local_state) VALUES (?, ?)", clips
        )
        connection.executemany(
            "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, ?)",
            relations,
        )


def _unavailable(clip_store: Path, clip_id: str, refs: tuple[str, ...]) -> Path:
    final = clip_store / "clips" / clip_id
    final.mkdir()
    manifest = UnavailableClipManifest(
        clip_id=clip_id,
        camera_id="camera-a",
        event_refs=refs,
        clip_start_at=TIMESTAMP,
        clip_end_at=TIMESTAMP,
        finalized_at=TIMESTAMP,
        reason_code=EvidenceReasonCode.NO_FRAMES,
    )
    path = final / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return path


def _ready(clip_store: Path, clip_id: str, refs: tuple[str, ...], ffprobe: Path) -> Path:
    final = clip_store / "clips" / clip_id
    final.mkdir()
    media = b"\x00\x00\x00\x08moov\x00\x00\x00\x09mdatx"
    (final / "clip.mp4").write_bytes(media)
    manifest = ReadyClipManifest(
        clip_id=clip_id,
        camera_id="camera-a",
        event_refs=refs,
        clip_start_at=TIMESTAMP,
        clip_end_at=TIMESTAMP,
        finalized_at=TIMESTAMP,
        sha256=hashlib.sha256(media).hexdigest(),
        size_bytes=len(media),
        duration_ms=1000,
    )
    path = final / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    ffprobe.write_text(
        "#!/bin/sh\nprintf '%s\\n' "
        "'{\"streams\":[{\"codec_type\":\"video\",\"codec_name\":\"h264\","
        "\"pix_fmt\":\"yuv420p\"}],\"format\":{\"duration\":\"1.0\"}}'\n",
        encoding="utf-8",
    )
    ffprobe.chmod(0o700)
    return path


def _quiescence(path: Path, database: Path, clip_store: Path) -> None:
    now = int(time.time())
    payload = {
        "format_version": 1,
        "state_db": str(database.absolute()),
        "clip_store": str(clip_store.absolute()),
        "stopped_service": "ml-worker",
        "stopped_db_writers": ["event", "config", "fault"],
        "operator_uid": os.getuid(),
        "issued_at": now - 1,
        "expires_at": now + 3599,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)


def _request(
    database: Path,
    clip_store: Path,
    maintenance: Path,
    *,
    apply: bool = False,
    resume: bool = False,
    fault_hook: Any = None,
    ffprobe_bin: str = "ffprobe",
) -> RepairRequest:
    receipt = maintenance / "quiescence.json"
    if not receipt.exists():
        _quiescence(receipt, database, clip_store)
    return RepairRequest(
        state_db=database,
        clip_store=clip_store,
        apply=apply,
        resume=resume,
        maintenance_root=maintenance,
        journal_path=maintenance / "apply.json",
        quiescence_receipt=receipt,
        ffprobe_bin=ffprobe_bin,
        fault_hook=fault_hook,
    )


def _relations(database: Path) -> list[tuple[str, str, int]]:
    with sqlite3.connect(database) as connection:
        return [
            (str(row[0]), str(row[1]), int(row[2]))
            for row in connection.execute(
                "SELECT clip_id, edge_event_id, ordinal FROM clip_events "
                "ORDER BY clip_id, ordinal"
            )
        ]


def test_manifest_authority_repairs_only_relations_and_same_id_staging(
    tmp_path: Path,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    ffprobe = tmp_path / "ffprobe"
    ready_manifest = _ready(clip_store, "clip-ready", EVENTS[:2], ffprobe)
    unavailable_manifest = _unavailable(clip_store, "clip-unavailable", (EVENTS[2],))
    _seed(
        database,
        clips=(
            ("clip-ready", "VERIFIED"),
            ("clip-unavailable", "UNAVAILABLE"),
            ("clip-corrupt", "CORRUPT"),
            ("clip-awaiting", "AWAITING_FINALIZE"),
        ),
        relations=(
            ("clip-ready", EVENTS[1], 0),
            ("clip-ready", EVENTS[0], 1),
            ("clip-ready", EVENTS[3], 2),
            ("clip-awaiting", EVENTS[2], 0),
            ("clip-awaiting", EVENTS[5], 1),
            ("clip-corrupt", EVENTS[4], 0),
        ),
    )
    overlap = clip_store / "clips/.staging/clip-ready"
    overlap.mkdir()
    (overlap / "partial.bin").write_bytes(b"stale")
    unrelated = clip_store / "clips/.staging/unrelated"
    unrelated.mkdir()
    immutable_before = (
        ready_manifest.read_bytes(),
        (clip_store / "clips/clip-ready/clip.mp4").read_bytes(),
        unavailable_manifest.read_bytes(),
    )

    dry = repair_clip_consistency(
        _request(database, clip_store, maintenance, ffprobe_bin=str(ffprobe))
    )

    assert dry.counters.mismatch_clips == 2
    assert dry.counters.mismatch_tuples == 7
    assert dry.counters.sql_relations_deleted == 4
    assert dry.counters.sql_relations_inserted == 3
    assert dry.counters.relations_before == 6
    assert dry.counters.relations_after == 5
    assert overlap.exists()

    applied = repair_clip_consistency(
        _request(database, clip_store, maintenance, apply=True, ffprobe_bin=str(ffprobe))
    )

    assert applied.state == "DONE"
    assert _relations(database) == [
        ("clip-awaiting", EVENTS[5], 1),
        ("clip-corrupt", EVENTS[4], 0),
        ("clip-ready", EVENTS[0], 0),
        ("clip-ready", EVENTS[1], 1),
        ("clip-unavailable", EVENTS[2], 0),
    ]
    assert not overlap.exists() and unrelated.is_dir()
    immutable_after = (
        ready_manifest.read_bytes(),
        (clip_store / "clips/clip-ready/clip.mp4").read_bytes(),
        unavailable_manifest.read_bytes(),
    )
    assert immutable_after == immutable_before
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evidence_events").fetchone()[0] == 8
        assert connection.execute("SELECT COUNT(*) FROM evidence_clips").fetchone()[0] == 4
    assert applied.backup_receipt_path is not None
    backup_payload = json.loads(Path(applied.backup_receipt_path).read_text(encoding="utf-8"))
    backup = Path(backup_payload["backup_path"])
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert backup_payload["source_wal_path"] == f"{database.absolute()}-wal"
    assert backup_payload["backup_state_sha256"] == backup_payload["source_state_sha256"]
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == backup_payload["backup_file_sha256"]

    second = repair_clip_consistency(
        _request(database, clip_store, maintenance, ffprobe_bin=str(ffprobe))
    )
    assert second.counters.changes == 0


def test_central_edge_sqlite_accepts_relation_repair_against_shared_tables(
    tmp_path: Path,
) -> None:
    database, clip_store, maintenance = _edge_layout(tmp_path)
    _unavailable(clip_store, "clip-a", EVENTS[:2])
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )

    dry = repair_clip_consistency(_request(database, clip_store, maintenance))
    assert dry.schema_version == CURRENT_SCHEMA_RANGE.maximum
    assert dry.counters.sql_relations_deleted == 1
    assert dry.counters.sql_relations_inserted == 2

    applied = repair_clip_consistency(
        _request(database, clip_store, maintenance, apply=True)
    )
    assert applied.state == "DONE"
    assert applied.schema_version == CURRENT_SCHEMA_RANGE.maximum
    assert _relations(database) == [
        ("clip-a", EVENTS[0], 0),
        ("clip-a", EVENTS[1], 1),
    ]


def test_wrong_ordinals_report_tuple_mismatch_separately_from_sql_mutations(
    tmp_path: Path,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _unavailable(clip_store, "clip-a", EVENTS[:2])
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[0], 3), ("clip-a", EVENTS[1], 4)),
    )

    dry = repair_clip_consistency(_request(database, clip_store, maintenance))

    assert dry.counters.mismatch_tuples == 4
    assert dry.counters.sql_relations_deleted == 2
    assert dry.counters.sql_relations_inserted == 2


def test_active_lease_and_worker_store_lock_are_refused(tmp_path: Path) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _seed(database, clips=(("clip-awaiting", "AWAITING_FINALIZE"),))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence_events SET state='IN_FLIGHT', lease_owner='sender', "
            "lease_expires_at=? WHERE edge_event_id=?",
            (time.time() + 3600, EVENTS[0]),
        )
    with pytest.raises(ClipConsistencyError, match="active_lease"):
        repair_clip_consistency(_request(database, clip_store, maintenance))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence_events SET state='READY', lease_owner=NULL, lease_expires_at=NULL"
        )
    with ClipStoreLock.acquire(clip_store):
        with pytest.raises(ClipStoreLockedError):
            repair_clip_consistency(
                _request(database, clip_store, maintenance, apply=True)
            )


@pytest.mark.parametrize("fault", ("malformed", "corrupt-ready", "manifest-symlink"))
def test_malformed_corrupt_and_symlinked_finals_are_refused(
    tmp_path: Path, fault: str
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _seed(database, clips=(("clip-bad", "CORRUPT"),))
    final = clip_store / "clips/clip-bad"
    final.mkdir()
    if fault == "malformed":
        (final / "manifest.json").write_text("{broken", encoding="utf-8")
    elif fault == "corrupt-ready":
        final.rmdir()
        ffprobe = tmp_path / "ffprobe"
        _ready(clip_store, "clip-bad", (EVENTS[0],), ffprobe)
        (final / "clip.mp4").write_bytes(b"changed")
    else:
        outside = tmp_path / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        (final / "manifest.json").symlink_to(outside)
    with pytest.raises(ClipConsistencyError, match="final_invalid|unsafe_path"):
        repair_clip_consistency(
            _request(
                database,
                clip_store,
                maintenance,
                ffprobe_bin=str(tmp_path / "ffprobe"),
            )
        )


_PRECOMMIT_STAGES = (
    "backup:file_fsynced",
    "backup:directory_fsynced",
    "backup_receipt:write",
    "backup_receipt:fsync_file",
    "backup_receipt:replace",
    "backup_receipt:fsync_directory",
    "quarantine:rename",
    "quarantine:rename_fsync",
    "journal_prepared:write",
    "journal_prepared:fsync_file",
    "journal_prepared:replace",
    "journal_prepared:fsync_directory",
    "apply:before_relations",
    "apply:before_commit",
)


@pytest.mark.parametrize("stage", _PRECOMMIT_STAGES)
def test_every_precommit_failure_restores_staging_and_rolls_back(
    tmp_path: Path, stage: str
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _unavailable(clip_store, "clip-a", (EVENTS[0],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    staging = clip_store / "clips/.staging/clip-a"
    staging.mkdir()
    before = _relations(database)

    def fail(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=stage):
        repair_clip_consistency(
            _request(database, clip_store, maintenance, apply=True, fault_hook=fail)
        )

    assert staging.is_dir()
    assert _relations(database) == before
    journal = maintenance / "apply.json"
    if journal.exists():
        assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "ABORTED"


_POSTCOMMIT_STAGES = (
    "apply:after_commit",
    "journal_db_committed:write",
    "journal_db_committed:fsync_file",
    "journal_db_committed:replace",
    "journal_db_committed:fsync_directory",
    "quarantine:before_remove",
    "quarantine:after_remove",
    "quarantine:fsync_directory",
    "journal_done:write",
    "journal_done:fsync_file",
    "journal_done:replace",
    "journal_done:fsync_directory",
)


@pytest.mark.parametrize("stage", _POSTCOMMIT_STAGES)
def test_every_postcommit_failure_is_durably_resumable(
    tmp_path: Path, stage: str
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _unavailable(clip_store, "clip-a", (EVENTS[0],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    staging = clip_store / "clips/.staging/clip-a"
    staging.mkdir()
    fired = False

    def fail(observed: str) -> None:
        nonlocal fired
        if observed == stage and not fired:
            fired = True
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=stage):
        repair_clip_consistency(
            _request(database, clip_store, maintenance, apply=True, fault_hook=fail)
        )

    assert _relations(database) == [("clip-a", EVENTS[0], 0)]
    journal_path = maintenance / "apply.json"
    assert journal_path.is_file()
    state = json.loads(journal_path.read_text(encoding="utf-8"))["state"]
    if stage in {
        "apply:after_commit",
        "journal_db_committed:write",
        "journal_db_committed:fsync_file",
    }:
        assert state == "PREPARED"
    elif stage in {
        "journal_db_committed:replace",
        "journal_db_committed:fsync_directory",
        "quarantine:before_remove",
        "quarantine:after_remove",
        "quarantine:fsync_directory",
        "journal_done:write",
        "journal_done:fsync_file",
    }:
        assert state == "DB_COMMITTED"
    else:
        assert state == "DONE"

    resumed = repair_clip_consistency(
        _request(database, clip_store, maintenance, resume=True)
    )
    repeated = repair_clip_consistency(
        _request(database, clip_store, maintenance, resume=True)
    )

    assert resumed.state == repeated.state == "DONE"
    assert not staging.exists()
    assert not tuple((clip_store / "clips/.staging").glob(".clip-consistency-*"))


@pytest.mark.parametrize(
    "stage,expected_state",
    (
        ("journal_prepared:fsync_directory", "PREPARED"),
        ("quarantine:rename", "PREPARED"),
        ("apply:after_commit", "PREPARED"),
        ("journal_db_committed:fsync_directory", "DB_COMMITTED"),
        ("quarantine:after_remove", "DB_COMMITTED"),
        ("journal_done:fsync_directory", "DONE"),
    ),
)
def test_process_crash_at_durable_phase_boundary_resumes_idempotently(
    tmp_path: Path,
    stage: str,
    expected_state: str,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _unavailable(clip_store, "clip-a", (EVENTS[0],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    staging = clip_store / "clips/.staging/clip-a"
    staging.mkdir()
    quiescence = maintenance / "quiescence.json"
    _quiescence(quiescence, database, clip_store)
    journal = maintenance / "apply.json"
    program = """
import os
import sys
from pathlib import Path
from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import RepairRequest

def crash(observed: str) -> None:
    if observed == sys.argv[6]:
        os._exit(91)

repair_clip_consistency(RepairRequest(
    Path(sys.argv[1]), Path(sys.argv[2]), apply=True,
    maintenance_root=Path(sys.argv[3]), journal_path=Path(sys.argv[4]),
    quiescence_receipt=Path(sys.argv[5]), fault_hook=crash,
))
"""

    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(database),
            str(clip_store),
            str(maintenance),
            str(journal),
            str(quiescence),
            stage,
        ],
        cwd=Path(__file__).parents[1],
        check=False,
    )

    assert crashed.returncode == 91
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == expected_state
    resumed = repair_clip_consistency(
        _request(database, clip_store, maintenance, resume=True)
    )
    repeated = repair_clip_consistency(
        _request(database, clip_store, maintenance, resume=True)
    )
    assert resumed.state == repeated.state == "DONE"
    assert _relations(database) == [("clip-a", EVENTS[0], 0)]
    assert not staging.exists()


@pytest.mark.parametrize("writer", ("event", "config", "fault"))
def test_prepared_resume_rejects_any_non_relation_database_write(
    tmp_path: Path,
    writer: str,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _unavailable(clip_store, "clip-a", (EVENTS[0],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    staging = clip_store / "clips/.staging/clip-a"
    staging.mkdir()
    quiescence = maintenance / "quiescence.json"
    _quiescence(quiescence, database, clip_store)
    journal = maintenance / "apply.json"
    _crash_apply(
        database,
        clip_store,
        maintenance,
        journal,
        quiescence,
        "journal_prepared:fsync_directory",
    )
    with sqlite3.connect(database) as connection:
        if writer == "event":
            connection.execute(
                "UPDATE evidence_events SET payload_json = '{\"changed\":true}' "
                "WHERE edge_event_id = ?",
                (EVENTS[0],),
            )
        elif writer == "config":
            connection.execute(
                """INSERT INTO config_current
                   (id, generation, config_version, registry_version, payload_json, saved_at)
                   VALUES (1, 1, 1, 1, '{}', 1)"""
            )
        else:
            connection.execute(
                """INSERT INTO faults (
                    id, pid, boot_time_iso, profile, task, stage, camera_id,
                    frame_index, pts, frame_shape_json, frame_hash_sha256,
                    model_artifact_digest, invocation_seq, exception_type,
                    exception_message, exit_code, action, fault_time_iso
                ) VALUES (
                    1, 1, 'boot', 'cpu', 'task', 'stage', 'camera',
                    NULL, NULL, NULL, NULL, NULL, 1, 'Error', 'changed', 1,
                    'stop', 'fault-time'
                )"""
            )

    with pytest.raises(ClipConsistencyError, match="resume_conflict"):
        repair_clip_consistency(
            _request(database, clip_store, maintenance, resume=True)
        )

    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "PREPARED"
    assert staging.is_dir()
    assert _relations(database) == [("clip-a", EVENTS[1], 0)]


@pytest.mark.parametrize(
    "outcome,expected_state",
    (("commit", "DB_COMMITTED"), ("rollback", "ABORTED"), ("partial", "UNKNOWN")),
)
def test_ambiguous_commit_exception_is_classified_from_fresh_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_state: str,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _unavailable(clip_store, "clip-a", (EVENTS[0],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    staging = clip_store / "clips/.staging/clip-a"
    staging.mkdir()

    def ambiguous_commit(connection: sqlite3.Connection) -> None:
        if outcome == "commit":
            connection.commit()
        elif outcome == "rollback":
            connection.rollback()
        else:
            connection.execute(
                "DELETE FROM clip_events WHERE edge_event_id = ?", (EVENTS[0],)
            )
            connection.commit()
        raise OSError(f"{outcome}-then-raise")

    monkeypatch.setattr(apply_module, "_commit_connection", ambiguous_commit)

    with pytest.raises(OSError, match=f"{outcome}-then-raise"):
        repair_clip_consistency(
            _request(database, clip_store, maintenance, apply=True)
        )

    journal = maintenance / "apply.json"
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == expected_state
    if outcome == "rollback":
        assert staging.is_dir()
        assert _relations(database) == [("clip-a", EVENTS[1], 0)]
    elif outcome == "commit":
        assert not staging.exists()
        assert tuple((clip_store / "clips/.staging").glob(".clip-consistency-*"))
        resumed = repair_clip_consistency(
            _request(database, clip_store, maintenance, resume=True)
        )
        assert resumed.state == "DONE"
        assert _relations(database) == [("clip-a", EVENTS[0], 0)]
    else:
        assert not staging.exists()
        assert tuple((clip_store / "clips/.staging").glob(".clip-consistency-*"))
        with pytest.raises(ClipConsistencyError, match="unknown|corrupt"):
            repair_clip_consistency(
                _request(database, clip_store, maintenance, resume=True)
            )


@pytest.mark.parametrize(
    "tamper",
    ("held-final", "original-final", "changed-id", "missing", "extra", "duplicate"),
)
def test_prepared_journal_rejects_noncanonical_quarantine_authority(
    tmp_path: Path,
    tamper: str,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    manifest = _unavailable(clip_store, "clip-a", (EVENTS[0],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    staging = clip_store / "clips/.staging/clip-a"
    staging.mkdir()
    quiescence = maintenance / "quiescence.json"
    _quiescence(quiescence, database, clip_store)
    journal = maintenance / "apply.json"
    _crash_apply(
        database,
        clip_store,
        maintenance,
        journal,
        quiescence,
        "journal_prepared:fsync_directory",
    )
    payload = json.loads(journal.read_text(encoding="utf-8"))
    row = payload["quarantine"][0]
    if tamper == "held-final":
        row[1] = "clips/clip-a"
    elif tamper == "original-final":
        row[0] = "clips/clip-a"
    elif tamper == "changed-id":
        payload["quarantine_clip_ids"][0] = "clip-other"
        row[0] = "clips/.staging/clip-other"
        row[1] = row[1].replace("clip-a", "clip-other")
    elif tamper == "missing":
        payload["quarantine"] = []
    elif tamper == "extra":
        payload["quarantine"].append(
            [
                "clips/.staging/clip-other",
                "clips/.staging/.clip-consistency-x-clip-other",
            ]
        )
    else:
        payload["quarantine"].append(list(row))
    journal.write_text(json.dumps(payload), encoding="utf-8")
    journal.chmod(0o600)
    manifest_before = manifest.read_bytes()

    with pytest.raises(ClipConsistencyError, match="journal_invalid"):
        repair_clip_consistency(
            _request(database, clip_store, maintenance, resume=True)
        )

    assert manifest.read_bytes() == manifest_before
    assert staging.is_dir()


def test_prepared_journal_rejects_reordered_quarantine_set(tmp_path: Path) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _unavailable(clip_store, "clip-a", (EVENTS[0],))
    _unavailable(clip_store, "clip-b", (EVENTS[2],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"), ("clip-b", "UNAVAILABLE")),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    (clip_store / "clips/.staging/clip-a").mkdir()
    (clip_store / "clips/.staging/clip-b").mkdir()
    quiescence = maintenance / "quiescence.json"
    _quiescence(quiescence, database, clip_store)
    journal = maintenance / "apply.json"
    _crash_apply(
        database,
        clip_store,
        maintenance,
        journal,
        quiescence,
        "journal_prepared:fsync_directory",
    )
    payload = json.loads(journal.read_text(encoding="utf-8"))
    payload["quarantine"].reverse()
    journal.write_text(json.dumps(payload), encoding="utf-8")
    journal.chmod(0o600)

    with pytest.raises(ClipConsistencyError, match="journal_invalid"):
        repair_clip_consistency(
            _request(database, clip_store, maintenance, resume=True)
        )


@pytest.mark.parametrize(
    "outcome,expected_state",
    (("commit", "DB_COMMITTED"), ("rollback", "ABORTED"), ("partial", "UNKNOWN")),
)
def test_prepared_resume_classifies_ambiguous_commit_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_state: str,
) -> None:
    database, clip_store, maintenance = _layout(tmp_path)
    _unavailable(clip_store, "clip-a", (EVENTS[0],))
    _seed(
        database,
        clips=(("clip-a", "UNAVAILABLE"),),
        relations=(("clip-a", EVENTS[1], 0),),
    )
    staging = clip_store / "clips/.staging/clip-a"
    staging.mkdir()
    quiescence = maintenance / "quiescence.json"
    _quiescence(quiescence, database, clip_store)
    journal = maintenance / "apply.json"
    _crash_apply(
        database,
        clip_store,
        maintenance,
        journal,
        quiescence,
        "journal_prepared:fsync_directory",
    )

    def ambiguous_commit(connection: sqlite3.Connection) -> None:
        if outcome == "commit":
            connection.commit()
        elif outcome == "rollback":
            connection.rollback()
        else:
            connection.execute(
                "DELETE FROM clip_events WHERE edge_event_id = ?", (EVENTS[0],)
            )
            connection.commit()
        raise OSError(f"resume-{outcome}-then-raise")

    monkeypatch.setattr(apply_module, "_commit_connection", ambiguous_commit)

    with pytest.raises(OSError, match=f"resume-{outcome}-then-raise"):
        repair_clip_consistency(
            _request(database, clip_store, maintenance, resume=True)
        )

    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == expected_state
    if outcome == "rollback":
        assert staging.is_dir()
        assert _relations(database) == [("clip-a", EVENTS[1], 0)]
    else:
        assert not staging.exists()
        assert tuple((clip_store / "clips/.staging").glob(".clip-consistency-*"))


def _crash_apply(
    database: Path,
    clip_store: Path,
    maintenance: Path,
    journal: Path,
    quiescence: Path,
    stage: str,
) -> None:
    program = """
import os
import sys
from pathlib import Path
from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import RepairRequest

def crash(observed: str) -> None:
    if observed == sys.argv[6]:
        os._exit(91)

repair_clip_consistency(RepairRequest(
    Path(sys.argv[1]), Path(sys.argv[2]), apply=True,
    maintenance_root=Path(sys.argv[3]), journal_path=Path(sys.argv[4]),
    quiescence_receipt=Path(sys.argv[5]), fault_hook=crash,
))
"""
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(database),
            str(clip_store),
            str(maintenance),
            str(journal),
            str(quiescence),
            stage,
        ],
        cwd=Path(__file__).parents[1],
        check=False,
    )
    assert crashed.returncode == 91
