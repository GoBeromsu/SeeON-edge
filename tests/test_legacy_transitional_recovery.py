from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.legacy_transitional_recovery import LegacyTransitionalRecovery
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import MIGRATIONS, SchemaV17MigrationError

NOW = "2026-08-22T00:00:00Z"


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:16])
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_events VALUES "
            "('event:one', ?, '{}', 'ACKED', 0, 0, 0, NULL, NULL, 'ACKED', NULL, NULL)",
            (NOW,),
        )
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, local_state, media_relpath) VALUES "
            "('clip:one', 'VERIFIED', 'clips/clip:one/clip.mp4')"
        )
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                provenance_state, provenance_missing_reason, lifecycle_state, failure_reason,
                created_at, updated_at
            ) VALUES ('incident:one', 'event:one', 'camera:one', 'fall', ?,
                      'MISSING', 'LEGACY', 'FAILED', 'MISSING', ?, ?)
            """,
            (NOW, NOW, NOW),
        )
        connection.execute(
            """
            INSERT INTO derivative_jobs (
                incident_id, derivative_kind, request_id, state, created_at, updated_at
            ) VALUES ('incident:one', 'STILL', ?, 'PENDING', ?, ?)
            """,
            ("d" * 64, NOW, NOW),
        )
        connection.execute(
            """
            INSERT INTO evidence_retention_states (clip_id, state, requested_at, updated_at)
            VALUES ('clip:one', 'PENDING', ?, ?)
            """,
            (NOW, NOW),
        )
    return database


def _store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    (store / "clips" / ".staging").mkdir(parents=True)
    return store


def test_transitional_recovery_terminalizes_only_truthful_states_and_unblocks_schema17(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    store = _store(tmp_path)

    with pytest.raises(SchemaV17MigrationError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        migrate_database(database)

    result = LegacyTransitionalRecovery(database, store).run()

    assert (
        result.derivatives_cancelled,
        result.retention_purged,
        result.retention_failed,
    ) == (1, 1, 0)
    assert result.unresolved == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT state, reason FROM derivative_jobs").fetchone() == (
            "CANCELLED",
            "LEGACY_DERIVATIVE_EXECUTOR_RETIRED",
        )
        retention = connection.execute(
            "SELECT state, reason FROM evidence_retention_states"
        ).fetchone()
        assert retention == (
            "PURGED",
            None,
        )
    # A re-run resolves nothing further and leaves the gate clear.
    assert LegacyTransitionalRecovery(database, store).run() == type(result)(
        derivatives_cancelled=0,
        slots_terminalized=0,
        retention_purged=0,
        retention_failed=0,
        unresolved=0,
    )
    assert migrate_database(database).current_version == 17


def test_retention_recovery_never_claims_a_purge_while_media_exists(tmp_path: Path) -> None:
    database = _database(tmp_path)
    store = _store(tmp_path)
    media = store / "clips" / "clip:one" / "clip.mp4"
    media.parent.mkdir()
    media.write_bytes(b"still present")

    result = LegacyTransitionalRecovery(database, store).run()

    assert result.retention_failed == 1
    with sqlite3.connect(database) as connection:
        retention = connection.execute(
            "SELECT state, reason FROM evidence_retention_states"
        ).fetchone()
        assert retention == (
            "FAILED",
            "LEGACY_RETENTION_MEDIA_STILL_PRESENT",
        )


@pytest.mark.parametrize("media_relpath", ["/outside/clip.mp4", "clips/../../outside.mp4"])
def test_retention_recovery_refuses_uncontained_media_paths(
    tmp_path: Path, media_relpath: str
) -> None:
    database = _database(tmp_path)
    store = _store(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE evidence_clips SET media_relpath = ? WHERE clip_id = 'clip:one'",
            (media_relpath,),
        )

    result = LegacyTransitionalRecovery(database, store).run()

    assert (result.retention_purged, result.retention_failed) == (0, 1)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT state, reason FROM evidence_retention_states"
        ).fetchone() == ("FAILED", "LEGACY_RETENTION_MEDIA_PATH_UNVERIFIABLE")
