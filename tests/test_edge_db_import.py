from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from legacy_import_fixtures import legacy_paths

from backend.app.edge_db.importer import (
    ImportProgress,
    import_legacy_databases,
)
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import SchemaV17MigrationError


@pytest.mark.parametrize("worker_version", [6, 7, 8, 9, 10])
def test_import_preserves_owned_data_and_forward_migrates_outbox(
    tmp_path: Path, worker_version: int
) -> None:
    sources = legacy_paths(tmp_path, worker_version=worker_version)
    target = tmp_path / "edge-state" / "edge.sqlite3"

    source_bytes = {
        path: path.read_bytes() for path in (sources.catalog, sources.connection, sources.worker)
    }

    result = import_legacy_databases(target, sources)

    assert result.imported_sources == ("catalog", "connection", "worker")
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    database = sqlite3.connect(target)
    try:
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        salt, password_hash = database.execute(
            "SELECT salt,password_hash FROM credentials"
        ).fetchone()
        assert salt == b"\x00salt\xff"
        assert password_hash == b"\x10hash\x00"
        cameras_json = database.execute("SELECT cameras_json FROM camera_registry").fetchone()[0]
        assert "hub:camera/opaque:%2Fbyte-case" in cameras_json
        assert '"error_class":"conflict"' in cameras_json
        assert database.execute(
            "SELECT legacy_canonical_space_id FROM camera_topology_rooms"
        ).fetchone() == ("space:\u00ff/opaque",)
        assert database.execute(
            "SELECT pause_reason,pending_body FROM edge_topology_sync_state"
        ).fetchone() == ("conflict", b"\x00snapshot\xff")
        assert database.execute(
            "SELECT facility_id,facility_token,client_installation_ref FROM connection_settings"
        ).fetchone() == ("facility:\u00ff", "token\x00opaque", "client:opaque/ref")
        event_columns = {
            str(row[1]) for row in database.execute("PRAGMA table_info(evidence_events)")
        }
        objects = {
            str(row[0])
            for row in database.execute(
                "SELECT name FROM sqlite_schema WHERE type IN ('table','index')"
            )
        }
        assert set(database.execute("SELECT edge_event_id FROM evidence_events")) == {
            ("event:ordinary/%2F",)
        }
        assert "operator_only" not in event_columns
        assert "system_test_runs" not in objects
        assert "evidence_events_operator_claim_idx" not in objects
        assert database.execute("SELECT payload_json FROM config_current").fetchone() == (
            '{"policy":"opaque"}',
        )
        assert database.execute(
            "SELECT camera_id,event_type,provenance_state,provenance_missing_reason,"
            "lifecycle_state,failure_reason FROM evidence_incidents "
            "WHERE edge_event_id='event:ordinary/%2F'"
        ).fetchone() == (
            "camera:opaque",
            "fall",
            "MISSING",
            "LEGACY_PROVENANCE_NOT_RECORDED",
            "FAILED",
            "MISSING",
        )
        assert database.execute(
            "SELECT slot_name,state,reason FROM evidence_artifact_slots "
            "WHERE incident_id='event:ordinary/%2F' ORDER BY slot_name"
        ).fetchall() == [
            ("PRIMARY_CLIP", "UNAVAILABLE", "LEGACY_CLIP_RELATION_NOT_RECORDED"),
            ("SNAPSHOT", "UNAVAILABLE", "LEGACY_SNAPSHOT_NOT_RECORDED"),
        ]
        receipts = database.execute(
            "SELECT source_name,source_schema,source_sha256,row_count "
            "FROM schema_import_sources ORDER BY source_name"
        ).fetchall()
        assert [row[0] for row in receipts] == ["catalog", "connection", "worker"]
        assert [row[1] for row in receipts] == ["3", "connection-v2", str(worker_version)]
        assert all(len(row[2]) == 64 and row[3] > 0 for row in receipts)
    finally:
        database.close()

    backups = tuple((target.parent / "legacy-backups").glob("*.sqlite3"))
    assert len(backups) == 3
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() in path.name for path in backups)


@pytest.mark.parametrize("event_state", ["READY", "STAGED", "IN_FLIGHT"])
def test_import_refuses_undelivered_legacy_evidence_before_schema17(
    tmp_path: Path, event_state: str
) -> None:
    sources = legacy_paths(tmp_path, event_state=event_state)
    target = tmp_path / "edge-state" / "edge.sqlite3"

    with pytest.raises(SchemaV17MigrationError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        import_legacy_databases(target, sources)

    with sqlite3.connect(target) as database:
        assert database.execute("PRAGMA user_version").fetchone() == (16,)


def test_every_receipt_barrier_is_resumable_without_duplication(tmp_path: Path) -> None:
    sources = legacy_paths(tmp_path)
    probe_target = tmp_path / "probe" / "edge.sqlite3"
    barriers: list[tuple[str, str]] = []
    import_legacy_databases(
        probe_target, sources, on_receipt=lambda source, barrier: barriers.append((source, barrier))
    )
    assert barriers

    def interrupt_at(receipt_number: int) -> ImportProgress:
        seen = 0

        def interrupt(_source: str, _barrier: str) -> None:
            nonlocal seen
            seen += 1
            if seen == receipt_number:
                raise InterruptedError

        return interrupt

    for index in range(len(barriers)):
        target = tmp_path / f"resume-{index}" / "edge.sqlite3"

        with pytest.raises(InterruptedError):
            import_legacy_databases(target, sources, on_receipt=interrupt_at(index + 1))
        import_legacy_databases(target, sources)
        import_legacy_databases(target, sources)
        connection = sqlite3.connect(target)
        try:
            assert connection.execute("SELECT count(*) FROM credentials").fetchone() == (1,)
            assert connection.execute("SELECT count(*) FROM connection_settings").fetchone() == (1,)
            assert connection.execute("SELECT count(*) FROM evidence_events").fetchone() == (1,)
            assert connection.execute("SELECT count(*) FROM schema_import_sources").fetchone() == (
                3,
            )
        finally:
            connection.close()


def test_digest_or_count_change_after_receipt_is_refused(tmp_path: Path) -> None:
    sources = legacy_paths(tmp_path)
    target = tmp_path / "edge" / "edge.sqlite3"
    import_legacy_databases(target, sources)
    source = sqlite3.connect(sources.catalog)
    source.execute("UPDATE camera_registry SET registry_version = 10")
    source.commit()
    source.close()

    with pytest.raises(ValueError, match="changed after import receipt"):
        import_legacy_databases(target, sources)


def test_fresh_migration_has_exact_compact_schema(
    tmp_path: Path,
) -> None:
    target = tmp_path / "edge" / "edge.sqlite3"
    migrate_database(target)
    connection = sqlite3.connect(target)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {
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
        }
    finally:
        connection.close()
