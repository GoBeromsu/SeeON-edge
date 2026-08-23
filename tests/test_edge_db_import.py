from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.importer import (
    ImportProgress,
    LegacyDatabasePaths,
    import_legacy_databases,
)
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import SchemaV17MigrationError


def _catalog(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA user_version = 3;
        CREATE TABLE credentials (
            id INTEGER PRIMARY KEY CHECK (id = 1), username TEXT NOT NULL,
            algorithm TEXT NOT NULL, salt BLOB NOT NULL, password_hash BLOB NOT NULL,
            updated_at TEXT NOT NULL
        ) STRICT;
        CREATE TABLE camera_registry (
            id INTEGER PRIMARY KEY CHECK (id = 1), registry_version INTEGER NOT NULL,
            cameras_json TEXT NOT NULL
        ) STRICT;
        CREATE TABLE camera_topology_floors (
            edge_ref TEXT PRIMARY KEY, name TEXT NOT NULL, order_index INTEGER NOT NULL
        ) STRICT;
        CREATE TABLE camera_topology_rooms (
            edge_ref TEXT PRIMARY KEY, floor_edge_ref TEXT NOT NULL, name TEXT NOT NULL,
            room_type TEXT NOT NULL, capacity INTEGER NOT NULL,
            legacy_canonical_space_id TEXT UNIQUE
        ) STRICT;
        CREATE TABLE edge_topology_sync_state (
            id INTEGER PRIMARY KEY, pause_reason TEXT, pending_body BLOB
        ) STRICT;
        CREATE TABLE edge_topology_confirmation_preview (
            id INTEGER PRIMARY KEY, confirmation_id TEXT NOT NULL, digest TEXT NOT NULL,
            expires_at TEXT NOT NULL, snapshot_id TEXT NOT NULL, client_revision INTEGER NOT NULL,
            server_revision INTEGER NOT NULL, registry_version INTEGER NOT NULL,
            edge_installation_id TEXT NOT NULL, enrollment_generation INTEGER NOT NULL,
            cameras INTEGER NOT NULL, rooms INTEGER NOT NULL, floors INTEGER NOT NULL,
            confirmed INTEGER NOT NULL, terminal_response TEXT
        ) STRICT;
        CREATE TABLE runtime_latency (
            facility_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL
        ) STRICT;
        """
    )
    opaque = "hub:camera/opaque:%2Fbyte-case"
    cameras = json.dumps(
        [{"id": opaque, "status": "error", "error_class": "conflict"}],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    connection.execute(
        "INSERT INTO credentials VALUES (1,?,?,?,?,?)",
        ("operator", "scrypt", b"\x00salt\xff", b"\x10hash\x00", "2026-01-01T00:00:00Z"),
    )
    connection.execute("INSERT INTO camera_registry VALUES (1,9,?)", (cameras,))
    connection.execute(
        "INSERT INTO camera_topology_floors VALUES (?,?,?)", ("floor-edge-ref", "Floor", 0)
    )
    connection.execute(
        "INSERT INTO camera_topology_rooms VALUES (?,?,?,?,?,?)",
        ("room-edge-ref", "floor-edge-ref", "Room", "ROOM", 1, "space:\u00ff/opaque"),
    )
    connection.execute(
        "INSERT INTO edge_topology_sync_state VALUES (1,'conflict',?)", (b"\x00snapshot\xff",)
    )
    connection.execute(
        "INSERT INTO edge_topology_confirmation_preview VALUES "
        "(1,'confirm:opaque','digest','expires','snapshot',1,2,9,'edge',7,1,1,1,1,?)",
        ('{"status":"accepted"}',),
    )
    connection.execute(
        "INSERT INTO runtime_latency VALUES (?,?)", ("facility:opaque", '{"max_sec":1.25}')
    )
    connection.commit()
    connection.close()


def _connection(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE connection_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1), events_url TEXT, config_url TEXT,
            facility_id TEXT, facility_token TEXT, updated_at TEXT, facility_code TEXT,
            client_installation_ref TEXT, edge_installation_id TEXT,
            enrollment_generation INTEGER, enrollment_created_at TEXT,
            enrollment_updated_at TEXT
        ) STRICT;
        CREATE TABLE connection_store_migrations (
            version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL,
            backup_filename TEXT, backup_sha256 TEXT, backup_size_bytes INTEGER
        ) STRICT;
        """
    )
    connection.execute(
        "INSERT INTO connection_settings VALUES (1,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "https://hub.invalid/events",
            "https://hub.invalid/config",
            "facility:\u00ff",
            "token\x00opaque",
            "updated",
            "FC-01",
            "client:opaque/ref",
            "edge:opaque/ref",
            7,
            "created",
            "enrolled",
        ),
    )
    connection.commit()
    connection.close()


def _worker(path: Path, *, version: int, event_state: str = "ACKED") -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE evidence_events (
            edge_event_id TEXT PRIMARY KEY, detected_at TEXT NOT NULL, payload_json TEXT NOT NULL,
            state TEXT NOT NULL, queued_at REAL NOT NULL, next_attempt_at REAL NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0, lease_owner TEXT, lease_expires_at REAL,
            delivery_state TEXT NOT NULL DEFAULT 'PENDING', backend_event_id TEXT,
            last_error_code TEXT
        ) STRICT;
        CREATE TABLE evidence_clips (clip_id TEXT PRIMARY KEY) STRICT;
        CREATE TABLE clip_events (
            clip_id TEXT NOT NULL REFERENCES evidence_clips(clip_id),
            edge_event_id TEXT NOT NULL UNIQUE REFERENCES evidence_events(edge_event_id),
            ordinal INTEGER NOT NULL, PRIMARY KEY (clip_id, ordinal)
        ) STRICT;
        CREATE TABLE config_current (
            id INTEGER PRIMARY KEY, generation INTEGER NOT NULL, config_version INTEGER NOT NULL,
            registry_version INTEGER NOT NULL, payload_json TEXT NOT NULL, saved_at REAL NOT NULL
        ) STRICT;
        CREATE TABLE config_history (
            config_version INTEGER PRIMARY KEY, generation INTEGER NOT NULL,
            registry_version INTEGER NOT NULL, payload_json TEXT NOT NULL, saved_at REAL NOT NULL
        ) STRICT;
        CREATE TABLE faults (
            id INTEGER PRIMARY KEY, pid INTEGER NOT NULL, boot_time_iso TEXT NOT NULL,
            profile TEXT NOT NULL, task TEXT NOT NULL, stage TEXT NOT NULL,
            camera_id TEXT NOT NULL, frame_index INTEGER, pts REAL, frame_shape_json TEXT,
            frame_hash_sha256 TEXT, model_artifact_digest TEXT, invocation_seq INTEGER NOT NULL,
            exception_type TEXT NOT NULL, exception_message TEXT NOT NULL,
            exit_code INTEGER NOT NULL,
            action TEXT NOT NULL, fault_time_iso TEXT NOT NULL
        ) STRICT;
        """
    )
    # Released main retired operator-only state at schema 9. Only 7/8 carry it.
    if version in {7, 8}:
        connection.execute(
            "ALTER TABLE evidence_events ADD COLUMN operator_only INTEGER NOT NULL DEFAULT 0"
        )
    if version == 8:
        connection.execute(
            "CREATE TABLE system_test_runs (validation_run_id TEXT PRIMARY KEY, "
            "edge_event_id TEXT NOT NULL UNIQUE REFERENCES evidence_events(edge_event_id)) STRICT"
        )
    if version >= 10:
        connection.execute("ALTER TABLE evidence_clips ADD COLUMN unavailable_reason_code TEXT")
    connection.execute(f"PRAGMA user_version = {version}")
    columns = (
        "edge_event_id,detected_at,payload_json,state,queued_at,next_attempt_at,attempt_count,"
        "lease_owner,lease_expires_at,delivery_state,backend_event_id,last_error_code"
    )
    lease_owner = "worker:legacy" if event_state == "IN_FLIGHT" else None
    lease_expires_at = 10.0 if event_state == "IN_FLIGHT" else None
    ordinary: tuple[object, ...] = (
        "event:ordinary/%2F",
        "detected",
        '{"camera_id":"camera:opaque","event_type":"fall"}',
        event_state,
        1.0,
        1.0,
        0,
        lease_owner,
        lease_expires_at,
        "PENDING",
        None,
        None,
    )
    if version in {7, 8}:
        columns += ",operator_only"
        ordinary += (0,)
    connection.execute(
        f"INSERT INTO evidence_events ({columns}) VALUES ({','.join('?' for _ in ordinary)})",
        ordinary,
    )
    if version in {7, 8}:
        operator_only_row = (
            "event:system-test/%2F",
            "detected",
            '{"type":"SYSTEM_TEST","validation_run_id":"run:opaque/%2F"}',
            "READY",
            2.0,
            2.0,
            0,
            None,
            None,
            "PENDING",
            None,
            None,
            1,
        )
        connection.execute(
            f"INSERT INTO evidence_events ({columns}) "
            f"VALUES ({','.join('?' for _ in operator_only_row)})",
            operator_only_row,
        )
    connection.execute(
        "INSERT INTO config_current VALUES (1,2,3,9,?,4.0)", ('{"policy":"opaque"}',)
    )
    connection.execute("INSERT INTO config_history VALUES (3,2,9,?,4.0)", ('{"policy":"opaque"}',))
    connection.execute(
        "INSERT INTO faults VALUES (1,1,'boot','cpu','task','stage','camera:opaque',NULL,NULL,NULL,"
        "NULL,NULL,1,'Error','opaque',4,'exit','fault')"
    )
    if version == 8:
        connection.execute(
            "INSERT INTO system_test_runs VALUES (?,?)",
            ("run:opaque/%2F", "event:system-test/%2F"),
        )
    connection.commit()
    connection.close()


def _paths(
    tmp_path: Path, *, worker_version: int = 8, event_state: str = "ACKED"
) -> LegacyDatabasePaths:
    catalog = tmp_path / "api" / "catalog.sqlite3"
    connection = tmp_path / "api" / "connection-settings.sqlite3"
    worker = tmp_path / "worker" / "worker-state.sqlite3"
    for path in (catalog, connection, worker):
        path.parent.mkdir(parents=True, exist_ok=True)
    _catalog(catalog)
    _connection(connection)
    _worker(worker, version=worker_version, event_state=event_state)
    return LegacyDatabasePaths(catalog=catalog, connection=connection, worker=worker)


@pytest.mark.parametrize("worker_version", [6, 7, 8, 9, 10])
def test_import_preserves_owned_data_and_forward_migrates_outbox(
    tmp_path: Path, worker_version: int
) -> None:
    sources = _paths(tmp_path, worker_version=worker_version)
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
        assert {row for row in database.execute("SELECT edge_event_id FROM evidence_events")} == {
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
    sources = _paths(tmp_path, event_state=event_state)
    target = tmp_path / "edge-state" / "edge.sqlite3"

    with pytest.raises(SchemaV17MigrationError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        import_legacy_databases(target, sources)

    with sqlite3.connect(target) as database:
        assert database.execute("PRAGMA user_version").fetchone() == (16,)


def test_every_receipt_barrier_is_resumable_without_duplication(tmp_path: Path) -> None:
    sources = _paths(tmp_path)
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
    sources = _paths(tmp_path)
    target = tmp_path / "edge" / "edge.sqlite3"
    import_legacy_databases(target, sources)
    source = sqlite3.connect(sources.catalog)
    source.execute("UPDATE camera_registry SET registry_version = 10")
    source.commit()
    source.close()

    with pytest.raises(ValueError, match="changed after import receipt"):
        import_legacy_databases(target, sources)


def test_fresh_migration_has_complete_schema_and_runtime_actions_need_no_ddl(
    tmp_path: Path,
) -> None:
    target = tmp_path / "edge" / "edge.sqlite3"
    migrate_database(target)
    connection = sqlite3.connect(target)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "connection_settings",
            "camera_registry",
            "evidence_events",
            "config_current",
            "faults",
        } <= tables
        assert "system_test_runs" not in tables
        assert "operator_only" not in {
            str(row[1]) for row in connection.execute("PRAGMA table_info(evidence_events)")
        }
    finally:
        connection.close()
