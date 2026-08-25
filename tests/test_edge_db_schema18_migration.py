from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.compatibility import CANONICAL_MIGRATION_LEDGER
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import MIGRATIONS, SCHEMA_VERSION
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION

SCHEMA17_APPLICATION_TABLE_COUNT = 72
SCHEMA18_APPLICATION_TABLES = (
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
HISTORICAL_LEDGER = (
    (
        1,
        "edge_database_foundation",
        "a4b4147ac858c3bdc9c4438e14b8165258e6d032c93f588aeda0067e4fdb20a5",
    ),
    (
        2,
        "single_edge_application_schema",
        "201bbc542e31350e3fdb76c57972b0d2b2e15aa70a9614ea959e2fb078e6123f",
    ),
    (
        3,
        "initialize_clip_listing_generation",
        "61189f418332f22918587b0f72395caef504b77bb93da08bf3d8a4979f613e08",
    ),
    (
        4,
        "versioned_numeric_detection_policies",
        "50021b0d36d25508cfdc9931f58e5f27a409c7b9b7757f70623dec76bd99dc35",
    ),
    (
        5,
        "applied_runtime_provenance_manifests",
        "57bbf42982b9c02307b87b4fbe04de6d25779c91178c6ca55a30b5ffd8b8ed57",
    ),
    (
        6,
        "bounded_analysis_decision_traces",
        "083dbb6457739d46e36248df9a65d1c505a49596395e0ae27eb1b3e43a306819",
    ),
    (
        7,
        "trace_persistence_integrity_and_bounds",
        "0296bbe4fa10eb324a606051ad57d05cd7415ece47c1643960761ab70e1a670a",
    ),
    (
        8,
        "truthful_trace_component_states",
        "f2190fb59a685aa60a13e439d90787bc489b3a27754168a7dbecdf568456a93d",
    ),
    (
        9,
        "authoritative_central_evidence_records",
        "c698903ad864a78ef134a91084afe9cf91488bf5c53141b72e3c6465305c0319",
    ),
    (
        10,
        "versioned_operator_evidence_reviews",
        "e52017eb2f393d4d654f60f1d1c7ac16bb441e069d9da86ce9265d439ca8ddc0",
    ),
    (
        11,
        "exhaustive_evidence_unavailable_reasons",
        "0b00e127d29bfd60202e96cd242d8926902e6e0f9bd4cec01eaf5b75eacdf257",
    ),
    (
        12,
        "retire_legacy_system_test_operator_state",
        "5bb3aca4d85ef7f0f448747dd6b6903c768d69c9be157da12e26c3dea095ff92",
    ),
    (
        13,
        "canonical_overlay_scenes_and_derivatives",
        "f77b5154932bfb8ecbe0fd1dd63a906c0d6f90e3541c347fc049a37cbc32809d",
    ),
    (
        14,
        "still_video_derivative_lifecycle",
        "dcdf072bda75f38169ca796e9b1d7c66c0c2e8a507b54c1eca590431ba073845",
    ),
    (
        15,
        "internal_replay_qa",
        "c29d47b081fc8920e5b0ca77bff51913cb38d6dd8f353df9b847dccb9d95d375",
    ),
    (
        16,
        "live_runtime_clip_export_settings",
        "a2a515c71e1ca57d62c423ed88c95ddec6a1bd5d4c22c6e02d4494312f4b8270",
    ),
    (
        17,
        "backend_only_application_ownership",
        "5bcd20e69f38bc5af6280099f5d8a7740a44027de3dae392dbaa0135d7935fa1",
    ),
)


def _application_tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM pragma_table_list() "
            "WHERE schema = 'main' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    )


def _strict_flags(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        str(row[0]): int(row[1])
        for row in connection.execute(
            "SELECT name, strict FROM pragma_table_list() "
            "WHERE schema = 'main' AND name NOT LIKE 'sqlite_%'"
        )
    }


def test_schema17_characterization_reports_seventy_two_tables(tmp_path: Path) -> None:
    database = tmp_path / "schema17.sqlite3"

    migrate_database(database, migrations=MIGRATIONS[:17])

    with sqlite3.connect(database) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = _application_tables(connection)
    assert version == 17
    assert len(tables) == SCHEMA17_APPLICATION_TABLE_COUNT


def test_fresh_migration_exposes_schema18_exact_ten_strict_contract(tmp_path: Path) -> None:
    database = tmp_path / "schema18.sqlite3"

    result = migrate_database(database)

    with sqlite3.connect(database) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = _application_tables(connection)
        listed = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM pragma_table_list() WHERE schema = 'main'"
            )
        }
        strict = _strict_flags(connection)
        fk_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        ledger = connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(schema_migrations)")
        }

    assert result.current_version == 18
    assert version == 18
    assert SCHEMA_VERSION == 18
    assert EDGE_DATABASE_SCHEMA_VERSION == 18
    assert tables == SCHEMA18_APPLICATION_TABLES
    assert listed - set(tables) <= {"sqlite_sequence", "sqlite_schema"}
    assert all(strict[name] == 1 for name in tables)
    assert fk_violations == []
    assert [row[0] for row in ledger] == list(range(1, 19))
    assert tuple(ledger[:17]) == HISTORICAL_LEDGER
    assert {
        "source_schema_version",
        "source_db_sha256",
        "reconciliation_sha256",
    } <= columns


def test_compiled_v1_v17_checksum_tuples_remain_byte_identical() -> None:
    compiled = tuple(
        (migration.version, migration.name, migration.checksum) for migration in MIGRATIONS[:17]
    )
    assert compiled == HISTORICAL_LEDGER
    assert CANONICAL_MIGRATION_LEDGER[:17] == HISTORICAL_LEDGER
    assert len(CANONICAL_MIGRATION_LEDGER) == 18
    assert CANONICAL_MIGRATION_LEDGER[17][0] == 18
    assert MIGRATIONS[17].version == 18
    assert MIGRATIONS[17].name


def test_schema18_refuses_v17_in_flight_publish_without_mutating_version(
    tmp_path: Path,
) -> None:
    database = tmp_path / "in-flight.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:17])
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, local_state, publish_state) "
            "VALUES ('clip:inflight', 'VERIFIED', 'IN_FLIGHT')"
        )
        connection.commit()
    before = database.read_bytes()

    with pytest.raises(Exception, match="EDGE_DB_DRAIN_INCOMPLETE"):
        migrate_database(database)

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_schema18_rejects_second_camera_in_one_room(tmp_path: Path) -> None:
    database = tmp_path / "room.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO locations "
            "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
            "created_at, updated_at) VALUES "
            "('floor-1', 'FLOOR', NULL, NULL, 'Floor 1', 0, "
            "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO locations "
            "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
            "created_at, updated_at) VALUES "
            "('room-1', 'ROOM', 'floor-1', 'FLOOR', 'Room 1', 0, "
            "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO cameras ("
            "camera_id, label, rtsp_url, normalized_stream_identity, "
            "room_location_id, room_location_kind, edge_ref, mapping_state, "
            "never_connected, revision, created_at, updated_at"
            ") VALUES ("
            "'cam-1', 'Cam 1', 'rtsp://127.0.0.1/one', 'stream-one', "
            "'room-1', 'ROOM', 'edge-1', 'UNMAPPED', 1, 1, "
            "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO cameras ("
                "camera_id, label, rtsp_url, normalized_stream_identity, "
                "room_location_id, room_location_kind, edge_ref, mapping_state, "
                "never_connected, revision, created_at, updated_at"
                ") VALUES ("
                "'cam-2', 'Cam 2', 'rtsp://127.0.0.1/two', 'stream-two', "
                "'room-1', 'ROOM', 'edge-2', 'UNMAPPED', 1, 1, "
                "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
            )


def test_schema18_rejects_occupied_location_deletion(tmp_path: Path) -> None:
    database = tmp_path / "occupied.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO locations "
            "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
            "created_at, updated_at) VALUES "
            "('floor-1', 'FLOOR', NULL, NULL, 'Floor 1', 0, "
            "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO locations "
            "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
            "created_at, updated_at) VALUES "
            "('room-1', 'ROOM', 'floor-1', 'FLOOR', 'Room 1', 0, "
            "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
        )
        connection.execute(
            "INSERT INTO cameras ("
            "camera_id, label, rtsp_url, normalized_stream_identity, "
            "room_location_id, room_location_kind, edge_ref, mapping_state, "
            "never_connected, revision, created_at, updated_at"
            ") VALUES ("
            "'cam-1', 'Cam 1', 'rtsp://127.0.0.1/one', 'stream-one', "
            "'room-1', 'ROOM', 'edge-1', 'UNMAPPED', 1, 1, "
            "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM locations WHERE location_id = 'room-1' AND kind = 'ROOM'"
            )


def test_schema18_contract_rejects_eleventh_application_table(tmp_path: Path) -> None:
    database = tmp_path / "table11.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE extra_table (id INTEGER PRIMARY KEY) STRICT")
        connection.commit()

    with pytest.raises(Exception, match="application table"):
        open_runtime_database(database, actor=RuntimeActor.API)


def test_historical_checksum_mutation_fails_schema18_runtime(tmp_path: Path) -> None:
    database = tmp_path / "mutated.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum = ? WHERE version = 7",
            ("0" * 64,),
        )
        connection.commit()

    with pytest.raises(Exception, match="ledger"):
        open_runtime_database(database, actor=RuntimeActor.API)


def test_runtime_constructor_remains_ddl_free_on_schema18(tmp_path: Path) -> None:
    database = tmp_path / "runtime.sqlite3"
    migrate_database(database)
    connection = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            connection.execute("CREATE TABLE extra_runtime (id INTEGER PRIMARY KEY)")
    finally:
        connection.close()
