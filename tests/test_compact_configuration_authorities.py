from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras.bed_zone_store import BedZoneStore
from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.detection_settings.store import (
    DetectionSettingsStore,
    DomainDetectionSetting,
)
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.shared.dashboard_credentials import DashboardCredentialsStore
from contracts.edge_provisioning_v1 import MachinePrincipal

COMPACT_TABLES = {
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


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return database


def test_configuration_authorities_round_trip_on_only_compact_tables(tmp_path: Path) -> None:
    # Given: a fresh real schema-18 database and every Task 7 authority.
    database = _database(tmp_path)
    credentials = DashboardCredentialsStore(database)
    connection = ConnectionSettingsStore(database)
    cameras = CameraRegistryStore(database)
    bed_zones = BedZoneStore(database)
    runtime = RuntimeSettingsStore(database)
    detection = DetectionSettingsStore(database)

    # When: each authority writes through its real store surface.
    credentials.save(username="operator", password="secret-pass")
    connection.save(
        {
            "facility_code": "NH-1234",
            "client_installation_ref": "install-1",
            "facility_id": "facility-1",
            "facility_token": "token-1",
            "edge_installation_id": "edge-1",
            "enrollment_generation": 1,
        }
    )
    cameras.create_floor(edge_ref="floor-1", name="Floor 1", order_index=1)
    cameras.create_room(edge_ref="room-1", floor_edge_ref="floor-1", name="Room 1")
    cameras.create(
        camera_id="camera-1",
        edge_ref="camera-edge-1",
        room_edge_ref="room-1",
        label="Bed camera",
        rtsp_url="rtsp://camera.invalid/live",
        space_id=None,
        status="offline",
    )
    bed_zones.put(
        "camera-1",
        polygon=[[1, 2], [3, 4], [5, 6]],
        image_width=640,
        image_height=480,
        recognized_at="2026-08-24T00:00:00Z",
    )
    runtime.set_clip_export_enabled(True, expected_version=0)
    detection.replace_all(
        {"fall": DomainDetectionSetting(on=True, mode="always", start=None, end=None)}
    )
    EdgeTopologySyncStateStore(database).ensure_principal(MachinePrincipal("edge-1", 1))

    # Then: state survives reopening and no feature-local table appeared.
    reloaded_credentials = DashboardCredentialsStore(database).load()
    assert reloaded_credentials is not None
    assert reloaded_credentials.username == "operator"
    assert ConnectionSettingsStore(database).load().facility_id == "facility-1"
    assert CameraRegistryStore(database).topology_snapshot().floors[0].rooms[0].name == "Room 1"
    assert BedZoneStore(database).get("camera-1") is not None
    assert RuntimeSettingsStore(database).get().clip_export_enabled is True
    assert DetectionSettingsStore(database).get_all()["fall"].on is True
    with sqlite3.connect(database) as raw:
        tables = {
            str(row[0])
            for row in raw.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert tables == COMPACT_TABLES


def test_store_construction_executes_no_ddl(tmp_path: Path) -> None:
    # Given: a fresh compact database with SQLite authorizer-visible schema.
    database = _database(tmp_path)
    before = database.read_bytes()

    # When: every configuration store is constructed and read.
    assert DashboardCredentialsStore(database).load() is None
    assert ConnectionSettingsStore(database).load().facility_id is None
    assert CameraRegistryStore(database).snapshot() == {"registry_version": 0, "cameras": []}
    assert BedZoneStore(database).get_all() == {}
    assert RuntimeSettingsStore(database).get().version == 0
    assert DetectionSettingsStore(database).get_all() == {}
    EdgeTopologySyncStateStore(database).load()

    # Then: construction/read did not mutate the database file or add schema.
    assert database.read_bytes() == before
