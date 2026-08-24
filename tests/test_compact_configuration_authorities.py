from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.configuration import open_configuration_database
from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras.bed_zone_store import BedZoneStore
from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_confirmation_state import TopologyConfirmationStore
from backend.app.features.clips.storage_location_store import ClipStorageLocationStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.detection_settings.policy_store import DetectionPolicyStore
from backend.app.features.detection_settings.store import (
    DetectionSettingsStore,
    DomainDetectionSetting,
)
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.shared.dashboard_credentials import DashboardCredentialsStore
from contracts.edge_provisioning_v1 import MachinePrincipal

CAMERA_STORE_EXPORTS = [
    "CameraRegistryStore",
    "CameraStatus",
    "DEFAULT_FLOOR",
    "DuplicateCameraError",
    "FLOOR_MAX",
    "FLOOR_MIN",
    "FLOOR_VALUES",
    "ProbeResult",
    "floor_label",
    "is_valid_floor",
    "mask_rtsp_url",
    "normalize_stream_identity",
    "parse_legacy_floor",
    "public_camera",
    "registry_expected_cameras",
    "status_from_probe",
    "utc_now_iso",
]

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


def test_camera_store_preserves_curated_public_exports() -> None:
    module = importlib.import_module("backend.app.features.cameras.store")
    namespace: dict[str, object] = {}

    exec("from backend.app.features.cameras.store import *", {}, namespace)

    assert module.__all__ == CAMERA_STORE_EXPORTS
    assert list(namespace) == CAMERA_STORE_EXPORTS
    assert all(namespace[name] is getattr(module, name) for name in CAMERA_STORE_EXPORTS)


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


def test_store_construction_executes_no_ddl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a migrated schema-18 DB and a direct DDL-denying capture seam.
    database = _database(tmp_path)
    ddl_attempts: list[tuple[int, str | None]] = []
    connections: list[sqlite3.Connection] = []
    ddl_actions = {
        value
        for name in (
            "SQLITE_ALTER_TABLE",
            "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TABLE",
            "SQLITE_CREATE_TRIGGER",
            "SQLITE_CREATE_VIEW",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_DROP_VIEW",
        )
        if (value := getattr(sqlite3, name, None)) is not None
    }

    def captured_open(path: Path) -> sqlite3.Connection:
        connection = open_configuration_database(path)

        def authorize(
            action: int,
            argument_one: str | None,
            _argument_two: str | None,
            _database_name: str | None,
            _source: str | None,
        ) -> int:
            if action in ddl_actions:
                ddl_attempts.append((action, argument_one))
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        connections.append(connection)
        return connection

    for module_name in (
        "backend.app.shared.dashboard_credentials",
        "backend.app.features.connection.sqlite_store",
        "backend.app.features.cameras.store",
        "backend.app.features.cameras.bed_zone_store",
        "backend.app.features.cameras.edge_topology_sync_state",
        "backend.app.features.cameras.topology_confirmation_state",
        "backend.app.features.clips.storage_location_store",
        "backend.app.features.runtime_settings.store",
        "backend.app.features.detection_settings.store",
        "backend.app.features.detection_settings.policy_store",
    ):
        monkeypatch.setattr(
            importlib.import_module(module_name),
            "open_configuration_database",
            captured_open,
        )

    # Mutation proof: this seam detects and rejects a constructor-style CREATE.
    mutation_connection = captured_open(database)
    with pytest.raises(sqlite3.DatabaseError):
        mutation_connection.execute("CREATE TABLE forbidden_authority(id INTEGER)")
    assert ddl_attempts and ddl_attempts[-1][1] == "forbidden_authority"
    mutation_connection.close()
    ddl_attempts.clear()

    # When: every Task 7 authority is constructed and read.
    assert DashboardCredentialsStore(database).load() is None
    assert ConnectionSettingsStore(database).load().facility_id is None
    assert CameraRegistryStore(database).snapshot() == {"registry_version": 0, "cameras": []}
    assert BedZoneStore(database).get_all() == {}
    assert RuntimeSettingsStore(database).get().version == 0
    assert DetectionSettingsStore(database).get_all() == {}
    assert DetectionPolicyStore(database).generation(None) == 0
    assert TopologyConfirmationStore(database).load() is None
    assert ClipStorageLocationStore(database).get() == ""
    assert EdgeTopologySyncStateStore(database).load().principal is None

    # Then: no constructor/read attempted DDL. Monkeypatch restores the real seam.
    assert ddl_attempts == []
    for connection in connections:
        connection.close()
