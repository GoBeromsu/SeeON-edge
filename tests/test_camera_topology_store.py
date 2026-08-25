from __future__ import annotations

import json
import sqlite3

import pytest

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology import TopologyConflictError, TopologyErrorCode
from contracts.edge_provisioning_models import EdgeErrorCode


@pytest.fixture(autouse=True)
def _migrated_compact_database(tmp_path) -> None:
    migrate_database(tmp_path / "catalog.sqlite3")


def test_topology_identity_survives_rename_and_restart(tmp_path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    store = CameraRegistryStore(path)
    store.create_floor(edge_ref="floor-a", name="First", order_index=1)
    store.create_room(
        edge_ref="room-a",
        floor_edge_ref="floor-a",
        name="101",
        legacy_canonical_space_id="a2222222-2222-4222-8222-222222222222",
    )
    store.create(
        camera_id="camera-local",
        edge_ref="camera-a",
        room_edge_ref="room-a",
        label="Bed camera",
        rtsp_url="rtsp://operator:private@10.0.0.9/live",
        space_id="a2222222-2222-4222-8222-222222222222",
        status="online",
    )

    # When
    store.update_floor("floor-a", name="Renamed floor", order_index=2)
    store.update_room("room-a", name="Renamed room")
    restarted = CameraRegistryStore(path)
    snapshot = restarted.topology_snapshot()

    # Then
    assert snapshot.registry_version == 5
    assert snapshot.dirty is not None
    assert snapshot.dirty.registry_version == 5
    assert snapshot.readiness_error is None
    floor = snapshot.floors[0]
    assert (floor.edge_ref, floor.name, floor.order_index) == ("floor-a", "Renamed floor", 2)
    room = floor.rooms[0]
    assert (room.edge_ref, room.name) == ("room-a", "Renamed room")
    assert room.cameras[0].edge_ref == "camera-a"
    encoded = json.dumps(snapshot.cloud_topology())
    assert "rtsp" not in encoded.lower()
    assert "private" not in encoded
    assert "10.0.0.9" not in encoded


def test_unmapped_camera_stays_local_with_typed_readiness_error(tmp_path) -> None:
    # Given
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    # When
    camera = store.create(
        camera_id="camera-local",
        label="Local only",
        rtsp_url="rtsp://camera/live",
        space_id=None,
        status="offline",
    )
    snapshot = store.topology_snapshot()

    # Then
    assert camera["id"] == "camera-local"
    assert store.get("camera-local") is not None
    assert snapshot.readiness_error is EdgeErrorCode.LEGACY_MAPPING_REQUIRED
    assert snapshot.unmapped_camera_ids == ("camera-local",)


@pytest.mark.parametrize(
    ("operation", "expected_code"),
    [
        ("duplicate_floor", TopologyErrorCode.DUPLICATE_REF),
        ("missing_parent", TopologyErrorCode.MISSING_PARENT),
        ("occupied_room", TopologyErrorCode.ROOM_OCCUPIED),
    ],
)
def test_topology_conflict_rolls_back_version_and_dirty_marker(
    tmp_path, operation: str, expected_code: TopologyErrorCode
) -> None:
    # Given
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create_floor(edge_ref="floor-a", name="First", order_index=1)
    store.create_room(edge_ref="room-a", floor_edge_ref="floor-a", name="101")
    store.create(
        camera_id="camera-a",
        edge_ref="camera-a",
        room_edge_ref="room-a",
        label="A",
        rtsp_url="rtsp://camera-a/live",
        space_id=None,
        status="online",
    )
    before = store.topology_snapshot()

    # When
    with pytest.raises(TopologyConflictError) as exc_info:
        if operation == "duplicate_floor":
            store.create_floor(edge_ref="floor-a", name="Again", order_index=2)
        elif operation == "missing_parent":
            store.create_room(edge_ref="room-b", floor_edge_ref="missing", name="102")
        else:
            store.create(
                camera_id="camera-b",
                edge_ref="camera-b",
                room_edge_ref="room-a",
                label="B",
                rtsp_url="rtsp://camera-b/live",
                space_id=None,
                status="online",
            )

    # Then
    assert exc_info.value.code is expected_code
    after = store.topology_snapshot()
    assert after.registry_version == before.registry_version
    assert after.dirty == before.dirty
    assert store.get("camera-b") is None


def test_legacy_space_id_is_validated_and_never_inferred(tmp_path) -> None:
    # Given
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create_floor(edge_ref="floor-a", name="192.168.0.1", order_index=1)

    # When / Then
    with pytest.raises(TopologyConflictError) as exc_info:
        store.create_room(
            edge_ref="room-a",
            floor_edge_ref="floor-a",
            name="Room 192.168.0.2",
            legacy_canonical_space_id="not a valid id",
        )
    assert exc_info.value.code is TopologyErrorCode.INVALID_LEGACY_SPACE_ID

    store.create_room(edge_ref="room-a", floor_edge_ref="floor-a", name="Room 192.168.0.2")
    assert store.topology_snapshot().floors[0].rooms[0].legacy_canonical_space_id is None


def test_camera_write_and_dirty_marker_are_one_sqlite_transaction(tmp_path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    store = CameraRegistryStore(path)
    store.create_floor(edge_ref="floor-a", name="First", order_index=1)
    store.create_room(edge_ref="room-a", floor_edge_ref="floor-a", name="101")

    # When
    store.create(
        camera_id="camera-a",
        edge_ref="camera-a",
        room_edge_ref="room-a",
        label="A",
        rtsp_url="rtsp://camera-a/live",
        space_id=None,
        status="online",
    )

    # Then
    with sqlite3.connect(path) as connection:
        registry_version, dirty_version = connection.execute(
            "SELECT registry_version,topology_dirty_registry_version FROM edge_site WHERE id=1"
        ).fetchone()
    assert registry_version == dirty_version == 3
