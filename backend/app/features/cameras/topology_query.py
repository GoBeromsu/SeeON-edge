from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from contracts.edge_provisioning_models import (
    EdgeErrorCode,
    JsonRecord,
    TopologyCamera,
    TopologyFloor,
    TopologyRoom,
)


@dataclass(frozen=True, slots=True)
class TopologyDirtyMarker:
    registry_version: int
    created_at: str


@dataclass(frozen=True, slots=True)
class RegistryTopologySnapshot:
    registry_version: int
    floors: tuple[TopologyFloor, ...]
    dirty: TopologyDirtyMarker | None
    readiness_error: EdgeErrorCode | None
    unmapped_camera_ids: tuple[str, ...]

    def cloud_topology(self) -> JsonRecord:
        return {
            "floors": [
                {
                    "edgeRef": floor.edge_ref,
                    "name": floor.name,
                    "orderIndex": floor.order_index,
                    "rooms": [
                        _cloud_room(room)
                        for room in sorted(floor.rooms, key=lambda item: item.edge_ref)
                    ],
                }
                for floor in sorted(self.floors, key=lambda item: item.edge_ref)
            ]
        }


def read_topology_snapshot(
    connection: sqlite3.Connection, *, registry_version: int, camera_ids: tuple[str, ...]
) -> RegistryTopologySnapshot:
    bindings = {
        str(row[0]): (str(row[1]), str(row[2]))
        for row in connection.execute(
            "SELECT camera_id, edge_ref, room_edge_ref FROM camera_topology_cameras"
        )
    }
    floors = _floors(connection, bindings, _camera_labels(connection))
    unmapped = tuple(sorted(camera_id for camera_id in camera_ids if camera_id not in bindings))
    dirty_row = connection.execute(
        "SELECT registry_version, created_at FROM topology_dirty WHERE id = 1"
    ).fetchone()
    dirty = (
        None
        if dirty_row is None
        else TopologyDirtyMarker(
            registry_version=int(dirty_row[0]), created_at=str(dirty_row[1])
        )
    )
    readiness = EdgeErrorCode.LEGACY_MAPPING_REQUIRED if unmapped else None
    return RegistryTopologySnapshot(registry_version, floors, dirty, readiness, unmapped)


def _camera_labels(connection: sqlite3.Connection) -> dict[str, str]:
    row = connection.execute("SELECT cameras_json FROM camera_registry WHERE id = 1").fetchone()
    if row is None:
        return {}
    records = json.loads(str(row[0]))
    return {
        str(record["id"]): str(record["label"])
        for record in records
        if isinstance(record, dict)
    }


def _floors(
    connection: sqlite3.Connection,
    bindings: dict[str, tuple[str, str]],
    labels: dict[str, str],
) -> tuple[TopologyFloor, ...]:
    camera_by_room = {
        room_ref: TopologyCamera(edge_ref=edge_ref, label=labels.get(camera_id, ""))
        for camera_id, (edge_ref, room_ref) in bindings.items()
    }
    rooms_by_floor: dict[str, list[TopologyRoom]] = {}
    for row in connection.execute(
        "SELECT edge_ref, floor_edge_ref, name, room_type, capacity, legacy_canonical_space_id "
        "FROM camera_topology_rooms ORDER BY edge_ref"
    ):
        camera = camera_by_room.get(str(row[0]))
        rooms_by_floor.setdefault(str(row[1]), []).append(
            TopologyRoom(
                str(row[0]),
                str(row[2]),
                str(row[3]),
                int(row[4]),
                () if camera is None else (camera,),
                None if row[5] is None else str(row[5]),
            )
        )
    return tuple(
        TopologyFloor(
            str(row[0]),
            str(row[1]),
            int(row[2]),
            tuple(rooms_by_floor.get(str(row[0]), ())),
        )
        for row in connection.execute(
            "SELECT edge_ref, name, order_index FROM camera_topology_floors ORDER BY edge_ref"
        )
    )


def _cloud_room(room: TopologyRoom) -> JsonRecord:
    body: JsonRecord = {
        "edgeRef": room.edge_ref,
        "name": room.name,
        "type": room.room_type,
        "capacity": room.capacity,
        "cameras": [{"edgeRef": camera.edge_ref, "label": camera.label} for camera in room.cameras],
    }
    if room.legacy_canonical_space_id is not None:
        body["legacyCanonicalSpaceId"] = room.legacy_canonical_space_id
    return body


__all__ = ["RegistryTopologySnapshot", "TopologyDirtyMarker", "read_topology_snapshot"]
