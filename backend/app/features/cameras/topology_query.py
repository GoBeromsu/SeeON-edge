from __future__ import annotations

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
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute(
            "SELECT camera_id,edge_ref,room_location_id,label FROM cameras "
            "WHERE edge_ref IS NOT NULL"
        )
    }
    cameras_by_room = {
        room_ref: TopologyCamera(edge_ref=edge_ref, label=label)
        for _, (edge_ref, room_ref, label) in bindings.items()
    }
    rooms_by_floor: dict[str, list[TopologyRoom]] = {}
    for row in connection.execute(
        "SELECT location_id,parent_location_id,name,capacity,legacy_space_id "
        "FROM locations WHERE kind='ROOM' ORDER BY location_id"
    ):
        room_ref = str(row[0])
        camera = cameras_by_room.get(room_ref)
        rooms_by_floor.setdefault(str(row[1]), []).append(
            TopologyRoom(
                room_ref,
                str(row[2]),
                "ROOM",
                int(row[3]),
                () if camera is None else (camera,),
                None if row[4] is None else str(row[4]),
            )
        )
    floors = tuple(
        TopologyFloor(
            str(row[0]),
            str(row[1]),
            int(row[2]),
            tuple(rooms_by_floor.get(str(row[0]), ())),
        )
        for row in connection.execute(
            "SELECT location_id,name,order_index FROM locations "
            "WHERE kind='FLOOR' ORDER BY location_id"
        )
    )
    dirty_row = connection.execute(
        "SELECT topology_dirty_registry_version,topology_dirty_created_at FROM edge_site WHERE id=1"
    ).fetchone()
    dirty = None
    if dirty_row is not None and dirty_row[0] is not None:
        dirty = TopologyDirtyMarker(int(dirty_row[0]), str(dirty_row[1]))
    unmapped = tuple(sorted(camera_id for camera_id in camera_ids if camera_id not in bindings))
    readiness = EdgeErrorCode.LEGACY_MAPPING_REQUIRED if unmapped else None
    return RegistryTopologySnapshot(registry_version, floors, dirty, readiness, unmapped)


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
