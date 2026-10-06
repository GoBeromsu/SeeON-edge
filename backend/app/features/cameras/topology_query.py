from __future__ import annotations

from dataclasses import dataclass

from contracts.edge_provisioning_models import (
    EdgeErrorCode,
    JsonRecord,
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


__all__ = ["RegistryTopologySnapshot", "TopologyDirtyMarker"]
