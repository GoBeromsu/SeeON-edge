from __future__ import annotations

from typing import Final

from contracts.edge_provisioning_models import (
    EnrollmentVerification,
    JsonRecord,
    TopologyCamera,
    TopologyConfirmation,
    TopologyFloor,
    TopologyRoom,
    TopologySnapshot,
)

CANONICAL_SCHEMA_VERSION: Final = 1


def serialize_enrollment_verification(
    verification: EnrollmentVerification,
) -> JsonRecord:
    return {
        "schemaVersion": CANONICAL_SCHEMA_VERSION,
        "facilityCode": verification.facility_code,
        "clientInstallationRef": verification.client_installation_ref,
    }


def serialize_topology_snapshot(snapshot: TopologySnapshot) -> JsonRecord:
    return {
        "schemaVersion": CANONICAL_SCHEMA_VERSION,
        "edgeInstallationId": snapshot.principal.edge_installation_id,
        "enrollmentGeneration": snapshot.principal.enrollment_generation,
        "clientRevision": snapshot.client_revision,
        "expectedServerRevision": snapshot.expected_server_revision,
        "floors": [_serialize_floor(floor) for floor in snapshot.floors],
    }


def serialize_topology_confirmation(
    confirmation: TopologyConfirmation,
) -> JsonRecord:
    return {
        "schemaVersion": CANONICAL_SCHEMA_VERSION,
        "confirmationId": confirmation.confirmation_id,
        "digest": confirmation.digest,
        "expectedServerRevision": confirmation.expected_server_revision,
    }


def _serialize_floor(floor: TopologyFloor) -> JsonRecord:
    return {
        "edgeRef": floor.edge_ref,
        "name": floor.name,
        "orderIndex": floor.order_index,
        "rooms": [_serialize_room(room) for room in floor.rooms],
    }


def _serialize_room(room: TopologyRoom) -> JsonRecord:
    return {
        "edgeRef": room.edge_ref,
        "name": room.name,
        "type": room.room_type,
        "capacity": room.capacity,
        "cameras": [_serialize_camera(camera) for camera in room.cameras],
    }


def _serialize_camera(camera: TopologyCamera) -> JsonRecord:
    return {"edgeRef": camera.edge_ref, "label": camera.label}


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "serialize_enrollment_verification",
    "serialize_topology_confirmation",
    "serialize_topology_snapshot",
]
