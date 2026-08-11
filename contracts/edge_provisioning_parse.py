from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from contracts.edge_provisioning_models import (
    ContractViolation,
    JsonValue,
    MachinePrincipal,
    TopologyCamera,
    TopologyConfirmation,
    TopologyFloor,
    TopologyKind,
    TopologyManifestEntry,
    TopologyRoom,
    TopologySnapshot,
)
from contracts.edge_provisioning_validation import (
    require_edge_ref,
    require_fields,
    require_initial_alias_scope,
    require_list,
    require_nonnegative,
    require_positive,
    require_record,
    require_schema_version,
    require_sha256,
    require_string,
    require_uuid,
    require_uuid_v7,
)

_PRINCIPAL_FIELDS: Final = frozenset({"edgeInstallationId", "enrollmentGeneration"})
_SNAPSHOT_FIELDS: Final = frozenset(
    {
        "schemaVersion",
        "edgeInstallationId",
        "enrollmentGeneration",
        "clientRevision",
        "expectedServerRevision",
        "floors",
    }
)
_ROOM_REQUIRED: Final = frozenset({"edgeRef", "name", "cameras"})
_ROOM_OPTIONAL: Final = frozenset({"type", "capacity", "legacyCanonicalSpaceId"})
_CONFIRM_FIELDS: Final = frozenset(
    {"schemaVersion", "confirmationId", "digest", "expectedServerRevision"}
)
_TOPOLOGY_KINDS: Final[dict[str, TopologyKind]] = {
    "FLOOR": "FLOOR",
    "ROOM": "ROOM",
    "CAMERA": "CAMERA",
}
_PARENT_KIND: Final[dict[TopologyKind, TopologyKind | None]] = {
    "FLOOR": None,
    "ROOM": "FLOOR",
    "CAMERA": "ROOM",
}


def parse_machine_principal(value: Mapping[str, JsonValue]) -> MachinePrincipal:
    require_fields(value, _PRINCIPAL_FIELDS)
    return MachinePrincipal(
        edge_installation_id=require_uuid(value["edgeInstallationId"]),
        enrollment_generation=require_positive(value["enrollmentGeneration"]),
    )


def parse_topology_snapshot(value: Mapping[str, JsonValue]) -> TopologySnapshot:
    require_fields(value, _SNAPSHOT_FIELDS)
    require_schema_version(value)
    principal = parse_machine_principal(
        {
            "edgeInstallationId": value["edgeInstallationId"],
            "enrollmentGeneration": value["enrollmentGeneration"],
        }
    )
    floors = tuple(_parse_floor(item) for item in require_list(value["floors"]))
    _reject_duplicate_refs(floors)
    snapshot = TopologySnapshot(
        principal=principal,
        client_revision=require_positive(value["clientRevision"]),
        expected_server_revision=require_nonnegative(value["expectedServerRevision"]),
        floors=floors,
    )
    require_initial_alias_scope(snapshot)
    return snapshot


def parse_topology_confirmation(value: Mapping[str, JsonValue]) -> TopologyConfirmation:
    require_fields(value, _CONFIRM_FIELDS)
    require_schema_version(value)
    return TopologyConfirmation(
        confirmation_id=require_uuid_v7(value["confirmationId"]),
        digest=require_sha256(value["digest"]),
        expected_server_revision=require_nonnegative(value["expectedServerRevision"]),
    )


def parse_topology_manifest(
    value: Sequence[Mapping[str, JsonValue]],
) -> tuple[TopologyManifestEntry, ...]:
    entries = tuple(_parse_manifest_entry(item) for item in value)
    canonical_ids = {entry.canonical_id for entry in entries}
    kind_refs = {(entry.kind, entry.edge_ref) for entry in entries}
    if len(canonical_ids) != len(entries) or len(kind_refs) != len(entries):
        raise ContractViolation("manifest identities must be unique within kind")
    kind_by_id = {entry.canonical_id: entry.kind for entry in entries}
    for entry in entries:
        expected_parent_kind = _PARENT_KIND[entry.kind]
        parent_kind = (
            None
            if entry.parent_canonical_id is None
            else kind_by_id.get(entry.parent_canonical_id)
        )
        if expected_parent_kind is None and entry.parent_canonical_id is not None:
            raise ContractViolation("floors cannot have parents")
        if expected_parent_kind is not None and parent_kind != expected_parent_kind:
            raise ContractViolation("manifest parent relationship is invalid")
    return entries


def _parse_floor(value: JsonValue) -> TopologyFloor:
    record = require_record(value)
    require_fields(record, frozenset({"edgeRef", "name", "orderIndex", "rooms"}))
    return TopologyFloor(
        edge_ref=require_edge_ref(record["edgeRef"]),
        name=require_string(record["name"], 120),
        order_index=require_nonnegative(record["orderIndex"]),
        rooms=tuple(_parse_room(item) for item in require_list(record["rooms"])),
    )


def _parse_room(value: JsonValue) -> TopologyRoom:
    record = require_record(value)
    require_fields(record, _ROOM_REQUIRED, _ROOM_OPTIONAL)
    cameras = tuple(_parse_camera(item) for item in require_list(record["cameras"]))
    if len(cameras) > 1:
        raise ContractViolation("a room can contain at most one camera")
    room_type = require_string(record.get("type", "ROOM"), 4)
    if room_type != "ROOM":
        raise ContractViolation("room type must be ROOM")
    legacy_id = (
        require_uuid(record["legacyCanonicalSpaceId"])
        if "legacyCanonicalSpaceId" in record
        else None
    )
    return TopologyRoom(
        edge_ref=require_edge_ref(record["edgeRef"]),
        name=require_string(record["name"], 120),
        room_type=room_type,
        capacity=require_positive(record.get("capacity", 1)),
        cameras=cameras,
        legacy_canonical_space_id=legacy_id,
    )


def _parse_camera(value: JsonValue) -> TopologyCamera:
    record = require_record(value)
    require_fields(record, frozenset({"edgeRef", "label"}))
    return TopologyCamera(
        edge_ref=require_edge_ref(record["edgeRef"]),
        label=require_string(record["label"], 120),
    )


def _parse_manifest_entry(value: Mapping[str, JsonValue]) -> TopologyManifestEntry:
    require_fields(value, frozenset({"kind", "edgeRef", "canonicalId", "parentCanonicalId"}))
    parent_value = value["parentCanonicalId"]
    parent_id = None if parent_value is None else require_uuid(parent_value)
    raw_kind = require_string(value["kind"], 6)
    try:
        kind = _TOPOLOGY_KINDS[raw_kind]
    except KeyError as error:
        raise ContractViolation("manifest kind is invalid") from error
    return TopologyManifestEntry(
        kind=kind,
        edge_ref=require_edge_ref(value["edgeRef"]),
        canonical_id=require_uuid(value["canonicalId"]),
        parent_canonical_id=parent_id,
    )


def _reject_duplicate_refs(floors: tuple[TopologyFloor, ...]) -> None:
    refs_by_kind = (
        [floor.edge_ref for floor in floors],
        [room.edge_ref for floor in floors for room in floor.rooms],
        [camera.edge_ref for floor in floors for room in floor.rooms for camera in room.cameras],
    )
    if any(len(refs) != len(set(refs)) for refs in refs_by_kind):
        raise ContractViolation("topology references must be unique within entity kind")


__all__ = [
    "parse_machine_principal",
    "parse_topology_confirmation",
    "parse_topology_manifest",
    "parse_topology_snapshot",
]
