from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from contracts.edge_provisioning_codec import CANONICAL_SCHEMA_VERSION
from contracts.edge_provisioning_models import (
    ContractViolation,
    ErrorEnvelope,
    JsonValue,
    MachinePrincipal,
    MutationCounts,
    OmissionPreview,
    TopologyCamera,
    TopologyConfirmation,
    TopologyFloor,
    TopologyKind,
    TopologyManifestEntry,
    TopologyMutationResult,
    TopologyRoom,
    TopologySnapshot,
    TopologySuccessEnvelope,
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
_CONFIRM_FIELDS: Final = frozenset(
    {"schemaVersion", "confirmationId", "digest", "expectedServerRevision"}
)
_ERROR_FIELDS: Final = frozenset({"schemaVersion", "error"})
_ERROR_DETAIL_FIELDS: Final = frozenset({"code", "message", "retryable", "requestId"})
_SUCCESS_FIELDS: Final = frozenset(
    {"schemaVersion", "snapshotId", "clientRevision", "serverRevision", "result", "omissions"}
)
_TOPOLOGY_KINDS: Final[dict[str, TopologyKind]] = {
    "FLOOR": "FLOOR",
    "ROOM": "ROOM",
    "CAMERA": "CAMERA",
}


def parse_machine_principal(value: Mapping[str, JsonValue]) -> MachinePrincipal:
    _require_fields(value, _PRINCIPAL_FIELDS)
    return MachinePrincipal(
        edge_installation_id=_require_string(value["edgeInstallationId"]),
        enrollment_generation=_require_positive(value["enrollmentGeneration"]),
    )


def parse_topology_snapshot(value: Mapping[str, JsonValue]) -> TopologySnapshot:
    _require_fields(value, _SNAPSHOT_FIELDS)
    _require_schema_version(value)
    principal = parse_machine_principal(
        {
            "edgeInstallationId": value["edgeInstallationId"],
            "enrollmentGeneration": value["enrollmentGeneration"],
        }
    )
    floors = tuple(_parse_floor(item) for item in _require_list(value["floors"]))
    _reject_duplicate_refs(floors)
    return TopologySnapshot(
        principal=principal,
        client_revision=_require_positive(value["clientRevision"]),
        expected_server_revision=_require_nonnegative(value["expectedServerRevision"]),
        floors=floors,
    )


def parse_topology_confirmation(value: Mapping[str, JsonValue]) -> TopologyConfirmation:
    _require_fields(value, _CONFIRM_FIELDS)
    _require_schema_version(value)
    digest = _require_string(value["digest"])
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractViolation("digest must be a lowercase SHA-256 hex string")
    return TopologyConfirmation(
        confirmation_id=_require_string(value["confirmationId"]),
        digest=digest,
        expected_server_revision=_require_nonnegative(value["expectedServerRevision"]),
    )


def parse_error_envelope(value: Mapping[str, JsonValue]) -> ErrorEnvelope:
    _require_fields(value, _ERROR_FIELDS)
    _require_schema_version(value)
    error = _require_record(value["error"])
    _require_fields(error, _ERROR_DETAIL_FIELDS)
    retryable = error["retryable"]
    if type(retryable) is not bool:
        raise ContractViolation("retryable must be a boolean")
    return ErrorEnvelope(
        code=_require_string(error["code"]),
        message=_require_string(error["message"]),
        retryable=retryable,
        request_id=_require_string(error["requestId"]),
    )


def parse_topology_success_envelope(value: Mapping[str, JsonValue]) -> TopologySuccessEnvelope:
    _require_fields(value, _SUCCESS_FIELDS)
    _require_schema_version(value)
    result = _require_record(value["result"])
    _require_fields(result, frozenset({"floors", "rooms", "cameras"}))
    omissions_value = value["omissions"]
    omissions = None if omissions_value is None else _parse_omission_preview(omissions_value)
    return TopologySuccessEnvelope(
        snapshot_id=_require_string(value["snapshotId"]),
        client_revision=_require_positive(value["clientRevision"]),
        server_revision=_require_nonnegative(value["serverRevision"]),
        result=TopologyMutationResult(
            floors=_parse_mutation_counts(result["floors"]),
            rooms=_parse_mutation_counts(result["rooms"]),
            cameras=_parse_mutation_counts(result["cameras"]),
        ),
        omissions=omissions,
    )


def parse_topology_manifest(
    value: Sequence[Mapping[str, JsonValue]],
) -> tuple[TopologyManifestEntry, ...]:
    entries = tuple(_parse_manifest_entry(item) for item in value)
    canonical_ids = {entry.canonical_id for entry in entries}
    refs = {entry.edge_ref for entry in entries}
    if len(canonical_ids) != len(entries) or len(refs) != len(entries):
        raise ContractViolation("manifest references must be unique")
    for entry in entries:
        if entry.kind == "FLOOR" and entry.parent_canonical_id is not None:
            raise ContractViolation("floors cannot have parents")
        if entry.kind != "FLOOR" and entry.parent_canonical_id not in canonical_ids:
            raise ContractViolation("manifest parent relationship is missing")
    return entries


def _parse_floor(value: JsonValue) -> TopologyFloor:
    record = _require_record(value)
    _require_fields(record, frozenset({"edgeRef", "name", "orderIndex", "rooms"}))
    return TopologyFloor(
        edge_ref=_require_string(record["edgeRef"]),
        name=_require_string(record["name"]),
        order_index=_require_nonnegative(record["orderIndex"]),
        rooms=tuple(_parse_room(item) for item in _require_list(record["rooms"])),
    )


def _parse_room(value: JsonValue) -> TopologyRoom:
    record = _require_record(value)
    _require_fields(record, frozenset({"edgeRef", "name", "type", "capacity", "cameras"}))
    cameras = tuple(_parse_camera(item) for item in _require_list(record["cameras"]))
    if len(cameras) > 1:
        raise ContractViolation("a room can contain at most one camera")
    return TopologyRoom(
        edge_ref=_require_string(record["edgeRef"]),
        name=_require_string(record["name"]),
        room_type=_require_string(record["type"]),
        capacity=_require_positive(record["capacity"]),
        cameras=cameras,
    )


def _parse_camera(value: JsonValue) -> TopologyCamera:
    record = _require_record(value)
    _require_fields(record, frozenset({"edgeRef", "label"}))
    return TopologyCamera(
        edge_ref=_require_string(record["edgeRef"]),
        label=_require_string(record["label"]),
    )


def _parse_manifest_entry(value: Mapping[str, JsonValue]) -> TopologyManifestEntry:
    _require_fields(value, frozenset({"kind", "edgeRef", "canonicalId", "parentCanonicalId"}))
    parent = value["parentCanonicalId"]
    if parent is not None and not isinstance(parent, str):
        raise ContractViolation("parentCanonicalId must be a string or null")
    raw_kind = _require_string(value["kind"])
    try:
        kind = _TOPOLOGY_KINDS[raw_kind]
    except KeyError as error:
        raise ContractViolation("manifest kind is invalid") from error
    return TopologyManifestEntry(
        kind=kind,
        edge_ref=_require_string(value["edgeRef"]),
        canonical_id=_require_string(value["canonicalId"]),
        parent_canonical_id=parent,
    )


def _parse_mutation_counts(value: JsonValue) -> MutationCounts:
    record = _require_record(value)
    _require_fields(record, frozenset({"created", "updated", "unchanged"}))
    return MutationCounts(
        created=_require_nonnegative(record["created"]),
        updated=_require_nonnegative(record["updated"]),
        unchanged=_require_nonnegative(record["unchanged"]),
    )


def _parse_omission_preview(value: JsonValue) -> OmissionPreview:
    record = _require_record(value)
    _require_fields(
        record,
        frozenset({"confirmationId", "digest", "expiresAt", "cameras", "rooms", "floors"}),
    )
    return OmissionPreview(
        confirmation_id=_require_string(record["confirmationId"]),
        digest=_require_string(record["digest"]),
        expires_at=_require_string(record["expiresAt"]),
        cameras=tuple(_require_string(item) for item in _require_list(record["cameras"])),
        rooms=tuple(_require_string(item) for item in _require_list(record["rooms"])),
        floors=tuple(_require_string(item) for item in _require_list(record["floors"])),
    )


def _reject_duplicate_refs(floors: tuple[TopologyFloor, ...]) -> None:
    refs = [floor.edge_ref for floor in floors]
    refs.extend(room.edge_ref for floor in floors for room in floor.rooms)
    refs.extend(
        camera.edge_ref for floor in floors for room in floor.rooms for camera in room.cameras
    )
    if len(refs) != len(set(refs)):
        raise ContractViolation("topology edge references must be unique")


def _require_schema_version(value: Mapping[str, JsonValue]) -> None:
    if value["schemaVersion"] != CANONICAL_SCHEMA_VERSION:
        raise ContractViolation("schemaVersion must be 1")


def _require_fields(value: Mapping[str, JsonValue], expected: frozenset[str]) -> None:
    if frozenset(value) != expected:
        raise ContractViolation("document contains missing or unknown fields")


def _require_record(value: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise ContractViolation("value must be an object")
    return value


def _require_list(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ContractViolation("value must be an array")
    return value


def _require_string(value: JsonValue) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation("value must be a non-empty string")
    return value


def _require_positive(value: JsonValue) -> int:
    if type(value) is not int or value < 1:
        raise ContractViolation("value must be a positive integer")
    return value


def _require_nonnegative(value: JsonValue) -> int:
    if type(value) is not int or value < 0:
        raise ContractViolation("value must be a non-negative integer")
    return value


__all__ = [
    "parse_error_envelope",
    "parse_machine_principal",
    "parse_topology_confirmation",
    "parse_topology_manifest",
    "parse_topology_snapshot",
    "parse_topology_success_envelope",
]
