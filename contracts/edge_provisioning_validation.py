from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Final

from contracts.edge_provisioning_models import (
    ContractViolation,
    JsonValue,
    TopologySnapshot,
)

_UUID: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_UUID_V7: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_EDGE_REF: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,63}$")
_SHA256: Final = re.compile(r"^[a-f0-9]{64}$")
_RFC3339_MILLIS_UTC: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)
_EMPTY_FIELDS: Final[frozenset[str]] = frozenset()


def require_fields(
    value: Mapping[str, JsonValue],
    required: frozenset[str],
    optional: frozenset[str] = _EMPTY_FIELDS,
) -> None:
    fields = frozenset(value)
    if not required <= fields or not fields <= required | optional:
        raise ContractViolation("document contains missing or unknown fields")


def require_schema_version(value: Mapping[str, JsonValue]) -> None:
    version = value["schemaVersion"]
    if type(version) is not int or version != 1:
        raise ContractViolation("schemaVersion must be integer 1")


def require_record(value: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise ContractViolation("value must be an object")
    return value


def require_list(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ContractViolation("value must be an array")
    return value


def require_string(value: JsonValue, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ContractViolation("value must be a bounded non-empty string")
    return value


def require_uuid(value: JsonValue) -> str:
    parsed = require_string(value, 36)
    if _UUID.fullmatch(parsed) is None:
        raise ContractViolation("value must be a lowercase UUID")
    return parsed


def require_uuid_v7(value: JsonValue) -> str:
    parsed = require_string(value, 36)
    if _UUID_V7.fullmatch(parsed) is None:
        raise ContractViolation("value must be a lowercase UUIDv7")
    return parsed


def require_edge_ref(value: JsonValue) -> str:
    parsed = require_string(value, 64)
    if _EDGE_REF.fullmatch(parsed) is None:
        raise ContractViolation("value must be an EdgeRef")
    return parsed


def require_sha256(value: JsonValue) -> str:
    parsed = require_string(value, 64)
    if _SHA256.fullmatch(parsed) is None:
        raise ContractViolation("value must be lowercase SHA-256 hex")
    return parsed


def require_rfc3339_millis_utc(value: JsonValue) -> str:
    parsed = require_string(value, 24)
    if _RFC3339_MILLIS_UTC.fullmatch(parsed) is None:
        raise ContractViolation("value must be RFC3339 UTC milliseconds")
    try:
        _ = datetime.strptime(parsed, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as error:
        raise ContractViolation("value must be a valid UTC timestamp") from error
    return parsed


def require_positive(value: JsonValue) -> int:
    if type(value) is not int or value < 1:
        raise ContractViolation("value must be a positive integer")
    return value


def require_nonnegative(value: JsonValue) -> int:
    if type(value) is not int or value < 0:
        raise ContractViolation("value must be a non-negative integer")
    return value


def require_initial_alias_scope(snapshot: TopologySnapshot) -> None:
    aliases_present = any(
        room.legacy_canonical_space_id is not None
        for floor in snapshot.floors
        for room in floor.rooms
    )
    generation = snapshot.principal.enrollment_generation
    revision = snapshot.client_revision
    is_initial = (
        type(generation) is int
        and generation == 1
        and type(revision) is int
        and revision == 1
    )
    if aliases_present and not is_initial:
        raise ContractViolation("legacy aliases require generation 1 client revision 1")


__all__ = [
    "require_edge_ref",
    "require_fields",
    "require_initial_alias_scope",
    "require_list",
    "require_nonnegative",
    "require_positive",
    "require_record",
    "require_rfc3339_millis_utc",
    "require_schema_version",
    "require_sha256",
    "require_string",
    "require_uuid",
    "require_uuid_v7",
]
