from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from contracts.edge_provisioning_codec import CANONICAL_SCHEMA_VERSION
from contracts.edge_provisioning_models import (
    ContractViolation,
    EnrollmentVerification,
    EnrollmentVerificationResult,
    FacilityIdentity,
    JsonValue,
    MachinePrincipal,
)

_REQUEST_FIELDS: Final = frozenset({"schemaVersion", "facilityCode", "clientInstallationRef"})
_RESPONSE_FIELDS: Final = frozenset(
    {"schemaVersion", "edgeInstallationId", "enrollmentGeneration", "facility", "serverRevision"}
)
_FACILITY_FIELDS: Final = frozenset({"id", "displayName"})
_FACILITY_CODE: Final = re.compile(r"^NH-[0-9A-HJKMNP-TV-Z]{10}$")
_UUID: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def parse_enrollment_verification(value: Mapping[str, JsonValue]) -> EnrollmentVerification:
    _require_fields(value, _REQUEST_FIELDS)
    _require_schema_version(value)
    facility_code = _require_string(value["facilityCode"])
    client_ref = _require_string(value["clientInstallationRef"])
    if _FACILITY_CODE.fullmatch(facility_code) is None or _UUID.fullmatch(client_ref) is None:
        raise ContractViolation("enrollment verification identity is invalid")
    return EnrollmentVerification(facility_code, client_ref)


def parse_enrollment_verification_result(
    value: Mapping[str, JsonValue],
) -> EnrollmentVerificationResult:
    _require_fields(value, _RESPONSE_FIELDS)
    _require_schema_version(value)
    facility = _require_record(value["facility"])
    _require_fields(facility, _FACILITY_FIELDS)
    return EnrollmentVerificationResult(
        principal=MachinePrincipal(
            edge_installation_id=_require_uuid(value["edgeInstallationId"]),
            enrollment_generation=_require_positive(value["enrollmentGeneration"]),
        ),
        facility=FacilityIdentity(
            facility_id=_require_uuid(facility["id"]),
            display_name=_require_string(facility["displayName"]),
        ),
        server_revision=_require_nonnegative(value["serverRevision"]),
    )


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


def _require_string(value: JsonValue) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation("value must be a non-empty string")
    return value


def _require_uuid(value: JsonValue) -> str:
    parsed = _require_string(value)
    if _UUID.fullmatch(parsed) is None:
        raise ContractViolation("value must be a UUID")
    return parsed


def _require_positive(value: JsonValue) -> int:
    if type(value) is not int or value < 1:
        raise ContractViolation("value must be a positive integer")
    return value


def _require_nonnegative(value: JsonValue) -> int:
    if type(value) is not int or value < 0:
        raise ContractViolation("value must be a non-negative integer")
    return value


__all__ = ["parse_enrollment_verification", "parse_enrollment_verification_result"]
