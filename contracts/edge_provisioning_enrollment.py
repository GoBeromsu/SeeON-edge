from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from contracts.edge_provisioning_models import (
    ContractViolation,
    EnrollmentVerification,
    EnrollmentVerificationResult,
    FacilityIdentity,
    JsonValue,
    MachinePrincipal,
)
from contracts.edge_provisioning_validation import (
    require_canonical_id,
    require_fields,
    require_nonnegative,
    require_positive,
    require_record,
    require_schema_version,
    require_string,
    require_uuid,
)

_REQUEST_FIELDS: Final = frozenset({"schemaVersion", "facilityCode", "clientInstallationRef"})
_RESPONSE_FIELDS: Final = frozenset(
    {"schemaVersion", "edgeInstallationId", "enrollmentGeneration", "facility", "serverRevision"}
)
_FACILITY_FIELDS: Final = frozenset({"id", "displayName"})
_FACILITY_CODE: Final = re.compile(r"^NH-[0-9A-HJKMNP-TV-Z]{10}$")


def parse_enrollment_verification(value: Mapping[str, JsonValue]) -> EnrollmentVerification:
    require_fields(value, _REQUEST_FIELDS)
    require_schema_version(value)
    facility_code = require_string(value["facilityCode"], 13)
    client_ref = require_uuid(value["clientInstallationRef"])
    if _FACILITY_CODE.fullmatch(facility_code) is None:
        raise ContractViolation("enrollment verification identity is invalid")
    return EnrollmentVerification(facility_code, client_ref)


def parse_enrollment_verification_result(
    value: Mapping[str, JsonValue],
) -> EnrollmentVerificationResult:
    require_fields(value, _RESPONSE_FIELDS)
    require_schema_version(value)
    facility = require_record(value["facility"])
    require_fields(facility, _FACILITY_FIELDS)
    return EnrollmentVerificationResult(
        principal=MachinePrincipal(
            edge_installation_id=require_uuid(value["edgeInstallationId"]),
            enrollment_generation=require_positive(value["enrollmentGeneration"]),
        ),
        facility=FacilityIdentity(
            facility_id=require_canonical_id(facility["id"]),
            display_name=require_string(facility["displayName"], 120),
        ),
        server_revision=require_nonnegative(value["serverRevision"]),
    )


__all__ = ["parse_enrollment_verification", "parse_enrollment_verification_result"]
