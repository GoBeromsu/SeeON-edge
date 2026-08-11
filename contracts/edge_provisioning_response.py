from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from contracts.edge_provisioning_models import (
    ContractViolation,
    EdgeErrorCode,
    ErrorEnvelope,
    JsonValue,
    MutationCounts,
    OmissionPreview,
    OwnershipTransferPreview,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)
from contracts.edge_provisioning_parse import parse_topology_manifest
from contracts.edge_provisioning_validation import (
    require_edge_ref,
    require_fields,
    require_list,
    require_nonnegative,
    require_positive,
    require_record,
    require_rfc3339_millis_utc,
    require_schema_version,
    require_sha256,
    require_string,
    require_uuid_v7,
)

_ERROR_FIELDS: Final = frozenset({"schemaVersion", "error"})
_ERROR_DETAIL_FIELDS: Final = frozenset({"code", "message", "retryable", "requestId"})
_SUCCESS_REQUIRED: Final = frozenset(
    {"schemaVersion", "snapshotId", "clientRevision", "serverRevision", "result", "omissions"}
)
_SUCCESS_OPTIONAL: Final = frozenset({"ownershipTransferRequired"})
_COUNT_REQUIRED: Final = frozenset({"created", "updated", "unchanged"})
_COUNT_OPTIONAL: Final = frozenset({"reactivated", "deactivated"})
_RETRYABLE_CODES: Final[frozenset[EdgeErrorCode]] = frozenset(
    {EdgeErrorCode.ENROLLMENT_RATE_LIMITED, EdgeErrorCode.EDGE_AUTH_NOT_CONFIGURED}
)


def parse_error_envelope(value: Mapping[str, JsonValue]) -> ErrorEnvelope:
    require_fields(value, _ERROR_FIELDS)
    require_schema_version(value)
    error = require_record(value["error"])
    require_fields(error, _ERROR_DETAIL_FIELDS)
    retryable = error["retryable"]
    if type(retryable) is not bool:
        raise ContractViolation("retryable must be a boolean")
    raw_code = require_string(error["code"], 64)
    try:
        code = EdgeErrorCode(raw_code)
    except ValueError as parse_error:
        raise ContractViolation("error code is not frozen") from parse_error
    if retryable != (code in _RETRYABLE_CODES):
        raise ContractViolation("retryable does not match frozen error code")
    return ErrorEnvelope(
        code=code,
        message=require_string(error["message"], 256),
        retryable=retryable,
        request_id=require_string(error["requestId"], 128),
    )


def parse_topology_success_envelope(
    value: Mapping[str, JsonValue],
) -> TopologySuccessEnvelope:
    require_fields(value, _SUCCESS_REQUIRED, _SUCCESS_OPTIONAL)
    require_schema_version(value)
    result = require_record(value["result"])
    require_fields(result, frozenset({"floors", "rooms", "cameras"}))
    omissions_value = value["omissions"]
    omissions = None if omissions_value is None else _parse_omission_preview(omissions_value)
    transfer_value = value.get("ownershipTransferRequired")
    transfer = None if transfer_value is None else _parse_transfer_preview(transfer_value)
    return TopologySuccessEnvelope(
        snapshot_id=require_uuid_v7(value["snapshotId"]),
        client_revision=require_positive(value["clientRevision"]),
        server_revision=require_nonnegative(value["serverRevision"]),
        result=TopologyMutationResult(
            floors=_parse_mutation_counts(result["floors"]),
            rooms=_parse_mutation_counts(result["rooms"]),
            cameras=_parse_mutation_counts(result["cameras"]),
        ),
        omissions=omissions,
        ownership_transfer_required=transfer,
    )


def _parse_mutation_counts(value: JsonValue) -> MutationCounts:
    record = require_record(value)
    require_fields(record, _COUNT_REQUIRED, _COUNT_OPTIONAL)
    return MutationCounts(
        created=require_nonnegative(record["created"]),
        updated=require_nonnegative(record["updated"]),
        unchanged=require_nonnegative(record["unchanged"]),
        reactivated=require_nonnegative(record.get("reactivated", 0)),
        deactivated=require_nonnegative(record.get("deactivated", 0)),
    )


def _parse_omission_preview(value: JsonValue) -> OmissionPreview:
    record = require_record(value)
    require_fields(
        record,
        frozenset({"confirmationId", "digest", "expiresAt", "cameras", "rooms", "floors"}),
    )
    return OmissionPreview(
        confirmation_id=require_uuid_v7(record["confirmationId"]),
        digest=require_sha256(record["digest"]),
        expires_at=require_rfc3339_millis_utc(record["expiresAt"]),
        cameras=_parse_refs(record["cameras"]),
        rooms=_parse_refs(record["rooms"]),
        floors=_parse_refs(record["floors"]),
    )


def _parse_transfer_preview(value: JsonValue) -> OwnershipTransferPreview:
    record = require_record(value)
    require_fields(record, frozenset({"manifestDigest", "items"}))
    items = tuple(require_record(item) for item in require_list(record["items"]))
    return OwnershipTransferPreview(
        manifest_digest=require_sha256(record["manifestDigest"]),
        items=parse_topology_manifest(items),
    )


def _parse_refs(value: JsonValue) -> tuple[str, ...]:
    return tuple(require_edge_ref(item) for item in require_list(value))


__all__ = ["parse_error_envelope", "parse_topology_success_envelope"]
