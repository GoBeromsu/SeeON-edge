from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest

from contracts.edge_provisioning_models import JsonRecord, JsonValue
from contracts.edge_provisioning_v1 import (
    ContractViolation,
    parse_error_envelope,
    parse_topology_confirmation,
    parse_topology_success_envelope,
)

SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81a"
CONFIRMATION_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81b"
FLOOR_ID = "f1111111-1111-4111-8111-111111111111"
ROOM_ID = "a2222222-2222-4222-8222-222222222222"
CAMERA_ID = "b3333333-3333-4333-8333-333333333333"


def _record(value: JsonValue) -> JsonRecord:
    assert isinstance(value, dict)
    return value


def _root(body: JsonRecord) -> JsonRecord:
    return body


def _omissions(body: JsonRecord) -> JsonRecord:
    return _record(body["omissions"])


def _counts(body: JsonRecord) -> JsonRecord:
    return _record(_record(body["result"])["floors"])


_TARGETS: Final[dict[str, Callable[[JsonRecord], JsonRecord]]] = {
    "root": _root,
    "omissions": _omissions,
    "counts": _counts,
}


def test_success_parser_accepts_optional_counts_and_transfer_preview() -> None:
    body = _success_body()
    result = _record(body["result"])
    for counts in result.values():
        parsed_counts = _record(counts)
        parsed_counts["reactivated"] = 2
        parsed_counts["deactivated"] = 1
    transfer: JsonRecord = {
        "manifestDigest": "b" * 64,
        "items": _manifest(),
    }
    body["ownershipTransferRequired"] = transfer

    parsed = parse_topology_success_envelope(body)

    assert parsed.result.floors.reactivated == 2
    assert parsed.result.rooms.deactivated == 1
    assert parsed.ownership_transfer_required is not None
    assert parsed.ownership_transfer_required.items[2].kind == "CAMERA"


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("NOT_FROZEN", False),
        ("INVALID_SCHEMA", True),
        ("ENROLLMENT_RATE_LIMITED", False),
    ],
)
def test_error_parser_rejects_unknown_or_inconsistent_retryability(
    code: str,
    retryable: bool,
) -> None:
    body: JsonRecord = {
        "schemaVersion": 1,
        "error": {
            "code": code,
            "message": "Frozen error",
            "retryable": retryable,
            "requestId": "request-test-final",
        },
    }

    with pytest.raises(ContractViolation):
        _ = parse_error_envelope(body)


@pytest.mark.parametrize("code", ["ENROLLMENT_RATE_LIMITED", "EDGE_AUTH_NOT_CONFIGURED"])
def test_error_parser_accepts_frozen_retryable_codes(code: str) -> None:
    body: JsonRecord = {
        "schemaVersion": 1,
        "error": {
            "code": code,
            "message": "Retry later",
            "retryable": True,
            "requestId": "request-test-retryable",
        },
    }

    assert parse_error_envelope(body).retryable is True


@pytest.mark.parametrize("state", ["absent", "null"])
def test_success_parser_defaults_optional_fields(state: str) -> None:
    body = _success_body()
    if state == "null":
        body["ownershipTransferRequired"] = None

    parsed = parse_topology_success_envelope(body)

    assert parsed.result.cameras.reactivated == 0
    assert parsed.result.cameras.deactivated == 0
    assert parsed.ownership_transfer_required is None


@pytest.mark.parametrize(
    ("target_name", "field", "value"),
    [
        ("root", "schemaVersion", True),
        ("root", "snapshotId", "not-uuid-v7"),
        ("omissions", "confirmationId", "not-uuid-v7"),
        ("omissions", "digest", "A" * 64),
        ("omissions", "expiresAt", "2026-01-01T00:15:00Z"),
        ("omissions", "cameras", [" invalid "]),
        ("counts", "reactivated", True),
        ("root", "ownershipTransferRequired", {"manifestDigest": "x"}),
    ],
)
def test_success_parser_rejects_invalid_optional_and_identity_formats(
    target_name: str,
    field: str,
    value: JsonValue,
) -> None:
    body = _success_body()
    _TARGETS[target_name](body)[field] = value

    with pytest.raises(ContractViolation):
        _ = parse_topology_success_envelope(body)


@pytest.mark.parametrize(
    "field, value",
    [
        ("schemaVersion", True),
        ("confirmationId", "0197f671-3a31-4a6c-a6e4-83ed412de81b"),
        ("digest", "A" * 64),
        ("expectedServerRevision", True),
    ],
)
def test_confirmation_parser_rejects_invalid_frozen_formats(field: str, value: str | bool) -> None:
    body: JsonRecord = {
        "schemaVersion": 1,
        "confirmationId": CONFIRMATION_ID,
        "digest": "a" * 64,
        "expectedServerRevision": 1,
    }
    body[field] = value

    with pytest.raises(ContractViolation):
        _ = parse_topology_confirmation(body)


def _success_body() -> JsonRecord:
    counts: JsonRecord = {"created": 1, "updated": 0, "unchanged": 0}
    return {
        "schemaVersion": 1,
        "snapshotId": SNAPSHOT_ID,
        "clientRevision": 1,
        "serverRevision": 1,
        "result": {"floors": dict(counts), "rooms": dict(counts), "cameras": dict(counts)},
        "omissions": {
            "confirmationId": CONFIRMATION_ID,
            "digest": "a" * 64,
            "expiresAt": "2026-01-01T00:15:00.000Z",
            "cameras": ["camera-old"],
            "rooms": ["room-old"],
            "floors": ["floor-old"],
        },
    }


def _manifest() -> list[JsonValue]:
    return [
        {"kind": "FLOOR", "edgeRef": "floor-2", "canonicalId": FLOOR_ID, "parentCanonicalId": None},
        {
            "kind": "ROOM",
            "edgeRef": "room-201",
            "canonicalId": ROOM_ID,
            "parentCanonicalId": FLOOR_ID,
        },
        {
            "kind": "CAMERA",
            "edgeRef": "camera-1",
            "canonicalId": CAMERA_ID,
            "parentCanonicalId": ROOM_ID,
        },
    ]
