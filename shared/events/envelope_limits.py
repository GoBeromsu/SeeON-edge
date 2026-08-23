"""Finite limits for durable delivery-queue envelopes."""

from __future__ import annotations

from typing import Final

# Every string is restricted to printable ASCII, so its serialized JSON size is
# exactly its character count.  Binary decision data is base64 encoded.
ENTRY_ID_MAX_CHARS = 128
EDGE_EVENT_ID_MAX_CHARS = 128
EVENT_TYPE_MAX_CHARS = 32
DETECTED_AT_MAX_CHARS = 64
CAMERA_ID_MAX_CHARS = 128
FACILITY_ID_MAX_CHARS = 128
DECISION_TRACE_BYTES_MAX = 16 * 1024
VALUES_BYTES_MAX = 32 * 1024
SNAPSHOT_ID_MAX_CHARS = 128
SHA256_MAX_CHARS = 64
MEDIA_REFERENCE_MAX_CHARS = 1024
MIME_TYPE_MAX_CHARS = 128
SNAPSHOT_SIZE_BYTES_MAX = 9_223_372_036_854_775_807
DISPOSITION_MAX_CHARS = 64
DISPOSITION_REASON_MAX_CHARS = 1024

# ``contracts.relay`` intentionally exposes only the legacy unvalidated
# ``RelayAlertPayload`` mapping, not the relay request schema or its required
# keys.  Importing the Pydantic model from ``backend`` would violate this
# shared-leaf boundary, so the relay-model drift test is the strongest legal
# guard until the frozen contract exports an importable schema.
REQUIRED_ALERT_FIELDS: Final = frozenset(
    {"camera_id", "detected_at", "event_type", "facility_id", "probability"}
)
RELAY_AUDIT_FIELDS: Final = frozenset(
    {
        "clock_source",
        "config_version",
        "decision_trace_id",
        "detector_version",
        "model_version",
        "operating_threshold",
        "runtime_manifest_sha256",
    }
)


def _base64_chars(byte_count: int) -> int:
    return 4 * ((byte_count + 2) // 3)


def maximum_serialized_envelope_bytes() -> int:
    """Return the exact worst-case canonical JSON envelope size.

    This deliberately builds the envelope from the named maxima instead of
    maintaining an unrelated guessed queue-byte limit.  The queue uses compact,
    sorted, ASCII JSON, and all textual inputs are printable ASCII.
    """
    import json

    common = {
        "edge_event_id": "x" * EDGE_EVENT_ID_MAX_CHARS,
        "entry_id": "x" * ENTRY_ID_MAX_CHARS,
    }
    envelopes = (
        {
            **common,
            "camera_id": "x" * CAMERA_ID_MAX_CHARS,
            "decision_trace_b64": "A" * _base64_chars(DECISION_TRACE_BYTES_MAX),
            "detected_at": "x" * DETECTED_AT_MAX_CHARS,
            "event_type": "x" * EVENT_TYPE_MAX_CHARS,
            "facility_id": "x" * FACILITY_ID_MAX_CHARS,
            "kind": "EVENT",
            "values_b64": "A" * _base64_chars(VALUES_BYTES_MAX),
        },
        {
            **common,
            "kind": "SNAPSHOT_ATTACHMENT",
            "media_reference": "x" * MEDIA_REFERENCE_MAX_CHARS,
            "mime_type": "x" * MIME_TYPE_MAX_CHARS,
            "sha256": "x" * SHA256_MAX_CHARS,
            "size_bytes": SNAPSHOT_SIZE_BYTES_MAX,
            "snapshot_id": "x" * SNAPSHOT_ID_MAX_CHARS,
        },
        {
            **common,
            "disposition": "x" * DISPOSITION_MAX_CHARS,
            "kind": "SNAPSHOT_DISPOSITION",
            "reason": "x" * DISPOSITION_REASON_MAX_CHARS,
            "snapshot_id": "x" * SNAPSHOT_ID_MAX_CHARS,
        },
    )
    return max(
        len(
            json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        )
        for envelope in envelopes
    )


__all__ = [name for name in globals() if name.endswith(("_MAX_CHARS", "_MAX", "_MAX_BYTES"))] + [
    "REQUIRED_ALERT_FIELDS",
    "RELAY_AUDIT_FIELDS",
    "maximum_serialized_envelope_bytes",
]
