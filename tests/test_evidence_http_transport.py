from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, tzinfo
from email.utils import format_datetime

import pytest

import shared.events.evidence_http_transport as transport_module
from shared.events.evidence_export_contract import (
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    DeliveryFailureCode,
    EventReceipt,
)
from shared.events.evidence_http_transport import classify_http_failure


@pytest.mark.parametrize("status", (401, 403))
def test_ambient_auth_failures_are_retried_not_dead_lettered(status: int) -> None:
    """401/403 are ambient auth/facility-config state, not a property of the
    payload that was sent -- see #183, #202. The config gets fixed
    out-of-band and the event is still perfectly valid, so it must stay
    retryable rather than being dead-lettered forever."""
    failure = classify_http_failure(status, {})

    assert failure.disposition is DeliveryDisposition.RETRY
    assert failure.code == f"HTTP_{status}"
    assert failure.status_code == status


@pytest.mark.parametrize("status", (400, 413, 415, 422))
def test_payload_specific_failures_remain_permanent(status: int) -> None:
    """400/413/415/422 are genuinely permanent: retrying the exact same
    bytes cannot change a schema, size, or media-type rejection. These stay
    dead-lettered -- only the ambient-auth codes above move."""
    failure = classify_http_failure(status, {})

    assert failure.disposition is DeliveryDisposition.PERMANENT
    assert failure.code == f"HTTP_{status}"


@pytest.mark.parametrize("status", (404, 405))
def test_compatibility_classes_are_unaffected(status: int) -> None:
    failure = classify_http_failure(status, {})

    assert failure.disposition is DeliveryDisposition.COMPATIBILITY


@pytest.mark.parametrize("status", (408, 425, 429, 500, 503, 599))
def test_transient_classes_remain_retry(status: int) -> None:
    failure = classify_http_failure(status, {})

    assert failure.disposition is DeliveryDisposition.RETRY


def test_retry_after_header_is_preserved_for_ambient_auth_failures() -> None:
    failure = classify_http_failure(403, {"Retry-After": "30"})

    assert failure.disposition is DeliveryDisposition.RETRY
    assert failure.retry_after_seconds == 30.0


def test_named_local_accept_is_terminal_and_absent_status_is_not() -> None:
    """A receiptless 2xx wedged the durable queue forever (#431).

    The edge backend deliberately accepts an event locally when the camera has
    no Hub mapping or no cloud client exists. It records the event and will
    never push it upstream, so no upstream id can ever be echoed. The worker
    demanded one, retried indefinitely, and every newer event queued behind the
    oldest undeliverable entry never left the edge.

    Terminal acceptance must be STATED by the party that knows -- the backend --
    never inferred by the worker from a missing field, because an absent id is
    equally consistent with a mangled response from a broken proxy.
    """
    from shared.events.evidence_export_contract import DeliveryFailure, EventReceipt
    from shared.events.evidence_http_transport import parse_event_result

    named = parse_event_result(
        (202, {}, b'{"status": "accepted_local", "edge_event_id": "edge-1"}'), "edge-1"
    )
    assert isinstance(named, EventReceipt), (
        "a named local accept must be terminal; treating it as a failure is the "
        "defect that wedged the queue"
    )
    assert named.status == "accepted_local"
    assert named.edge_event_id == "edge-1"
    assert named.event_id == "", "a local accept has no upstream id to fabricate"

    # The old body, which says nothing about the decision, must NOT become
    # terminal by accident -- that would silently drop genuinely mangled
    # responses instead of retrying them.
    bare = parse_event_result((202, {}, b'{"status": "accepted"}'), "edge-1")
    assert isinstance(bare, DeliveryFailure), "an unnamed 2xx is still malformed"
    assert bare.code == "MALFORMED_RECEIPT"

    # A named local accept for a DIFFERENT event must not satisfy this one.
    wrong = parse_event_result(
        (202, {}, b'{"status": "accepted_local", "edge_event_id": "other"}'), "edge-1"
    )
    assert isinstance(wrong, DeliveryFailure), (
        "a local accept naming another event must not acknowledge this one"
    )


def test_a_terminal_local_accept_requires_that_something_was_persisted() -> None:
    """A terminal receipt tells the worker to DELETE its only other copy.

    On the local-accept path nothing is pushed upstream, so the backend's own
    record is the only copy that will ever exist. Review caught that the first
    version of `accepted_local` was returned even when the projection failed AND
    the catalog fallback failed, which destroyed the alert on both sides at once
    -- a fall event with no trace anywhere.

    This pins the sender half: a 503 must stay retryable so the worker keeps its
    copy. The backend half raises that 503 in `_local_accept_body`.
    """
    from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
    from shared.events.evidence_http_transport import parse_event_result

    refused = parse_event_result(
        (503, {}, b'{"detail": "edge-local persistence failed"}'), "edge-1"
    )
    assert isinstance(refused, DeliveryFailure)
    assert refused.disposition is DeliveryDisposition.RETRY, (
        "a backend that could not persist the alert must not cause the worker "
        "to drop it; the event would then exist nowhere"
    )


@pytest.mark.parametrize("status", (200, 202, 299, 409))
def test_matching_event_receipt_acknowledges_success_or_conflict(status: int) -> None:
    body = json.dumps(
        {"status": "accepted", "edge_event_id": "edge-1", "event_id": "hub-1"}
    ).encode()

    assert transport_module.parse_event_result((status, {}, body), "edge-1") == EventReceipt(
        "accepted", "edge-1", "hub-1"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "accepted", "edge_event_id": "other", "event_id": "hub-1"},
        {"edge_event_id": "edge-1"},
        {"status": "accepted", "edge_event_id": "edge-1", "event_id": ""},
    ],
    ids=("wrong-identity", "missing-status", "empty-hub-id"),
)
@pytest.mark.parametrize("status", (202, 409))
def test_invalid_event_receipt_never_acknowledges(payload: dict[str, str], status: int) -> None:
    result = transport_module.parse_event_result(
        (status, {}, json.dumps(payload).encode()), "edge-1"
    )

    expected = (
        DeliveryFailure(DeliveryDisposition.RETRY, "MALFORMED_RECEIPT")
        if status == 202
        else DeliveryFailure(DeliveryDisposition.PERMANENT, "HTTP_409", status_code=409)
    )
    assert result == expected


@pytest.mark.parametrize("status", (200, 409))
@pytest.mark.parametrize(
    ("state", "version"),
    [("READY", 2), ("UNAVAILABLE", 2), ("EXPIRED", 2), ("EXPIRED", 3)],
)
def test_clip_receipt_requires_matching_version_unless_expired(
    status: int, state: str, version: int
) -> None:
    body = json.dumps(
        {
            "clip_id": "clip-1",
            "state": state,
            "state_version": version,
            "sha256": "a" * 64,
            "size_bytes": 12,
        }
    ).encode()

    assert transport_module.parse_clip_result((status, {}, body), "clip-1", 2) == ClipReceipt(
        "clip-1", state, version, "a" * 64, 12
    )


@pytest.mark.parametrize(
    ("clip_id", "state", "version"),
    [
        ("other", "READY", 2),
        ("clip-1", "READY", 1),
        ("clip-1", "READY", 3),
        ("clip-1", "EXPIRED", 1),
    ],
)
def test_clip_identity_and_version_mismatch_remain_retryable(
    clip_id: str, state: str, version: int
) -> None:
    body = json.dumps({"clip_id": clip_id, "state": state, "state_version": version}).encode()

    assert transport_module.parse_clip_result((200, {}, body), "clip-1", 2) == DeliveryFailure(
        DeliveryDisposition.RETRY, "MALFORMED_RECEIPT"
    )


@pytest.mark.parametrize(
    ("status", "disposition"),
    [
        (199, DeliveryDisposition.PERMANENT),
        (300, DeliveryDisposition.PERMANENT),
        (503, DeliveryDisposition.RETRY),
    ],
)
def test_matching_receipts_do_not_override_unsuccessful_http_status(
    status: int, disposition: DeliveryDisposition
) -> None:
    event = b'{"status":"accepted","edge_event_id":"edge-1","event_id":"hub-1"}'
    clip = b'{"clip_id":"clip-1","state":"READY","state_version":1}'
    expected = DeliveryFailure(disposition, f"HTTP_{status}", status_code=status)

    assert transport_module.parse_event_result((status, {}, event), "edge-1") == expected
    assert transport_module.parse_clip_result((status, {}, clip), "clip-1", 1) == expected


def test_missing_camera_mapping_overrides_payload_rejection() -> None:
    failure = classify_http_failure(422, {}, b'{"detail":{"code":"CAMERA_MAPPING_MISSING"}}')

    assert failure == DeliveryFailure(
        DeliveryDisposition.RETRY, DeliveryFailureCode.CAMERA_MAPPING_MISSING, status_code=422
    )


@pytest.mark.parametrize(
    ("retry_after", "expected_seconds"),
    [
        (None, None),
        ("soon", None),
        ("-5", 0.0),
        ("0", 0.0),
        ("120", 120.0),
        ("900", 900.0),
        ("901", 900.0),
    ],
)
def test_retry_after_seconds_are_bounded(
    retry_after: str | None, expected_seconds: float | None
) -> None:
    headers = {} if retry_after is None else {"retry-after": retry_after}
    failure = classify_http_failure(503, headers)

    assert failure.disposition is DeliveryDisposition.RETRY
    assert failure.retry_after_seconds == expected_seconds


@pytest.mark.parametrize(("offset_seconds", "expected_seconds"), [(-60, 0), (300, 300), (901, 900)])
def test_retry_after_http_dates_use_injected_clock_and_bounds(
    monkeypatch: pytest.MonkeyPatch, offset_seconds: int, expected_seconds: int
) -> None:
    now = datetime(2026, 9, 29, 12, tzinfo=UTC)

    class _InjectedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return now.astimezone(tz or UTC)

    monkeypatch.setattr(transport_module, "datetime", _InjectedDatetime)
    header = format_datetime(now + timedelta(seconds=offset_seconds), usegmt=True)
    failure = classify_http_failure(429, {"Retry-After": header})

    assert failure.disposition is DeliveryDisposition.RETRY
    assert failure.retry_after_seconds == expected_seconds
