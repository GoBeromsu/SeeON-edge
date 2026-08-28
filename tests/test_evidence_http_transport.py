from __future__ import annotations

import pytest

from shared.events.evidence_export_contract import DeliveryDisposition
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
