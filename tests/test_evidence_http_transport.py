from __future__ import annotations

import json
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any

import pytest

import shared.events.evidence_http_transport as transport_module
from shared.events.evidence_export_contract import (
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from shared.events.evidence_http_transport import classify_http_failure

# d1b: the relay-disposition wire contract the Rust relay client (d1) also replays.
# The manifest names this file as its Python consumer.
_RELAY_GOLDEN_PATH = (
    Path(__file__).parent / "fixtures" / "worker-wire" / "r" / "relay-dispositions.json"
)
_RELAY_GOLDEN: dict[str, Any] = json.loads(_RELAY_GOLDEN_PATH.read_text(encoding="utf-8"))
_PARSER_SOURCE = "shared/events/evidence_http_transport.py"


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


@pytest.fixture
def injected_relay_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the parser's wall clock to the golden's injected_now_utc."""
    injected_now = datetime.fromisoformat(_RELAY_GOLDEN["injected_now_utc"])

    class _InjectedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return injected_now.astimezone(tz or UTC)

    monkeypatch.setattr(transport_module, "datetime", _InjectedDatetime)


def _replay_relay_row(row: dict[str, Any]) -> EventReceipt | ClipReceipt | DeliveryFailure:
    response = row["response"]
    body = response["body"]
    body_bytes = None if body is None else body.encode("utf-8")
    request = row["request"]
    parser = row["parser"]
    if parser == f"{_PARSER_SOURCE}:parse_event_result":
        return transport_module.parse_event_result(
            (response["status"], response["headers"], body_bytes or b""),
            request["expected_edge_event_id"],
        )
    if parser == f"{_PARSER_SOURCE}:parse_clip_result":
        return transport_module.parse_clip_result(
            (response["status"], response["headers"], body_bytes or b""),
            request["expected_clip_id"],
            request["expected_state_version"],
        )
    if parser == f"{_PARSER_SOURCE}:classify_http_failure":
        return transport_module.classify_http_failure(
            response["status"], response["headers"], body_bytes
        )
    pytest.fail(f"relay golden row {row['name']!r} names an unknown parser {parser!r}")


def _relay_result_view(result: EventReceipt | ClipReceipt | DeliveryFailure) -> dict[str, Any]:
    if isinstance(result, EventReceipt):
        return {
            "kind": "EventReceipt",
            "status": result.status,
            "edge_event_id": result.edge_event_id,
            "event_id": result.event_id,
        }
    if isinstance(result, ClipReceipt):
        return {
            "kind": "ClipReceipt",
            "clip_id": result.clip_id,
            "state": result.state,
            "state_version": result.state_version,
            "sha256": result.sha256,
            "size_bytes": result.size_bytes,
        }
    return {
        "kind": "failure",
        "code": str(result.code),
        "disposition": result.disposition.value,
        "status_code": result.status_code,
        "retry_after_seconds": result.retry_after_seconds,
    }


@pytest.mark.usefixtures("injected_relay_clock")
@pytest.mark.parametrize("row", _RELAY_GOLDEN["rows"], ids=lambda row: row["name"])
def test_relay_disposition_golden_row_replays_through_the_python_parser(
    row: dict[str, Any],
) -> None:
    result = _replay_relay_row(row)

    assert _relay_result_view(result) == row["result"]


@pytest.mark.usefixtures("injected_relay_clock")
@pytest.mark.parametrize(
    ("retry_after", "expected_seconds"),
    sorted(_RELAY_GOLDEN["retry_after_direct"].items()),
)
def test_relay_retry_after_golden_replays_through_the_python_parser(
    retry_after: str, expected_seconds: float | None
) -> None:
    failure = classify_http_failure(503, {"Retry-After": retry_after})

    assert failure.retry_after_seconds == expected_seconds
