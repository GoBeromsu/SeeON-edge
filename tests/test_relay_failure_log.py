from __future__ import annotations

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.relay_failure_log import RelayFailureClass, classify_relay_failure


def test_server_error_hint_names_the_edge_api_not_a_vague_upstream() -> None:
    # ml-api is the worker's "relay": a 5xx it returns (e.g. from SQLite
    # write contention) must be attributed to ml-api itself, not logged as
    # if some other upstream service behind the relay were unreachable.
    outcome = classify_relay_failure(
        DeliveryFailure(DeliveryDisposition.RETRY, "HTTP_503", status_code=503)
    )
    assert outcome.failure_class is RelayFailureClass.SERVER_ERROR
    assert "ml-api" in outcome.hint
    assert "upstream relay is down" not in outcome.hint


def test_transport_failure_hint_is_unaffected_by_the_server_error_wording() -> None:
    outcome = classify_relay_failure(
        DeliveryFailure(
            DeliveryDisposition.RETRY, "URLError", transport_error="URLError: timed out"
        )
    )
    assert outcome.failure_class is RelayFailureClass.TRANSPORT
    assert outcome.hint == "cannot reach relay host; will keep retrying"
