from __future__ import annotations

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.relay_failure_log import RelayFailureClass, classify_relay_failure


def test_server_error_hint_is_neutral_and_names_the_status() -> None:
    # A 5xx from the edge API can originate from its own local contention
    # (e.g. SQLite write contention, see #579) just as easily as from
    # something genuinely upstream -- the hint must not presume which, only
    # name the status the worker actually saw.
    outcome = classify_relay_failure(
        DeliveryFailure(DeliveryDisposition.RETRY, "HTTP_503", status_code=503)
    )
    assert outcome.failure_class is RelayFailureClass.SERVER_ERROR
    assert outcome.hint == "edge API returned 503; will keep retrying"
    assert "upstream relay is down" not in outcome.hint
    assert "ml-api" not in outcome.hint


def test_transport_failure_hint_is_unaffected_by_the_server_error_wording() -> None:
    outcome = classify_relay_failure(
        DeliveryFailure(
            DeliveryDisposition.RETRY, "URLError", transport_error="URLError: timed out"
        )
    )
    assert outcome.failure_class is RelayFailureClass.TRANSPORT
    assert outcome.hint == "cannot reach relay host; will keep retrying"
