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
