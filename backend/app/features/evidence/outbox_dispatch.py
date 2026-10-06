"""The single alert sender: committed outbox obligations to the Hub ingest API.

The relay route (right after its admission COMMIT) and the lifespan sender loop
(for everything the route could not finish) share ``dispatch``. No SQL
transaction is open while the Hub request runs: ``OutboxDelivery.claim`` commits
the lease first and ``finish`` records the observation in a second transaction.
A lost ``finish`` leaves an expiring lease, so the row is re-claimed and resent
under the same edge event id, which the Hub deduplicates.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import psycopg

from backend.app.edge_db.authority import AuthorityFenced
from backend.app.edge_db.postgres import PostgresError
from backend.app.features.evidence.event_outbox import OutboxBudget
from backend.app.features.evidence.outbox_delivery import (
    DeliveryBudget,
    DeliveryClaim,
    DeliveryOutcome,
    DeliveryResponseConflict,
    OutboxDelivery,
)
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from shared.events.relay_failure_log import RelayFailureLog

_LOGGER = logging.getLogger(__name__)

RELAY_OUTBOX_BUDGET = OutboxBudget(max_entries=1_000_000, max_bytes=8 * 1024**3)
# One day of 5 s row retries. The lease must outlive one Hub request, including
# the optional snapshot upload that follows an accepted alert.
RELAY_DELIVERY_BUDGET = DeliveryBudget(
    max_attempts=17_280,
    lease_seconds=120.0,
    request_timeout_seconds=60.0,
    retry_seconds=5.0,
)
SENDER_BASE_INTERVAL_SEC = 1.0
SENDER_MAX_BACKOFF_SEC = 60.0
SENDER_BATCH_LIMIT = 32

_REASON_PATTERN = re.compile(r"[^A-Z0-9_]")
_FALLBACK_REASON = "DELIVERY_FAILED"
_KEEP_OBLIGATION = frozenset({DeliveryDisposition.RETRY, DeliveryDisposition.COMPATIBILITY})
_sender_failures = RelayFailureLog(_LOGGER, channel="backend outbox sender", method="POST")


class AlertIngestClient(Protocol):
    def send_alert_receipt(
        self,
        *,
        edge_event_id: str,
        event_type: Any,
        detected_at: str,
        probability: float,
        audit: dict[str, object] | None = None,
        snapshot_bytes: bytes | None = None,
        clip_id: str | None = None,
        on_accepted: Callable[[float], None] | None = None,
    ) -> EventReceipt | DeliveryFailure: ...


@dataclass(slots=True)
class OutboxSenderStatus:
    """Operator-visible sender state; written only by the sender thread."""

    enabled: bool
    reason: str | None = None
    consecutive_failures: int = 0
    last_failure_code: str | None = None
    sent: int = 0


def scoped_client(client: Any, backend_camera_id: str) -> AlertIngestClient:
    for_camera = getattr(client, "for_camera", None)
    if for_camera is not None:
        scoped: AlertIngestClient = for_camera(backend_camera_id)
        return scoped
    unscoped: AlertIngestClient = client
    return unscoped


def alert_kwargs(envelope: str) -> dict[str, Any]:
    """Rebuild the Hub request from the committed envelope, never from the request."""
    payload = json.loads(envelope)
    kwargs: dict[str, Any] = {
        "edge_event_id": payload["edge_event_id"],
        "event_type": payload["event_type"],
        "detected_at": payload["detected_at"],
        "probability": payload["probability"],
    }
    if payload.get("audit") is not None:
        kwargs["audit"] = payload["audit"]
    evidence = payload.get("evidence")
    clip_id = evidence.get("clip_id") if isinstance(evidence, dict) else None
    if isinstance(clip_id, str) and clip_id.strip() != "":
        kwargs["clip_id"] = clip_id
    inline = payload.get("snapshot_jpeg_base64")
    if isinstance(inline, str):
        kwargs["snapshot_bytes"] = base64.b64decode(inline, validate=True)
    return kwargs


def delivery_reason(code: str | None) -> str:
    reason = _REASON_PATTERN.sub("_", (code or "").upper())
    if not reason:
        return _FALLBACK_REASON
    if not reason[0].isalpha():
        reason = f"E_{reason}"
    return reason[:64]


def dispatch(
    client: Any,
    claim: DeliveryClaim,
    delivery: OutboxDelivery,
    *,
    on_accepted: Callable[[float], None] | None = None,
) -> EventReceipt | DeliveryFailure:
    """Send one claimed obligation and record what the Hub answered.

    The Hub answer is returned even when recording it fails: the incident is
    already committed, and the unfinished lease makes the sender resend later.
    """
    try:
        result = scoped_client(client, claim.backend_camera_id).send_alert_receipt(
            on_accepted=on_accepted, **alert_kwargs(claim.envelope)
        )
    except Exception:  # noqa: BLE001 - a send failure has an unknown outcome; the row is resent
        _LOGGER.exception("backend outbox send raised; outcome unknown")
        result = DeliveryFailure(DeliveryDisposition.RETRY, "TRANSPORT_ERROR")
        _finish(delivery, claim, DeliveryOutcome.UNKNOWN, reason="TRANSPORT_ERROR")
        return result
    if isinstance(result, EventReceipt):
        if result.event_id:
            _finish(
                delivery,
                claim,
                DeliveryOutcome.SENT,
                reason="ACCEPTED",
                backend_event_id=result.event_id,
            )
        else:
            _finish(delivery, claim, DeliveryOutcome.REJECTED, reason="ACCEPTED_LOCAL")
        return result
    http_status = (
        result.status_code
        if isinstance(result.status_code, int) and 100 <= result.status_code <= 599
        else None
    )
    if result.transport_error is not None:
        outcome = DeliveryOutcome.UNKNOWN
    elif result.disposition in _KEEP_OBLIGATION:
        # A Hub without the ingest route (404/405) may be upgraded later; keep
        # the obligation until max_attempts instead of dropping the alert.
        outcome = DeliveryOutcome.RETRY
    else:
        outcome = DeliveryOutcome.REJECTED
    _finish(delivery, claim, outcome, reason=delivery_reason(result.code), http_status=http_status)
    return result


def _finish(
    delivery: OutboxDelivery,
    claim: DeliveryClaim,
    outcome: DeliveryOutcome,
    *,
    reason: str,
    http_status: int | None = None,
    backend_event_id: str | None = None,
) -> None:
    try:
        delivery.finish(
            claim,
            outcome,
            reason=reason,
            http_status=http_status,
            backend_event_id=backend_event_id,
        )
    except (PostgresError, psycopg.Error, AuthorityFenced, DeliveryResponseConflict, ValueError):
        # The lease expires and the row is re-claimed; the Hub dedupes the resend.
        _LOGGER.warning(
            "backend outbox delivery result not recorded; lease will expire",
            extra={"edge_event_id": claim.edge_event_id, "outcome": outcome.value},
        )


def send_outbox_once(app: Any, *, limit: int = SENDER_BATCH_LIMIT) -> DeliveryFailure | None:
    """Drain due obligations; stop at the first retryable failure for backoff."""
    status: OutboxSenderStatus | None = getattr(app.state, "backend_outbox_sender_status", None)
    client = getattr(app.state, "backend_ingest_client", None)
    delivery = getattr(app.state, "event_outbox_delivery", None)
    if client is None or not isinstance(delivery, OutboxDelivery):
        return None
    for _ in range(limit):
        try:
            claim = delivery.claim()
        except (PostgresError, psycopg.Error, AuthorityFenced) as error:
            failure = DeliveryFailure(DeliveryDisposition.RETRY, type(error).__name__)
            _record(status, failure)
            return failure
        if claim is None:
            return None
        result = dispatch(client, claim, delivery)
        if isinstance(result, DeliveryFailure):
            _sender_failures.record_failure(result, path="alerts")
            if result.disposition in _KEEP_OBLIGATION:
                _record(status, result)
                return result
            continue
        _sender_failures.record_success(path="alerts")
        if status is not None:
            status.sent += 1
            status.consecutive_failures = 0
            status.last_failure_code = None
    return None


def _record(status: OutboxSenderStatus | None, failure: DeliveryFailure) -> None:
    if status is None:
        return
    status.consecutive_failures += 1
    status.last_failure_code = failure.code


def sender_delay(
    base: float,
    consecutive_failures: int,
    retry_after: float | None,
    jitter: float,
) -> float:
    """Exponential backoff with a jitter fraction in [0, 1), honoring Retry-After."""
    if consecutive_failures <= 0:
        return base
    exponential = min(SENDER_MAX_BACKOFF_SEC, base * 2 ** min(consecutive_failures, 16))
    delay = exponential * (0.5 + 0.5 * min(max(jitter, 0.0), 1.0))
    if retry_after is not None and retry_after > delay:
        delay = retry_after
    return min(SENDER_MAX_BACKOFF_SEC, max(base, delay))
