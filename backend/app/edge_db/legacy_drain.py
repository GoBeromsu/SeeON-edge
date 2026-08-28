"""Backend-owned delivery drain for schema-16 evidence events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.app.edge_db.connection import (
    open_legacy_evidence_drain_database,
    write_transaction,
)
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)


class LegacyEventTransport(Protocol):
    def send_event(
        self, payload_json: str, edge_event_id: str
    ) -> EventReceipt | DeliveryFailure: ...


@dataclass(frozen=True, slots=True)
class LegacyDrainResult:
    delivered: int
    permanent: int
    retryable: int


class LegacyEvidenceDrain:
    """Deliver schema-16 event rows without removing their local evidence facts."""

    def __init__(self, database: Path, transport: LegacyEventTransport) -> None:
        self._database = database
        self._transport = transport

    def run(self) -> LegacyDrainResult:
        delivered = permanent = retryable = 0
        connection = open_legacy_evidence_drain_database(self._database)
        try:
            rows = connection.execute(
                """
                SELECT edge_event_id, payload_json
                FROM evidence_events
                WHERE state IN ('STAGED', 'READY', 'IN_FLIGHT')
                ORDER BY queued_at, edge_event_id
                """
            ).fetchall()
        finally:
            connection.close()
        for edge_event_id, payload_json in rows:
            result = self._transport.send_event(str(payload_json), str(edge_event_id))
            if isinstance(result, EventReceipt):
                if result.edge_event_id != edge_event_id:
                    retryable += 1
                    continue
                # An accepted_local receipt carries no upstream id: the relay
                # persisted the row itself and will never push it. Store NULL,
                # not "" -- the compact schema CHECKs backend_event_id length
                # and gates "delivered upstream" on IS NOT NULL.
                self._acknowledge(str(edge_event_id), result.event_id or None)
                delivered += 1
            elif result.disposition is not DeliveryDisposition.RETRY:
                delivery_state = (
                    "PERMANENT"
                    if result.disposition is DeliveryDisposition.PERMANENT
                    else "COMPATIBILITY"
                )
                self._classify_failure(str(edge_event_id), delivery_state, result.code)
                permanent += 1
            else:
                retryable += 1
        return LegacyDrainResult(delivered, permanent, retryable)

    def _acknowledge(self, edge_event_id: str, backend_event_id: str | None) -> None:
        """Mark a row delivered. Only a matching durable receipt reaches here."""
        self._finish(edge_event_id, "ACKED", "ACKED", backend_event_id, None)

    def _classify_failure(
        self, edge_event_id: str, delivery_state: str, error_code: str | None
    ) -> None:
        """Record a non-retryable failure **without** claiming delivery.

        This previously routed through the same write as success, which set
        ``state = 'ACKED'`` unconditionally. A fleet-wide 422 would therefore
        have marked all 1143 legacy events delivered without any of them
        reaching the backend, emptied the pending set, and let the schema-17
        drain gate wave the migration through -- destroying the evidence and
        the only signal that it was gone.

        The row stays in ``READY`` so it is still counted as pending and still
        blocks migration. The classification and error code are recorded so an
        operator can see why, but resolving it is an explicit decision, never a
        side effect of running the drain.
        """
        self._finish(edge_event_id, "READY", delivery_state, None, error_code)

    def _finish(
        self,
        edge_event_id: str,
        state: str,
        delivery_state: str,
        backend_event_id: str | None,
        error_code: str | None,
    ) -> None:
        connection = open_legacy_evidence_drain_database(self._database)
        try:
            with write_transaction(connection):
                cursor = connection.execute(
                    """
                    UPDATE evidence_events
                    SET state = ?, delivery_state = ?, backend_event_id = ?,
                        last_error_code = ?, lease_owner = NULL, lease_expires_at = NULL,
                        attempt_count = attempt_count + 1
                    WHERE edge_event_id = ?
                      AND state IN ('STAGED', 'READY', 'IN_FLIGHT')
                    """,
                    (state, delivery_state, backend_event_id, error_code, edge_event_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"legacy evidence event {edge_event_id} is no longer pending"
                    )
        finally:
            connection.close()


__all__ = ["LegacyDrainResult", "LegacyEventTransport", "LegacyEvidenceDrain"]
