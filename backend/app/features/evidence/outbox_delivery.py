"""Bounded delivery claims and append-only outcomes; no network inside SQL transactions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID, uuid4

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase


class DeliveryResponseConflict(RuntimeError):
    """An attempt already has a different immutable request completion."""


class DeliveryOutcome(StrEnum):
    SENT = "SENT"
    RETRY = "RETRY"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DeliveryBudget:
    max_attempts: int
    lease_seconds: float
    request_timeout_seconds: float
    retry_seconds: float

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 0 < self.max_attempts < 2**31:
            raise ValueError("delivery attempts must have a positive finite bound")
        for value in (self.lease_seconds, self.request_timeout_seconds, self.retry_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 3600:
                raise ValueError("delivery deadlines must be positive and at most one hour")
        if self.lease_seconds <= self.request_timeout_seconds:
            raise ValueError("claim lease must exceed the network request deadline")


@dataclass(frozen=True, slots=True)
class DeliveryClaim:
    attempt_id: UUID
    edge_event_id: str
    ordinal: int
    backend_camera_id: str
    envelope: str = field(repr=False)


class OutboxDelivery:
    def __init__(
        self, database: PostgresDatabase, authority: AuthorityToken, budget: DeliveryBudget
    ) -> None:
        self.database, self.authority, self.budget = database, authority, budget

    def claim(self) -> DeliveryClaim | None:
        def claim_one(connection: psycopg.Connection) -> DeliveryClaim | None:
            require_authority(connection, self.authority, sender=True)
            row = connection.execute(
                "SELECT edge_event_id,envelope,backend_camera_id,"
                "state,attempt_count,active_attempt "
                "FROM event_outbox WHERE (state='PENDING' AND retry_at<=clock_timestamp()) "
                "OR (state='IN_FLIGHT' AND lease_until<=clock_timestamp()) "
                "ORDER BY retry_at,accepted_at,edge_event_id LIMIT 1 FOR UPDATE SKIP LOCKED"
            ).fetchone()
            if row is None:
                return None
            event_id, envelope, camera_id, state, count, active = row
            if state == "IN_FLIGHT":
                connection.execute(
                    "INSERT INTO event_delivery_results (attempt_id,finished_at,outcome,reason) "
                    "VALUES (%s,clock_timestamp(),'UNKNOWN','LEASE_EXPIRED')",
                    (active,),
                )
            if count >= self.budget.max_attempts:
                connection.execute(
                    "UPDATE event_outbox SET state='EXHAUSTED',"
                    "active_attempt=NULL,lease_until=NULL "
                    "WHERE edge_event_id=%s",
                    (event_id,),
                )
                return None
            attempt = uuid4()
            connection.execute(
                "INSERT INTO event_delivery_attempts "
                "(attempt_id,edge_event_id,ordinal,writer_generation,started_at) "
                "VALUES (%s,%s,%s,%s,clock_timestamp())",
                (attempt, event_id, count + 1, self.authority.generation),
            )
            connection.execute(
                "UPDATE event_outbox SET state='IN_FLIGHT',attempt_count=%s,active_attempt=%s,"
                "lease_until=clock_timestamp()+(%s * interval '1 second') WHERE edge_event_id=%s",
                (count + 1, attempt, self.budget.lease_seconds, event_id),
            )
            return DeliveryClaim(attempt, event_id, count + 1, camera_id, envelope)

        return self.database.transact(claim_one)

    def finish(
        self,
        claim: DeliveryClaim,
        outcome: DeliveryOutcome,
        *,
        reason: str,
        http_status: int | None = None,
        backend_event_id: str | None = None,
    ) -> bool:
        """Commit the response observation even when a stale lease returns False.

        Each persisted attempt permits exactly one actual HTTP request. Its first
        finish records only classified metadata, never response bodies or credentials.
        Exact repeats are idempotent; different responses raise DeliveryResponseConflict.
        A timeout/connection loss is UNKNOWN, not rejection or remote acceptance.
        The same edge_event_id remains the central idempotency key on every retry.
        Late responses cannot change the active claim or a terminal outbox state,
        and never replace an immutable UNKNOWN/LEASE_EXPIRED result.
        Fenced authority or an invalid event/attempt association refuses all writes.
        No result escapes failed/unknown COMMIT, and no transaction is replayed.
        """
        if not isinstance(outcome, DeliveryOutcome):
            raise TypeError("delivery outcome must be explicitly classified")
        if not isinstance(reason, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reason) is None:
            raise ValueError("delivery reason must be a bounded privacy-safe code")
        if http_status is not None and (
            type(http_status) is not int or not 100 <= http_status <= 599
        ):
            raise ValueError("delivery HTTP status is invalid")
        if (outcome is DeliveryOutcome.SENT) != (backend_event_id is not None):
            raise ValueError("only remote acceptance carries a central event identity")
        if backend_event_id is not None and (
            not isinstance(backend_event_id, str) or not 0 < len(backend_event_id) <= 128
        ):
            raise ValueError("central event identity is invalid")

        def complete(connection: psycopg.Connection) -> bool:
            require_authority(connection, self.authority, sender=True)
            row = connection.execute(
                "SELECT o.active_attempt,o.state,o.attempt_count "
                "FROM event_outbox o JOIN event_delivery_attempts a "
                "ON a.edge_event_id=o.edge_event_id "
                "WHERE o.edge_event_id=%s AND a.attempt_id=%s AND a.ordinal=%s "
                "FOR UPDATE OF o",
                (claim.edge_event_id, claim.attempt_id, claim.ordinal),
            ).fetchone()
            if row is None:
                raise ValueError("delivery claim does not match its persisted attempt")
            response = (str(outcome), reason, http_status, backend_event_id)
            observed = connection.execute(
                "SELECT outcome,reason,http_status,backend_event_id "
                "FROM event_delivery_observations WHERE attempt_id=%s",
                (claim.attempt_id,),
            ).fetchone()
            if observed is None:
                connection.execute(
                    "INSERT INTO event_delivery_observations "
                    "(attempt_id,edge_event_id,ordinal,observed_at,"
                    "outcome,reason,http_status,backend_event_id) "
                    "VALUES (%s,%s,%s,clock_timestamp(),%s,%s,%s,%s)",
                    (claim.attempt_id, claim.edge_event_id, claim.ordinal, *response),
                )
            elif observed != response:
                raise DeliveryResponseConflict("attempt already has a different response")
            if row[:2] != (claim.attempt_id, "IN_FLIGHT"):
                return False
            terminal = str(outcome)
            if outcome in (DeliveryOutcome.RETRY, DeliveryOutcome.UNKNOWN):
                terminal = "EXHAUSTED" if row[2] >= self.budget.max_attempts else "PENDING"
            connection.execute(
                "INSERT INTO event_delivery_results "
                "(attempt_id,finished_at,outcome,reason,http_status,backend_event_id) "
                "VALUES (%s,clock_timestamp(),%s,%s,%s,%s)",
                (claim.attempt_id, *response),
            )
            connection.execute(
                "UPDATE event_outbox SET state=%s,active_attempt=NULL,lease_until=NULL,"
                "retry_at=clock_timestamp()+(%s * interval '1 second') WHERE edge_event_id=%s",
                (terminal, self.budget.retry_seconds, claim.edge_event_id),
            )
            return True

        return self.database.transact(complete)
