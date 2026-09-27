"""Product incident, snapshot, audit and delivery obligation in one PostgreSQL commit."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.postgres_store import append_postgres_audit
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.features.evidence.relay_projection import (
    RelayEvent,
    RelaySnapshot,
    _validate_snapshot,
)


class EventIdentityConflict(RuntimeError):
    """An already accepted event cannot be rebound to different content."""


class OutboxCapacityExceeded(RuntimeError):
    """No acceptance occurred; the producer must retain its durable copy."""


@dataclass(frozen=True, slots=True)
class OutboxBudget:
    max_entries: int
    max_bytes: int

    def __post_init__(self) -> None:
        for value in (self.max_entries, self.max_bytes):
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError("outbox budgets must be positive signed 64-bit integers")


@dataclass(frozen=True, slots=True)
class AcceptedEvent:
    edge_event_id: str
    duplicate: bool
    delivery_state: str


def _envelope(
    event: RelayEvent, snapshot: RelaySnapshot | None, snapshot_bytes: bytes | None
) -> str:
    payload = asdict(event)
    if snapshot is not None:
        _validate_snapshot(snapshot)
        payload["snapshot"] = asdict(snapshot)
    if snapshot_bytes is not None:
        if snapshot is None or not 0 < len(snapshot_bytes) <= 200 * 1024:
            raise ValueError("inline evidence requires bounded snapshot metadata")
        if (
            snapshot.size_bytes != len(snapshot_bytes)
            or snapshot.sha256 != hashlib.sha256(snapshot_bytes).hexdigest()
        ):
            raise ValueError("inline evidence does not match its declared identity")
        payload["snapshot_jpeg_base64"] = base64.b64encode(snapshot_bytes).decode("ascii")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(encoded.encode("utf-8")) > 512 * 1024:
        raise ValueError("event envelope exceeds its admission bound")
    return encoded


def _snapshot(connection: psycopg.Connection, incident_id: str, snapshot: RelaySnapshot) -> None:
    expected = (
        snapshot.snapshot_id,
        "AVAILABLE",
        snapshot.path,
        snapshot.sha256,
        snapshot.size_bytes,
        snapshot.mime_type,
        snapshot.captured_at,
    )
    row = connection.execute(
        "SELECT artifact_id,state,contained_relpath,content_sha256,"
        "size_bytes,mime_type,captured_at "
        "FROM artifacts WHERE incident_id=%s AND kind='SNAPSHOT'",
        (incident_id,),
    ).fetchone()
    if row is not None:
        if tuple(row) != expected:
            raise EventIdentityConflict("snapshot identity conflicts with accepted content")
        return
    connection.execute(
        "INSERT INTO artifacts (incident_id,kind,artifact_id,state,"
        "contained_relpath,content_sha256,"
        "size_bytes,mime_type,captured_at,revision,created_at,updated_at) "
        "VALUES (%s,'SNAPSHOT',%s,'AVAILABLE',%s,%s,%s,%s,%s,1,%s,%s)",
        (
            incident_id,
            snapshot.snapshot_id,
            snapshot.path,
            snapshot.sha256,
            snapshot.size_bytes,
            snapshot.mime_type,
            snapshot.captured_at,
            snapshot.captured_at,
            snapshot.captured_at,
        ),
    )


class EventOutbox:
    def __init__(
        self, database: PostgresDatabase, authority: AuthorityToken, budget: OutboxBudget
    ) -> None:
        self.database, self.authority, self.budget = database, authority, budget

    def accept(
        self,
        event: RelayEvent,
        *,
        backend_camera_id: str | None,
        forward: bool,
        snapshot: RelaySnapshot | None = None,
        snapshot_bytes: bytes | None = None,
    ) -> AcceptedEvent:
        if type(forward) is not bool:
            raise TypeError("forwarding policy must be explicit")
        if forward and (not isinstance(backend_camera_id, str) or not backend_camera_id.strip()):
            raise ValueError("central forwarding requires a Hub-issued camera identity")
        envelope = _envelope(event, snapshot, snapshot_bytes)
        encoded = envelope.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        state = "PENDING" if forward else "LOCAL_ONLY"

        def admit(connection: psycopg.Connection) -> AcceptedEvent:
            require_authority(connection, self.authority)
            # Serialize quota reservation and duplicate decisions, not SDK callbacks.
            connection.execute(
                "SELECT pg_advisory_xact_lock('event_outbox'::regclass::oid::bigint)"
            )
            existing = connection.execute(
                "SELECT envelope_sha256,state FROM event_outbox WHERE edge_event_id=%s",
                (event.edge_event_id,),
            ).fetchone()
            if existing is not None:
                # Mapping/config may change after acceptance, but cannot alter the
                # already committed destination or invent a second delivery.
                if existing[0] != digest:
                    raise EventIdentityConflict("event ID conflicts with accepted content")
                return AcceptedEvent(event.edge_event_id, True, existing[1])
            used = connection.execute(
                "SELECT count(*),coalesce(sum(envelope_bytes),0) FROM event_outbox"
            ).fetchone()
            if used[0] >= self.budget.max_entries or used[1] + len(encoded) > self.budget.max_bytes:
                raise OutboxCapacityExceeded("accepted delivery capacity is exhausted")
            incident_id = f"incident:{event.edge_event_id}"
            expected = (
                incident_id,
                event.facility_id,
                event.camera_id,
                event.event_type,
                event.probability,
                event.detected_at,
            )
            incident = connection.execute(
                "SELECT incident_id,facility_id,camera_id,event_type,probability,detected_at "
                "FROM incidents WHERE edge_event_id=%s FOR UPDATE",
                (event.edge_event_id,),
            ).fetchone()
            if incident is None:
                connection.execute(
                    "INSERT INTO incidents (incident_id,edge_event_id,facility_id,"
                    "camera_id,event_type,"
                    "probability,detected_at,lifecycle_state,provenance_state,provenance_missing_reason,"
                    "review_version,revision,created_at,updated_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,'OPEN','MISSING','NOT_RECORDED',0,1,%s,%s)",
                    (
                        incident_id,
                        event.edge_event_id,
                        event.facility_id,
                        event.camera_id,
                        event.event_type,
                        event.probability,
                        event.detected_at,
                        event.detected_at,
                        event.detected_at,
                    ),
                )
            elif tuple(incident) != expected:
                raise EventIdentityConflict("event ID conflicts with an existing incident")
            if snapshot is not None:
                _snapshot(connection, incident_id, snapshot)
            connection.execute(
                "INSERT INTO event_outbox (edge_event_id,envelope,envelope_sha256,envelope_bytes,"
                "backend_camera_id,state,accepted_generation,accepted_at,retry_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,clock_timestamp(),clock_timestamp())",
                (
                    event.edge_event_id,
                    envelope,
                    digest,
                    len(encoded),
                    backend_camera_id,
                    state,
                    self.authority.generation,
                ),
            )
            append_postgres_audit(
                connection,
                AuditEvent(
                    occurred_at=utc_now(),
                    actor_id="worker-relay",
                    action=AuditAction.RELAY_ALERT,
                    target_id=event.edge_event_id,
                    detail=empty_detail(AuditAction.RELAY_ALERT),
                    actor_type=AuditActorType.SERVICE,
                    auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
                ),
            )
            return AcceptedEvent(event.edge_event_id, False, state)

        # There is no network send, response construction, or retry in this transaction.
        # The value cannot escape a failed/unknown COMMIT or failed pool release.
        return self.database.transact(admit)
