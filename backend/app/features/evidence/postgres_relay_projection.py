"""Native late-snapshot facts; initial acceptance belongs to EventOutbox.

The injected database owns the complete transaction and pool lease. A supplied
callback runs once even for matching facts, and remains tentative until the
owner returns. This owner neither publishes audit work nor changes delivery
obligations. HTTP/lifespan activation is a separate atomic integration.
"""

from __future__ import annotations

from collections.abc import Callable

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.features.evidence.relay_projection import (
    RelayEvidenceProjectionConflict,
    RelayEvidenceProjectionMissingEvent,
    RelaySnapshot,
    _bounded_reason,
    _snapshot_timestamp,
    _validate_snapshot,
)


def put_snapshot(connection: psycopg.Connection, incident_id: str, snapshot: RelaySnapshot) -> None:
    """Borrow the caller's authority-checked transaction and locked incident."""
    _validate_snapshot(snapshot)
    expected = (
        snapshot.snapshot_id,
        "AVAILABLE",
        snapshot.path,
        snapshot.sha256,
        snapshot.size_bytes,
        snapshot.mime_type,
        snapshot.captured_at,
    )
    existing = connection.execute(
        "SELECT artifact_id,state,contained_relpath,content_sha256,"
        "size_bytes,mime_type,captured_at "
        "FROM artifacts WHERE incident_id=%s AND kind='SNAPSHOT'",
        (incident_id,),
    ).fetchone()
    if existing is not None:
        if tuple(existing) != expected:
            raise RelayEvidenceProjectionConflict(
                "snapshot attachment conflicts with existing content identity"
            )
        return
    connection.execute(
        "INSERT INTO artifacts (incident_id,kind,artifact_id,state,contained_relpath,"
        "content_sha256,size_bytes,mime_type,captured_at,revision,created_at,updated_at) "
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


class PostgresRelayEvidenceProjection:
    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("native snapshot projection requires a PostgreSQL owner") from None
        if not isinstance(authority, AuthorityToken):
            raise TypeError("native snapshot projection requires deployment authority") from None
        self.database, self.authority = database, authority

    def attach_snapshot(
        self,
        *,
        edge_event_id: str,
        snapshot_id: str,
        sha256: str,
        media_reference: str,
        size_bytes: int,
        mime_type: str,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> None:
        snapshot = RelaySnapshot(
            snapshot_id=snapshot_id,
            path=media_reference,
            sha256=sha256,
            size_bytes=size_bytes,
            mime_type=mime_type,
            captured_at=_snapshot_timestamp(snapshot_id),
        )

        def attach(connection: psycopg.Connection) -> None:
            require_authority(connection, self.authority)
            incident_id = _incident_for_event(connection, edge_event_id)
            put_snapshot(connection, incident_id, snapshot)
            if after_write is not None:
                after_write(connection)

        self.database.transact(attach)

    def record_snapshot_disposition(
        self,
        *,
        edge_event_id: str,
        snapshot_id: str,
        disposition: str,
        reason: str,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> None:
        terminal_reason = _bounded_reason(disposition, reason)
        captured_at = _snapshot_timestamp(snapshot_id)

        def record(connection: psycopg.Connection) -> None:
            require_authority(connection, self.authority)
            incident_id = _incident_for_event(connection, edge_event_id)
            existing = connection.execute(
                "SELECT artifact_id,state,reason FROM artifacts "
                "WHERE incident_id=%s AND kind='SNAPSHOT'",
                (incident_id,),
            ).fetchone()
            expected = (None, "UNAVAILABLE", terminal_reason)
            if existing is None:
                connection.execute(
                    "INSERT INTO artifacts (incident_id,kind,state,reason,captured_at,"
                    "revision,created_at,updated_at) "
                    "VALUES (%s,'SNAPSHOT','UNAVAILABLE',%s,%s,1,%s,%s)",
                    (incident_id, terminal_reason, captured_at, captured_at, captured_at),
                )
            elif tuple(existing) != expected:
                raise RelayEvidenceProjectionConflict(
                    "snapshot disposition conflicts with existing terminal fact"
                )
            if after_write is not None:
                after_write(connection)

        self.database.transact(record)


def _incident_for_event(connection: psycopg.Connection, edge_event_id: str) -> str:
    # Serialize competing terminal decisions for this incident. The outbox's
    # initial-snapshot path takes the same incident lock (or inserts its row).
    row = connection.execute(
        "SELECT incident_id FROM incidents WHERE edge_event_id=%s FOR UPDATE",
        (edge_event_id,),
    ).fetchone()
    if row is None:
        raise RelayEvidenceProjectionMissingEvent(
            "snapshot companion requires an already-projected incident"
        )
    return str(row[0])


__all__ = ["PostgresRelayEvidenceProjection", "put_snapshot"]
