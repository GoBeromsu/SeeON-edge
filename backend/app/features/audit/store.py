"""Transactional immutable audit append store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from backend.app.edge_db import EDGE_DATABASE_PATH, RuntimeActor, open_runtime_database
from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.connection import write_transaction
from backend.app.edge_db.functions import audit_record_hash
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    AuditDetail,
)
from backend.app.features.audit.verification import (
    GENESIS_HASH,
    MAX_AUDIT_ROWS,
    AuditVerificationError,
    VerificationCheckpoint,
    database_identity,
    verify_connection,
)


@dataclass(frozen=True, slots=True)
class AuditEvent:
    occurred_at: str
    actor_id: str
    action: AuditAction
    target_id: str
    detail: AuditDetail
    actor_type: AuditActorType = AuditActorType.USER
    auth_mechanism: AuditAuthMechanism = AuditAuthMechanism.DASHBOARD_SESSION


@dataclass(frozen=True, slots=True)
class AuditRecord:
    audit_id: int
    occurred_at: str
    recorded_at: str
    actor_id: str
    action: AuditAction
    target_type: str
    target_id: str
    detail: AuditDetail
    previous_hash: str
    record_hash: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _observer_data_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA data_version").fetchone()
    if row is None or type(row[0]) is not int:
        raise AuditVerificationError("audit data version is unreadable")
    return row[0]


def _target_type(action: AuditAction) -> str:
    return action.value.partition(".")[0]


def _payload(event: AuditEvent, recorded_at: str, previous_hash: str) -> dict[str, str | None]:
    return {
        "action": event.action.value,
        "actor_id": event.actor_id,
        "actor_type": event.actor_type.value,
        "auth_mechanism": event.auth_mechanism.value,
        "clock_quality": "trusted",
        "detail_json": event.detail.json,
        "hold_reference": None,
        "interaction_id": None,
        "occurred_at": event.occurred_at,
        "outcome": "success",
        "previous_hash": previous_hash,
        "reason": None,
        "recorded_at": recorded_at,
        "request_id": None,
        "retention_class": "standard",
        "target_id": event.target_id,
        "target_type": _target_type(event.action),
    }


class AuditStore:
    """Append registered events on either an owned or caller-owned transaction."""

    def __init__(self, path: Path | None = None) -> None:
        self.path: Path = EDGE_DATABASE_PATH if path is None else path
        self._verifier_connection: sqlite3.Connection | None = None
        self._verifier_identity: tuple[int, int] | None = None
        self._verifier_id = uuid4().hex
        self._verifier_lock = Lock()

    def append(
        self, event: AuditEvent, *, connection: sqlite3.Connection | None = None
    ) -> AuditRecord:
        if connection is None:
            with closing(open_runtime_database(self.path, actor=RuntimeActor.API)) as owned:
                with write_transaction(owned):
                    return self._append(owned, event)
        if not connection.in_transaction:
            raise AuditVerificationError("caller-owned audit append requires an active transaction")
        return self._append(connection, event)

    def append_batch(self, events: Sequence[AuditEvent]) -> tuple[AuditRecord, ...]:
        """Commit a bounded event group atomically under SQLite serialization."""
        with closing(open_runtime_database(self.path, actor=RuntimeActor.API)) as connection:
            with write_transaction(connection):
                return tuple(self._append(connection, event) for event in events)

    def _append(self, connection: sqlite3.Connection, event: AuditEvent) -> AuditRecord:
        if event.detail.action is not event.action:
            raise AuditVerificationError("audit action/detail variants do not match")
        previous_row = connection.execute(
            "SELECT record_hash FROM audit_events ORDER BY audit_id DESC LIMIT 1"
        ).fetchone()
        previous_hash = GENESIS_HASH if previous_row is None else str(previous_row[0])
        recorded_at = utc_now()
        payload = _payload(event, recorded_at, previous_hash)
        payload_json = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        record_hash = audit_record_hash(previous_hash, payload_json)
        insert_values = payload | {"record_hash": record_hash}
        cursor = connection.execute(
            "INSERT INTO audit_events(occurred_at,recorded_at,clock_quality,actor_type,"
            "actor_id,auth_mechanism,action,target_type,target_id,outcome,reason,request_id,"
            "interaction_id,detail_json,previous_hash,record_hash,retention_class,hold_reference) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(insert_values[key] for key in (
                "occurred_at", "recorded_at", "clock_quality", "actor_type", "actor_id",
                "auth_mechanism", "action", "target_type", "target_id", "outcome", "reason",
                "request_id", "interaction_id", "detail_json", "previous_hash", "record_hash",
                "retention_class", "hold_reference",
            )),
        )
        audit_id = cursor.lastrowid
        if audit_id is None:
            raise AuditVerificationError("audit insert did not return an identity")
        return AuditRecord(
            audit_id=audit_id, occurred_at=event.occurred_at,
            recorded_at=recorded_at, actor_id=event.actor_id, action=event.action,
            target_type=_target_type(event.action), target_id=event.target_id,
            detail=event.detail, previous_hash=previous_hash, record_hash=record_hash,
        )

    def verify(
        self, checkpoint: VerificationCheckpoint | None = None
    ) -> VerificationCheckpoint:
        try:
            if checkpoint is None:
                with closing(open_runtime_database(self.path, actor=RuntimeActor.API)):
                    pass
            with self._verifier_lock:
                identity = database_identity(self.path)
                connection = self._verifier(identity, checkpoint)
                connection.execute("BEGIN")
                try:
                    observed_version = _observer_data_version(connection)
                    verified = verify_connection(
                        connection, checkpoint, identity,
                        observer_id=self._verifier_id, data_version=observed_version,
                    )
                    connection.execute("COMMIT")
                    return verified
                finally:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
        except (OSError, sqlite3.Error, EdgeDatabaseError, ValueError) as error:
            raise AuditVerificationError(str(error)) from error

    def close_verifier(self) -> None:
        with self._verifier_lock:
            if self._verifier_connection is not None:
                self._verifier_connection.close()
                self._verifier_connection = None
                self._verifier_identity = None

    def _verifier(
        self,
        identity: tuple[int, int],
        checkpoint: VerificationCheckpoint | None,
    ) -> sqlite3.Connection:
        if self._verifier_connection is not None and self._verifier_identity != identity:
            if checkpoint is not None:
                raise AuditVerificationError("audit checkpoint database identity changed")
            self._verifier_connection.close()
            self._verifier_connection = None
        if self._verifier_connection is None:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            self._verifier_connection = sqlite3.connect(
                uri, uri=True, check_same_thread=False, isolation_level=None,
            )
            self._verifier_connection.execute("PRAGMA query_only=ON")
            self._verifier_identity = identity
        return self._verifier_connection

    def _verify(
        self, connection: sqlite3.Connection, checkpoint: VerificationCheckpoint | None
    ) -> VerificationCheckpoint:
        return verify_connection(connection, checkpoint, database_identity(self.path))


__all__ = [
    "GENESIS_HASH", "MAX_AUDIT_ROWS", "AuditEvent", "AuditRecord", "AuditStore",
    "AuditVerificationError", "VerificationCheckpoint", "utc_now",
]
