"""Transactional immutable audit append and bounded hash-chain verification."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from backend.app.edge_db import EDGE_DATABASE_PATH, RuntimeActor, open_runtime_database
from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.connection import write_transaction
from backend.app.edge_db.functions import audit_record_hash
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    AuditDetail,
    parse_detail,
)

GENESIS_HASH: Final = "0" * 64
MAX_AUDIT_ROWS: Final = 1_000_000
_VERIFY_PAGE_SIZE: Final = 1_000

SqlValue = str | int | float | bytes | None
_REQUIRED_TRIGGERS: Final = frozenset(
    {
        "audit_events_immutable_update",
        "audit_events_immutable_delete",
        "audit_events_chain",
        "audit_events_record_hash",
        "audit_events_capacity",
    }
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


@dataclass(frozen=True, slots=True)
class VerificationCheckpoint:
    audit_id: int
    record_hash: str


@dataclass(frozen=True, slots=True)
class AuditVerificationError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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
        """Commit a bounded operation event and optional recovery fence atomically."""
        with closing(open_runtime_database(self.path, actor=RuntimeActor.API)) as connection:
            with write_transaction(connection):
                return tuple(self._append(connection, event) for event in events)

    def _append(self, connection: sqlite3.Connection, event: AuditEvent) -> AuditRecord:
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
            with closing(open_runtime_database(self.path, actor=RuntimeActor.API)) as connection:
                return self._verify(connection, checkpoint)
        except (OSError, sqlite3.Error, EdgeDatabaseError, ValueError) as error:
            raise AuditVerificationError(str(error)) from error

    def _verify(
        self, connection: sqlite3.Connection, checkpoint: VerificationCheckpoint | None
    ) -> VerificationCheckpoint:
        count = int(connection.execute("SELECT COUNT(audit_id) FROM audit_events").fetchone()[0])
        if count >= MAX_AUDIT_ROWS:
            raise AuditVerificationError("audit history reached the one-million-row refusal limit")
        triggers = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='audit_events'"
            )
        }
        if not _REQUIRED_TRIGGERS <= triggers:
            raise AuditVerificationError("audit immutable trigger contract is incomplete")
        last_id = 0 if checkpoint is None else checkpoint.audit_id
        previous_hash = GENESIS_HASH if checkpoint is None else checkpoint.record_hash
        while True:
            rows = connection.execute(
                "SELECT audit_id,occurred_at,recorded_at,clock_quality,actor_type,actor_id,"
                "auth_mechanism,action,target_type,target_id,outcome,reason,request_id,"
                "interaction_id,detail_json,previous_hash,record_hash,retention_class,"
                "hold_reference FROM audit_events WHERE audit_id>? ORDER BY audit_id LIMIT ?",
                (last_id, _VERIFY_PAGE_SIZE),
            ).fetchall()
            if not rows:
                return VerificationCheckpoint(last_id, previous_hash)
            for row in rows:
                last_id, previous_hash = self._verify_row(row, previous_hash)

    @staticmethod
    def _verify_row(row: tuple[SqlValue, ...], expected_previous: str) -> tuple[int, str]:
        action = AuditAction(str(row[7]))
        detail_raw = {} if row[14] is None else json.loads(str(row[14]))
        detail = parse_detail(action, detail_raw)
        event = AuditEvent(
            str(row[1]), str(row[5]), action, str(row[9]), detail,
            actor_type=AuditActorType(str(row[4])),
            auth_mechanism=AuditAuthMechanism(str(row[6])),
        )
        previous_hash = str(row[15])
        if previous_hash != expected_previous:
            raise AuditVerificationError("audit previous hash does not match")
        payload = {
            "action": action.value, "actor_id": str(row[5]), "actor_type": str(row[4]),
            "auth_mechanism": str(row[6]), "clock_quality": str(row[3]),
            "detail_json": detail.json, "hold_reference": row[18], "interaction_id": row[13],
            "occurred_at": str(row[1]), "outcome": str(row[10]), "previous_hash": previous_hash,
            "reason": row[11], "recorded_at": str(row[2]), "request_id": row[12],
            "retention_class": str(row[17]), "target_id": event.target_id,
            "target_type": str(row[8]),
        }
        expected_hash = audit_record_hash(previous_hash, json.dumps(payload))
        if str(row[16]) != expected_hash:
            raise AuditVerificationError("audit record hash does not match")
        audit_id = row[0]
        if not isinstance(audit_id, int) or isinstance(audit_id, bool):
            raise AuditVerificationError("audit identity is not an integer")
        return audit_id, str(row[16])


__all__ = [
    "GENESIS_HASH", "MAX_AUDIT_ROWS", "AuditEvent", "AuditRecord", "AuditStore",
    "AuditVerificationError", "VerificationCheckpoint", "utc_now",
]
