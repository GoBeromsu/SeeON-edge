"""Bounded audit-chain verification and schema-bound checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Final

from backend.app.edge_db.compact_schema_ddl import COMPACT_SCHEMA_CREATE_STATEMENTS
from backend.app.edge_db.functions import audit_record_hash
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    parse_detail_json,
)

GENESIS_HASH: Final = "0" * 64
MAX_AUDIT_ROWS: Final = 1_000_000
_VERIFY_PAGE_SIZE: Final = 1_000
def _canonical_triggers() -> dict[str, str]:
    triggers: dict[str, str] = {}
    for statement in COMPACT_SCHEMA_CREATE_STATEMENTS:
        normalized = " ".join(statement.split())
        if not normalized.startswith("CREATE TRIGGER audit_events_"):
            continue
        triggers[normalized.split(maxsplit=3)[2]] = normalized
    return triggers


_CANONICAL_TRIGGERS: Final = _canonical_triggers()
_ROW_SELECT: Final = (
    "SELECT audit_id,occurred_at,recorded_at,clock_quality,actor_type,actor_id,"
    "auth_mechanism,action,target_type,target_id,outcome,reason,request_id,"
    "interaction_id,detail_json,previous_hash,record_hash,retention_class,hold_reference "
    "FROM audit_events"
)

SqlValue = str | int | float | bytes | None
DatabaseIdentity = tuple[int, int]


@dataclass(frozen=True, slots=True)
class VerificationCheckpoint:
    audit_id: int
    record_hash: str
    anchor_previous_hash: str
    schema_version: int
    trigger_fingerprint: str
    database_identity: DatabaseIdentity


class AuditVerificationError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def database_identity(path: os.PathLike[str]) -> DatabaseIdentity:
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino


def verify_connection(
    connection: sqlite3.Connection,
    checkpoint: VerificationCheckpoint | None,
    identity: DatabaseIdentity,
) -> VerificationCheckpoint:
    """Verify the unchanged suffix or restart from genesis after a schema epoch change."""
    count_row = connection.execute("SELECT COUNT(audit_id) FROM audit_events").fetchone()
    count = 0 if count_row is None else int(count_row[0])
    if count >= MAX_AUDIT_ROWS:
        raise AuditVerificationError("audit history reached the one-million-row refusal limit")
    schema_version, trigger_fingerprint = _schema_state(connection)
    if checkpoint is not None and checkpoint.database_identity != identity:
        raise AuditVerificationError("audit checkpoint database identity changed")
    full = checkpoint is None or (
        checkpoint.schema_version != schema_version
        or checkpoint.trigger_fingerprint != trigger_fingerprint
    )
    if full:
        last_id, previous_hash = 0, GENESIS_HASH
        anchor_previous_hash = GENESIS_HASH
    else:
        assert checkpoint is not None
        last_id, previous_hash, anchor_previous_hash = _verify_anchor(connection, checkpoint)
    while True:
        rows = connection.execute(
            _ROW_SELECT + " WHERE audit_id>? ORDER BY audit_id LIMIT ?",
            (last_id, _VERIFY_PAGE_SIZE),
        ).fetchall()
        if not rows:
            return VerificationCheckpoint(
                last_id, previous_hash, anchor_previous_hash,
                schema_version, trigger_fingerprint, identity,
            )
        for row in rows:
            anchor_previous_hash = previous_hash
            last_id, previous_hash = verify_row(row, previous_hash)


def _schema_state(connection: sqlite3.Connection) -> tuple[int, str]:
    version_row = connection.execute("PRAGMA schema_version").fetchone()
    if version_row is None or not isinstance(version_row[0], int):
        raise AuditVerificationError("audit schema epoch is unreadable")
    trigger_rows = connection.execute(
        "SELECT name,sql FROM sqlite_master "
        "WHERE type='trigger' AND tbl_name='audit_events' ORDER BY name"
    ).fetchall()
    triggers = {
        str(name): " ".join(str(sql).split()) for name, sql in trigger_rows if sql is not None
    }
    if triggers != _CANONICAL_TRIGGERS:
        raise AuditVerificationError("audit immutable trigger contract is not canonical")
    canonical = json.dumps(triggers, sort_keys=True, separators=(",", ":")).encode()
    return version_row[0], hashlib.sha256(canonical).hexdigest()


def _verify_anchor(
    connection: sqlite3.Connection, checkpoint: VerificationCheckpoint
) -> tuple[int, str, str]:
    if checkpoint.audit_id == 0:
        if checkpoint.record_hash != GENESIS_HASH:
            raise AuditVerificationError("empty audit checkpoint hash is invalid")
        return 0, GENESIS_HASH, GENESIS_HASH
    row = connection.execute(_ROW_SELECT + " WHERE audit_id=?", (checkpoint.audit_id,)).fetchone()
    if row is None:
        raise AuditVerificationError("audit checkpoint anchor is missing")
    audit_id, record_hash = verify_row(row, checkpoint.anchor_previous_hash)
    if audit_id != checkpoint.audit_id or record_hash != checkpoint.record_hash:
        raise AuditVerificationError("audit checkpoint anchor changed")
    return audit_id, record_hash, checkpoint.anchor_previous_hash


def verify_row(row: tuple[SqlValue, ...], expected_previous: str) -> tuple[int, str]:
    action = AuditAction(str(row[7]))
    if row[14] is None:
        raise AuditVerificationError("audit detail version is missing")
    detail = parse_detail_json(action, str(row[14]))
    _ = AuditActorType(str(row[4]))
    _ = AuditAuthMechanism(str(row[6]))
    previous_hash = str(row[15])
    if previous_hash != expected_previous:
        raise AuditVerificationError("audit previous hash does not match")
    payload = {
        "action": action.value, "actor_id": str(row[5]), "actor_type": str(row[4]),
        "auth_mechanism": str(row[6]), "clock_quality": str(row[3]),
        "detail_json": detail.json, "hold_reference": row[18], "interaction_id": row[13],
        "occurred_at": str(row[1]), "outcome": str(row[10]), "previous_hash": previous_hash,
        "reason": row[11], "recorded_at": str(row[2]), "request_id": row[12],
        "retention_class": str(row[17]), "target_id": str(row[9]),
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
    "GENESIS_HASH", "MAX_AUDIT_ROWS", "AuditVerificationError", "DatabaseIdentity",
    "SqlValue", "VerificationCheckpoint", "database_identity", "verify_connection",
]
