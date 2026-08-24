"""Bounded audit-chain verification and connection-observed checkpoints.

The local threat boundary detects commits visible to the retained SQLite observer.
It cannot defend an attacker who can rewrite both the database and process memory
(or a future external anchor); that requires an authority outside this process.
"""

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
    observer_id: str
    data_version: int
    rolling_audit_id: int
    rolling_previous_hash: str


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
    *,
    observer_id: str = "ephemeral",
    data_version: int | None = None,
) -> VerificationCheckpoint:
    """Verify a suffix and one rolling historical page from one observed snapshot."""
    count_row = connection.execute("SELECT COUNT(audit_id) FROM audit_events").fetchone()
    count = 0 if count_row is None else int(count_row[0])
    if count >= MAX_AUDIT_ROWS:
        raise AuditVerificationError("audit history reached the one-million-row refusal limit")
    observed_version = _data_version(connection) if data_version is None else data_version
    schema_version, trigger_fingerprint = _schema_state(connection)
    if checkpoint is not None and checkpoint.database_identity != identity:
        raise AuditVerificationError("audit checkpoint database identity changed")
    same_observer = checkpoint is not None and checkpoint.observer_id == observer_id
    schema_changed = checkpoint is not None and (
        checkpoint.schema_version != schema_version
        or checkpoint.trigger_fingerprint != trigger_fingerprint
    )
    data_changed = (
        checkpoint is not None
        and same_observer
        and checkpoint.data_version != observed_version
    )
    # A valid new audit tail accounts for a governed caller-owned commit because
    # the business write and audit INSERT share that SQLite transaction. A commit
    # without such a tail is unexplained and must complete a full verification.
    has_tail = checkpoint is not None and connection.execute(
        "SELECT 1 FROM audit_events WHERE audit_id>? LIMIT 1", (checkpoint.audit_id,)
    ).fetchone() is not None
    full = (
        checkpoint is None
        or not same_observer
        or schema_changed
        or (data_changed and not has_tail)
    )
    if full:
        last_id, previous_hash, anchor_previous_hash = _verify_all(connection)
        rolling_id, rolling_hash = 0, GENESIS_HASH
    else:
        assert checkpoint is not None
        last_id, previous_hash, anchor_previous_hash = _verify_anchor(connection, checkpoint)
        last_id, previous_hash, anchor_previous_hash = _verify_suffix(
            connection, last_id, previous_hash, anchor_previous_hash
        )
        rolling_id, rolling_hash = _verify_rolling_page(
            connection,
            checkpoint.rolling_audit_id,
            checkpoint.rolling_previous_hash,
            last_id,
        )
    return VerificationCheckpoint(
        last_id, previous_hash, anchor_previous_hash, schema_version,
        trigger_fingerprint, identity, observer_id, observed_version,
        rolling_id, rolling_hash,
    )


def _data_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA data_version").fetchone()
    if row is None or type(row[0]) is not int:
        raise AuditVerificationError("audit data version is unreadable")
    return row[0]


def _verify_all(connection: sqlite3.Connection) -> tuple[int, str, str]:
    return _verify_suffix(connection, 0, GENESIS_HASH, GENESIS_HASH)


def _verify_suffix(
    connection: sqlite3.Connection,
    last_id: int,
    previous_hash: str,
    anchor_previous_hash: str,
) -> tuple[int, str, str]:
    while True:
        rows = connection.execute(
            _ROW_SELECT + " WHERE audit_id>? ORDER BY audit_id LIMIT ?",
            (last_id, _VERIFY_PAGE_SIZE),
        ).fetchall()
        if not rows:
            return last_id, previous_hash, anchor_previous_hash
        for row in rows:
            anchor_previous_hash = previous_hash
            last_id, previous_hash = verify_row(row, previous_hash)


def _verify_rolling_page(
    connection: sqlite3.Connection,
    rolling_id: int,
    rolling_previous_hash: str,
    terminal_id: int,
) -> tuple[int, str]:
    rows = connection.execute(
        _ROW_SELECT + " WHERE audit_id>? AND audit_id<=? ORDER BY audit_id LIMIT ?",
        (rolling_id, terminal_id, _VERIFY_PAGE_SIZE),
    ).fetchall()
    if not rows:
        return 0, GENESIS_HASH
    previous_hash = rolling_previous_hash
    last_id = rolling_id
    for row in rows:
        last_id, previous_hash = verify_row(row, previous_hash)
    if last_id >= terminal_id:
        return 0, GENESIS_HASH
    return last_id, previous_hash


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
