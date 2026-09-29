"""Audit-chain row verification shared by the PostgreSQL audit verifier."""

from __future__ import annotations

import json
from typing import Final

from backend.app.edge_db.functions import audit_record_hash
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    parse_detail_json,
)

GENESIS_HASH: Final = "0" * 64
MAX_AUDIT_ROWS: Final = 1_000_000
AUDIT_ROW_COLUMNS: Final = (
    "audit_id",
    "occurred_at",
    "recorded_at",
    "clock_quality",
    "actor_type",
    "actor_id",
    "auth_mechanism",
    "action",
    "target_type",
    "target_id",
    "outcome",
    "reason",
    "request_id",
    "interaction_id",
    "detail_json",
    "previous_hash",
    "record_hash",
    "retention_class",
    "hold_reference",
)

SqlValue = str | int | float | bytes | None


class AuditVerificationError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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
        "action": action.value,
        "actor_id": str(row[5]),
        "actor_type": str(row[4]),
        "auth_mechanism": str(row[6]),
        "clock_quality": str(row[3]),
        "detail_json": detail.json,
        "hold_reference": row[18],
        "interaction_id": row[13],
        "occurred_at": str(row[1]),
        "outcome": str(row[10]),
        "previous_hash": previous_hash,
        "reason": row[11],
        "recorded_at": str(row[2]),
        "request_id": row[12],
        "retention_class": str(row[17]),
        "target_id": str(row[9]),
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
    "AUDIT_ROW_COLUMNS",
    "GENESIS_HASH",
    "MAX_AUDIT_ROWS",
    "AuditVerificationError",
    "SqlValue",
    "verify_row",
]
