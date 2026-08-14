from __future__ import annotations

import re
import sqlite3
from typing import Final

DECISION_TRACE_ID_KEY: Final = "decision_trace_id"
_SHA256: Final = re.compile(r"[0-9a-f]{64}")


class DecisionTraceReferenceError(ValueError):
    pass


def validate_decision_trace_id(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecisionTraceReferenceError(
            "decision trace reference must be a lowercase SHA-256 identity"
        )
    return value


def require_decision_trace(
    connection: sqlite3.Connection,
    decision_trace_id: str,
    *,
    runtime_manifest_sha256: str | None,
) -> None:
    row = connection.execute(
        "SELECT runtime_manifest_sha256 FROM evidence_decision_traces WHERE trace_id = ?",
        (decision_trace_id,),
    ).fetchone()
    if row is None:
        raise DecisionTraceReferenceError("decision trace reference does not resolve")
    if runtime_manifest_sha256 is not None and str(row[0]) != runtime_manifest_sha256:
        raise DecisionTraceReferenceError(
            "decision trace reference contradicts the event runtime manifest"
        )


__all__ = [
    "DECISION_TRACE_ID_KEY",
    "DecisionTraceReferenceError",
    "require_decision_trace",
    "validate_decision_trace_id",
]
