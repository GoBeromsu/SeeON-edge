from __future__ import annotations

import re
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


__all__ = [
    "DECISION_TRACE_ID_KEY",
    "DecisionTraceReferenceError",
    "validate_decision_trace_id",
]
