"""Strict scalar extraction for backup receipt JSON."""

from __future__ import annotations

from collections.abc import Mapping

from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError


def integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ClipConsistencyError("backup_receipt_invalid", "receipt field type differs")
    return value


def string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ClipConsistencyError("backup_receipt_invalid", "receipt field type differs")
    return value


def boolean(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ClipConsistencyError("backup_receipt_invalid", "receipt field type differs")
    return value


__all__ = ["boolean", "integer", "string"]
