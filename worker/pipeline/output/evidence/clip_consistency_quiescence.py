"""Strict operator assertion that every worker-state writer is stopped."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from worker.pipeline.output.evidence.clip_consistency_io import (
    read_strict_json,
    validate_under_root,
)
from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError

_KEYS = frozenset(
    {
        "format_version",
        "state_db",
        "clip_store",
        "stopped_service",
        "stopped_db_writers",
        "operator_uid",
        "issued_at",
        "expires_at",
    }
)
_WRITERS = ("event", "config", "fault")


def validate_quiescence_receipt(
    path: Path,
    *,
    state_db: Path,
    clip_store: Path,
    maintenance_root: Path,
    expected_uid: int,
    now: float | None = None,
) -> None:
    validate_under_root(path, maintenance_root, allow_missing_leaf=False)
    payload = read_strict_json(
        path,
        expected_uid=expected_uid,
        exact_mode=0o600,
        error_code="quiescence_invalid",
    )
    if frozenset(payload) != _KEYS:
        raise ClipConsistencyError("quiescence_invalid", "receipt key set differs")
    issued_at = _integer(payload, "issued_at")
    expires_at = _integer(payload, "expires_at")
    observed = time.time() if now is None else now
    writers = payload.get("stopped_db_writers")
    valid = (
        _integer(payload, "format_version") == 1
        and _integer(payload, "operator_uid") == expected_uid
        and _string(payload, "state_db") == str(state_db.resolve(strict=True))
        and _string(payload, "clip_store") == str(clip_store.resolve(strict=True))
        and _string(payload, "stopped_service") == "ml-worker"
        and isinstance(writers, list)
        and tuple(cast("list[object]", writers)) == _WRITERS
        and issued_at <= observed <= expires_at
        and 0 < expires_at - issued_at <= 3600
    )
    if not valid:
        raise ClipConsistencyError("quiescence_invalid", "receipt assertion differs")


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ClipConsistencyError("quiescence_invalid", "receipt field type differs")
    return value


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ClipConsistencyError("quiescence_invalid", "receipt field type differs")
    return value


__all__ = ["validate_quiescence_receipt"]
