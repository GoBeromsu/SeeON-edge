"""Environment-sourced execution-record export settings (ADR-0002).

``None`` means the feature is off. Enabling without every capacity/batch/flush
value refuses to start; those integers have no defaults.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from worker.runtime.config.errors import WorkerConfigError

ML_WORKER_EXECUTION_RECORDS_ENABLED_ENV: Final = "ML_WORKER_EXECUTION_RECORDS_ENABLED"
ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY_ENV: Final = "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY"
ML_WORKER_EXECUTION_RECORDS_BATCH_MAX_ENV: Final = "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX"
ML_WORKER_EXECUTION_RECORDS_FLUSH_MS_ENV: Final = "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_FALSY: Final = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class ExecutionRecordsSettings:
    lane_capacity: int
    batch_max: int
    flush_ms: int


def execution_records_settings_from_environment(
    environ: Mapping[str, str] | None = None,
) -> ExecutionRecordsSettings | None:
    """Return settings when enabled; ``None`` is the seam default (feature off)."""
    env = os.environ if environ is None else environ
    enabled = _bool_env(ML_WORKER_EXECUTION_RECORDS_ENABLED_ENV, env)
    if enabled is None or not enabled:
        return None
    because = "when ML_WORKER_EXECUTION_RECORDS_ENABLED=1"
    return ExecutionRecordsSettings(
        lane_capacity=_required_positive_int(
            ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY_ENV, env, because=because
        ),
        batch_max=_required_positive_int(
            ML_WORKER_EXECUTION_RECORDS_BATCH_MAX_ENV, env, because=because
        ),
        flush_ms=_required_positive_int(
            ML_WORKER_EXECUTION_RECORDS_FLUSH_MS_ENV, env, because=because
        ),
    )


def _bool_env(name: str, env: Mapping[str, str]) -> bool | None:
    raw = env.get(name, "").strip().lower()
    if raw == "":
        return None
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    raise WorkerConfigError(f"{name} must be a boolean ({sorted(_TRUTHY | _FALSY)}), got {raw!r}")


def _required_positive_int(name: str, env: Mapping[str, str], *, because: str) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        raise WorkerConfigError(f"{name} is required {because}")
    try:
        value = int(raw)
    except ValueError as error:
        raise WorkerConfigError(f"{name} must be an integer, got {raw!r}") from error
    if value < 1:
        raise WorkerConfigError(f"{name} must be a positive integer, got {raw!r}")
    return value


__all__ = [
    "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX_ENV",
    "ML_WORKER_EXECUTION_RECORDS_ENABLED_ENV",
    "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS_ENV",
    "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY_ENV",
    "ExecutionRecordsSettings",
    "execution_records_settings_from_environment",
]
