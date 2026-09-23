"""Retention budget, unit terminal rules (Gate R), and coherent prune.

``total_bytes`` is required explicit config. Fractions below are design
constants, not deployment numbers.

    control_reserve = total_bytes // 16
    high_water = total_bytes - control_reserve
    low_water = (high_water * 7) // 8
    segment_bytes = total_bytes // 64
    max_record_bytes = total_bytes // 256
    coverage_rows_per_epoch = 512  (design constant; RetentionBudget field)

Minimum ``total_bytes`` is 256 because ``max_record_bytes = total_bytes // 256``
must be at least 1. Deployments must choose ``total_bytes`` large enough that
``control_reserve`` can hold live control rows; that is not a baked default.

``unit_horizon_ns`` defaults to 60_000_000_000 (60 s): documented as 2x the
deployed 30-frame/15fps window bound, to be replaced by a measured Gate M
value.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

from backend.app.features.diagnostics.prune import (
    coarsen_coverage,
    next_prunable_unit,
    prune_unit,
)
from backend.app.features.diagnostics.terminals import (
    force_oldest_units_terminal,
    refresh_unit_terminals,
)

COVERAGE_ROWS_PER_EPOCH: Final = 512
DEFAULT_UNIT_HORIZON_NS: Final = 60_000_000_000
_COVERAGE_ROW_BYTES: Final = 256
_BATCH_ROW_BYTES: Final = 256
_CONTROL_ROW_BYTES: Final = 128


@dataclass(frozen=True, slots=True)
class RetentionBudget:
    total_bytes: int
    unit_horizon_ns: int = DEFAULT_UNIT_HORIZON_NS
    coverage_rows_per_epoch: int = COVERAGE_ROWS_PER_EPOCH

    def __post_init__(self) -> None:
        if type(self.total_bytes) is not int or self.total_bytes < 256:
            raise ValueError(
                "total_bytes must be an explicit integer >= 256 "
                "(max_record_bytes = total_bytes // 256 must be >= 1)"
            )
        if type(self.unit_horizon_ns) is not int or self.unit_horizon_ns <= 0:
            raise ValueError("unit_horizon_ns must be a positive integer")
        if type(self.coverage_rows_per_epoch) is not int or self.coverage_rows_per_epoch < 1:
            raise ValueError("coverage_rows_per_epoch must be a positive integer")

    @property
    def control_reserve(self) -> int:
        return self.total_bytes // 16

    @property
    def high_water(self) -> int:
        return self.total_bytes - self.control_reserve

    @property
    def low_water(self) -> int:
        return (self.high_water * 7) // 8

    @property
    def segment_bytes(self) -> int:
        return self.total_bytes // 64

    @property
    def max_record_bytes(self) -> int:
        return self.total_bytes // 256


def logical_bytes(connection: sqlite3.Connection) -> int:
    payload = connection.execute(
        "SELECT COALESCE(SUM(payload_bytes), 0) FROM execution_records"
    ).fetchone()
    return int(payload[0]) + control_envelope_bytes(connection)


def control_envelope_bytes(connection: sqlite3.Connection) -> int:
    coverage = connection.execute("SELECT COUNT(*) FROM execution_coverage").fetchone()
    batches = connection.execute("SELECT COUNT(*) FROM execution_batches").fetchone()
    units = connection.execute("SELECT COUNT(*) FROM execution_units").fetchone()
    segments = connection.execute("SELECT COUNT(*) FROM execution_segments").fetchone()
    provenance = connection.execute("SELECT COUNT(*) FROM execution_provenance").fetchone()
    return (
        int(coverage[0]) * _COVERAGE_ROW_BYTES
        + int(batches[0]) * _BATCH_ROW_BYTES
        + (int(units[0]) + int(segments[0]) + int(provenance[0])) * _CONTROL_ROW_BYTES
    )


def enforce_budget(connection: sqlite3.Connection, budget: RetentionBudget, now_ns: int) -> bool:
    """Prune whole units until <= low_water. False when the envelope still fails."""
    refresh_unit_terminals(connection, budget.unit_horizon_ns)
    coarsen_coverage(connection, budget.coverage_rows_per_epoch, now_ns)
    while logical_bytes(connection) > budget.high_water:
        unit_id = next_prunable_unit(connection)
        if unit_id is None:
            if force_oldest_units_terminal(connection, 1) == 0:
                break
            unit_id = next_prunable_unit(connection)
            if unit_id is None:
                break
        prune_unit(connection, unit_id, now_ns)
        coarsen_coverage(connection, budget.coverage_rows_per_epoch, now_ns)
        if logical_bytes(connection) <= budget.low_water:
            break
    return logical_bytes(connection) <= budget.high_water


__all__ = [
    "COVERAGE_ROWS_PER_EPOCH",
    "DEFAULT_UNIT_HORIZON_NS",
    "RetentionBudget",
    "coarsen_coverage",
    "control_envelope_bytes",
    "enforce_budget",
    "force_oldest_units_terminal",
    "logical_bytes",
    "next_prunable_unit",
    "prune_unit",
    "refresh_unit_terminals",
]
