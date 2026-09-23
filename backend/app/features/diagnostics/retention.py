"""Retention budget, unit terminal rules (Gate R), and coherent prune.

``total_bytes`` is required explicit config. Fractions below are design
constants, not deployment numbers.

    control_reserve = total_bytes // 16
    high_water = total_bytes - control_reserve
    low_water = (high_water * 7) // 8
    segment_bytes = total_bytes // 64
    max_record_bytes = total_bytes // 256
    coverage_rows_per_epoch = 512  (design constant; RetentionBudget field)

``total_bytes`` is the on-disk byte envelope of the execution_* tables and
their indexes (not the whole edge.sqlite3 file, not the WAL). Empty b-trees
already occupy pages, so the smallest meaningful budget is a few MiB; the
integer floor ``total_bytes >= 256`` exists only so
``max_record_bytes = total_bytes // 256`` is at least 1 and is not a
deployment size.

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
    drop_orphan_batches,
    next_prunable_unit,
    prune_unit,
)
from backend.app.features.diagnostics.terminals import (
    force_oldest_units_terminal,
    refresh_unit_terminals,
)

COVERAGE_ROWS_PER_EPOCH: Final = 512
DEFAULT_UNIT_HORIZON_NS: Final = 60_000_000_000
_DBSTAT_MISSING: Final = (
    "dbstat is unavailable; SQLite must be compiled with SQLITE_ENABLE_DBSTAT_VTAB"
)


@dataclass(frozen=True, slots=True)
class RetentionBudget:
    """On-disk capacity envelope for the execution_* tables and indexes.

    ``total_bytes`` is compared against ``used_bytes`` (dbstat page sizes of
    every execution_* table and every index whose ``tbl_name`` is an
    execution_* table). It is not the size of edge.sqlite3 and does not
    include the WAL.
    """

    total_bytes: int
    unit_horizon_ns: int = DEFAULT_UNIT_HORIZON_NS
    coverage_rows_per_epoch: int = COVERAGE_ROWS_PER_EPOCH

    def __post_init__(self) -> None:
        if type(self.total_bytes) is not int or self.total_bytes < 256:
            raise ValueError(
                "total_bytes must be an explicit integer >= 256 "
                "(max_record_bytes = total_bytes // 256 must be >= 1; "
                "this floor is not a deployment size — empty execution_* "
                "b-trees already occupy pages, so a meaningful budget is a few MiB)"
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


def _execution_object_names(connection: sqlite3.Connection) -> tuple[str, ...]:
    rows = connection.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type IN ('table', 'index')
          AND (name LIKE 'execution_%' OR tbl_name LIKE 'execution_%')
        """
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def used_bytes(connection: sqlite3.Connection) -> int:
    """Exact on-disk bytes of every execution_* table and its indexes."""
    names = _execution_object_names(connection)
    if not names:
        return 0
    placeholders = ",".join("?" * len(names))
    try:
        row = connection.execute(
            f"SELECT COALESCE(SUM(pgsize), 0) FROM dbstat "
            f"WHERE aggregate = TRUE AND name IN ({placeholders})",
            names,
        ).fetchone()
    except sqlite3.OperationalError as error:
        if "no such table: dbstat" in str(error).lower():
            raise RuntimeError(_DBSTAT_MISSING) from error
        raise
    return 0 if row is None else int(row[0])


#: Whole units pruned per enforce_budget call. Bounds the work one ingest
#: request can do so a large backlog is drained across requests instead of
#: one request pruning for minutes while the loop is blocked. Not a budget
#: number: it only shapes latency.
MAX_UNITS_PER_ENFORCE: Final = 8


def enforce_budget(
    connection: sqlite3.Connection,
    budget: RetentionBudget,
    now_ns: int,
    *,
    max_units: int = MAX_UNITS_PER_ENFORCE,
) -> bool:
    """Prune up to ``max_units`` whole units toward ``low_water``.

    Returns True when the ingest may commit: usage is within ``high_water``,
    or it is over but this call made progress (pruned at least one unit), so
    the envelope converges over the next calls. Returns False only when usage
    is over ``high_water`` and nothing is prunable - the honest
    STORAGE_UNAVAILABLE case.
    """
    refresh_unit_terminals(connection, budget.unit_horizon_ns)
    coarsen_coverage(connection, budget.coverage_rows_per_epoch, now_ns)
    occupied = used_bytes(connection)
    pruned = 0
    while occupied > budget.high_water and pruned < max_units:
        unit_id = next_prunable_unit(connection)
        if unit_id is None:
            if force_oldest_units_terminal(connection, 1) == 0:
                break
            unit_id = next_prunable_unit(connection)
            if unit_id is None:
                break
        prune_unit(connection, unit_id, now_ns)
        pruned += 1
        occupied = used_bytes(connection)
        if occupied <= budget.low_water:
            break
    if pruned:
        coarsen_coverage(connection, budget.coverage_rows_per_epoch, now_ns)
        drop_orphan_batches(connection)
    return occupied <= budget.high_water or pruned > 0


__all__ = [
    "COVERAGE_ROWS_PER_EPOCH",
    "DEFAULT_UNIT_HORIZON_NS",
    "MAX_UNITS_PER_ENFORCE",
    "RetentionBudget",
    "coarsen_coverage",
    "enforce_budget",
    "force_oldest_units_terminal",
    "next_prunable_unit",
    "prune_unit",
    "refresh_unit_terminals",
    "used_bytes",
]
