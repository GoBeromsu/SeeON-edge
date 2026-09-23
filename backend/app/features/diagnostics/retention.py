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
MAX_UNITS_PER_ENFORCE: Final = 32

#: How long a dbstat measurement may be reused before it is taken again.
#: Between measurements the meter adds the bytes it has been told were
#: written, so usage is never reported below what is known. A page walk of a
#: ~1 GB table costs ~0.4 s; taking it on every ingest call serialised the
#: writer at ~1 call/s on the reference edge.
USAGE_REMEASURE_NS: Final = 1_000_000_000


class UsageMeter:
    """Measured on-disk usage with bounded staleness.

    ``value()`` returns the last dbstat measurement plus the bytes accrued
    since, and re-measures when the measurement is older than
    ``USAGE_REMEASURE_NS`` or was invalidated by a prune. Accrual scales the
    payload bytes written by the *measured* on-disk/payload ratio taken at the
    same instant (rows plus five indexes were 3.8x payload on the reference
    edge), so the bridge between two measurements is derived from a
    measurement, not from a constant. The estimate is only ever used to
    decide WHEN to measure: ``enforce_budget`` re-measures exactly before it
    prunes anything, and again after pruning. Usage can exceed high_water by
    at most one interval of accrual error, which is what control_reserve
    (total minus high_water) exists to absorb.
    """

    __slots__ = ("_accrued", "_measured", "_measured_at", "_ratio", "_remeasure_ns")

    def __init__(self, *, remeasure_ns: int = USAGE_REMEASURE_NS) -> None:
        self._remeasure_ns = remeasure_ns
        self._measured: int | None = None
        self._measured_at = 0
        self._ratio = 1.0
        self._accrued = 0

    def accrue(self, payload_bytes: int) -> None:
        self._accrued += int(max(0, payload_bytes) * self._ratio)

    def invalidate(self) -> None:
        self._measured = None

    def release(self, freed_payload_bytes: int) -> int:
        """Account bytes a prune just freed (measured ratio) and return the estimate.

        Keeps the measurement usable across a draining sequence of calls so
        the page walk runs at most once per USAGE_REMEASURE_NS while pruning,
        instead of once per call.
        """
        if self._measured is None:
            return 0
        self._accrued -= int(max(0, freed_payload_bytes) * self._ratio)
        return max(0, self._measured + self._accrued)

    def measured_over(self, high_water: int) -> bool:
        """True when the last exact measurement itself exceeded ``high_water``.

        Pruning on that basis needs no fresh walk; only an estimate-driven
        crossing does.
        """
        return self._measured is not None and self._measured > high_water

    def value(self, connection: sqlite3.Connection, now_ns: int) -> int:
        stale = self._measured is None or now_ns - self._measured_at >= self._remeasure_ns
        if stale:
            self._measured = used_bytes(connection)
            self._measured_at = now_ns
            self._accrued = 0
            payload = connection.execute(
                "SELECT COALESCE(SUM(payload_bytes), 0) FROM execution_segments"
            ).fetchone()
            payload_total = int(payload[0]) if payload else 0
            self._ratio = self._measured / payload_total if payload_total > 0 else 1.0
        return self._measured + self._accrued


def enforce_budget(
    connection: sqlite3.Connection,
    budget: RetentionBudget,
    now_ns: int,
    *,
    max_units: int = MAX_UNITS_PER_ENFORCE,
    meter: UsageMeter | None = None,
) -> bool:
    """Prune up to ``max_units`` whole units toward ``low_water``.

    Returns True when the ingest may commit: usage is within ``high_water``,
    or it is over but this call made progress (pruned at least one unit), so
    the envelope converges over the next calls. Returns False only when usage
    is over ``high_water`` and nothing is prunable - the honest
    STORAGE_UNAVAILABLE case. Usage is measured at most once per call (on
    entry, or reused from ``meter`` within its staleness bound), never once
    per pruned unit; a call that pruned leaves the next call to re-measure.
    """
    gauge = meter if meter is not None else UsageMeter()
    refresh_unit_terminals(connection, budget.unit_horizon_ns)
    coarsen_coverage(connection, budget.coverage_rows_per_epoch, now_ns)
    occupied = gauge.value(connection, now_ns)
    if occupied > budget.high_water and not gauge.measured_over(budget.high_water):
        # The accrued estimate only decides when to pay for a measurement.
        # Pruning is never driven by an estimate: re-measure exactly first.
        gauge.invalidate()
        occupied = gauge.value(connection, now_ns)
    pruned = 0
    while occupied > budget.high_water and pruned < max_units:
        unit_id = next_prunable_unit(connection)
        if unit_id is None:
            if force_oldest_units_terminal(connection, 1) == 0:
                break
            unit_id = next_prunable_unit(connection)
            if unit_id is None:
                break
        freed_payload = prune_unit(connection, unit_id, now_ns)
        pruned += 1
        # Stop near low_water using the measured ratio for the bytes just
        # freed; stopping early is always safe, the next call re-checks.
        occupied = gauge.release(freed_payload)
        if occupied <= budget.low_water:
            break
    if pruned:
        coarsen_coverage(connection, budget.coverage_rows_per_epoch, now_ns)
        drop_orphan_batches(connection)
        return True
    return occupied <= budget.high_water


__all__ = [
    "COVERAGE_ROWS_PER_EPOCH",
    "DEFAULT_UNIT_HORIZON_NS",
    "MAX_UNITS_PER_ENFORCE",
    "USAGE_REMEASURE_NS",
    "RetentionBudget",
    "UsageMeter",
    "coarsen_coverage",
    "enforce_budget",
    "force_oldest_units_terminal",
    "next_prunable_unit",
    "prune_unit",
    "refresh_unit_terminals",
    "used_bytes",
]
