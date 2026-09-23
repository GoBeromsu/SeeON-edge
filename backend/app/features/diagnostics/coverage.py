"""Bounded coverage summary and query-truth availability ranges."""

from __future__ import annotations

import sqlite3
from bisect import bisect_right
from dataclasses import dataclass
from typing import Final

from backend.app.features.diagnostics.records import (
    AvailabilityKind,
    CoverageKind,
)

_INF: Final = 1 << 62

_AVAIL_EXACT: Final = {
    CoverageKind.MISSING_NOT_RECORDED: AvailabilityKind.MISSING_NOT_RECORDED,
    CoverageKind.DELETED_BY_CAPACITY: AvailabilityKind.DELETED_BY_CAPACITY,
    CoverageKind.UNKNOWN: AvailabilityKind.UNKNOWN,
}


@dataclass(frozen=True, slots=True)
class AvailabilityRange:
    from_ns: int
    to_ns: int
    kind: AvailabilityKind


@dataclass(frozen=True, slots=True)
class QueryableRange:
    min_observed_at_ns: int | None
    max_observed_at_ns: int | None


def insert_coverage(
    connection: sqlite3.Connection,
    *,
    camera_id: str,
    worker_boot_id: str,
    source_generation: int,
    stream_epoch: int,
    kind: CoverageKind,
    producer: str | None,
    from_sequence: int | None,
    to_sequence: int | None,
    from_ns: int,
    to_ns: int,
    record_count: int,
    exact: bool,
    cause: str,
    recorded_at_ns: int,
) -> None:
    connection.execute(
        """
        INSERT INTO execution_coverage (
            camera_id, worker_boot_id, source_generation, stream_epoch,
            coverage_kind, producer, from_sequence, to_sequence,
            from_ns, to_ns, record_count, exact, cause, recorded_at_ns
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            camera_id,
            worker_boot_id,
            source_generation,
            stream_epoch,
            str(kind),
            producer,
            from_sequence,
            to_sequence,
            from_ns,
            to_ns,
            record_count,
            1 if exact else 0,
            cause,
            recorded_at_ns,
        ),
    )


def parent_loss_kind(
    connection: sqlite3.Connection, camera_id: str, observed_at_ns: int
) -> CoverageKind | None:
    deleted = connection.execute(
        """
        SELECT coverage_id FROM execution_coverage
        WHERE camera_id = ? AND coverage_kind = ? AND exact = 1
          AND from_ns <= ? AND to_ns >= ?
        LIMIT 1
        """,
        (camera_id, str(CoverageKind.DELETED_BY_CAPACITY), observed_at_ns, observed_at_ns),
    ).fetchone()
    if deleted is not None:
        return CoverageKind.DELETED_BY_CAPACITY
    coarsened = connection.execute(
        """
        SELECT coverage_id FROM execution_coverage
        WHERE camera_id = ? AND coverage_kind = ? AND exact = 0
          AND from_ns <= ? AND to_ns >= ?
        LIMIT 1
        """,
        (camera_id, str(CoverageKind.UNKNOWN_COARSENED), observed_at_ns, observed_at_ns),
    ).fetchone()
    return CoverageKind.UNKNOWN_COARSENED if coarsened is not None else None


def queryable_range(connection: sqlite3.Connection, camera_id: str) -> QueryableRange:
    row = connection.execute(
        """
        SELECT MIN(observed_at_ns), MAX(observed_at_ns)
        FROM execution_records WHERE camera_id = ?
        """,
        (camera_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return QueryableRange(None, None)
    return QueryableRange(int(row[0]), int(row[1]))


def availability(
    connection: sqlite3.Connection, camera_id: str, from_ns: int, to_ns: int
) -> tuple[AvailabilityRange, ...]:
    """Paint the query window with the five availability words.

    AVAILABLE is a *span*, not an instant: two consecutive records of one
    (boot, producer) lane with adjacent ``producer_sequence`` prove nothing was
    lost between them (the worker counts every drop as a coverage gap), so the
    whole interval between them is available. That is an exact rule with no
    cadence threshold. A sequence discontinuity, a lane boundary, or the edge
    of the retained evidence ends a span. Everything else inside the retained
    bounds is painted from coverage rows (MISSING_NOT_RECORDED /
    DELETED_BY_CAPACITY exact, UNKNOWN_COARSENED otherwise) or is UNKNOWN;
    outside the bounds it is UNKNOWN.
    """
    if to_ns < from_ns:
        return ()
    rows = connection.execute(
        """
        SELECT observed_at_ns, worker_boot_id, producer, producer_sequence
        FROM execution_records
        WHERE camera_id = ? AND observed_at_ns >= ? AND observed_at_ns <= ?
        ORDER BY worker_boot_id, producer, producer_sequence
        """,
        (camera_id, from_ns, to_ns),
    ).fetchall()
    coverage = connection.execute(
        """
        SELECT coverage_kind, exact, from_ns, to_ns FROM execution_coverage
        WHERE camera_id = ? AND from_ns <= ? AND to_ns >= ?
        ORDER BY from_ns, coverage_id
        """,
        (camera_id, to_ns, from_ns),
    ).fetchall()
    bounds = connection.execute(
        """
        SELECT
          (SELECT MIN(observed_at_ns) FROM execution_records WHERE camera_id = ?),
          (SELECT MAX(observed_at_ns) FROM execution_records WHERE camera_id = ?),
          (SELECT MIN(from_ns) FROM execution_coverage WHERE camera_id = ?),
          (SELECT MAX(to_ns) FROM execution_coverage WHERE camera_id = ?)
        """,
        (camera_id, camera_id, camera_id, camera_id),
    ).fetchone()
    earliest_parts = [value for value in (bounds[0], bounds[2]) if value is not None]
    latest_parts = [value for value in (bounds[1], bounds[3]) if value is not None]
    earliest = min(int(value) for value in earliest_parts) if earliest_parts else None
    latest = max(int(value) for value in latest_parts) if latest_parts else None

    available = _merge(_lane_spans(rows))
    endpoints = {from_ns, to_ns + 1}
    for span_from, span_to in available:
        endpoints.add(span_from)
        endpoints.add(span_to + 1)
    for _kind, _exact, cover_from, cover_to in coverage:
        endpoints.add(int(cover_from))
        endpoints.add(int(cover_to) + 1)
    if earliest is not None:
        endpoints.add(earliest)
    if latest is not None:
        endpoints.add(latest + 1)
    ordered = sorted(point for point in endpoints if from_ns <= point <= to_ns + 1)
    painted: list[AvailabilityRange] = []
    for index in range(len(ordered) - 1):
        start, end_exclusive = ordered[index], ordered[index + 1]
        if end_exclusive <= start:
            continue
        last = end_exclusive - 1
        kind = _atom_kind(start, last, earliest, latest, available, coverage)
        if painted and painted[-1].kind is kind and painted[-1].to_ns + 1 == start:
            painted[-1] = AvailabilityRange(painted[-1].from_ns, last, kind)
        else:
            painted.append(AvailabilityRange(start, last, kind))
    return tuple(painted)


def _lane_spans(rows: list[tuple[object, ...]]) -> list[tuple[int, int]]:
    """[from, to] spans of contiguous producer_sequence within one lane.

    ``rows`` are ordered by (boot, producer, producer_sequence). A single
    record is a zero-length span at its own time.
    """
    spans: list[tuple[int, int]] = []
    lane: tuple[object, object] | None = None
    span_from = span_to = 0
    previous_sequence: int | None = None
    for observed_raw, boot, producer, sequence_raw in rows:
        observed, sequence = int(observed_raw), int(sequence_raw)
        key = (boot, producer)
        contiguous = (
            key == lane and previous_sequence is not None and sequence == previous_sequence + 1
        )
        if contiguous:
            span_to = max(span_to, observed)
        else:
            if lane is not None:
                spans.append((span_from, span_to))
            lane, span_from, span_to = key, observed, observed
        previous_sequence = sequence
    if lane is not None:
        spans.append((span_from, span_to))
    return spans


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of spans across lanes: any lane available makes the time available."""
    merged: list[tuple[int, int]] = []
    for span_from, span_to in sorted(spans):
        if merged and span_from <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span_to))
        else:
            merged.append((span_from, span_to))
    return merged


def _atom_kind(
    start: int,
    last: int,
    earliest: int | None,
    latest: int | None,
    available: list[tuple[int, int]],
    coverage: list[tuple[object, ...]],
) -> AvailabilityKind:
    if earliest is None or latest is None or last < earliest or start > latest:
        return AvailabilityKind.UNKNOWN
    # ``available`` is sorted and disjoint; an atom never straddles a span
    # boundary because every boundary is an endpoint.
    index = bisect_right(available, (start, _INF)) - 1
    if index >= 0 and available[index][0] <= start and last <= available[index][1]:
        return AvailabilityKind.AVAILABLE
    exact_kind: AvailabilityKind | None = None
    coarsened = False
    for kind_raw, exact, cover_from, cover_to in coverage:
        if int(cover_from) > last or int(cover_to) < start:
            continue
        kind = CoverageKind(str(kind_raw))
        if int(exact) == 1 and kind in _AVAIL_EXACT:
            exact_kind = _AVAIL_EXACT[kind]
            break
        if kind is CoverageKind.UNKNOWN_COARSENED:
            coarsened = True
    if exact_kind is not None:
        return exact_kind
    if coarsened:
        return AvailabilityKind.UNKNOWN_COARSENED
    return AvailabilityKind.UNKNOWN


__all__ = [
    "AvailabilityRange",
    "QueryableRange",
    "availability",
    "insert_coverage",
    "parent_loss_kind",
    "queryable_range",
]
