"""Query result types and camera-range reads."""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from backend.app.features.diagnostics.coverage import (
    AvailabilityRange,
    QueryableRange,
    availability,
    queryable_range,
)
from backend.app.features.diagnostics.records import (
    CoverageKind,
    RecordKind,
    UnitCausalState,
)

_CURSOR_SEP = "\0"


@dataclass(frozen=True, slots=True)
class StoredRecord:
    record_id: str
    record_kind: RecordKind
    camera_id: str
    worker_boot_id: str
    source_generation: int
    stream_epoch: int
    producer: str
    producer_sequence: int
    frame_seq: int | None
    source_pts_ns: int | None
    observed_at_ns: int
    time_quality: str
    causal_unit_id: str
    parent_record_id: str | None
    outcome: str
    reason: str | None
    payload: Mapping[str, object]
    payload_bytes: int
    committed_at_ns: int
    segment_id: int
    provenance_id: str


@dataclass(frozen=True, slots=True)
class UnitView:
    causal_unit_id: str
    camera_id: str
    worker_boot_id: str
    source_generation: int
    stream_epoch: int
    causal_state: UnitCausalState
    terminal: bool
    first_observed_ns: int
    last_observed_ns: int
    record_count: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class CoverageRow:
    coverage_id: int
    camera_id: str
    worker_boot_id: str
    source_generation: int
    stream_epoch: int
    coverage_kind: CoverageKind
    producer: str | None
    from_sequence: int | None
    to_sequence: int | None
    from_ns: int
    to_ns: int
    record_count: int
    exact: bool
    cause: str
    recorded_at_ns: int


@dataclass(frozen=True, slots=True)
class QueryResult:
    records: tuple[StoredRecord, ...]
    units: tuple[UnitView, ...]
    coverage: tuple[CoverageRow, ...]
    availability: tuple[AvailabilityRange, ...]
    queryable_range: QueryableRange
    next_cursor: str | None


def encode_cursor(observed_at_ns: int, producer_sequence: int, record_id: str) -> str:
    payload = f"{observed_at_ns}{_CURSOR_SEP}{producer_sequence}{_CURSOR_SEP}{record_id}"
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> tuple[int, int, str]:
    raw = base64.urlsafe_b64decode(cursor.encode()).decode().split(_CURSOR_SEP, 2)
    return int(raw[0]), int(raw[1]), raw[2]


def load_coverage_row(row: tuple[object, ...]) -> CoverageRow:
    return CoverageRow(
        coverage_id=int(row[0]),
        camera_id=str(row[1]),
        worker_boot_id=str(row[2]),
        source_generation=int(row[3]),
        stream_epoch=int(row[4]),
        coverage_kind=CoverageKind(str(row[5])),
        producer=None if row[6] is None else str(row[6]),
        from_sequence=None if row[7] is None else int(row[7]),
        to_sequence=None if row[8] is None else int(row[8]),
        from_ns=int(row[9]),
        to_ns=int(row[10]),
        record_count=int(row[11]),
        exact=int(row[12]) == 1,
        cause=str(row[13]),
        recorded_at_ns=int(row[14]),
    )


def load_unit_view(row: tuple[object, ...]) -> UnitView:
    return UnitView(
        causal_unit_id=str(row[0]),
        camera_id=str(row[1]),
        worker_boot_id=str(row[2]),
        source_generation=int(row[3]),
        stream_epoch=int(row[4]),
        causal_state=UnitCausalState(str(row[5])),
        terminal=int(row[6]) == 1,
        first_observed_ns=int(row[7]),
        last_observed_ns=int(row[8]),
        record_count=int(row[9]),
        payload_bytes=int(row[10]),
    )


def load_stored_record(row: tuple[object, ...]) -> StoredRecord:
    return StoredRecord(
        record_id=str(row[0]),
        record_kind=RecordKind(str(row[1])),
        camera_id=str(row[2]),
        worker_boot_id=str(row[3]),
        source_generation=int(row[4]),
        stream_epoch=int(row[5]),
        producer=str(row[6]),
        producer_sequence=int(row[7]),
        frame_seq=None if row[8] is None else int(row[8]),
        source_pts_ns=None if row[9] is None else int(row[9]),
        observed_at_ns=int(row[10]),
        time_quality=str(row[11]),
        causal_unit_id=str(row[12]),
        parent_record_id=None if row[13] is None else str(row[13]),
        outcome=str(row[14]),
        reason=None if row[15] is None else str(row[15]),
        payload=json.loads(str(row[16])),
        payload_bytes=int(row[17]),
        committed_at_ns=int(row[18]),
        segment_id=int(row[19]),
        provenance_id=str(row[20]),
    )


def execute_query(
    connection: sqlite3.Connection,
    *,
    camera_id: str,
    from_ns: int,
    to_ns: int,
    limit: int,
    cursor: str | None,
) -> QueryResult:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    params: list[object] = [camera_id, from_ns, to_ns]
    where = "camera_id = ? AND observed_at_ns >= ? AND observed_at_ns <= ?"
    if cursor is not None:
        observed, sequence, record_id = decode_cursor(cursor)
        where += (
            " AND (observed_at_ns > ? OR "
            "(observed_at_ns = ? AND producer_sequence > ?) OR "
            "(observed_at_ns = ? AND producer_sequence = ? "
            "AND record_id > ?))"
        )
        params.extend((observed, observed, sequence, observed, sequence, record_id))
    params.append(limit + 1)
    rows = connection.execute(
        f"""
        SELECT record_id, record_kind, camera_id, worker_boot_id, source_generation,
               stream_epoch, producer, producer_sequence, frame_seq, source_pts_ns,
               observed_at_ns, time_quality, causal_unit_id, parent_record_id, outcome,
               reason, payload, payload_bytes, committed_at_ns, segment_id, provenance_id
        FROM execution_records
        WHERE {where}
        ORDER BY observed_at_ns, producer_sequence, record_id
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    records = tuple(load_stored_record(row) for row in rows[:limit])
    next_cursor = None
    if len(rows) > limit and records:
        last = records[-1]
        next_cursor = encode_cursor(last.observed_at_ns, last.producer_sequence, last.record_id)
    unit_rows = connection.execute(
        """
        SELECT causal_unit_id, camera_id, worker_boot_id, source_generation, stream_epoch,
               causal_state, terminal, first_observed_ns, last_observed_ns,
               record_count, payload_bytes
        FROM execution_units
        WHERE camera_id = ? AND first_observed_ns <= ? AND last_observed_ns >= ?
        ORDER BY first_observed_ns, causal_unit_id
        """,
        (camera_id, to_ns, from_ns),
    ).fetchall()
    coverage_rows = connection.execute(
        """
        SELECT coverage_id, camera_id, worker_boot_id, source_generation, stream_epoch,
               coverage_kind, producer, from_sequence, to_sequence, from_ns, to_ns,
               record_count, exact, cause, recorded_at_ns
        FROM execution_coverage
        WHERE camera_id = ? AND from_ns <= ? AND to_ns >= ?
        ORDER BY from_ns, coverage_id
        """,
        (camera_id, to_ns, from_ns),
    ).fetchall()
    return QueryResult(
        records=records,
        units=tuple(load_unit_view(row) for row in unit_rows),
        coverage=tuple(load_coverage_row(row) for row in coverage_rows),
        availability=availability(connection, camera_id, from_ns, to_ns),
        queryable_range=queryable_range(connection, camera_id),
        next_cursor=next_cursor,
    )


__all__ = [
    "CoverageRow",
    "QueryResult",
    "StoredRecord",
    "UnitView",
    "execute_query",
]
