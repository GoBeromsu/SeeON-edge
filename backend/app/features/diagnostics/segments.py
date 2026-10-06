"""OPEN segment assignment and byte-threshold sealing for one ingest batch."""

from __future__ import annotations

import psycopg

from backend.app.features.diagnostics.records import (
    ExecutionRecordInput,
    SegmentStorageState,
)
from backend.app.features.diagnostics.retention import RetentionBudget

_Lane = tuple[str, str, int, int]


class SegmentAllocator:
    """Assign records to their lane's OPEN segment within one ingest transaction.

    The OPEN segment of each lane is read once and then tracked in memory;
    count and byte increments are buffered until ``flush``. Sealing and
    opening the next segment are written immediately, so segment ids are
    allocated in the same order as a per-record assignment would. Only this
    allocator may write execution_segments while it is live.
    """

    def __init__(
        self, connection: psycopg.Connection, budget: RetentionBudget, now_ns: int
    ) -> None:
        self._connection = connection
        self._budget = budget
        self._now_ns = now_ns
        # lane -> [segment_id, payload_bytes, segment_ordinal] of its OPEN segment
        self._open: dict[_Lane, list[int] | None] = {}
        # segment_id -> [record_count, payload_bytes] not yet written
        self._pending: dict[int, list[int]] = {}

    def assign(self, record: ExecutionRecordInput, payload_bytes: int) -> int:
        key = (
            record.camera_id,
            record.worker_boot_id,
            record.source_generation,
            record.stream_epoch,
        )
        if key not in self._open:
            self._open[key] = self._load_open(key)
        current = self._open[key]
        if current is not None and current[1] + payload_bytes <= self._budget.segment_bytes:
            current[1] += payload_bytes
            pending = self._pending.setdefault(current[0], [0, 0])
            pending[0] += 1
            pending[1] += payload_bytes
            return current[0]
        if current is not None:
            self._connection.execute(
                "UPDATE execution_segments SET storage_state = %s, sealed_at_ns = %s "
                "WHERE segment_id = %s",
                (str(SegmentStorageState.SEALED_PENDING), self._now_ns, current[0]),
            )
            ordinal = current[2] + 1
        else:
            last = self._connection.execute(
                """
                SELECT COALESCE(MAX(segment_ordinal), -1) FROM execution_segments
                WHERE camera_id = %s AND worker_boot_id = %s AND source_generation = %s
                  AND stream_epoch = %s
                """,
                key,
            ).fetchone()
            ordinal = int(last[0]) + 1
        row = self._connection.execute(
            """
            INSERT INTO execution_segments (
                camera_id, worker_boot_id, source_generation, stream_epoch, segment_ordinal,
                storage_state, opened_at_ns, sealed_at_ns, record_count, payload_bytes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, 1, %s)
            RETURNING segment_id
            """,
            (*key, ordinal, str(SegmentStorageState.OPEN), self._now_ns, payload_bytes),
        ).fetchone()
        segment_id = int(row[0])
        self._open[key] = [segment_id, payload_bytes, ordinal]
        return segment_id

    def flush(self) -> None:
        if not self._pending:
            return
        with self._connection.cursor() as cursor:
            cursor.executemany(
                """
                UPDATE execution_segments
                SET record_count = record_count + %s, payload_bytes = payload_bytes + %s
                WHERE segment_id = %s
                """,
                [(count, size, segment_id) for segment_id, (count, size) in self._pending.items()],
            )
        self._pending.clear()

    def _load_open(self, key: _Lane) -> list[int] | None:
        row = self._connection.execute(
            """
            SELECT segment_id, payload_bytes, segment_ordinal FROM execution_segments
            WHERE camera_id = %s AND worker_boot_id = %s AND source_generation = %s
              AND stream_epoch = %s AND storage_state = %s
            ORDER BY segment_ordinal DESC LIMIT 1
            """,
            (*key, str(SegmentStorageState.OPEN)),
        ).fetchone()
        if row is None:
            return None
        return [int(row[0]), int(row[1]), int(row[2])]


__all__ = ["SegmentAllocator"]
