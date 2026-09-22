"""OPEN segment assignment and byte-threshold sealing."""

from __future__ import annotations

import sqlite3

from backend.app.features.diagnostics.records import (
    ExecutionRecordInput,
    SegmentStorageState,
)
from backend.app.features.diagnostics.retention import RetentionBudget


def assign_segment(
    connection: sqlite3.Connection,
    record: ExecutionRecordInput,
    payload_bytes: int,
    budget: RetentionBudget,
    now_ns: int,
) -> int:
    key = (record.camera_id, record.worker_boot_id, record.source_generation, record.stream_epoch)
    open_row = connection.execute(
        """
        SELECT segment_id, payload_bytes, segment_ordinal FROM execution_segments
        WHERE camera_id = ? AND worker_boot_id = ? AND source_generation = ?
          AND stream_epoch = ? AND storage_state = ?
        ORDER BY segment_ordinal DESC LIMIT 1
        """,
        (*key, str(SegmentStorageState.OPEN)),
    ).fetchone()
    if open_row is not None and int(open_row[1]) + payload_bytes <= budget.segment_bytes:
        segment_id = int(open_row[0])
        connection.execute(
            """
            UPDATE execution_segments
            SET record_count = record_count + 1, payload_bytes = payload_bytes + ?
            WHERE segment_id = ?
            """,
            (payload_bytes, segment_id),
        )
        return segment_id
    if open_row is not None:
        connection.execute(
            "UPDATE execution_segments SET storage_state = ?, sealed_at_ns = ? "
            "WHERE segment_id = ?",
            (str(SegmentStorageState.SEALED_PENDING), now_ns, int(open_row[0])),
        )
        ordinal = int(open_row[2]) + 1
    else:
        last = connection.execute(
            """
            SELECT COALESCE(MAX(segment_ordinal), -1) FROM execution_segments
            WHERE camera_id = ? AND worker_boot_id = ? AND source_generation = ?
              AND stream_epoch = ?
            """,
            key,
        ).fetchone()
        ordinal = int(last[0]) + 1
    cursor = connection.execute(
        """
        INSERT INTO execution_segments (
            camera_id, worker_boot_id, source_generation, stream_epoch, segment_ordinal,
            storage_state, opened_at_ns, sealed_at_ns, record_count, payload_bytes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 1, ?)
        """,
        (*key, ordinal, str(SegmentStorageState.OPEN), now_ns, payload_bytes),
    )
    return int(cursor.lastrowid)


__all__ = ["assign_segment"]
