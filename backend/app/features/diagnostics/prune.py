"""Coherent unit prune and coverage coarsening."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from backend.app.features.diagnostics.coverage import insert_coverage
from backend.app.features.diagnostics.records import (
    CoverageKind,
    SegmentStorageState,
)


def next_prunable_unit(connection: sqlite3.Connection) -> str | None:
    """Oldest terminal unit by last observation - the order execution_units_prune_order
    indexes, so this is a single index seek rather than a sort of every
    terminal unit on each ingest."""
    row = connection.execute(
        """
        SELECT causal_unit_id FROM execution_units
        WHERE terminal = 1
        ORDER BY last_observed_ns, causal_unit_id
        LIMIT 1
        """
    ).fetchone()
    return None if row is None else str(row[0])


def prune_unit(
    connection: sqlite3.Connection, unit_id: str, now_ns: int
) -> tuple[int, Lane | None]:
    """Delete one whole unit; returns (payload bytes it held, its lane) or (0, None)."""
    unit = connection.execute(
        """
        SELECT camera_id, worker_boot_id, source_generation, stream_epoch,
               first_observed_ns, last_observed_ns, record_count
        FROM execution_units WHERE causal_unit_id = ?
        """,
        (unit_id,),
    ).fetchone()
    if unit is None:
        return 0, None
    camera_id, boot, gen, epoch = str(unit[0]), str(unit[1]), int(unit[2]), int(unit[3])
    first_ns, last_ns, record_count = int(unit[4]), int(unit[5]), int(unit[6])
    producers = connection.execute(
        """
        SELECT producer, MIN(producer_sequence), MAX(producer_sequence),
               MIN(observed_at_ns), MAX(observed_at_ns), COUNT(*)
        FROM execution_records WHERE causal_unit_id = ?
        GROUP BY producer
        """,
        (unit_id,),
    ).fetchall()
    segment_deltas = connection.execute(
        """
        SELECT segment_id, COUNT(*), COALESCE(SUM(payload_bytes), 0)
        FROM execution_records WHERE causal_unit_id = ?
        GROUP BY segment_id
        """,
        (unit_id,),
    ).fetchall()
    freed = sum(int(row[2]) for row in segment_deltas)
    connection.execute("DELETE FROM execution_units WHERE causal_unit_id = ?", (unit_id,))
    for segment_id, count, bytes_removed in segment_deltas:
        connection.execute(
            """
            UPDATE execution_segments
            SET record_count = record_count - ?, payload_bytes = payload_bytes - ?
            WHERE segment_id = ?
            """,
            (int(count), int(bytes_removed), int(segment_id)),
        )
        emptied = connection.execute(
            "SELECT record_count FROM execution_segments WHERE segment_id = ?",
            (int(segment_id),),
        ).fetchone()
        if emptied is not None and int(emptied[0]) <= 0:
            connection.execute(
                """
                UPDATE execution_segments
                SET storage_state = ?, record_count = 0, payload_bytes = 0
                WHERE segment_id = ?
                """,
                (str(SegmentStorageState.PRUNED_SUMMARY), int(segment_id)),
            )
    if not producers:
        insert_coverage(
            connection,
            camera_id=camera_id,
            worker_boot_id=boot,
            source_generation=gen,
            stream_epoch=epoch,
            kind=CoverageKind.DELETED_BY_CAPACITY,
            producer=None,
            from_sequence=None,
            to_sequence=None,
            from_ns=first_ns,
            to_ns=last_ns,
            record_count=record_count,
            exact=True,
            cause="capacity",
            recorded_at_ns=now_ns,
        )
        return freed, (camera_id, boot, gen, epoch)
    for producer, from_seq, to_seq, from_ns, to_ns, count in producers:
        if _extend_contiguous_deletion(
            connection,
            camera_id=camera_id,
            worker_boot_id=boot,
            source_generation=gen,
            stream_epoch=epoch,
            producer=str(producer),
            from_sequence=int(from_seq),
            to_sequence=int(to_seq),
            to_ns=int(to_ns),
            record_count=int(count),
            recorded_at_ns=now_ns,
        ):
            continue
        insert_coverage(
            connection,
            camera_id=camera_id,
            worker_boot_id=boot,
            source_generation=gen,
            stream_epoch=epoch,
            kind=CoverageKind.DELETED_BY_CAPACITY,
            producer=str(producer),
            from_sequence=int(from_seq),
            to_sequence=int(to_seq),
            from_ns=int(from_ns),
            to_ns=int(to_ns),
            record_count=int(count),
            exact=True,
            cause="capacity",
            recorded_at_ns=now_ns,
        )
    return freed, (camera_id, boot, gen, epoch)


def _extend_contiguous_deletion(
    connection: sqlite3.Connection,
    *,
    camera_id: str,
    worker_boot_id: str,
    source_generation: int,
    stream_epoch: int,
    producer: str,
    from_sequence: int,
    to_sequence: int,
    to_ns: int,
    record_count: int,
    recorded_at_ns: int,
) -> bool:
    """Grow the adjacent exact deletion row when this range continues it.

    A proven contiguous deletion prefix is still exact evidence (plan §7), so
    merging keeps the control envelope bounded without widening any claim: the
    merged row covers exactly the sequences that were deleted, no more.
    """
    row = connection.execute(
        """
        SELECT coverage_id FROM execution_coverage
        WHERE camera_id = ? AND worker_boot_id = ? AND source_generation = ?
          AND stream_epoch = ? AND producer = ? AND coverage_kind = ? AND exact = 1
          AND to_sequence = ? - 1
        """,
        (
            camera_id,
            worker_boot_id,
            source_generation,
            stream_epoch,
            producer,
            str(CoverageKind.DELETED_BY_CAPACITY),
            from_sequence,
        ),
    ).fetchone()
    if row is None:
        return False
    connection.execute(
        """
        UPDATE execution_coverage
        SET to_sequence = ?, to_ns = MAX(to_ns, ?), record_count = record_count + ?,
            recorded_at_ns = ?
        WHERE coverage_id = ?
        """,
        (to_sequence, to_ns, record_count, recorded_at_ns, int(row[0])),
    )
    return True


def drop_orphan_batches(connection: sqlite3.Connection) -> None:
    """Delete batch receipts none of whose records can still exist.

    A receipt exists so a retried batch is answered idempotently. Every record
    of a batch was observed no later than the batch was received
    (``observed_at_ns <= committed_at_ns == received_at_ns``), so a receipt
    received before the camera's oldest surviving observation has no records
    left and only pins control bytes. Per camera so the
    (camera_id, observed_at_ns) index answers the MIN in O(log n); slightly
    conservative (keeps a receipt a little longer than strictly needed),
    never wrong.
    """
    cameras = connection.execute("SELECT DISTINCT camera_id FROM execution_batches").fetchall()
    for (camera_id,) in cameras:
        connection.execute(
            """
            DELETE FROM execution_batches
            WHERE camera_id = ? AND received_at_ns < (
                SELECT COALESCE(MIN(observed_at_ns), 0) FROM execution_records
                WHERE camera_id = ?
            )
            """,
            (camera_id, camera_id),
        )


Lane = tuple[str, str, int, int]


def coarsen_coverage(
    connection: sqlite3.Connection,
    coverage_rows_per_epoch: int,
    now_ns: int,
    lanes: Iterable[Lane] | None = None,
) -> None:
    """Fold each over-bound (camera, boot, generation, epoch) lane's oldest
    coverage rows into one UNKNOWN_COARSENED row.

    ``lanes`` restricts the check to lanes that just gained rows; a full
    GROUP BY over every coverage row on every ingest was a third of each
    request on the reference edge. ``None`` checks every lane (startup, tests).
    """
    if lanes is None:
        groups = connection.execute(
            """
            SELECT camera_id, worker_boot_id, source_generation, stream_epoch, COUNT(*)
            FROM execution_coverage
            GROUP BY camera_id, worker_boot_id, source_generation, stream_epoch
            HAVING COUNT(*) > ?
            """,
            (coverage_rows_per_epoch,),
        ).fetchall()
    else:
        groups = []
        for camera_id, boot, gen, epoch in set(lanes):
            count = connection.execute(
                """
                SELECT COUNT(*) FROM execution_coverage
                WHERE camera_id = ? AND worker_boot_id = ? AND source_generation = ?
                  AND stream_epoch = ?
                """,
                (camera_id, boot, gen, epoch),
            ).fetchone()
            if count is not None and int(count[0]) > coverage_rows_per_epoch:
                groups.append((camera_id, boot, gen, epoch, int(count[0])))
    for camera_id, boot, gen, epoch, count in groups:
        overflow = int(count) - coverage_rows_per_epoch
        if overflow <= 0:
            continue
        rows = connection.execute(
            """
            SELECT coverage_id, from_ns, to_ns, record_count FROM execution_coverage
            WHERE camera_id = ? AND worker_boot_id = ? AND source_generation = ?
              AND stream_epoch = ?
            ORDER BY from_ns, coverage_id
            LIMIT ?
            """,
            (camera_id, boot, gen, epoch, overflow + 1),
        ).fetchall()
        if len(rows) < 2:
            continue
        ids = [int(row[0]) for row in rows]
        connection.execute(
            f"DELETE FROM execution_coverage WHERE coverage_id IN ({','.join('?' * len(ids))})",
            ids,
        )
        insert_coverage(
            connection,
            camera_id=str(camera_id),
            worker_boot_id=str(boot),
            source_generation=int(gen),
            stream_epoch=int(epoch),
            kind=CoverageKind.UNKNOWN_COARSENED,
            producer=None,
            from_sequence=None,
            to_sequence=None,
            from_ns=min(int(row[1]) for row in rows),
            to_ns=max(int(row[2]) for row in rows),
            record_count=sum(int(row[3]) for row in rows),
            exact=False,
            cause="coarsened",
            recorded_at_ns=now_ns,
        )


__all__ = [
    "Lane",
    "coarsen_coverage",
    "next_prunable_unit",
    "prune_unit",
]
