"""Gate R unit terminal rules and segment final-seal."""

from __future__ import annotations

import sqlite3

from backend.app.features.diagnostics.records import (
    CoverageKind,
    SegmentStorageState,
    UnitCausalState,
)


def refresh_unit_terminals(connection: sqlite3.Connection, unit_horizon_ns: int) -> None:
    units = connection.execute(
        """
        SELECT causal_unit_id, camera_id, worker_boot_id, source_generation, stream_epoch,
               first_observed_ns, last_observed_ns, causal_state
        FROM execution_units
        """
    ).fetchall()
    if not units:
        return
    newest: dict[str, tuple[str, int]] = {}
    later_first: dict[tuple[object, ...], list[int]] = {}
    for row in units:
        camera_id, boot, epoch = str(row[1]), str(row[2]), int(row[4])
        current = newest.get(camera_id)
        if current is None or (boot, epoch) > current:
            newest[camera_id] = (boot, epoch)
        later_first.setdefault((row[1], row[2], row[3], row[4]), []).append(int(row[5]))
    for values in later_first.values():
        values.sort()
    for row in units:
        unit_id = str(row[0])
        camera_id = str(row[1])
        boot = str(row[2])
        gen = int(row[3])
        epoch = int(row[4])
        first_ns = int(row[5])
        last_ns = int(row[6])
        state = str(row[7])
        successors = [
            value for value in later_first[(row[1], row[2], row[3], row[4])] if value > first_ns
        ]
        by_horizon = bool(successors) and min(successors) > first_ns + unit_horizon_ns
        by_epoch = (boot, epoch) < newest[camera_id]
        if not by_horizon and not by_epoch:
            continue
        if by_horizon:
            new_state = _horizon_state(connection, camera_id, boot, gen, epoch, first_ns, last_ns)
        elif state == UnitCausalState.COMPLETE:
            new_state = UnitCausalState.COMPLETE
        else:
            new_state = UnitCausalState.INCOMPLETE_UNKNOWN
        connection.execute(
            "UPDATE execution_units SET terminal = 1, causal_state = ? WHERE causal_unit_id = ?",
            (str(new_state), unit_id),
        )
    seal_final_segments(connection)


def _horizon_state(
    connection: sqlite3.Connection,
    camera_id: str,
    boot: str,
    gen: int,
    epoch: int,
    first_ns: int,
    last_ns: int,
) -> UnitCausalState:
    gap = connection.execute(
        """
        SELECT 1 FROM execution_coverage
        WHERE camera_id = ? AND worker_boot_id = ? AND source_generation = ?
          AND stream_epoch = ? AND coverage_kind = ? AND exact = 1
          AND from_ns <= ? AND to_ns >= ?
        LIMIT 1
        """,
        (
            camera_id,
            boot,
            gen,
            epoch,
            str(CoverageKind.MISSING_NOT_RECORDED),
            last_ns,
            first_ns,
        ),
    ).fetchone()
    return UnitCausalState.INCOMPLETE_KNOWN if gap is not None else UnitCausalState.COMPLETE


def seal_final_segments(connection: sqlite3.Connection) -> None:
    pending = connection.execute(
        """
        SELECT segment_id, camera_id, worker_boot_id, source_generation, stream_epoch
        FROM execution_segments WHERE storage_state = ?
        """,
        (str(SegmentStorageState.SEALED_PENDING),),
    ).fetchall()
    for segment_id, camera_id, boot, gen, epoch in pending:
        open_units = connection.execute(
            """
            SELECT 1 FROM execution_units
            WHERE camera_id = ? AND worker_boot_id = ? AND source_generation = ?
              AND stream_epoch = ? AND terminal = 0
            LIMIT 1
            """,
            (camera_id, boot, gen, epoch),
        ).fetchone()
        if open_units is None:
            connection.execute(
                "UPDATE execution_segments SET storage_state = ? WHERE segment_id = ?",
                (str(SegmentStorageState.SEALED_FINAL), segment_id),
            )


def force_oldest_units_terminal(connection: sqlite3.Connection, count: int = 1) -> int:
    rows = connection.execute(
        """
        SELECT causal_unit_id FROM execution_units
        WHERE terminal = 0
        ORDER BY camera_id, worker_boot_id, source_generation, stream_epoch, first_observed_ns
        LIMIT ?
        """,
        (count,),
    ).fetchall()
    for (unit_id,) in rows:
        connection.execute(
            "UPDATE execution_units SET terminal = 1, causal_state = ? WHERE causal_unit_id = ?",
            (str(UnitCausalState.INCOMPLETE_UNKNOWN), unit_id),
        )
    if rows:
        seal_final_segments(connection)
    return len(rows)


__all__ = [
    "force_oldest_units_terminal",
    "refresh_unit_terminals",
    "seal_final_segments",
]
