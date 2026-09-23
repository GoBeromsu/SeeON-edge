"""Gate R unit terminal rules and segment final-seal."""

from __future__ import annotations

import sqlite3

from backend.app.features.diagnostics.records import (
    CoverageKind,
    SegmentStorageState,
    UnitCausalState,
)


def refresh_unit_terminals(connection: sqlite3.Connection, unit_horizon_ns: int) -> None:
    """Mark units terminal by horizon or by a newer (boot, epoch) on the camera.

    Only non-terminal units are candidates (a terminal unit never changes),
    and the per-lane successor lookup is one window query in SQLite rather
    than a Python pass over every unit per candidate: the previous shape was
    O(units^2) per ingest and cost half of every request on a 50k-unit DB.
    """
    # The newest (boot, epoch) lane per camera is the one observed most
    # recently. Boot ids are opaque (UUIDs): comparing them as strings marked
    # a NEWER boot's units terminal whenever its id happened to sort lower,
    # so live data was pruned first and a dead boot's units were kept.
    lanes = connection.execute(
        """
        SELECT camera_id, worker_boot_id, stream_epoch, MAX(last_observed_ns)
        FROM execution_units
        GROUP BY camera_id, worker_boot_id, stream_epoch
        """
    ).fetchall()
    if not lanes:
        return
    newest: dict[str, tuple[str, int]] = {}
    newest_seen: dict[str, int] = {}
    for camera_id_raw, boot_raw, epoch_raw, last_raw in lanes:
        camera_id, last_seen = str(camera_id_raw), int(last_raw)
        if camera_id not in newest_seen or last_seen > newest_seen[camera_id]:
            newest_seen[camera_id] = last_seen
            newest[camera_id] = (str(boot_raw), int(epoch_raw))
    candidates = connection.execute(
        """
        WITH ordered AS (
            SELECT causal_unit_id, camera_id, worker_boot_id, source_generation, stream_epoch,
                   first_observed_ns, last_observed_ns, causal_state, terminal,
                   LEAD(first_observed_ns) OVER (
                       PARTITION BY camera_id, worker_boot_id, source_generation, stream_epoch
                       ORDER BY first_observed_ns
                   ) AS next_first_ns
            FROM execution_units
        )
        SELECT causal_unit_id, camera_id, worker_boot_id, source_generation, stream_epoch,
               first_observed_ns, last_observed_ns, causal_state, next_first_ns
        FROM ordered
        WHERE terminal = 0
        """
    ).fetchall()
    for row in candidates:
        unit_id, camera_id, boot = str(row[0]), str(row[1]), str(row[2])
        gen, epoch = int(row[3]), int(row[4])
        first_ns, last_ns, state = int(row[5]), int(row[6]), str(row[7])
        next_first = None if row[8] is None else int(row[8])
        by_horizon = next_first is not None and next_first > first_ns + unit_horizon_ns
        by_epoch = (boot, epoch) != newest[camera_id]
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
