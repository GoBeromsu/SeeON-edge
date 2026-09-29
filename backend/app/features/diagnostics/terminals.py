"""Gate R unit terminal rules and segment final-seal."""

from __future__ import annotations

import psycopg

from backend.app.features.diagnostics.coverage import UNSCOPED_GAP_CAUSE
from backend.app.features.diagnostics.records import (
    CoverageKind,
    SegmentStorageState,
    UnitCausalState,
)


def refresh_unit_terminals(connection: psycopg.Connection, unit_horizon_ns: int) -> None:
    """Mark units terminal by a lane watermark or by a newer (boot, epoch).

    Horizon closure is scoped to one (camera, boot, generation, epoch) lane.
    The lane watermark is its maximum ``last_observed_ns``. A unit closes only
    when that watermark is strictly more than ``unit_horizon_ns`` past the
    unit's own last observation. The next unit's ``first_observed_ns`` is not
    evidence: dense neighbours never clear the horizon, and a unit that is
    still receiving observations would look finished because it started long
    ago. Another generation's observations are not this lane's watermark.

    A different (boot, epoch) on the same camera closes only when some other
    (boot, epoch) was observed strictly later. Boot ids are opaque, so they
    are not ordered as text. Abrupt closure cannot prove completeness beyond
    already COMPLETE units. Later coverage can only lower a terminal unit's
    certainty; it never reopens the unit or upgrades forced UNKNOWN.
    """
    known = str(UnitCausalState.INCOMPLETE_KNOWN)
    connection.execute(
        """
        UPDATE execution_units
        SET terminal = 1,
            causal_state = CASE
                WHEN EXISTS (
                    SELECT 1 FROM execution_coverage AS gap
                    WHERE gap.camera_id = execution_units.camera_id
                      AND gap.worker_boot_id = execution_units.worker_boot_id
                      AND (
                          (gap.source_generation = execution_units.source_generation
                           AND gap.stream_epoch = execution_units.stream_epoch)
                          OR (gap.cause = %s AND gap.exact = 0)
                      )
                      AND (gap.coverage_kind != %s OR gap.exact = 0)
                      AND gap.from_ns <= execution_units.last_observed_ns
                      AND gap.to_ns >= execution_units.first_observed_ns
                ) THEN %s
                WHEN execution_units.causal_state = %s THEN execution_units.causal_state
                WHEN EXISTS (
                    SELECT 1 FROM execution_coverage AS gap
                    WHERE gap.camera_id = execution_units.camera_id
                      AND gap.worker_boot_id = execution_units.worker_boot_id
                      AND gap.source_generation = execution_units.source_generation
                      AND gap.stream_epoch = execution_units.stream_epoch
                      AND gap.coverage_kind = %s
                      AND gap.exact = 1
                      AND gap.from_ns <= execution_units.last_observed_ns
                      AND gap.to_ns >= execution_units.first_observed_ns
                ) THEN %s
                ELSE %s
            END
        FROM (
            SELECT camera_id, worker_boot_id, source_generation, stream_epoch,
                   MAX(last_observed_ns) AS watermark_ns
            FROM execution_units
            GROUP BY camera_id, worker_boot_id, source_generation, stream_epoch
        ) AS lane
        WHERE execution_units.camera_id = lane.camera_id
          AND execution_units.worker_boot_id = lane.worker_boot_id
          AND execution_units.source_generation = lane.source_generation
          AND execution_units.stream_epoch = lane.stream_epoch
          AND (
              (execution_units.terminal = 0
               AND lane.watermark_ns - execution_units.last_observed_ns > %s)
              OR (execution_units.terminal = 1 AND execution_units.causal_state IN (%s, %s))
          )
        """,
        (
            UNSCOPED_GAP_CAUSE,
            str(CoverageKind.MISSING_NOT_RECORDED),
            str(UnitCausalState.INCOMPLETE_UNKNOWN),
            known,
            str(CoverageKind.MISSING_NOT_RECORDED),
            known,
            str(UnitCausalState.COMPLETE),
            unit_horizon_ns,
            str(UnitCausalState.COMPLETE),
            known,
        ),
    )
    connection.execute(
        """
        UPDATE execution_units
        SET terminal = 1,
            causal_state = CASE WHEN execution_units.causal_state = %s
                THEN %s ELSE %s END
        FROM (
            SELECT lane.camera_id AS camera_id,
                   lane.worker_boot_id AS worker_boot_id,
                   lane.stream_epoch AS stream_epoch
            FROM (
                SELECT camera_id, worker_boot_id, stream_epoch,
                       MAX(last_observed_ns) AS seen_ns
                FROM execution_units
                GROUP BY camera_id, worker_boot_id, stream_epoch
            ) AS lane
            JOIN (
                SELECT camera_id, MAX(last_observed_ns) AS newest_ns
                FROM execution_units
                GROUP BY camera_id
            ) AS camera
              ON camera.camera_id = lane.camera_id
             AND camera.newest_ns > lane.seen_ns
        ) AS stale
        WHERE execution_units.terminal = 0
          AND execution_units.camera_id = stale.camera_id
          AND execution_units.worker_boot_id = stale.worker_boot_id
          AND execution_units.stream_epoch = stale.stream_epoch
        """,
        (
            str(UnitCausalState.COMPLETE),
            str(UnitCausalState.COMPLETE),
            str(UnitCausalState.INCOMPLETE_UNKNOWN),
        ),
    )
    seal_final_segments(connection)


def seal_final_segments(connection: psycopg.Connection) -> None:
    """Promote a pending segment only when every member causal unit is terminal.

    An open unit elsewhere on the lane does not hold an older segment pending.
    A segment that still contains a non-terminal unit stays pending, including
    when that unit also has rows in another segment. Sealing does not delete
    rows; a unit that spans segments is removed only as a whole unit.
    """
    connection.execute(
        """
        UPDATE execution_segments
        SET storage_state = %s
        WHERE storage_state = %s
          AND NOT EXISTS (
              SELECT 1
              FROM execution_records AS record
              JOIN execution_units AS unit
                ON unit.causal_unit_id = record.causal_unit_id
              WHERE record.segment_id = execution_segments.segment_id
                AND unit.terminal = 0
          )
        """,
        (str(SegmentStorageState.SEALED_FINAL), str(SegmentStorageState.SEALED_PENDING)),
    )


def force_oldest_units_terminal(connection: psycopg.Connection, count: int = 1) -> int:
    """Pressure fallback: close ``count`` globally oldest non-terminal units.

    Oldest matches ``next_prunable_unit`` once the row is terminal: least
    ``last_observed_ns``, then ``causal_unit_id``.
    Camera and boot ids are not a priority. The closed state is
    INCOMPLETE_UNKNOWN: even a known gap cannot exclude additional unknown
    loss when pressure forces closure. Existing coverage is retained.
    This does not raise the count or the byte budget.
    """
    rows = connection.execute(
        """
        SELECT causal_unit_id FROM execution_units
        WHERE terminal = 0
        ORDER BY last_observed_ns, causal_unit_id COLLATE pg_catalog."C"
        LIMIT %s
        """,
        (count,),
    ).fetchall()
    connection.execute(
        """
        UPDATE execution_units
        SET terminal = 1, causal_state = %s
        WHERE causal_unit_id = ANY(%s) AND terminal = 0
        """,
        (str(UnitCausalState.INCOMPLETE_UNKNOWN), [str(row[0]) for row in rows]),
    )
    if rows:
        seal_final_segments(connection)
    return len(rows)


__all__ = [
    "force_oldest_units_terminal",
    "refresh_unit_terminals",
    "seal_final_segments",
]
