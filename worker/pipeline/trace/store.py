from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from worker.pipeline.trace.models import (
    AnalysisTrace,
    DecisionTrace,
    OptionalNumber,
    RecoveredCameraTrace,
    TraceBed,
    TraceComponent,
    TraceFrame,
    TraceKeypoint,
    TracePersistenceError,
    TracePerson,
    TraceTruncation,
    trace_frame_size_bytes,
)
from worker.types import DecisionTraceSnapshot


@dataclass(frozen=True, slots=True)
class _RetentionRow:
    trace_id: str
    boot_id: str
    camera_id: str
    epoch: int
    seq: int
    source_time: float | None
    storage_bytes: int
    row_count: int
    protected: bool

    @property
    def frame_key(self) -> tuple[str, str, int, int]:
        return (self.boot_id, self.camera_id, self.epoch, self.seq)


class TraceStore:
    """DDL-free worker writer/reader for bounded normalized trace rows."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def persist_batch(
        self,
        frames: Sequence[TraceFrame],
        *,
        max_frames_per_camera: int,
        max_age_seconds: float,
        max_cameras: int,
        max_total_frames: int,
        max_total_rows: int,
        max_total_bytes: int,
        dropped_by_camera: Mapping[str, int],
        failed_by_camera: Mapping[str, int] | None = None,
    ) -> int:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        failed = failed_by_camera or {}
        try:
            with write_transaction(connection):
                inserted = sum(self._insert_frame(connection, frame) for frame in frames)
                cameras = (
                    {frame.analysis.frame_key[1] for frame in frames}
                    | set(dropped_by_camera)
                    | set(failed)
                )
                pruned, blocked = self._prune(
                    connection,
                    max_frames_per_camera=max_frames_per_camera,
                    max_age_seconds=max_age_seconds,
                    max_cameras=max_cameras,
                    max_total_frames=max_total_frames,
                    max_total_rows=max_total_rows,
                    max_total_bytes=max_total_bytes,
                )
                cameras |= set(pruned) | set(blocked)
                for camera_id in cameras:
                    self._update_cursor(
                        connection,
                        camera_id,
                        handoff_dropped=dropped_by_camera.get(camera_id, 0),
                        persistence_failed=failed.get(camera_id, 0),
                        pruned=pruned.get(camera_id, 0),
                        blocked=blocked.get(camera_id, 0),
                    )
        finally:
            connection.close()
        return inserted

    def _insert_frame(self, connection: sqlite3.Connection, frame: TraceFrame) -> int:
        execute = connection.execute
        analysis = frame.analysis
        boot_id, camera_id, epoch, seq = analysis.frame_key
        cursor = execute(
            """
            INSERT OR IGNORE INTO runtime_analysis_traces (
                trace_id, trace_schema_version, worker_boot_id, camera_id,
                stream_epoch, frame_seq, pts, pts_missing_reason,
                source_time_sec, source_time_missing_reason, frame_width,
                frame_height, bed_region_provenance, storage_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis.trace_id,
                analysis.schema_version,
                boot_id,
                camera_id,
                epoch,
                seq,
                analysis.pts.value,
                analysis.pts.missing_reason,
                analysis.source_time.value,
                analysis.source_time.missing_reason,
                analysis.frame_width,
                analysis.frame_height,
                analysis.bed_region_provenance,
                trace_frame_size_bytes(frame),
            ),
        )
        if cursor.rowcount == 0:
            existing = execute(
                "SELECT worker_boot_id, camera_id, stream_epoch, frame_seq "
                "FROM runtime_analysis_traces WHERE trace_id = ?",
                (analysis.trace_id,),
            ).fetchone()
            if existing != (boot_id, camera_id, epoch, seq):
                raise TracePersistenceError("analysis trace identity resolves to another frame")
            return 0
        for component in analysis.components:
            execute(
                "INSERT INTO runtime_analysis_components VALUES (?, ?, ?, ?)",
                (
                    analysis.trace_id,
                    component.ordinal,
                    component.qualified_id,
                    component.observation_state,
                ),
            )
        for person in analysis.persons:
            execute(
                "INSERT INTO runtime_analysis_persons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis.trace_id,
                    person.ordinal,
                    person.track_id.value,
                    person.track_id.missing_reason,
                    *person.box,
                    person.confidence,
                ),
            )
        for person in analysis.persons:
            for point in person.keypoints:
                execute(
                    "INSERT INTO runtime_analysis_keypoints VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        analysis.trace_id,
                        person.ordinal,
                        point.index,
                        point.x,
                        point.y,
                        point.confidence,
                    ),
                )
        for bed in analysis.beds:
            execute(
                "INSERT INTO runtime_analysis_beds VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis.trace_id,
                    bed.ordinal,
                    *bed.box,
                    bed.confidence,
                    bed.provenance,
                ),
            )
            for point_index, (x, y) in enumerate(bed.polygon):
                execute(
                    "INSERT INTO runtime_analysis_bed_points VALUES (?, ?, ?, ?, ?)",
                    (analysis.trace_id, bed.ordinal, point_index, x, y),
                )
        for decision in frame.decisions:
            self._insert_decision(connection, decision)
        return 1

    def _insert_decision(self, connection: sqlite3.Connection, decision: DecisionTrace) -> None:
        snapshot = decision.snapshot
        connection.execute(
            """
            INSERT INTO evidence_decision_traces (
                trace_id, trace_schema_version, analysis_trace_id,
                module_qualified_id, policy_qualified_id, effective_policy_id,
                runtime_manifest_sha256, reason, previous_state, current_state,
                triggered, track_id, track_missing_reason, bed_id, bed_missing_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.trace_id,
                decision.schema_version,
                decision.analysis_trace_id,
                decision.module_qualified_id,
                decision.policy_qualified_id,
                decision.effective_policy_id,
                decision.runtime_manifest_sha256,
                snapshot.reason,
                snapshot.previous_state,
                snapshot.current_state,
                int(snapshot.triggered),
                snapshot.track_id,
                "not-applicable" if snapshot.track_id is None else None,
                snapshot.bed_id,
                "not-applicable" if snapshot.bed_id is None else None,
            ),
        )
        for name, value in snapshot.values.items():
            connection.execute(
                "INSERT INTO evidence_decision_values VALUES (?, ?, ?, NULL)",
                (decision.trace_id, name, value),
            )
        for name, reason in snapshot.missing_values.items():
            connection.execute(
                "INSERT INTO evidence_decision_values VALUES (?, ?, NULL, ?)",
                (decision.trace_id, name, reason),
            )

    def _retention_rows(self, connection: sqlite3.Connection) -> list[_RetentionRow]:
        rows = connection.execute(
            """
            SELECT analysis.trace_id, analysis.worker_boot_id, analysis.camera_id,
                   analysis.stream_epoch, analysis.frame_seq, analysis.source_time_sec,
                   analysis.storage_bytes,
                   1
                     + (SELECT count(*) FROM runtime_analysis_components
                        WHERE analysis_trace_id = analysis.trace_id)
                     + (SELECT count(*) FROM runtime_analysis_persons
                        WHERE analysis_trace_id = analysis.trace_id)
                     + (SELECT count(*) FROM runtime_analysis_beds
                        WHERE analysis_trace_id = analysis.trace_id)
                     + (SELECT count(*) FROM runtime_analysis_keypoints
                        WHERE analysis_trace_id = analysis.trace_id)
                     + (SELECT count(*) FROM runtime_analysis_bed_points
                        WHERE analysis_trace_id = analysis.trace_id)
                     + (SELECT count(*) FROM evidence_decision_traces
                        WHERE analysis_trace_id = analysis.trace_id)
                     + (SELECT count(*) FROM evidence_decision_values AS value
                        JOIN evidence_decision_traces AS decision
                          ON decision.trace_id = value.decision_trace_id
                        WHERE decision.analysis_trace_id = analysis.trace_id),
                   EXISTS (
                       SELECT 1 FROM evidence_decision_traces AS decision
                       LEFT JOIN evidence_event_trace_refs AS event_ref
                         ON event_ref.decision_trace_id = decision.trace_id
                       WHERE decision.analysis_trace_id = analysis.trace_id
                         AND (decision.triggered = 1 OR event_ref.edge_event_id IS NOT NULL)
                   )
            FROM runtime_analysis_traces AS analysis
            ORDER BY analysis.source_time_sec IS NULL, analysis.source_time_sec DESC,
                     analysis.worker_boot_id DESC, analysis.camera_id DESC,
                     analysis.stream_epoch DESC, analysis.frame_seq DESC,
                     analysis.trace_id DESC
            """
        ).fetchall()
        return [
            _RetentionRow(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                int(row[3]),
                int(row[4]),
                None if row[5] is None else float(row[5]),
                int(row[6]),
                int(row[7]),
                bool(row[8]),
            )
            for row in rows
        ]

    def _prune(
        self,
        connection: sqlite3.Connection,
        *,
        max_frames_per_camera: int,
        max_age_seconds: float,
        max_cameras: int,
        max_total_frames: int,
        max_total_rows: int,
        max_total_bytes: int,
    ) -> tuple[dict[str, int], dict[str, int]]:
        rows = self._retention_rows(connection)
        targets: set[str] = set()
        blocked: defaultdict[str, int] = defaultdict(int)
        newest_by_camera: dict[str, float] = {}
        for row in rows:
            if row.source_time is not None:
                newest_by_camera.setdefault(row.camera_id, row.source_time)
        for row in rows:
            newest = newest_by_camera.get(row.camera_id)
            if newest is None or row.source_time is None:
                continue
            if row.source_time < newest - max_age_seconds:
                if row.protected:
                    blocked[row.camera_id] += 1
                else:
                    targets.add(row.trace_id)

        remaining = [row for row in rows if row.trace_id not in targets]
        grouped: defaultdict[str, list[_RetentionRow]] = defaultdict(list)
        for row in remaining:
            grouped[row.camera_id].append(row)
        for camera_rows in grouped.values():
            self._retain_newest_unprotected(
                camera_rows,
                max_frames_per_camera,
                targets,
                blocked,
            )

        remaining = [row for row in rows if row.trace_id not in targets]
        protected_cameras = {row.camera_id for row in remaining if row.protected}
        camera_order = tuple(dict.fromkeys(row.camera_id for row in remaining))
        allowed_cameras = set(protected_cameras)
        for camera_id in camera_order:
            if camera_id in allowed_cameras:
                continue
            if len(allowed_cameras) < max_cameras:
                allowed_cameras.add(camera_id)
        if len(protected_cameras) > max_cameras:
            raise TracePersistenceError("referenced trace cameras exceed max_cameras")
        targets.update(
            row.trace_id
            for row in remaining
            if row.camera_id not in allowed_cameras and not row.protected
        )

        remaining = [row for row in rows if row.trace_id not in targets]
        self._retain_newest_unprotected(
            remaining,
            max_total_frames,
            targets,
            blocked,
        )
        remaining = [row for row in rows if row.trace_id not in targets]
        protected_rows = [row for row in remaining if row.protected]
        protected_row_count = sum(row.row_count for row in protected_rows)
        protected_bytes = sum(row.storage_bytes for row in protected_rows)
        if protected_row_count > max_total_rows or protected_bytes > max_total_bytes:
            raise TracePersistenceError("referenced traces exceed global row or byte bounds")
        used_rows = protected_row_count
        used_bytes = protected_bytes
        for row in remaining:
            if row.protected:
                continue
            if (
                used_rows + row.row_count <= max_total_rows
                and used_bytes + row.storage_bytes <= max_total_bytes
            ):
                used_rows += row.row_count
                used_bytes += row.storage_bytes
            else:
                targets.add(row.trace_id)

        pruned: defaultdict[str, int] = defaultdict(int)
        by_id = {row.trace_id: row for row in rows}
        for trace_id in sorted(targets):
            row = by_id[trace_id]
            connection.execute(
                "DELETE FROM evidence_decision_traces "
                "WHERE analysis_trace_id = ? AND triggered = 0 "
                "AND NOT EXISTS (SELECT 1 FROM evidence_event_trace_refs "
                "WHERE decision_trace_id = evidence_decision_traces.trace_id)",
                (trace_id,),
            )
            deleted = connection.execute(
                "DELETE FROM runtime_analysis_traces WHERE trace_id = ?", (trace_id,)
            ).rowcount
            if deleted:
                pruned[row.camera_id] += 1
        return dict(pruned), dict(blocked)

    @staticmethod
    def _retain_newest_unprotected(
        rows: Sequence[_RetentionRow],
        maximum: int,
        targets: set[str],
        blocked: defaultdict[str, int],
    ) -> None:
        protected = [row for row in rows if row.protected]
        if len(protected) > maximum:
            for row in protected[maximum:]:
                blocked[row.camera_id] += 1
            raise TracePersistenceError("referenced traces exceed frame retention bounds")
        allowance = maximum - len(protected)
        unprotected = [row for row in rows if not row.protected]
        targets.update(row.trace_id for row in unprotected[allowance:])

    def _update_cursor(
        self,
        connection: sqlite3.Connection,
        camera_id: str,
        *,
        handoff_dropped: int,
        persistence_failed: int,
        pruned: int,
        blocked: int,
    ) -> None:
        bounds = connection.execute(
            """
            SELECT worker_boot_id, camera_id, stream_epoch, frame_seq, trace_id,
                   source_time_sec
            FROM runtime_analysis_traces WHERE camera_id = ?
            ORDER BY source_time_sec IS NULL, source_time_sec,
                     worker_boot_id, stream_epoch, frame_seq, trace_id
            """,
            (camera_id,),
        ).fetchall()
        oldest = None if not bounds else bounds[0]
        newest = None if not bounds else bounds[-1]
        connection.execute(
            """
            INSERT INTO runtime_trace_cursors (
                camera_id, handoff_dropped_frames, pruned_frames,
                oldest_retained_seq, newest_retained_seq, updated_at_source_sec,
                persistence_failed_frames, retention_blocked_frames,
                oldest_retained_boot_id, oldest_retained_stream_epoch,
                oldest_retained_trace_id, newest_retained_boot_id,
                newest_retained_stream_epoch, newest_retained_trace_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                handoff_dropped_frames = handoff_dropped_frames
                    + excluded.handoff_dropped_frames,
                pruned_frames = pruned_frames + excluded.pruned_frames,
                oldest_retained_seq = excluded.oldest_retained_seq,
                newest_retained_seq = excluded.newest_retained_seq,
                updated_at_source_sec = excluded.updated_at_source_sec,
                persistence_failed_frames = persistence_failed_frames
                    + excluded.persistence_failed_frames,
                retention_blocked_frames = retention_blocked_frames
                    + excluded.retention_blocked_frames,
                oldest_retained_boot_id = excluded.oldest_retained_boot_id,
                oldest_retained_stream_epoch = excluded.oldest_retained_stream_epoch,
                oldest_retained_trace_id = excluded.oldest_retained_trace_id,
                newest_retained_boot_id = excluded.newest_retained_boot_id,
                newest_retained_stream_epoch = excluded.newest_retained_stream_epoch,
                newest_retained_trace_id = excluded.newest_retained_trace_id
            """,
            (
                camera_id,
                handoff_dropped,
                pruned,
                None if oldest is None else oldest[3],
                None if newest is None else newest[3],
                None if newest is None else newest[5],
                persistence_failed,
                blocked,
                None if oldest is None else oldest[0],
                None if oldest is None else oldest[2],
                None if oldest is None else oldest[4],
                None if newest is None else newest[0],
                None if newest is None else newest[2],
                None if newest is None else newest[4],
            ),
        )

    def recover_camera(self, camera_id: str) -> RecoveredCameraTrace:
        """Load one camera's analysis/decision timeline in real boot chronology.

        Stream-relative ``source_time_sec`` is not a cross-boot clock: values
        restart per ingest session and must never interleave boots. Ordering is
        therefore:

        1. ``runtime_manifest_boots.applied_at`` (real process-boot chronology),
           unknown boots last via ``applied_at IS NULL``;
        2. ``worker_boot_id`` (stable tie-break when applied_at collides);
        3. ``stream_epoch``, ``frame_seq``, ``trace_id`` (production in-boot order).

        Retention age pruning still uses source_time; only recovery/replay order
        is boot-partitioned.
        """
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            rows = connection.execute(
                """
                SELECT analysis.trace_id, analysis.trace_schema_version,
                       analysis.worker_boot_id, analysis.camera_id,
                       analysis.stream_epoch, analysis.frame_seq, analysis.pts,
                       analysis.pts_missing_reason, analysis.source_time_sec,
                       analysis.source_time_missing_reason, analysis.frame_width,
                       analysis.frame_height, analysis.bed_region_provenance
                FROM runtime_analysis_traces AS analysis
                LEFT JOIN runtime_manifest_boots AS boot
                  ON boot.boot_instance_id = analysis.worker_boot_id
                WHERE analysis.camera_id = ?
                ORDER BY boot.applied_at IS NULL, boot.applied_at,
                         analysis.worker_boot_id, analysis.stream_epoch,
                         analysis.frame_seq, analysis.trace_id
                """,
                (camera_id,),
            ).fetchall()
            frames = tuple(self._analysis_from_row(connection, row) for row in rows)
            decision_rows = connection.execute(
                """
                SELECT decision.trace_id, decision.trace_schema_version,
                       decision.analysis_trace_id, decision.module_qualified_id,
                       decision.policy_qualified_id, decision.effective_policy_id,
                       decision.runtime_manifest_sha256, decision.reason,
                       decision.previous_state, decision.current_state,
                       decision.triggered, decision.track_id, decision.bed_id
                FROM evidence_decision_traces AS decision
                JOIN runtime_analysis_traces AS analysis
                  ON analysis.trace_id = decision.analysis_trace_id
                LEFT JOIN runtime_manifest_boots AS boot
                  ON boot.boot_instance_id = analysis.worker_boot_id
                WHERE analysis.camera_id = ?
                ORDER BY boot.applied_at IS NULL, boot.applied_at,
                         analysis.worker_boot_id, analysis.stream_epoch,
                         analysis.frame_seq, analysis.trace_id, decision.trace_id
                """,
                (camera_id,),
            ).fetchall()
            decisions = tuple(
                self._decision_from_row(connection, row, index)
                for index, row in enumerate(decision_rows)
            )
            cursor = connection.execute(
                """
                SELECT handoff_dropped_frames, pruned_frames,
                       oldest_retained_seq, newest_retained_seq,
                       persistence_failed_frames, retention_blocked_frames,
                       oldest_retained_boot_id, oldest_retained_stream_epoch,
                       newest_retained_boot_id, newest_retained_stream_epoch
                FROM runtime_trace_cursors WHERE camera_id = ?
                """,
                (camera_id,),
            ).fetchone()
        finally:
            connection.close()
        if cursor is None:
            truncation = TraceTruncation(0, 0, None, None)
        else:
            oldest_key = (
                None
                if cursor[6] is None or cursor[2] is None
                else (str(cursor[6]), camera_id, int(cursor[7]), int(cursor[2]))
            )
            newest_key = (
                None
                if cursor[8] is None or cursor[3] is None
                else (str(cursor[8]), camera_id, int(cursor[9]), int(cursor[3]))
            )
            truncation = TraceTruncation(
                int(cursor[0]),
                int(cursor[1]),
                None if cursor[2] is None else int(cursor[2]),
                None if cursor[3] is None else int(cursor[3]),
                int(cursor[4]),
                int(cursor[5]),
                oldest_key,
                newest_key,
            )
        return RecoveredCameraTrace(frames, decisions, truncation)

    def _analysis_from_row(
        self, connection: sqlite3.Connection, row: Sequence[Any]
    ) -> AnalysisTrace:
        trace_id = str(row[0])
        components = tuple(
            TraceComponent(int(item[0]), str(item[1]), str(item[2]))
            for item in connection.execute(
                "SELECT ordinal, component_qualified_id, observation_state "
                "FROM runtime_analysis_components WHERE analysis_trace_id = ? ORDER BY ordinal",
                (trace_id,),
            ).fetchall()
        )
        persons = tuple(
            TracePerson(
                int(item[0]),
                OptionalNumber(
                    None if item[1] is None else int(item[1]),
                    None if item[2] is None else str(item[2]),
                ),
                (int(item[3]), int(item[4]), int(item[5]), int(item[6])),
                float(item[7]),
                tuple(
                    TraceKeypoint(int(point[0]), int(point[1]), int(point[2]), float(point[3]))
                    for point in connection.execute(
                        "SELECT keypoint_index, x, y, confidence "
                        "FROM runtime_analysis_keypoints WHERE analysis_trace_id = ? "
                        "AND person_ordinal = ? ORDER BY keypoint_index",
                        (trace_id, int(item[0])),
                    ).fetchall()
                ),
            )
            for item in connection.execute(
                "SELECT ordinal, track_id, track_missing_reason, x1, y1, x2, y2, confidence "
                "FROM runtime_analysis_persons WHERE analysis_trace_id = ? ORDER BY ordinal",
                (trace_id,),
            ).fetchall()
        )
        beds = tuple(
            TraceBed(
                int(item[0]),
                (int(item[1]), int(item[2]), int(item[3]), int(item[4])),
                float(item[5]),
                str(item[6]),
                tuple(
                    (int(point[0]), int(point[1]))
                    for point in connection.execute(
                        "SELECT x, y FROM runtime_analysis_bed_points "
                        "WHERE analysis_trace_id = ? AND bed_ordinal = ? ORDER BY point_index",
                        (trace_id, int(item[0])),
                    ).fetchall()
                ),
            )
            for item in connection.execute(
                "SELECT ordinal, x1, y1, x2, y2, confidence, provenance "
                "FROM runtime_analysis_beds WHERE analysis_trace_id = ? ORDER BY ordinal",
                (trace_id,),
            ).fetchall()
        )
        return AnalysisTrace(
            trace_id,
            (str(row[2]), str(row[3]), int(row[4]), int(row[5])),
            _optional_number(row[6], row[7]),
            _optional_number(row[8], row[9]),
            int(row[10]),
            int(row[11]),
            str(row[12]),
            persons,
            beds,
            components,
            int(row[1]),
        )

    def _decision_from_row(
        self,
        connection: sqlite3.Connection,
        row: Sequence[Any],
        identity_index: int,
    ) -> DecisionTrace:
        values: dict[str, int | float] = {}
        missing: dict[str, str] = {}
        for name, value, reason in connection.execute(
            "SELECT name, numeric_value, missing_reason FROM evidence_decision_values "
            "WHERE decision_trace_id = ? ORDER BY name",
            (row[0],),
        ).fetchall():
            if value is None:
                missing[str(name)] = str(reason)
            else:
                values[str(name)] = float(value)
        snapshot = DecisionTraceSnapshot(
            reason=str(row[7]),
            previous_state=str(row[8]),
            current_state=str(row[9]),
            triggered=bool(row[10]),
            track_id=None if row[11] is None else int(row[11]),
            bed_id=None if row[12] is None else int(row[12]),
            values=values,
            missing_values=missing,
        )
        return DecisionTrace(
            str(row[0]),
            str(row[2]),
            identity_index,
            str(row[3]),
            str(row[4]),
            str(row[5]),
            str(row[6]),
            snapshot,
            int(row[1]),
        )


def _optional_number(value: object, reason: object) -> OptionalNumber:
    if value is None:
        return OptionalNumber(None, str(reason))
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError("stored trace numeric value is invalid")
    return OptionalNumber(value)


__all__ = ["TraceStore"]
