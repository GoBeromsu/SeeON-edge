"""Backend-owned persistence and recovery for image-free analysis timelines."""
# ruff: noqa: E501

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from shared.events.replay_wire import MAX_TRACE_FRAMES, ReplayTrace, ReplayWireError


def _rows(value: object, field: str) -> list[dict[str, object]]:
    """Narrow a nested wire field to the row list it is contracted to be.

    The wire envelope is `dict[str, object]` by construction, so every nested
    access lands on `object`. Narrowing here keeps the type checker honest
    without scattering casts, and turns a malformed payload into a named error
    instead of an attribute error deep inside an INSERT.
    """
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ReplayWireError(f"{field} must be a list of objects")
    return [dict(row) for row in value]


def _mapping(value: object, field: str) -> dict[str, object]:
    """Narrow a nested wire field to the mapping it is contracted to be."""
    if not isinstance(value, dict):
        raise ReplayWireError(f"{field} must be an object")
    return dict(value)


def _sequence(value: object, field: str) -> list[object]:
    """Narrow a nested wire field to the sequence it is contracted to be."""
    if not isinstance(value, list):
        raise ReplayWireError(f"{field} must be a list")
    return list(value)


def _pair(value: object, field: str) -> tuple[object, object]:
    """Narrow an optional-value wire field to its (present, missing) pair."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ReplayWireError(f"{field} must carry exactly two elements")
    return (value[0], value[1])


class ReplayInputUnavailable(ReplayWireError):
    """No complete, reproducible input is available for an operator replay."""


class RuntimeTraceConflict(ReplayWireError):
    """A trace identity was resent with facts different from the stored trace."""


class RuntimeAnalysisStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def ingest(self, trace: ReplayTrace) -> None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        connection.row_factory = sqlite3.Row
        try:
            with write_transaction(connection):
                for frame in trace.frames:
                    self._insert_frame(connection, trace.camera_id, frame)
                self._upsert_cursor(connection, trace.camera_id, trace.truncation)
                self._prune_beyond_window(connection, trace.camera_id)
        finally:
            connection.close()

    @staticmethod
    def _prune_beyond_window(connection: sqlite3.Connection, camera_id: str) -> None:
        """Keep only the window recovery can actually return.

        Ingest previously carried no DELETE at all, so `runtime_analysis_*` grew
        without bound on every camera publishing traces -- unbounded growth of
        the very database this ownership change centres on. Recovery is limited
        to `MAX_TRACE_FRAMES`, so anything older is storage nobody can read.

        Pruned oldest-first so the retained window stays contiguous: trimming the
        front is a window that starts late, which replay accepts, while removing
        from the middle would be a hole and would correctly refuse.
        """
        connection.execute(
            """
            DELETE FROM runtime_analysis_traces
            WHERE camera_id = ?
              AND rowid NOT IN (
                  SELECT rowid FROM runtime_analysis_traces
                  WHERE camera_id = ?
                  ORDER BY rowid DESC
                  LIMIT ?
              )
            """,
            (camera_id, camera_id, MAX_TRACE_FRAMES),
        )

    def recover(self, camera_id: str) -> ReplayTrace:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            cursor = connection.execute(
                "SELECT * FROM runtime_trace_cursors WHERE camera_id = ?", (camera_id,)
            ).fetchone()
            if cursor is None:
                raise ReplayInputUnavailable(f"no captured analysis trace for camera {camera_id!r}")
            rows = tuple(
                reversed(
                    connection.execute(
                        # Ordered by insertion, not by worker_boot_id. Boot ids
                        # are uuid4 hex (worker/pipeline/ingest/rtsp.py), so
                        # sorting them lexically picks an arbitrary boot rather
                        # than the most recent one -- with a LIMIT that means
                        # recovering an old window while newer frames exist.
                        # rowid is the only monotonic signal available without
                        # a schema change.
                        """SELECT * FROM runtime_analysis_traces WHERE camera_id = ?
                        ORDER BY rowid DESC
                        LIMIT ?""",
                        (camera_id, MAX_TRACE_FRAMES),
                    ).fetchall()
                )
            )
            if not rows:
                raise ReplayInputUnavailable(f"no captured analysis trace for camera {camera_id!r}")
            frames = tuple(self._frame(connection, row) for row in rows)
            truncation = {
                "handoff_dropped_frames": cursor["handoff_dropped_frames"],
                "pruned_frames": cursor["pruned_frames"],
                "persistence_failed_frames": cursor["persistence_failed_frames"],
                "retention_blocked_frames": cursor["retention_blocked_frames"],
                "oldest_retained_seq": cursor["oldest_retained_seq"],
                "newest_retained_seq": cursor["newest_retained_seq"],
                "oldest_retained_key": self._cursor_key(cursor, "oldest"),
                "newest_retained_key": self._cursor_key(cursor, "newest"),
                "detail_unavailable_reason": None,
            }
            trace = ReplayTrace(camera_id, frames, truncation)
            if any(
                int(truncation[key])
                for key in (
                    "handoff_dropped_frames",
                    "pruned_frames",
                    "persistence_failed_frames",
                    "retention_blocked_frames",
                )
            ):
                raise ReplayInputUnavailable("captured replay input is truncated")
            return trace
        finally:
            connection.close()

    @staticmethod
    def _cursor_key(row: sqlite3.Row, prefix: str) -> list[object] | None:
        boot = row[f"{prefix}_retained_boot_id"]
        epoch = row[f"{prefix}_retained_stream_epoch"]
        sequence = row[f"{prefix}_retained_seq"]
        return (
            [boot, row["camera_id"], epoch, sequence]
            if boot is not None and epoch is not None and sequence is not None
            else None
        )

    def _insert_frame(
        self, connection: sqlite3.Connection, camera_id: str, frame: dict[str, object]
    ) -> None:
        required = {
            "trace_id",
            "frame_key",
            "pts",
            "source_time",
            "frame_width",
            "frame_height",
            "bed_region_provenance",
            "persons",
            "beds",
            "components",
            "schema_version",
        }
        if set(frame) != required:
            raise ReplayWireError("analysis frame has undeclared or missing fields")
        key = frame["frame_key"]
        if not isinstance(key, list) or len(key) != 4 or key[1] != camera_id:
            raise ReplayWireError("analysis frame key does not match camera")
        pts = self._optional(frame["pts"], "pts")
        source = self._optional(frame["source_time"], "source_time")
        values = (
            frame["trace_id"],
            frame["schema_version"],
            key[0],
            camera_id,
            key[2],
            key[3],
            *pts,
            *source,
            frame["frame_width"],
            frame["frame_height"],
            frame["bed_region_provenance"],
        )
        trace_id = frame["trace_id"]
        if not isinstance(trace_id, str):
            raise ReplayWireError("analysis trace_id must be a string")
        existing = connection.execute(
            "SELECT * FROM runtime_analysis_traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if existing is not None:
            if self._frame(connection, existing) != frame:
                raise RuntimeTraceConflict(
                    f"analysis trace {trace_id!r} conflicts with the stored trace"
                )
            return
        connection.execute(
            """INSERT INTO runtime_analysis_traces (trace_id,trace_schema_version,worker_boot_id,camera_id,stream_epoch,frame_seq,pts,pts_missing_reason,source_time_sec,source_time_missing_reason,frame_width,frame_height,bed_region_provenance) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            values,
        )
        for component in _rows(frame["components"], "components"):
            connection.execute(
                "INSERT INTO runtime_analysis_components VALUES (?,?,?,?)",
                (
                    trace_id,
                    component["ordinal"],
                    component["qualified_id"],
                    component["observation_state"],
                ),
            )
        for person in _rows(frame["persons"], "persons"):
            track = self._optional(person["track_id"], "track_id")
            connection.execute(
                "INSERT INTO runtime_analysis_persons VALUES (?,?,?,?,?,?,?,?,?)",
                (trace_id, person["ordinal"], *track, *_sequence(person["box"], "person box"), person["confidence"]),
            )
            for point in _rows(person["keypoints"], "keypoints"):
                connection.execute(
                    "INSERT INTO runtime_analysis_keypoints VALUES (?,?,?,?,?,?)",
                    (
                        trace_id,
                        person["ordinal"],
                        point["index"],
                        point["x"],
                        point["y"],
                        point["confidence"],
                    ),
                )
        for bed in _rows(frame["beds"], "beds"):
            connection.execute(
                "INSERT INTO runtime_analysis_beds VALUES (?,?,?,?,?,?,?,?)",
                (trace_id, bed["ordinal"], *_sequence(bed["box"], "bed box"), bed["confidence"], bed["provenance"]),
            )
            for index, polygon_point in enumerate(_sequence(bed["polygon"], "polygon")):
                connection.execute(
                    "INSERT INTO runtime_analysis_bed_points VALUES (?,?,?,?,?)",
                    (trace_id, bed["ordinal"], index, *_sequence(polygon_point, "polygon point")),
                )

    @staticmethod
    def _optional(value: object, name: str) -> tuple[object, object]:
        if not isinstance(value, dict) or set(value) != {"value", "missing_reason"}:
            raise ReplayWireError(f"{name} must carry value and missing_reason")
        present, missing = value["value"], value["missing_reason"]
        if (present is None) == (missing is None):
            raise ReplayWireError(f"{name} must contain exactly one value or missing reason")
        return present, missing

    def _upsert_cursor(
        self, connection: sqlite3.Connection, camera_id: str, value: dict[str, object]
    ) -> None:
        keys = (
            "handoff_dropped_frames",
            "pruned_frames",
            "persistence_failed_frames",
            "retention_blocked_frames",
            "oldest_retained_seq",
            "newest_retained_seq",
            "oldest_retained_key",
            "newest_retained_key",
            "detail_unavailable_reason",
        )
        if set(value) != set(keys):
            raise ReplayWireError("truncation fields are incomplete")
        oldest_row = _sequence(value["oldest_retained_key"], "oldest_retained_key") \
            if value["oldest_retained_key"] is not None else None
        newest_row = _sequence(value["newest_retained_key"], "newest_retained_key") \
            if value["newest_retained_key"] is not None else None
        connection.execute(
            """INSERT INTO runtime_trace_cursors (camera_id,handoff_dropped_frames,pruned_frames,oldest_retained_seq,newest_retained_seq,persistence_failed_frames,retention_blocked_frames,oldest_retained_boot_id,oldest_retained_stream_epoch,newest_retained_boot_id,newest_retained_stream_epoch) VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(camera_id) DO UPDATE SET handoff_dropped_frames=excluded.handoff_dropped_frames,pruned_frames=excluded.pruned_frames,oldest_retained_seq=excluded.oldest_retained_seq,newest_retained_seq=excluded.newest_retained_seq,persistence_failed_frames=excluded.persistence_failed_frames,retention_blocked_frames=excluded.retention_blocked_frames,oldest_retained_boot_id=excluded.oldest_retained_boot_id,oldest_retained_stream_epoch=excluded.oldest_retained_stream_epoch,newest_retained_boot_id=excluded.newest_retained_boot_id,newest_retained_stream_epoch=excluded.newest_retained_stream_epoch""",
            (
                camera_id,
                value["handoff_dropped_frames"],
                value["pruned_frames"],
                value["oldest_retained_seq"],
                value["newest_retained_seq"],
                value["persistence_failed_frames"],
                value["retention_blocked_frames"],
                oldest_row[0] if oldest_row else None,
                oldest_row[2] if oldest_row else None,
                newest_row[0] if newest_row else None,
                newest_row[2] if newest_row else None,
            ),
        )

    def _frame(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, object]:
        trace_id = row["trace_id"]
        persons = []
        for person in connection.execute(
            "SELECT * FROM runtime_analysis_persons WHERE analysis_trace_id=? ORDER BY ordinal",
            (trace_id,),
        ):
            points = [
                dict(index=p["keypoint_index"], x=p["x"], y=p["y"], confidence=p["confidence"])
                for p in connection.execute(
                    "SELECT * FROM runtime_analysis_keypoints WHERE analysis_trace_id=? AND person_ordinal=? ORDER BY keypoint_index",
                    (trace_id, person["ordinal"]),
                )
            ]
            persons.append(
                dict(
                    ordinal=person["ordinal"],
                    track_id={
                        "value": person["track_id"],
                        "missing_reason": person["track_missing_reason"],
                    },
                    box=[person["x1"], person["y1"], person["x2"], person["y2"]],
                    confidence=person["confidence"],
                    keypoints=points,
                )
            )
        beds = []
        for bed in connection.execute(
            "SELECT * FROM runtime_analysis_beds WHERE analysis_trace_id=? ORDER BY ordinal",
            (trace_id,),
        ):
            polygon = [
                [p["x"], p["y"]]
                for p in connection.execute(
                    "SELECT * FROM runtime_analysis_bed_points WHERE analysis_trace_id=? AND bed_ordinal=? ORDER BY point_index",
                    (trace_id, bed["ordinal"]),
                )
            ]
            beds.append(
                dict(
                    ordinal=bed["ordinal"],
                    box=[bed["x1"], bed["y1"], bed["x2"], bed["y2"]],
                    confidence=bed["confidence"],
                    provenance=bed["provenance"],
                    polygon=polygon,
                )
            )
        components = [
            dict(
                ordinal=c["ordinal"],
                qualified_id=c["component_qualified_id"],
                observation_state=c["observation_state"],
            )
            for c in connection.execute(
                "SELECT * FROM runtime_analysis_components WHERE analysis_trace_id=? ORDER BY ordinal",
                (trace_id,),
            )
        ]
        return dict(
            trace_id=trace_id,
            frame_key=[
                row["worker_boot_id"],
                row["camera_id"],
                row["stream_epoch"],
                row["frame_seq"],
            ],
            pts={"value": row["pts"], "missing_reason": row["pts_missing_reason"]},
            source_time={
                "value": row["source_time_sec"],
                "missing_reason": row["source_time_missing_reason"],
            },
            frame_width=row["frame_width"],
            frame_height=row["frame_height"],
            bed_region_provenance=row["bed_region_provenance"],
            persons=persons,
            beds=beds,
            components=components,
            schema_version=row["trace_schema_version"],
        )


__all__ = ["ReplayInputUnavailable", "RuntimeAnalysisStore", "RuntimeTraceConflict"]
