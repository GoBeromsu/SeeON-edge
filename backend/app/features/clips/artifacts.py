"""Privacy-bounded central clip artifact projection and verified derivative reads."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.features.clips.descriptor_files import (
    OpenedRegularFile,
    open_contained_regular_file,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ClipAnalysis:
    decision_trace_id: str
    module_qualified_id: str
    policy_qualified_id: str
    effective_policy_id: str
    runtime_manifest_sha256: str
    reason: str
    previous_state: str
    current_state: str
    triggered: bool
    track_id: int | None
    bed_id: int | None
    values: tuple[tuple[str, float | None, str | None], ...]


@dataclass(frozen=True, slots=True)
class DerivativeProjection:
    kind: str
    state: str
    reason: str | None
    request_id: str
    sha256: str | None
    size_bytes: int | None
    mime_type: str | None
    relpath: str | None
    width: int | None
    height: int | None
    start_time_ms: int | None
    end_time_ms: int | None
    render_backend: str | None
    render_version: str | None
    scene_id: str | None
    primary_clip_id: str | None
    decision_trace_id: str | None
    runtime_manifest_sha256: str | None


@dataclass(frozen=True, slots=True)
class CentralClipArtifacts:
    incident_id: str
    clean_state: str
    analysis: ClipAnalysis | None
    annotated_state: str
    annotated_relpath: str | None
    annotated_sha256: str | None
    annotated_size_bytes: int | None
    still: DerivativeProjection | None
    video: DerivativeProjection | None


class CentralClipArtifactQuery:
    """Read stable service facts without exposing worker-owned table rows."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path: Path = EDGE_DATABASE_PATH if database_path is None else database_path

    def get(self, clip_id: str) -> CentralClipArtifacts | None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            row = connection.execute(
                """
                SELECT incident.incident_id, primary_slot.state,
                       trace.trace_id, trace.module_qualified_id,
                       trace.policy_qualified_id, trace.effective_policy_id,
                       trace.runtime_manifest_sha256, trace.reason,
                       trace.previous_state, trace.current_state, trace.triggered,
                       trace.track_id, trace.bed_id, derivative.state,
                       media.contained_relpath, media.content_sha256, media.size_bytes
                FROM evidence_primary_clips AS primary_record
                JOIN evidence_incidents AS incident USING (incident_id)
                LEFT JOIN evidence_artifact_slots AS primary_slot
                  ON primary_slot.incident_id = incident.incident_id
                 AND primary_slot.slot_name = 'PRIMARY_CLIP'
                LEFT JOIN evidence_clip_trace_refs AS trace_ref
                  ON trace_ref.clip_id = primary_record.clip_id
                LEFT JOIN evidence_decision_traces AS trace
                  ON trace.trace_id = trace_ref.decision_trace_id
                LEFT JOIN derivative_evidence_slots AS derivative
                  ON derivative.incident_id = incident.incident_id
                 AND derivative.derivative_kind = 'ANNOTATED_CLIP'
                LEFT JOIN derivative_render_records AS render
                  ON render.incident_id = derivative.incident_id
                 AND render.derivative_kind = derivative.derivative_kind
                LEFT JOIN evidence_media_objects AS media
                  ON media.media_id = render.media_id
                WHERE primary_record.clip_id = ?
                """,
                (clip_id,),
            ).fetchone()
            if row is None:
                return None
            analysis = None
            if row[2] is not None:
                values = connection.execute(
                    "SELECT name, numeric_value, missing_reason "
                    "FROM evidence_decision_values WHERE decision_trace_id=? ORDER BY name",
                    (str(row[2]),),
                ).fetchall()
                analysis = ClipAnalysis(
                    decision_trace_id=str(row[2]),
                    module_qualified_id=str(row[3]),
                    policy_qualified_id=str(row[4]),
                    effective_policy_id=str(row[5]),
                    runtime_manifest_sha256=str(row[6]),
                    reason=str(row[7]),
                    previous_state=str(row[8]),
                    current_state=str(row[9]),
                    triggered=bool(row[10]),
                    track_id=_optional_int(row[11]),
                    bed_id=_optional_int(row[12]),
                    values=tuple(
                        (str(value[0]), _optional_float(value[1]), _text(value[2]))
                        for value in values
                    ),
                )
            incident_id = str(row[0])
            still = _derivative_projection(connection, incident_id, "STILL")
            video = _derivative_projection(connection, incident_id, "VIDEO")
            return CentralClipArtifacts(
                incident_id=incident_id,
                clean_state=str(row[1] or "UNAVAILABLE"),
                analysis=analysis,
                annotated_state=(
                    video.state if video is not None else str(row[13] or "NOT_REQUESTED")
                ),
                annotated_relpath=(video.relpath if video is not None else _text(row[14])),
                annotated_sha256=(video.sha256 if video is not None else _text(row[15])),
                annotated_size_bytes=(
                    video.size_bytes if video is not None else _optional_int(row[16])
                ),
                still=still,
                video=video,
            )
        finally:
            connection.close()


def _derivative_projection(
    connection: sqlite3.Connection, incident_id: str, kind: str
) -> DerivativeProjection | None:
    row = connection.execute(
        "SELECT job.derivative_kind,job.state,job.reason,job.request_id,"
        "media.content_sha256,media.size_bytes,media.mime_type,media.contained_relpath,"
        "artifact.width,artifact.height,artifact.start_time_ms,artifact.end_time_ms,"
        "artifact.render_backend,artifact.render_version,artifact.scene_id,"
        "artifact.primary_clip_id,artifact.decision_trace_id,"
        "artifact.runtime_manifest_sha256 FROM derivative_jobs AS job "
        "LEFT JOIN derivative_artifacts AS artifact "
        "USING(incident_id,derivative_kind) LEFT JOIN evidence_media_objects AS media "
        "ON media.media_id=artifact.media_id WHERE job.incident_id=? "
        "AND job.derivative_kind=?",
        (incident_id, kind),
    ).fetchone()
    if row is None:
        return None
    return DerivativeProjection(
        str(row[0]),
        str(row[1]),
        _text(row[2]),
        str(row[3]),
        _text(row[4]),
        _optional_int(row[5]),
        _text(row[6]),
        _text(row[7]),
        _optional_int(row[8]),
        _optional_int(row[9]),
        _optional_int(row[10]),
        _optional_int(row[11]),
        _text(row[12]),
        _text(row[13]),
        _text(row[14]),
        _text(row[15]),
        _text(row[16]),
        _text(row[17]),
    )


def open_verified_annotated(
    store_root: Path, artifacts: CentralClipArtifacts
) -> OpenedRegularFile | None:
    """Open annotated media using publication-time identity, not a full rehash.

    Publication already persisted ``content_sha256`` + ``size_bytes`` after a
    verified write. Playback trusts that identity when the contained regular
    file's size still matches; a size mismatch (or missing/unreadable file)
    falls back to clean. Full-file hashing on every range request is avoided.
    """

    if (
        artifacts.annotated_state != "AVAILABLE"
        or artifacts.annotated_relpath is None
        or artifacts.annotated_sha256 is None
        or artifacts.annotated_size_bytes is None
    ):
        return None
    if not _SHA256_RE.fullmatch(artifacts.annotated_sha256):
        return None
    try:
        opened = open_contained_regular_file(
            store_root, store_root / Path(artifacts.annotated_relpath)
        )
    except FileNotFoundError:
        return None
    if opened.size_bytes != artifacts.annotated_size_bytes:
        opened.handle.close()
        return None
    return opened


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("stored clip artifact integer is invalid")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError("stored clip analysis value is invalid")
    return float(value)


def _text(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "CentralClipArtifactQuery",
    "CentralClipArtifacts",
    "ClipAnalysis",
    "DerivativeProjection",
    "open_verified_annotated",
]
