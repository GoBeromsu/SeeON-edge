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
                "SELECT incident_id,state FROM artifacts "
                "WHERE clip_id=? AND kind='PRIMARY_CLIP'",
                (clip_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return CentralClipArtifacts(
            incident_id=str(row[0]), clean_state=str(row[1]), analysis=None,
            annotated_state="NOT_REQUESTED", annotated_relpath=None,
            annotated_sha256=None, annotated_size_bytes=None, still=None, video=None,
        )


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
