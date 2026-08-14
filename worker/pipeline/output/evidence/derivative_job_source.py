from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from shared.edge_db.connection import RuntimeActor, open_runtime_database
from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    DerivativeKind,
    DerivativeUnavailableReason,
)
from worker.pipeline.output.overlay_scene import AppliedCameraProvenance, OverlaySceneBuilder
from worker.pipeline.trace.store import TraceStore


class DerivativeSourceUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CentralDerivativeJobSource:
    """Rebuild render jobs from immutable primary media and persisted traces."""

    def __init__(self, database_path: Path, store_root: Path) -> None:
        self.database_path = database_path
        self.store_root = store_root

    def for_clip(self, clip_id: str, kind: DerivativeKind) -> AnnotatedDerivativeJob:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            row = connection.execute(
                "SELECT incident.incident_id,incident.camera_id,primary_record.clip_id,"
                "media.contained_relpath,media.content_sha256,media.size_bytes,"
                "incident.decision_trace_id,incident.runtime_manifest_sha256,"
                "decision.analysis_trace_id,primary_record.time_origin_json "
                "FROM evidence_primary_clips AS primary_record "
                "JOIN evidence_incidents AS incident USING(incident_id) "
                "JOIN evidence_media_objects AS media ON media.media_id=primary_record.media_id "
                "LEFT JOIN evidence_decision_traces AS decision "
                "ON decision.trace_id=incident.decision_trace_id "
                "WHERE primary_record.clip_id=?",
                (clip_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise DerivativeSourceUnavailable(DerivativeUnavailableReason.SOURCE_MEDIA_MISSING)
        required = row[6:9]
        if any(value is None for value in required):
            raise DerivativeSourceUnavailable(DerivativeUnavailableReason.SOURCE_TRACE_MISSING)
        relative = Path(str(row[3]))
        source = self.store_root / relative
        try:
            source.resolve(strict=False).relative_to(self.store_root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as error:
            raise DerivativeSourceUnavailable(
                DerivativeUnavailableReason.SOURCE_MEDIA_MISSING
            ) from error
        if not source.is_file() or source.is_symlink():
            raise DerivativeSourceUnavailable(DerivativeUnavailableReason.SOURCE_MEDIA_MISSING)
        if _facts(source) != (str(row[4]), int(row[5])):
            raise DerivativeSourceUnavailable(DerivativeUnavailableReason.SOURCE_MEDIA_CORRUPT)
        camera_id = str(row[1])
        analysis_id = str(row[8])
        recovered = TraceStore(self.database_path).recover_camera(camera_id)
        analysis = next(
            (value for value in recovered.frames if value.trace_id == analysis_id),
            None,
        )
        if analysis is None:
            raise DerivativeSourceUnavailable(DerivativeUnavailableReason.SOURCE_TRACE_MISSING)
        decisions = tuple(
            decision
            for decision in recovered.decisions
            if decision.analysis_trace_id == analysis_id
        )
        runtime_manifest = str(row[7])
        scene = OverlaySceneBuilder().from_traces(
            analysis,
            decisions,
            provenance=AppliedCameraProvenance(
                runtime_manifest,
                hashlib.sha256(f"{runtime_manifest}\0{camera_id}".encode()).hexdigest(),
            ),
        )
        media_origin = _media_origin(row[9])
        return AnnotatedDerivativeJob(
            incident_id=str(row[0]),
            primary_clip_id=str(row[2]),
            primary_media_path=source,
            primary_sha256=str(row[4]),
            decision_trace_id=str(row[6]),
            runtime_manifest_sha256=runtime_manifest,
            scenes=(replace(scene),),
            source_size_bytes=int(row[5]),
            media_origin_pts_sec=media_origin,
            derivative_kind=kind,
        )

    def for_incident(self, incident_id: str, kind: DerivativeKind) -> AnnotatedDerivativeJob:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            row = connection.execute(
                "SELECT primary_clip_id FROM evidence_incidents WHERE incident_id=?",
                (incident_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None or row[0] is None:
            raise DerivativeSourceUnavailable(DerivativeUnavailableReason.SOURCE_MEDIA_MISSING)
        return self.for_clip(str(row[0]), kind)


def _media_origin(value: object) -> float:
    if value is None:
        return 0.0
    try:
        payload = json.loads(str(value))
        origin = payload.get("media_origin_pts_sec") if isinstance(payload, dict) else None
        return float(origin) if isinstance(origin, int | float) else 0.0
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0


def _facts(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


__all__ = ["CentralDerivativeJobSource", "DerivativeSourceUnavailable"]
