"""Crash-resumable publication of re-encoded derivative clips."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, final

from worker.adapters.encode.adapter_errors import ThumbnailGenerationError
from worker.adapters.encode.thumbnail import THUMBNAIL_FILENAME, FFmpegThumbnailGenerator
from worker.interfaces import ThumbnailGenerator
from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.pipeline.output.evidence.clip_publication_types import (
    ClipPublicationConflictError,
    ClipPublicationMetadata,
    ClipTimeOrigin,
    JsonValue,
    PublicationBarrier,
    PublicationStage,
    PublishedClip,
)
from worker.pipeline.output.evidence.durability import fsync_directory, fsync_file
from worker.pipeline.output.evidence.evidence_manifest import (
    ReadyClipManifest,
    UnavailableClipManifest,
    finalize_ready_manifest,
    unavailable_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode

LOGGER: Final = logging.getLogger(__name__)


def _no_barrier(_stage: PublicationStage, _path: Path) -> None:
    return


@final
class ClipPublisher:
    def __init__(
        self,
        store_dir: Path,
        *,
        barrier: PublicationBarrier = _no_barrier,
        ffprobe_bin: str = "ffprobe",
        thumbnail_generator: ThumbnailGenerator | None = None,
    ) -> None:
        self._store_dir = store_dir
        self._barrier = barrier
        self._ffprobe_bin = ffprobe_bin
        self._thumbnail_generator = thumbnail_generator or FFmpegThumbnailGenerator()

    def publish_ready(
        self,
        reservation: ClipReservation,
        artifact_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> PublishedClip:
        self._validate_reservation(reservation)
        video_path = self._publish_media(reservation, artifact_path)
        try:
            thumbnail_path = self._thumbnail_generator.generate(
                video_path,
                reservation.final_dir / THUMBNAIL_FILENAME,
                metadata.duration_s,
            )
        except ThumbnailGenerationError as exc:
            LOGGER.warning(
                "clip thumbnail generation failed camera_id=%r clip_id=%r error_type=%s",
                metadata.camera_id,
                str(reservation.clip_id),
                type(exc).__name__,
                extra={
                    "camera_id": metadata.camera_id,
                    "clip_id": str(reservation.clip_id),
                },
            )
        else:
            self._barrier(PublicationStage.THUMBNAIL_RENAMED, thumbnail_path)
        manifest = finalize_ready_manifest(
            video_path=video_path,
            clip_id=reservation.clip_id,
            camera_id=metadata.camera_id,
            event_refs=metadata.event_refs,
            clip_start_at=metadata.clip_start_at,
            clip_end_at=metadata.clip_end_at,
            finalized_at=metadata.finalized_at,
            ffprobe_bin=self._ffprobe_bin,
            runtime_manifest_sha256=metadata.runtime_manifest_sha256,
        )
        payload = _manifest_payload(
            manifest,
            metadata,
            path=f"clips/{reservation.clip_id}/{video_path.name}",
            video_available=True,
        )
        manifest_path = self._publish_manifest(reservation, payload)
        self._cleanup_staging(reservation)
        return PublishedClip(reservation.clip_id, manifest, manifest_path, video_path)

    def publish_unavailable(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
        reason_code: EvidenceReasonCode,
    ) -> PublishedClip:
        self._validate_reservation(reservation)
        reservation.final_dir.mkdir(parents=True, exist_ok=True)
        fsync_directory(reservation.final_dir.parent)
        manifest = unavailable_manifest(
            clip_id=reservation.clip_id,
            camera_id=metadata.camera_id,
            event_refs=metadata.event_refs,
            clip_start_at=metadata.clip_start_at,
            clip_end_at=metadata.clip_end_at,
            finalized_at=metadata.finalized_at,
            reason_code=reason_code,
            runtime_manifest_sha256=metadata.runtime_manifest_sha256,
        )
        payload = _manifest_payload(
            manifest,
            metadata,
            path=None,
            video_available=False,
        )
        manifest_path = self._publish_manifest(reservation, payload)
        self._cleanup_staging(reservation)
        return PublishedClip(reservation.clip_id, manifest, manifest_path, None)

    def _publish_media(
        self,
        reservation: ClipReservation,
        artifact_path: Path,
    ) -> Path:
        destination = reservation.final_dir / "clip.mp4"
        reservation.final_dir.mkdir(parents=True, exist_ok=True)
        fsync_directory(reservation.final_dir.parent)
        if destination.exists():
            if artifact_path.exists() and artifact_path.read_bytes() != destination.read_bytes():
                raise ClipPublicationConflictError(
                    reservation.clip_id,
                    "existing media differs from staged derivative",
                )
            fsync_file(destination)
            fsync_directory(reservation.final_dir)
            return destination
        if not artifact_path.is_file():
            raise ClipPublicationConflictError(
                reservation.clip_id,
                "staged derivative is missing",
            )
        fsync_file(artifact_path)
        self._barrier(PublicationStage.MEDIA_FSYNCED, artifact_path)
        os.replace(artifact_path, destination)
        self._barrier(PublicationStage.MEDIA_RENAMED, destination)
        fsync_file(destination)
        fsync_directory(reservation.final_dir)
        return destination

    def _publish_manifest(
        self,
        reservation: ClipReservation,
        payload: Mapping[str, JsonValue],
    ) -> Path:
        manifest_path = reservation.final_dir / "manifest.json"
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ClipPublicationConflictError(
                    reservation.clip_id,
                    "existing manifest is unreadable",
                ) from exc
            if existing != payload:
                raise ClipPublicationConflictError(
                    reservation.clip_id,
                    "existing manifest differs from resumed publication",
                )
            fsync_file(manifest_path)
            fsync_directory(reservation.final_dir)
            return manifest_path

        temporary = manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, separators=(",", ":"), sort_keys=True)
            _ = output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, manifest_path)
        self._barrier(PublicationStage.MANIFEST_RENAMED, manifest_path)
        fsync_file(manifest_path)
        fsync_directory(reservation.final_dir)
        return manifest_path

    def _cleanup_staging(self, reservation: ClipReservation) -> None:
        if reservation.staging_dir.exists():
            shutil.rmtree(reservation.staging_dir)
            fsync_directory(reservation.staging_dir.parent)

    def _validate_reservation(self, reservation: ClipReservation) -> None:
        clips_dir = self._store_dir / "clips"
        if reservation.final_dir != clips_dir / reservation.clip_id:
            raise ClipPublicationConflictError(reservation.clip_id, "final path mismatch")
        if reservation.staging_dir != clips_dir / ".staging" / reservation.clip_id:
            raise ClipPublicationConflictError(reservation.clip_id, "staging path mismatch")
        if reservation.camera_id.strip() == "":
            raise ClipPublicationConflictError(reservation.clip_id, "camera id is blank")


def _manifest_payload(
    manifest: ReadyClipManifest | UnavailableClipManifest,
    metadata: ClipPublicationMetadata,
    *,
    path: str | None,
    video_available: bool,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = manifest.model_dump(mode="json", exclude_none=True)
    payload.update(
        {
            "event_ref": str(metadata.event_refs[0]),
            "started_at": _utc_iso(metadata.started_at),
            "duration_s": metadata.duration_s,
            "encoder": metadata.encoder,
            "path": path,
            "finalized": True,
            "video_available": video_available,
            "recovery_state": "MEDIA_VERIFIED" if video_available else "UNAVAILABLE",
        }
    )
    if metadata.decision_trace_id is not None:
        payload["decision_trace_id"] = metadata.decision_trace_id
    if metadata.event_type is not None:
        payload["event_type"] = metadata.event_type
    if metadata.domain is not None:
        payload["domain"] = metadata.domain
    if metadata.source_media is not None:
        payload["source_media"] = metadata.source_media
    if metadata.source_error_reason is not None:
        payload["source_error_reason"] = metadata.source_error_reason
    if metadata.truncation_reasons:
        payload["truncation_reasons"] = list(metadata.truncation_reasons)
    if metadata.time_origin is not None:
        origin = metadata.time_origin
        payload["time_origin"] = {
            "worker_boot_id": origin.worker_boot_id,
            "camera_id": origin.camera_id,
            "stream_epoch": origin.stream_epoch,
            "generation": origin.generation,
            "media_origin_pts_sec": origin.media_origin_pts_sec,
            "event_pts_sec": origin.event_pts_sec,
            "requested_start_pts_sec": origin.requested_start_pts_sec,
            "requested_end_pts_sec": origin.requested_end_pts_sec,
            "event_media_time_ms": origin.event_media_time_ms,
        }
    return payload


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ClipPublicationConflictError",
    "ClipPublicationMetadata",
    "ClipPublisher",
    "ClipTimeOrigin",
    "PublicationBarrier",
    "PublicationStage",
    "PublishedClip",
]
