"""Crash-resumable publication of re-encoded derivative clips."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Final, final

from worker.adapters.encode.adapter_errors import ThumbnailGenerationError
from worker.adapters.encode.thumbnail import THUMBNAIL_FILENAME, FFmpegThumbnailGenerator
from worker.interfaces import ThumbnailGenerator
from worker.pipeline.output.evidence.clip_corrupt_publication import publish_existing_corrupt
from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.pipeline.output.evidence.clip_manifest_payload import manifest_payload
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
    finalize_ready_manifest,
    unavailable_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode
from worker.pipeline.output.evidence.manifest_media_models import SceneIndexFacts
from worker.pipeline.output.evidence.scene_index import SCENE_INDEX_FILENAME
from worker.pipeline.output.evidence.terminal_outcome import (
    TerminalClipOutcome,
    TerminalClipState,
    commit_terminal_outcome,
)

LOGGER: Final = logging.getLogger(__name__)


def _no_barrier(_stage: PublicationStage, _path: Path) -> None:
    return


def _scene_index_facts(path: Path) -> SceneIndexFacts:
    data = path.read_bytes()
    return SceneIndexFacts(
        path=SCENE_INDEX_FILENAME,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        schema=1,
        count=_scene_frame_count(data),
    )


def _scene_frame_count(data: bytes) -> int:
    try:
        value = json.loads(data)
        count = value["frame_count"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise OSError("scene index is invalid") from exc
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise OSError("scene index frame count is invalid")
    return count


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
        metadata = replace(
            metadata,
            scene_index=self._publish_scene_index(reservation, metadata),
        )
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
        payload = manifest_payload(
            manifest,
            metadata,
            path=f"clips/{reservation.clip_id}/{video_path.name}",
            video_available=True,
        )
        manifest_path = self._publish_manifest(reservation, payload)
        _ = commit_terminal_outcome(
            reservation.final_dir,
            TerminalClipOutcome(
                str(reservation.clip_id),
                tuple(str(value) for value in metadata.event_refs),
                TerminalClipState.READY,
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            ),
        )
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
        payload = manifest_payload(
            manifest,
            metadata,
            path=None,
            video_available=False,
        )
        manifest_path = self._publish_manifest(reservation, payload)
        _ = commit_terminal_outcome(
            reservation.final_dir,
            TerminalClipOutcome(
                str(reservation.clip_id),
                tuple(str(value) for value in metadata.event_refs),
                TerminalClipState.UNAVAILABLE,
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            ),
        )
        self._cleanup_staging(reservation)
        return PublishedClip(reservation.clip_id, manifest, manifest_path, None)

    def publish_corrupt(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
    ) -> PublishedClip:
        existing = publish_existing_corrupt(reservation, metadata)
        if existing is not None:
            return existing
        return self.publish_unavailable(reservation, metadata, EvidenceReasonCode.CORRUPT)

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
        au_index = artifact_path.with_name("au-index.cbor")
        if au_index.exists():
            fsync_file(au_index)
        fsync_file(artifact_path)
        self._barrier(PublicationStage.MEDIA_FSYNCED, artifact_path)
        os.replace(artifact_path, destination)
        if au_index.exists():
            os.replace(au_index, reservation.final_dir / "au-index.cbor")
        self._barrier(PublicationStage.MEDIA_RENAMED, destination)
        fsync_file(destination)
        final_au_index = reservation.final_dir / "au-index.cbor"
        if final_au_index.exists():
            fsync_file(final_au_index)
        fsync_directory(reservation.final_dir)
        return destination

    def _publish_scene_index(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
    ) -> SceneIndexFacts | None:
        """Best-effort sidecar promotion; failure must never block READY media."""
        expected = metadata.scene_index
        if expected is None:
            return None
        staged = reservation.staging_dir / SCENE_INDEX_FILENAME
        destination = reservation.final_dir / SCENE_INDEX_FILENAME
        try:
            if destination.exists():
                facts = _scene_index_facts(destination)
                if facts != expected:
                    self._scene_warning(metadata, reservation, "HASH_CONFLICT")
                    return None
                fsync_file(destination)
                fsync_directory(reservation.final_dir)
                return facts
            if not staged.is_file():
                self._scene_warning(metadata, reservation, "STAGED_MISSING")
                return None
            facts = _scene_index_facts(staged)
            if facts != expected:
                self._scene_warning(metadata, reservation, "HASH_CONFLICT")
                return None
            fsync_file(staged)
            os.replace(staged, destination)
            fsync_file(destination)
            fsync_directory(reservation.final_dir)
        except OSError as exc:
            self._scene_warning(metadata, reservation, "PROMOTION_FAILED", exc)
            return None
        else:
            return facts

    def _scene_warning(
        self,
        metadata: ClipPublicationMetadata,
        reservation: ClipReservation,
        reason: str,
        exc: Exception | None = None,
    ) -> None:
        LOGGER.warning(
            "clip scene index not published: camera_id=%s clip_id=%s reason=%s error_type=%s",
            metadata.camera_id,
            reservation.clip_id,
            reason,
            type(exc).__name__ if exc is not None else "None",
        )

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


__all__ = [
    "ClipPublicationConflictError",
    "ClipPublicationMetadata",
    "ClipPublisher",
    "ClipTimeOrigin",
    "PublicationBarrier",
    "PublicationStage",
    "PublishedClip",
]
