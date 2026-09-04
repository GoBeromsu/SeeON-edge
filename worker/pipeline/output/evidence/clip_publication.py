"""Crash-resumable publication of re-encoded derivative clips."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final, final

from shared.events.delivery_queue import ClipEntry, DeliveryQueue
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
    ClipManifest,
    finalize_ready_manifest,
    unavailable_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode
from worker.pipeline.output.evidence.manifest_models import ReadyClipManifest
from worker.pipeline.output.evidence.terminal_outcome import (
    TerminalClipOutcome,
    TerminalClipState,
    commit_terminal_outcome,
)

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
        delivery_queue_directory: Path | None = None,
    ) -> None:
        self._store_dir = store_dir
        self._barrier = barrier
        self._ffprobe_bin = ffprobe_bin
        self._thumbnail_generator = thumbnail_generator
        self._delivery_queue_directory = delivery_queue_directory

    def publish_ready(
        self,
        reservation: ClipReservation,
        artifact_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> PublishedClip:
        self._validate_reservation(reservation)
        video_path = self._publish_media(reservation, artifact_path)
        if self._thumbnail_generator is not None:
            thumbnail_path = self._thumbnail_generator.generate(
                video_path,
                reservation.final_dir / "thumbnail.jpg",
                metadata.duration_s,
            )
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
        self._enqueue_clip(manifest, metadata)
        self._cleanup_staging(reservation)
        return PublishedClip(reservation.clip_id, manifest, manifest_path, video_path)

    def publish_adopted_ready(
        self,
        reservation: ClipReservation,
        source_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> PublishedClip:
        """Copy externally-recorded media into store staging before publication.

        Smart Record owns its output path, which can be on another filesystem.
        Copying and fsyncing under the reserved staging directory makes the
        subsequent publication rename local and atomic without consuming the
        plane-owned source until a complete manifest exists.
        """
        if not source_path.is_file():
            raise ClipPublicationConflictError(reservation.clip_id, "recorded media is missing")
        adopted = reservation.staging_dir / "adopted.mp4"
        temporary = adopted.with_suffix(".mp4.tmp")
        _adopt_media(source_path, temporary)
        os.replace(temporary, adopted)
        fsync_file(adopted)
        fsync_directory(reservation.staging_dir)
        return self.publish_ready(reservation, adopted, metadata)

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
        self._enqueue_clip(manifest, metadata)
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

    def _enqueue_clip(self, manifest: ClipManifest, metadata: ClipPublicationMetadata) -> None:
        if self._delivery_queue_directory is None:
            return
        if metadata.facility_id is None:
            raise ClipPublicationConflictError(
                manifest.clip_id, "facility id is required for clip relay delivery"
            )
        if isinstance(manifest, ReadyClipManifest):
            entry = ClipEntry(
                clip_id=manifest.clip_id,
                event_ids=manifest.event_refs,
                camera_id=manifest.camera_id,
                facility_id=metadata.facility_id,
                local_state="VERIFIED",
                state_version=manifest.state_version,
                media_reference=f"clips/{manifest.clip_id}/clip.mp4",
                sha256=manifest.sha256,
                size_bytes=manifest.size_bytes,
                mime_type=manifest.mime_type,
                codec=manifest.codec,
                duration_ms=manifest.duration_ms,
                clip_start_at=manifest.clip_start_at,
                clip_end_at=manifest.clip_end_at,
                finalized_at=manifest.finalized_at,
                unavailable_reason=None,
            )
        else:
            entry = ClipEntry(
                clip_id=manifest.clip_id,
                event_ids=manifest.event_refs,
                camera_id=manifest.camera_id,
                facility_id=metadata.facility_id,
                local_state="UNAVAILABLE",
                state_version=manifest.state_version,
                media_reference=None,
                sha256=None,
                size_bytes=None,
                mime_type=None,
                codec=None,
                duration_ms=None,
                clip_start_at=manifest.clip_start_at,
                clip_end_at=manifest.clip_end_at,
                finalized_at=manifest.finalized_at,
                unavailable_reason=manifest.reason_code.value,
            )
        admitted = DeliveryQueue(self._delivery_queue_directory).try_admit(entry)
        if not admitted.accepted:
            raise ClipPublicationConflictError(
                manifest.clip_id, f"clip relay queue admission failed: {admitted.fault}"
            )

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


def _adopt_media(source_path: Path, destination: Path) -> None:
    """Copy adopted media into staging as a faststart MP4.

    DeepStream's Smart Record writes the moov atom last, but the evidence
    contract requires faststart so a player can start without the whole file.
    Remux (stream copy, no re-encode, so no NVENC session) when a remuxer is
    available and fall back to a plain copy, which the media inspection then
    rejects as CORRUPT rather than publishing something unplayable.
    """
    remuxer = shutil.which("ffmpeg")
    if remuxer is not None:
        result = subprocess.run(  # noqa: S603 - fixed argv, operator-owned binary
            [
                remuxer,
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(source_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                # The staging name ends in .tmp, so ffmpeg cannot infer the
                # container from the extension and must be told.
                "-f",
                "mp4",
                "-y",
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and destination.exists() and destination.stat().st_size > 0:
            with destination.open("rb") as handle:
                os.fsync(handle.fileno())
            return
        LOGGER.warning(
            "faststart remux failed for %s; adopting the original bytes: %s",
            source_path,
            (result.stderr or result.stdout).strip()[:200],
        )
        destination.unlink(missing_ok=True)
    with source_path.open("rb") as source, destination.open("xb") as handle:
        shutil.copyfileobj(source, handle)
        handle.flush()
        os.fsync(handle.fileno())
