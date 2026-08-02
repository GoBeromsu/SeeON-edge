from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Protocol, assert_never, final

from worker.adapters.encode.adapter_errors import EncoderPolicyError
from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.pipeline.output.evidence.clip_publication import (
    ClipPublicationConflictError,
    ClipPublicationMetadata,
    PublishedClip,
)
from worker.pipeline.output.evidence.clip_recorder_models import (
    ActiveClip,
    ClipRecorderConfig,
    ClipRecorderStats,
    EventMessage,
    FrameMessage,
)
from worker.pipeline.output.evidence.clip_recording import (
    ClipOutcome,
    ClipReady,
    ClipReasonCode,
    ClipUnavailable,
)
from worker.pipeline.output.evidence.evidence_media import ClipEvidenceError
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    EdgeEventId,
    EvidenceReasonCode,
)
from worker.types import BusinessEvent, FramePacket

LOGGER: Final = logging.getLogger(__name__)


class RecordingCoordinator(Protocol):
    def write(self, packet: FramePacket) -> bool: ...

    def seal(self, camera_id: str) -> bool: ...

    def finalize(
        self,
        *,
        camera_id: str,
        clip_id: str,
        event_time_sec: float,
        event: BusinessEvent,
        output_dir: Path | None = None,
    ) -> ClipOutcome: ...

    def close(self, camera_id: str) -> None: ...

    def close_all(self) -> None: ...


class ClipPublicationPort(Protocol):
    def publish_ready(
        self,
        reservation: ClipReservation,
        artifact_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> PublishedClip | None: ...

    def publish_unavailable(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
        reason_code: EvidenceReasonCode,
    ) -> PublishedClip | None: ...


class ClipReleaseHook(Protocol):
    def __call__(self, camera_id: str, clip_id: ClipId, /) -> None: ...


class ClipCancelHook(Protocol):
    def __call__(self, reservation: ClipReservation, /) -> None: ...


class ClipFinalizedHook(Protocol):
    def __call__(self, clip_id: ClipId, /) -> None: ...


@dataclass(frozen=True, slots=True)
class ClipActorDependencies:
    coordinator: RecordingCoordinator
    publisher: ClipPublicationPort
    release: ClipReleaseHook
    finalized: ClipFinalizedHook | None = None
    encoder_name: str = "libx264"
    cancel: ClipCancelHook | None = None


@final
class ClipActor:
    def __init__(
        self,
        config: ClipRecorderConfig,
        stats: ClipRecorderStats,
        dependencies: ClipActorDependencies,
    ) -> None:
        self._config = config
        self._stats = stats
        self._dependencies = dependencies
        self._active_by_camera: dict[str, ActiveClip] = {}
        self._latest_time_by_camera: dict[str, float] = {}

    def handle_frame(self, message: FrameMessage) -> None:
        packet = message.packet
        if not self._dependencies.coordinator.write(packet):
            self._stats.failed_writes += 1
        frame_time = packet.frame.time_sec if packet.pts is None else packet.pts
        self._latest_time_by_camera[packet.camera_id] = frame_time
        active = self._active_by_camera.get(packet.camera_id)
        if active is None:
            return
        active.last_time_sec = frame_time
        if frame_time >= active.cutoff_time_sec:
            self._finalize(active, forced=False)

    def handle_event(self, message: EventMessage) -> None:
        camera_id = message.reservation.camera_id
        active = self._active_by_camera.get(camera_id)
        if active is not None and active.reservation.clip_id == message.reservation.clip_id:
            if message.event_ref not in active.event_refs:
                active.event_refs.append(message.event_ref)
            return
        if active is not None:
            self._finalize(active, forced=True)
        if not message.allow_new_clip:
            self._stats.attach_missed_events += 1
            self._dependencies.release(camera_id, message.reservation.clip_id)
            return
        event_time = message.event.time_sec
        started_at = datetime.now(UTC) - timedelta(
            seconds=self._config.pre_event_seconds
        )
        active = ActiveClip(
            reservation=message.reservation,
            event_ref=message.event_ref,
            event_type=message.event_type,
            event=message.event,
            event_time_sec=event_time,
            cutoff_time_sec=event_time + self._config.post_event_seconds,
            started_at=started_at,
            start_time_sec=event_time - self._config.pre_event_seconds,
            last_time_sec=self._latest_time_by_camera.get(camera_id, event_time),
            event_refs=[message.event_ref],
            opened_monotonic=time.monotonic(),
        )
        self._active_by_camera[camera_id] = active
        self._stats.active_clips = len(self._active_by_camera)
        if active.last_time_sec >= active.cutoff_time_sec:
            self._finalize(active, forced=False)

    def expire(self) -> None:
        now = time.monotonic()
        maximum_age = (
            self._config.post_event_seconds + self._config.finalize_grace_seconds
        )
        for active in tuple(self._active_by_camera.values()):
            if now - active.opened_monotonic >= maximum_age:
                self._finalize(active, forced=True)

    def flush(self) -> None:
        for active in tuple(self._active_by_camera.values()):
            self._finalize(active, forced=True)
        self._dependencies.coordinator.close_all()

    def shutdown(self) -> None:
        self.flush()

    def _finalize(self, active: ActiveClip, *, forced: bool) -> None:
        published = False
        camera_id = active.reservation.camera_id
        if forced and active.last_time_sec < active.cutoff_time_sec:
            self._stats.forced_finalized += 1
        try:
            _ = self._dependencies.coordinator.seal(camera_id)
            outcome = self._dependencies.coordinator.finalize(
                camera_id=camera_id,
                clip_id=str(active.reservation.clip_id),
                event_time_sec=active.event_time_sec,
                event=active.event,
                output_dir=active.reservation.staging_dir,
            )
            match outcome:
                case ClipReady(artifact=artifact):
                    metadata = self._metadata(active, artifact.duration_s)
                    _ = self._dependencies.publisher.publish_ready(
                        active.reservation,
                        artifact.path,
                        metadata,
                    )
                case ClipUnavailable(reason_code=reason_code):
                    metadata = self._metadata(
                        active,
                        max(0.0, active.last_time_sec - active.start_time_sec),
                    )
                    _ = self._dependencies.publisher.publish_unavailable(
                        active.reservation,
                        metadata,
                        _evidence_reason(reason_code),
                    )
                    self._stats.video_unavailable_clips += 1
                case unreachable:
                    assert_never(unreachable)
            published = True
            self._stats.finalized_clips += 1
        except (
            ClipEvidenceError,
            ClipPublicationConflictError,
            EncoderPolicyError,
            OSError,
        ):
            self._stats.failed_writes += 1
            LOGGER.warning(
                "clip finalize failed",
                extra={
                    "camera_id": camera_id,
                    "clip_id": str(active.reservation.clip_id),
                },
                exc_info=True,
            )
            if self._dependencies.cancel is not None:
                self._dependencies.cancel(active.reservation)
        finally:
            self._active_by_camera.pop(camera_id, None)
            self._stats.active_clips = len(self._active_by_camera)
            self._dependencies.coordinator.close(camera_id)
            self._dependencies.release(camera_id, active.reservation.clip_id)
        if published and self._dependencies.finalized is not None:
            self._dependencies.finalized(active.reservation.clip_id)

    def _metadata(
        self,
        active: ActiveClip,
        duration_s: float,
    ) -> ClipPublicationMetadata:
        # `active.started_at` is a wall-clock anchor taken when the event
        # arrived, but `duration_s` is derived from source/stream time
        # (segment or frame timestamps), not wall-clock elapsed time. When
        # frames are ingested faster than real time (bursty catch-up,
        # non-realtime test/replay sources), `started_at + duration_s` can
        # land after `finalized_at`, which the manifest's `clip_start_at <=
        # clip_end_at <= finalized_at` contract forbids. Anchor the window
        # to the immutable `finalized_at` instead and derive both ends from
        # it, so the invariant holds unconditionally.
        finalized_at = datetime.now(UTC)
        clip_end_at = min(active.started_at + timedelta(seconds=duration_s), finalized_at)
        clip_start_at = clip_end_at - timedelta(seconds=duration_s)
        return ClipPublicationMetadata(
            camera_id=active.reservation.camera_id,
            event_refs=tuple(EdgeEventId(value) for value in active.event_refs),
            event_type=active.event_type,
            clip_start_at=clip_start_at,
            clip_end_at=clip_end_at,
            finalized_at=finalized_at,
            started_at=active.started_at,
            duration_s=duration_s,
            encoder=self._dependencies.encoder_name,
        )


def _evidence_reason(reason_code: ClipReasonCode) -> EvidenceReasonCode:
    match reason_code:
        case ClipReasonCode.ENCODER_FAILED:
            return EvidenceReasonCode.ENCODER_FAILED
        case ClipReasonCode.REMUX_FAILED:
            return EvidenceReasonCode.FINALIZE_FAILED
        case ClipReasonCode.NO_SEGMENTS:
            return EvidenceReasonCode.NO_FRAMES
        case unreachable:
            assert_never(unreachable)


__all__ = ["ClipActor", "ClipActorDependencies"]
