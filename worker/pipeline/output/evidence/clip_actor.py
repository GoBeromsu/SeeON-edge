from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Final, assert_never, final

from worker.adapters.encode.adapter_errors import EncoderPolicyError
from worker.pipeline.output.evidence.clip_actor_metadata import (
    evidence_reason,
    publication_metadata,
)
from worker.pipeline.output.evidence.clip_actor_types import ClipActorDependencies
from worker.pipeline.output.evidence.clip_publication import (
    ClipPublicationConflictError,
)
from worker.pipeline.output.evidence.clip_recorder_models import (
    ActiveClip,
    ClipRecorderConfig,
    ClipRecorderStats,
    EventMessage,
    FrameMessage,
)
from worker.pipeline.output.evidence.clip_recording import (
    ClipReady,
    ClipReasonCode,
    ClipUnavailable,
)
from worker.pipeline.output.evidence.evidence_media import ClipEvidenceError
from worker.pipeline.output.evidence.terminal_outcome import TerminalOutcomeConflictError
from worker.types.source_packet import StreamEpoch

LOGGER: Final = logging.getLogger(__name__)


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
            if (
                active.trigger_frame_key is None
                or active.trigger_frame_key.worker_boot_id
                != message.trigger_packet.worker_boot_id
                or active.trigger_frame_key.stream_epoch
                != message.trigger_packet.stream_epoch
                or active.trigger_frame_key.source_generation
                != message.trigger_packet.source_generation
            ):
                self._finalize(active, forced=True)
            else:
                event_time = (
                    message.trigger_packet.frame.time_sec
                    if message.trigger_packet.pts is None
                    else message.trigger_packet.pts
                )
                active.start_time_sec = min(
                    active.start_time_sec,
                    event_time - self._config.pre_event_seconds,
                )
                active.cutoff_time_sec = max(
                    active.cutoff_time_sec,
                    event_time + self._config.post_event_seconds,
                )
                if message.event_ref not in active.event_refs:
                    active.event_refs.append(message.event_ref)
                return
        if active is not None:
            self._finalize(active, forced=True)
        if not message.allow_new_clip:
            self._stats.attach_missed_events += 1
            self._dependencies.release(camera_id, message.reservation.clip_id)
            return
        event_time = (
            message.trigger_packet.frame.time_sec
            if message.trigger_packet.pts is None
            else message.trigger_packet.pts
        )
        started_at = datetime.now(UTC) - timedelta(seconds=self._config.pre_event_seconds)
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
            trigger_frame_key=message.trigger_packet.frame_key,
            opened_monotonic=time.monotonic(),
        )
        self._active_by_camera[camera_id] = active
        self._stats.active_clips = len(self._active_by_camera)
        if active.last_time_sec >= active.cutoff_time_sec:
            self._finalize(active, forced=False)

    def handle_epoch_roll(self, previous: StreamEpoch) -> None:
        for active in tuple(self._active_by_camera.values()):
            key = active.trigger_frame_key
            if (
                key is not None
                and key.worker_boot_id == previous.worker_boot_id
                and key.stream_epoch == previous.stream_epoch
                and key.source_generation == previous.source_generation
            ):
                self._finalize(active, forced=True, epoch_rolled=True)

    def expire(self) -> None:
        now = time.monotonic()
        maximum_age = self._config.post_event_seconds + self._config.finalize_grace_seconds
        for active in tuple(self._active_by_camera.values()):
            if now - active.opened_monotonic >= maximum_age:
                self._finalize(active, forced=True)

    def flush(self) -> None:
        for active in tuple(self._active_by_camera.values()):
            self._finalize(active, forced=True)
        self._dependencies.coordinator.close_all()

    def shutdown(self) -> None:
        self.flush()

    def _finalize(
        self, active: ActiveClip, *, forced: bool, epoch_rolled: bool = False
    ) -> None:
        published = False
        camera_id = active.reservation.camera_id
        if forced and active.last_time_sec < active.cutoff_time_sec:
            self._stats.forced_finalized += 1
        try:
            self._dependencies.close(camera_id, active.reservation.clip_id)
            _ = self._dependencies.coordinator.seal(camera_id)
            outcome = (
                ClipUnavailable(
                    str(active.reservation.clip_id),
                    ClipReasonCode.STREAM_EPOCH_MISMATCH,
                    "STREAM_EPOCH_ROLLED",
                    ("STREAM_EPOCH_ROLLED",),
                )
                if epoch_rolled
                else self._dependencies.coordinator.finalize(
                    camera_id=camera_id,
                    clip_id=str(active.reservation.clip_id),
                    event_time_sec=active.event_time_sec,
                    event=active.event,
                    output_dir=active.reservation.staging_dir,
                    trigger_frame_key=active.trigger_frame_key,
                    window_bounds=(active.start_time_sec, active.cutoff_time_sec),
                )
            )
            match outcome:
                case ClipReady(artifact=artifact):
                    metadata = publication_metadata(
                        active, artifact.duration_s, self._dependencies.encoder_name, artifact
                    )
                    _ = self._dependencies.publisher.publish_ready(
                        active.reservation,
                        artifact.path,
                        metadata,
                    )
                case ClipUnavailable(
                    reason_code=reason_code,
                    detail_reason=detail_reason,
                    truncation_reasons=truncation_reasons,
                ):
                    metadata = publication_metadata(
                        active,
                        max(0.0, active.last_time_sec - active.start_time_sec),
                        self._dependencies.encoder_name,
                        source_error_reason=detail_reason,
                        truncation_reasons=truncation_reasons,
                    )
                    _ = self._dependencies.publisher.publish_unavailable(
                        active.reservation,
                        metadata,
                        evidence_reason(reason_code),
                    )
                    self._stats.video_unavailable_clips += 1
                case unreachable:
                    assert_never(unreachable)
            published = True
            self._stats.finalized_clips += 1
        except (
            ClipEvidenceError,
            ClipPublicationConflictError,
            TerminalOutcomeConflictError,
            EncoderPolicyError,
            OSError,
        ):
            self._stats.failed_writes += 1
            LOGGER.warning(
                "clip finalize failed: camera_id=%s clip_id=%s",
                camera_id,
                str(active.reservation.clip_id),
                extra={
                    "camera_id": camera_id,
                    "clip_id": str(active.reservation.clip_id),
                },
                exc_info=True,
            )
            publish_corrupt = getattr(self._dependencies.publisher, "publish_corrupt", None)
            try:
                if callable(publish_corrupt):
                    _ = publish_corrupt(
                        active.reservation,
                        publication_metadata(
                            active,
                            max(0.0, active.last_time_sec - active.start_time_sec),
                            self._dependencies.encoder_name,
                            source_error_reason="FINALIZE_FAILED",
                        ),
                    )
                    published = True
                    self._stats.video_unavailable_clips += 1
                elif self._dependencies.cancel is not None:
                    self._dependencies.cancel(active.reservation)
            except (ClipPublicationConflictError, TerminalOutcomeConflictError, OSError):
                if self._dependencies.cancel is not None:
                    self._dependencies.cancel(active.reservation)
        finally:
            self._active_by_camera.pop(camera_id, None)
            self._stats.active_clips = len(self._active_by_camera)
            self._dependencies.coordinator.close(camera_id)
            self._dependencies.release(camera_id, active.reservation.clip_id)
        if published and self._dependencies.finalized is not None:
            self._dependencies.finalized(active.reservation.clip_id)



__all__ = ["ClipActor", "ClipActorDependencies"]
