from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Final, Protocol, final

from worker.adapters.encode.adapter_errors import ClipRemuxError
from worker.adapters.encode.models import ClipArtifact
from worker.pipeline.output.evidence.clip_recording import (
    ClipOutcome,
    ClipReady,
    ClipReasonCode,
    ClipUnavailable,
    ClipWindow,
)
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.types import BusinessEvent, FrameKey, FramePacket
from worker.types.source_packet import (
    PacketSelectionError,
    SourcePacket,
    SourceStreamConfiguration,
    StreamEpoch,
)

LOGGER: Final = logging.getLogger(__name__)


class PacketRemuxer(Protocol):
    def remux(
        self,
        packets: Sequence[SourcePacket],
        configuration: SourceStreamConfiguration,
        output_path: Path,
    ) -> ClipArtifact: ...


@final
class PacketClipRecordingCoordinator:
    """Select source packets and stream-copy them; decoded frames are timing taps only."""

    def __init__(
        self,
        repository: PacketRingRepository,
        remuxer: PacketRemuxer,
        *,
        window: ClipWindow,
    ) -> None:
        self._repository = repository
        self._remuxer = remuxer
        self._window = window

    def set_camera_fps(self, camera_id: str, fps: float) -> None:
        del camera_id, fps

    def write(self, packet: FramePacket) -> bool:
        del packet
        return True

    def seal(self, camera_id: str) -> bool:
        del camera_id
        return True

    def finalize(
        self,
        *,
        camera_id: str,
        clip_id: str,
        event_time_sec: float,
        event: BusinessEvent,
        output_dir: Path | None = None,
        trigger_frame_key: FrameKey | None = None,
        window_bounds: tuple[float, float] | None = None,
    ) -> ClipOutcome:
        del event_time_sec, event
        if output_dir is None:
            return ClipUnavailable(
                clip_id,
                ClipReasonCode.REMUX_FAILED,
                "OUTPUT_DIRECTORY_UNAVAILABLE",
            )
        if trigger_frame_key is None or not trigger_frame_key.worker_boot_id:
            return ClipUnavailable(
                clip_id,
                ClipReasonCode.STREAM_EPOCH_MISMATCH,
                "TRIGGER_STREAM_IDENTITY_UNAVAILABLE",
            )
        trigger_pts = _trigger_time(trigger_frame_key)
        if trigger_pts is None:
            return ClipUnavailable(
                clip_id,
                ClipReasonCode.REMUX_FAILED,
                "TRIGGER_SOURCE_PTS_UNAVAILABLE",
            )
        epoch = StreamEpoch(
            trigger_frame_key.worker_boot_id,
            camera_id,
            trigger_frame_key.stream_epoch,
            trigger_frame_key.source_generation,
        )
        if window_bounds is None:
            pre_seconds = Fraction(str(self._window.pre_event_seconds))
            post_seconds = Fraction(str(self._window.post_event_seconds))
        else:
            pre_seconds = trigger_pts - Fraction(str(window_bounds[0]))
            post_seconds = Fraction(str(window_bounds[1])) - trigger_pts
        try:
            selection = self._repository.ring(camera_id).select(
                trigger_epoch=epoch,
                trigger_pts=trigger_pts,
                pre_seconds=max(pre_seconds, Fraction()),
                post_seconds=max(post_seconds, Fraction()),
            )
        except ValueError:
            return ClipUnavailable(
                clip_id,
                ClipReasonCode.REMUX_FAILED,
                "SOURCE_PACKET_RING_UNAVAILABLE",
            )
        except PacketSelectionError as exc:
            return ClipUnavailable(
                clip_id,
                ClipReasonCode.REMUX_FAILED,
                exc.reason.value,
            )
        # Detach everything the remux needs, then release the ring lease BEFORE
        # remuxing. SourcePacket is immutable and already materialized, so the
        # lease buys nothing here -- but holding it across the remux pins the
        # oldest entries, which stops _trim_to_limits() from evicting, which
        # makes the ring drop ARRIVING packets under lease backpressure. Those
        # drops punch holes into the next clip window, and a hole makes the MP4
        # muxer stretch the pre-gap packet's stts duration to the next survivor,
        # which the remux verifier then correctly rejects. That is how this
        # deployment reached 1053 clips stuck at WAITING and 1102 undelivered
        # events. Releasing first is bitrate- and duration-independent: no
        # capacity tuning can substitute for not holding the lease.
        packets = selection.packets
        configuration = selection.configuration
        selected_start = selection.selected_start
        selected_end = selection.selected_end
        truncation_reasons = tuple(reason.value for reason in selection.truncations)
        selection.close()

        try:
            artifact = self._remuxer.remux(
                packets,
                configuration,
                output_dir / "clip.mp4",
            )
            return ClipReady(
                clip_id,
                replace(
                    artifact,
                    duration_s=max(float(selected_end - selected_start), 0.001),
                    selected_start_pts_sec=float(selected_start),
                    selected_end_pts_sec=float(selected_end),
                    truncation_reasons=truncation_reasons,
                ),
            )
        except (ClipRemuxError, OSError) as error:
            LOGGER.warning(
                "source packet clip remux failed: camera_id=%s error_type=%s",
                camera_id,
                type(error).__name__,
                exc_info=True,
            )
            return ClipUnavailable(
                clip_id,
                ClipReasonCode.REMUX_FAILED,
                "SOURCE_PACKET_REMUX_FAILED",
                truncation_reasons,
            )

    def close(self, camera_id: str) -> None:
        del camera_id

    def close_all(self) -> None:
        # ClipActor uses this for frame-encoder generations during routine
        # flushes. The packet ring is a camera-ingest resource and remains
        # live until the runtime closes the repository after ingest stops.
        return


def _trigger_time(frame_key: FrameKey) -> Fraction | None:
    if frame_key.source_pts is not None and frame_key.source_time_base is not None:
        return frame_key.source_pts * frame_key.source_time_base
    if frame_key.pts is None:
        return None
    return Fraction(str(frame_key.pts))


__all__ = ["PacketClipRecordingCoordinator", "PacketRemuxer"]
