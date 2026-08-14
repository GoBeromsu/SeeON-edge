from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from worker.adapters.encode import FFmpegThumbnailGenerator
from worker.adapters.encode.packet_remuxer import PyAvPacketRemuxer
from worker.pipeline.output.evidence.clip_actor import (
    ClipPublicationPort,
    RecordingCoordinator,
)
from worker.pipeline.output.evidence.clip_publication import ClipPublisher
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recording import ClipWindow
from worker.pipeline.output.evidence.packet_recording import PacketClipRecordingCoordinator
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository


class RecorderCoordinator(RecordingCoordinator, Protocol):
    def set_camera_fps(self, camera_id: str, fps: float) -> None: ...


@dataclass(frozen=True, slots=True)
class ClipRecorderServices:
    coordinator: RecorderCoordinator
    publisher: ClipPublicationPort
    encoder_name: str


def default_services(
    config: ClipRecorderConfig,
    repository: PacketRingRepository,
) -> ClipRecorderServices:
    """Build the sole production clean-clip path: source packet stream copy."""
    coordinator = PacketClipRecordingCoordinator(
        repository,
        PyAvPacketRemuxer(),
        window=ClipWindow(
            pre_event_seconds=config.pre_event_seconds,
            post_event_seconds=config.post_event_seconds,
        ),
    )
    ffprobe_bin = (
        f"{config.ffmpeg_bin[: -len('ffmpeg')]}ffprobe"
        if config.ffmpeg_bin.endswith("ffmpeg")
        else "ffprobe"
    )
    return ClipRecorderServices(
        coordinator,
        ClipPublisher(
            config.store_dir,
            ffprobe_bin=ffprobe_bin,
            thumbnail_generator=FFmpegThumbnailGenerator(ffmpeg_bin=config.ffmpeg_bin),
        ),
        "source-packet-remux",
    )


__all__ = ["ClipRecorderServices", "RecorderCoordinator", "default_services"]
