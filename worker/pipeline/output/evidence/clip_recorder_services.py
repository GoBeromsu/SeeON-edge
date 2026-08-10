from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

from worker.adapters.encode import (
    FFmpegConcatFinalizer,
    FFmpegSegmentEncoder,
    FFmpegThumbnailGenerator,
)
from worker.adapters.encode.models import SegmentEncoderConfig
from worker.pipeline.output.evidence.clip_actor import (
    ClipPublicationPort,
    RecordingCoordinator,
)
from worker.pipeline.output.evidence.clip_publication import ClipPublisher
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recording import (
    ClipRecordingCoordinator,
    ClipWindow,
)


class RecorderCoordinator(RecordingCoordinator, Protocol):
    def set_camera_fps(self, camera_id: str, fps: float) -> None: ...


@dataclass(frozen=True, slots=True)
class ClipRecorderServices:
    coordinator: RecorderCoordinator
    publisher: ClipPublicationPort
    encoder_name: str


def resolve_encoder() -> str:
    """Intentionally conservative encoder default for the services-less construction path.

    Only reached by ``ClipRecorder.start()`` (clip_recorder.py) when a
    ``ClipRecorder`` is built without an explicit ``services`` argument (e.g.
    ``ClipRecorder.from_env()``) -- production always constructs
    ``ClipRecorder`` with ``services`` built from ``boot.encode``
    (``worker/runtime/profile/boot.py``'s already-probed-and-fallback-resolved
    selection, wired in by the composition root), so this function never runs
    in production. It hardcodes ``libx264`` rather than probing for
    ``h264_nvenc`` on purpose: this path has no access to the boot-time
    profile/probe machinery, and per #53, guessing nvenc here without a real
    preflight would risk exactly the kind of un-probed, unverified encoder
    selection the boot-time preflight exists to prevent. If a caller needs
    nvenc outside the composition root, it must resolve
    ``worker.runtime.profile.boot.resolve_boot_context(...).encode`` itself
    and pass ``services`` explicitly instead of relying on this default.
    """
    return "libx264"


def default_services(
    config: ClipRecorderConfig,
    encoder_name: str,
) -> ClipRecorderServices:
    encoder = FFmpegSegmentEncoder(
        SegmentEncoderConfig(
            config.store_dir / "segments",
            segment_seconds=config.segment_seconds,
            ffmpeg_bin=config.ffmpeg_bin,
        )
    )
    coordinator = ClipRecordingCoordinator(
        encoder=encoder,
        finalizer_factory=partial(
            FFmpegConcatFinalizer,
            ffmpeg_bin=config.ffmpeg_bin,
        ),
        profile=encoder_name,
        fps=config.fps,
        window=ClipWindow(
            pre_event_seconds=config.pre_event_seconds,
            post_event_seconds=config.post_event_seconds,
        ),
    )
    ffprobe_bin = (
        f"{config.ffmpeg_bin[:-len('ffmpeg')]}ffprobe"
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
        encoder_name,
    )


__all__ = [
    "ClipRecorderServices",
    "RecorderCoordinator",
    "default_services",
    "resolve_encoder",
]
