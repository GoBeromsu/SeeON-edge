"""Browser-compatible, view-only renditions of immutable evidence clips."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import av

LOGGER: Final = logging.getLogger(__name__)
PLAYBACK_MANIFEST_NAME: Final = "clip.playback-h264.json"
PLAYBACK_RENDITION_PREFIX: Final = "clip.playback-h264."
_PLAYBACK_EXECUTOR: Final = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="evidence-playback-rendition",
)


class PlaybackRenditionError(RuntimeError):
    """A browser playback rendition could not be created."""


@dataclass(frozen=True, slots=True)
class VideoTiming:
    """Decoded video timing needed to bind a rendition to its source."""

    time_base_numerator: int
    time_base_denominator: int
    pts: tuple[int, ...]


def probe_video_codec(
    path: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Return the first video stream codec reported by ffprobe."""
    try:
        result = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PlaybackRenditionError("ffprobe failed") from exc
    codec = result.stdout.strip().lower()
    if result.returncode != 0 or not codec or "\n" in codec:
        raise PlaybackRenditionError("ffprobe did not report one video codec")
    return codec


def read_video_timing(path: Path) -> VideoTiming:
    """Decode one video stream single-threaded and return its complete PTS sequence."""
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = _require_time_base(stream.time_base)
            stream.thread_type = "NONE"
            stream.thread_count = 1
            pts = [_require_pts(frame.pts) for frame in container.decode(stream)]
    except PlaybackRenditionError:
        raise
    except Exception as exc:
        raise PlaybackRenditionError("could not decode video timing") from exc
    return VideoTiming(time_base.numerator, time_base.denominator, tuple(pts))


def schedule_playback_rendition(clip_path: Path, clip_id: str) -> Future[Path | None]:
    """Queue view-only transcode work outside evidence publication and relay delivery."""
    future = _PLAYBACK_EXECUTOR.submit(write_playback_rendition, clip_path)
    future.add_done_callback(lambda completed: _log_failure(completed, clip_id))
    return future


def _log_failure(future: Future[Path | None], clip_id: str) -> None:
    exception = future.exception()
    if exception is not None:
        LOGGER.warning(
            "playback rendition failed stage=transcode clip_id=%s exception_class=%s",
            clip_id,
            type(exception).__name__,
        )


def _require_time_base(time_base: Fraction | None) -> Fraction:
    if time_base is None or time_base.numerator <= 0 or time_base.denominator <= 0:
        raise PlaybackRenditionError("video stream has no valid time base")
    return time_base


def _require_pts(pts: int | None) -> int:
    if pts is None:
        raise PlaybackRenditionError("decoded frame has no PTS")
    return pts


def write_playback_rendition(
    clip_path: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_s: float = 120.0,
    read_timing: Callable[[Path], VideoTiming] | None = None,
) -> Path | None:
    """Publish a digest-addressed rendition bundle beside a sealed clip."""
    from worker.pipeline.output.evidence.playback_rendition_publish import (
        write_playback_rendition as publish,
    )

    return publish(clip_path, run=run, timeout_s=timeout_s, read_timing=read_timing)


__all__ = [
    "PLAYBACK_MANIFEST_NAME",
    "PLAYBACK_RENDITION_PREFIX",
    "PlaybackRenditionError",
    "VideoTiming",
    "probe_video_codec",
    "read_video_timing",
    "schedule_playback_rendition",
    "write_playback_rendition",
]
