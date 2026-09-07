"""Browser-compatible, view-only renditions of immutable evidence clips."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import av

from worker.pipeline.output.evidence.durability import fsync_directory

LOGGER: Final = logging.getLogger(__name__)
PLAYBACK_NAME: Final = "clip.playback-h264.mp4"
PLAYBACK_DIGEST_NAME: Final = "clip.playback-h264.mp4.sha256"
PLAYBACK_TIMING_NAME: Final = "clip.playback-h264.timing.json"
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


def write_playback_rendition(
    clip_path: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_s: float = 120.0,
    read_timing: Callable[[Path], VideoTiming] | None = None,
) -> Path | None:
    """Write an atomically-published H.264 rendition unless input is already H.264."""
    codec = probe_video_codec(clip_path, run)
    if codec in {"avc1", "h264"}:
        return None

    timing_reader = read_video_timing if read_timing is None else read_timing
    source_timing = timing_reader(clip_path)
    source_digest = _sha256(clip_path)
    rendition = clip_path.with_name(PLAYBACK_NAME)
    sidecar = clip_path.with_name(PLAYBACK_DIGEST_NAME)
    timing_sidecar = clip_path.with_name(PLAYBACK_TIMING_NAME)
    temporary = _temporary_path(rendition)
    renamed = False
    try:
        try:
            result = run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(clip_path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-fps_mode",
                    "passthrough",
                    "-enc_time_base",
                    "-1",
                    "-video_track_timescale",
                    str(source_timing.time_base_denominator),
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PlaybackRenditionError("ffmpeg failed") from exc
        _require_rendition_output(result.returncode, temporary)
        rendition_timing = timing_reader(temporary)
        rendition_digest = _sha256(temporary)
        timing_payload = _timing_payload(
            source_timing, rendition_timing, source_digest, rendition_digest
        )
        _fsync_file(temporary)
        _write_timing_sidecar(timing_sidecar, timing_payload)
        os.replace(temporary, rendition)
        renamed = True
        fsync_directory(rendition.parent)
        _write_digest_sidecar(sidecar, rendition_digest)
    except PlaybackRenditionError:
        temporary.unlink(missing_ok=True)
        if renamed:
            rendition.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        if renamed:
            rendition.unlink(missing_ok=True)
        raise PlaybackRenditionError("could not publish playback rendition") from exc
    return rendition


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


def _timing_payload(
    source: VideoTiming, rendition: VideoTiming, source_digest: str, rendition_digest: str
) -> dict[str, bool | int | str]:
    return {
        "source_sha256": source_digest,
        "rendition_sha256": rendition_digest,
        "pts_identical": source == rendition,
        "time_base": f"{source.time_base_numerator}/{source.time_base_denominator}",
        "frames": len(rendition.pts),
        "source_frames": len(source.pts),
    }


def _require_rendition_output(returncode: int, temporary: Path) -> None:
    if returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        raise PlaybackRenditionError("ffmpeg did not create a rendition")


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


def _temporary_path(rendition: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{rendition.stem}.",
        suffix=".mp4",
        dir=rendition.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def _write_digest_sidecar(sidecar: Path, digest: str) -> None:
    _write_atomic_text(sidecar, f"{digest}\n")


def _write_timing_sidecar(sidecar: Path, payload: dict[str, bool | int | str]) -> None:
    _write_atomic_text(sidecar, json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _write_atomic_text(sidecar: Path, text: str) -> None:
    temporary = _temporary_path(sidecar.with_suffix(".mp4"))
    try:
        with temporary.open("x", encoding="ascii") as output:
            _ = output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, sidecar)
        fsync_directory(sidecar.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PLAYBACK_DIGEST_NAME",
    "PLAYBACK_NAME",
    "PLAYBACK_TIMING_NAME",
    "PlaybackRenditionError",
    "VideoTiming",
    "probe_video_codec",
    "read_video_timing",
    "schedule_playback_rendition",
    "write_playback_rendition",
]


def _require_time_base(time_base: Fraction | None) -> Fraction:
    if time_base is None or time_base.numerator <= 0 or time_base.denominator <= 0:
        raise PlaybackRenditionError("video stream has no valid time base")
    return time_base


def _require_pts(pts: int | None) -> int:
    if pts is None:
        raise PlaybackRenditionError("decoded frame has no PTS")
    return pts
