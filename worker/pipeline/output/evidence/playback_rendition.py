"""Browser-compatible, view-only renditions of immutable evidence clips."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Final

from worker.pipeline.output.evidence.durability import fsync_directory

LOGGER: Final = logging.getLogger(__name__)
PLAYBACK_NAME: Final = "clip.playback-h264.mp4"
PLAYBACK_DIGEST_NAME: Final = "clip.playback-h264.mp4.sha256"
_PLAYBACK_EXECUTOR: Final = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="evidence-playback-rendition",
)


class PlaybackRenditionError(RuntimeError):
    """A browser playback rendition could not be created."""


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
) -> Path | None:
    """Write an atomically-published H.264 rendition unless input is already H.264."""
    codec = probe_video_codec(clip_path, run)
    if codec in {"avc1", "h264"}:
        return None

    rendition = clip_path.with_name(PLAYBACK_NAME)
    sidecar = clip_path.with_name(PLAYBACK_DIGEST_NAME)
    temporary = _temporary_path(rendition)
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
        _fsync_file(temporary)
        os.replace(temporary, rendition)
        fsync_directory(rendition.parent)
        _write_digest_sidecar(sidecar, _sha256(rendition))
    except PlaybackRenditionError:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise PlaybackRenditionError("could not publish playback rendition") from exc
    return rendition


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
    temporary = _temporary_path(sidecar.with_suffix(".mp4"))
    try:
        with temporary.open("x", encoding="ascii") as output:
            _ = output.write(f"{digest}\n")
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
    "PlaybackRenditionError",
    "probe_video_codec",
    "schedule_playback_rendition",
    "write_playback_rendition",
]
