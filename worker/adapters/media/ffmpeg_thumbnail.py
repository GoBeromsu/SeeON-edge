"""FFmpeg-backed thumbnails for published evidence clips."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Final


class ThumbnailUnavailable(RuntimeError):
    """A thumbnail could not be generated from an otherwise publishable clip."""


class FfmpegThumbnailGenerator:
    """Generate one atomically-published JPEG thumbnail from a video clip."""

    def __init__(
        self,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout_s: float = 30.0,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self._run = run
        self._timeout_s = timeout_s
        self._ffmpeg_bin = ffmpeg_bin

    def generate(
        self,
        video_path: Path,
        thumbnail_path: Path,
        duration_s: float,
    ) -> Path:
        # Smart Record includes a 15-second lookback, so this seeks to the
        # triggering moment while still leaving half a second for short clips.
        offset_s = min(15.0, max(0.0, duration_s - 0.5))
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".thumbnail.",
            suffix=".jpg",
            dir=thumbnail_path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            try:
                result = self._run(
                    [
                        self._ffmpeg_bin,
                        "-nostdin",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        str(offset_s),
                        "-i",
                        str(video_path),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=640:-2",
                        "-q:v",
                        "3",
                        str(temporary),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise ThumbnailUnavailable("ffmpeg thumbnail generation timed out") from exc
            except (OSError, subprocess.SubprocessError) as exc:
                raise ThumbnailUnavailable("ffmpeg thumbnail generation failed") from exc
            _require_thumbnail_output(result.returncode, temporary)
            _fsync_file(temporary)
            os.replace(temporary, thumbnail_path)
            _fsync_directory(thumbnail_path.parent)
        except ThumbnailUnavailable:
            temporary.unlink(missing_ok=True)
            raise
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ThumbnailUnavailable("could not publish thumbnail") from exc
        return thumbnail_path


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__: Final = ["FfmpegThumbnailGenerator", "ThumbnailUnavailable"]


def _require_thumbnail_output(returncode: int, temporary: Path) -> None:
    if returncode != 0:
        raise ThumbnailUnavailable("ffmpeg did not create a thumbnail")
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise ThumbnailUnavailable("ffmpeg created an empty thumbnail")
