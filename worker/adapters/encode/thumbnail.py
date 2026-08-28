"""Bounded FFmpeg thumbnail extraction and secure durable publication."""

from __future__ import annotations

import math
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, final

from worker.adapters.encode.adapter_errors import (
    ThumbnailGenerationError,
    ThumbnailPayloadError,
    ThumbnailSecurityError,
    ThumbnailTimeoutError,
)
from worker.adapters.encode.thumbnail_artifact import (
    MAX_THUMBNAIL_BYTES,
    THUMBNAIL_FILENAME,
    THUMBNAIL_HEIGHT,
    THUMBNAIL_WIDTH,
    fsync_existing_thumbnail,
    is_valid_jpeg,
    is_valid_thumbnail,
    publish_thumbnail,
)

THUMBNAIL_TIMEOUT_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True)
class ThumbnailCommandResult:
    returncode: int
    stdout: bytes


class ThumbnailRunner(Protocol):
    def __call__(
        self,
        args: tuple[str, ...],
        timeout_s: float,
    ) -> ThumbnailCommandResult: ...


def run_ffmpeg_thumbnail(
    args: tuple[str, ...],
    timeout_s: float,
) -> ThumbnailCommandResult:
    read_descriptor: int | None = None
    write_descriptor: int | None = None
    try:
        read_descriptor, write_descriptor = os.pipe()
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=write_descriptor,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
    except OSError as exc:
        if read_descriptor is not None:
            os.close(read_descriptor)
        if write_descriptor is not None:
            os.close(write_descriptor)
        raise ThumbnailGenerationError(
            f"failed to start ffmpeg thumbnail extraction ({type(exc).__name__})"
        ) from exc
    os.close(write_descriptor)
    output: bytes | None = None
    read_error: OSError | None = None

    def read_bounded_output() -> None:
        nonlocal output, read_error
        try:
            with os.fdopen(read_descriptor, "rb", closefd=True) as stream:
                output = stream.read(MAX_THUMBNAIL_BYTES + 1)
        except OSError as exc:
            read_error = exc

    reader = threading.Thread(target=read_bounded_output, name="thumbnail-stdout")
    reader.start()
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        _ = process.wait()
        raise ThumbnailTimeoutError(timeout_s) from exc
    finally:
        reader.join()
    if read_error is not None:
        raise ThumbnailGenerationError(
            f"failed to read ffmpeg thumbnail output ({type(read_error).__name__})"
        ) from read_error
    if output is None:
        raise ThumbnailGenerationError("ffmpeg thumbnail output reader failed")
    if len(output) > MAX_THUMBNAIL_BYTES:
        raise ThumbnailPayloadError(len(output))
    return ThumbnailCommandResult(returncode, output)


@final
class FFmpegThumbnailGenerator:
    def __init__(
        self,
        *,
        ffmpeg_bin: str = "ffmpeg",
        runner: ThumbnailRunner = run_ffmpeg_thumbnail,
        timeout_s: float = THUMBNAIL_TIMEOUT_SECONDS,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ThumbnailGenerationError("thumbnail timeout must be finite and positive")
        self._ffmpeg_bin = ffmpeg_bin
        self._runner = runner
        self._timeout_s = timeout_s

    def generate(
        self,
        video_path: Path,
        thumbnail_path: Path,
        duration_s: float,
    ) -> Path:
        if is_valid_thumbnail(thumbnail_path):
            fsync_existing_thumbnail(thumbnail_path)
            return thumbnail_path
        try:
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ThumbnailSecurityError("directory create", type(exc).__name__) from exc
        midpoint_s = max(0.0, duration_s) / 2.0
        args = (
            self._ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{midpoint_s:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            (
                "scale=640:360:force_original_aspect_ratio=decrease,"
                "pad=640:360:(ow-iw)/2:(oh-ih)/2"
            ),
            "-q:v",
            "2",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        )
        result = self._runner(args, self._timeout_s)
        if result.returncode != 0:
            raise ThumbnailGenerationError(
                "ffmpeg thumbnail extraction failed",
                returncode=result.returncode,
            )
        if not is_valid_jpeg(result.stdout):
            raise ThumbnailPayloadError(len(result.stdout))
        publish_thumbnail(result.stdout, thumbnail_path)
        return thumbnail_path


__all__ = [
    "MAX_THUMBNAIL_BYTES",
    "THUMBNAIL_FILENAME",
    "THUMBNAIL_HEIGHT",
    "THUMBNAIL_TIMEOUT_SECONDS",
    "THUMBNAIL_WIDTH",
    "FFmpegThumbnailGenerator",
    "ThumbnailCommandResult",
    "ThumbnailGenerationError",
    "ThumbnailPayloadError",
    "ThumbnailRunner",
    "ThumbnailSecurityError",
    "ThumbnailTimeoutError",
    "is_valid_thumbnail",
    "run_ffmpeg_thumbnail",
]
