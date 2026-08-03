from __future__ import annotations

import logging
import math
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, final

from worker.adapters.encode.adapter_errors import EncoderStartError, EncoderWriteError
from worker.adapters.encode.models import EncodePolicy, EncoderGeometry, SegmentEncoderConfig

LOGGER = logging.getLogger(__name__)

_WAIT_TIMEOUT_SECONDS: Final = 15.0
_TERMINATE_TIMEOUT_SECONDS: Final = 5.0

# FramePacket images are RGB (every decode adapter publishes rgb24). Declaring
# bgr24 here silently swaps red and blue in every stored clip.
INPUT_PIX_FMT: Final = "rgb24"


class EncoderProcess(Protocol):
    def write(self, payload: bytes) -> None: ...

    def reap(self) -> int | None: ...


class ProcessSpawner(Protocol):
    def __call__(self, args: tuple[str, ...]) -> EncoderProcess: ...


@dataclass(frozen=True, slots=True)
class SegmentProcessSpec:
    config: SegmentEncoderConfig
    geometry: EncoderGeometry
    encoder: EncodePolicy
    output_dir: Path

    @property
    def segment_list_path(self) -> Path:
        return self.output_dir / "segments.csv"


@final
class _PopenEncoderProcess:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process: subprocess.Popen[bytes] | None = process
        self._returncode: int | None = None

    def write(self, payload: bytes) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise EncoderWriteError("ffmpeg encoder process is closed")
        try:
            _ = process.stdin.write(payload)
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise EncoderWriteError("ffmpeg encoder rejected frame data") from exc

    def reap(self) -> int | None:
        process = self._process
        if process is None:
            return self._returncode
        self._process = None
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            _ = process.wait(timeout=_WAIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                _ = process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                _ = process.wait()
        self._returncode = process.returncode
        if self._returncode not in (0, None) and process.stderr is not None:
            # Only read here, once the process has already exited: this call
            # is on the same already-blocking reap() path as the wait() calls
            # above, so it adds no new synchronous wait to encoder session
            # open (see worker/adapters/encode/ffmpeg_segment_encoder.py's
            # #53 fallback, which never waits on this).
            with suppress(OSError, ValueError):
                stderr_bytes = process.stderr.read()
                if stderr_bytes:
                    LOGGER.warning(
                        "ffmpeg encoder process exited with code %s: %s",
                        self._returncode,
                        stderr_bytes.decode("utf-8", errors="replace").strip(),
                    )
        return self._returncode


def segment_process_args(spec: SegmentProcessSpec) -> tuple[str, ...]:
    geometry = spec.geometry
    segment_seconds = f"{spec.config.segment_seconds:g}"
    gop_frames = max(1, math.ceil(geometry.fps * spec.config.segment_seconds))
    return (
        spec.config.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        INPUT_PIX_FMT,
        "-s",
        f"{geometry.width}x{geometry.height}",
        "-r",
        f"{geometry.fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        spec.encoder,
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop_frames),
        "-force_key_frames",
        f"expr:gte(t,n_forced*{segment_seconds})",
        "-f",
        "segment",
        "-segment_time",
        segment_seconds,
        "-reset_timestamps",
        "1",
        "-segment_list",
        str(spec.segment_list_path),
        "-segment_list_type",
        "csv",
        "-segment_format",
        "mp4",
        "-segment_format_options",
        "movflags=+faststart",
        str(spec.output_dir / "seg-%05d.mp4"),
    )


def spawn_encoder_process(args: tuple[str, ...]) -> EncoderProcess:
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        raise EncoderStartError(
            f"failed to start ffmpeg encoder ({type(exc).__name__})"
        ) from exc
    return _PopenEncoderProcess(process)


__all__ = [
    "INPUT_PIX_FMT",
    "EncoderProcess",
    "ProcessSpawner",
    "SegmentProcessSpec",
    "segment_process_args",
    "spawn_encoder_process",
]
