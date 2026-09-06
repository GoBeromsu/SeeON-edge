"""Bounded native-resolution RTSP frame capture."""

from __future__ import annotations

import subprocess
from collections.abc import Callable


class NativeFrameUnavailable(RuntimeError):
    """A native camera frame could not be captured."""


def grab_native_jpeg(
    rtsp_url: str,
    *,
    timeout_s: float = 8.0,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> bytes:
    """Return the first decoded camera frame as a JPEG without exposing its URI."""
    command = (
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "2",
        "-",
    )
    try:
        completed = run(command, check=False, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise NativeFrameUnavailable("camera stream frame capture timed out") from exc
    except FileNotFoundError as exc:
        raise NativeFrameUnavailable("camera stream frame capture requires ffmpeg") from exc
    except OSError as exc:
        raise NativeFrameUnavailable("camera stream frame capture could not start") from exc
    if completed.returncode != 0:
        raise NativeFrameUnavailable("camera stream frame capture failed")
    if not completed.stdout:
        raise NativeFrameUnavailable("camera stream frame capture produced no frame")
    return completed.stdout
