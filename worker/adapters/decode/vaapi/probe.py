from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worker.adapters.decode.vaapi.errors import sanitized_vaapi_error
from worker.adapters.decode.vaapi.models import StreamDimensions, VaapiConfig


class ProbeRunner(Protocol):
    def __call__(self, args: tuple[str, ...], timeout_sec: float, /) -> str: ...


class _FFprobeStream(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    codec_name: str = Field(min_length=1)


class _FFprobeDocument(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    streams: tuple[_FFprobeStream, ...] = Field(min_length=1)


def ffprobe_binary(ffmpeg_bin: str) -> str:
    if ffmpeg_bin.endswith("ffmpeg"):
        return f"{ffmpeg_bin[: -len('ffmpeg')]}ffprobe"
    return "ffprobe"


def ffprobe_args(config: VaapiConfig) -> tuple[str, ...]:
    return (
        ffprobe_binary(config.ffmpeg_bin),
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name",
        "-of",
        "json",
        config.url,
    )


def run_ffprobe(args: tuple[str, ...], timeout_sec: float) -> str:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
    return completed.stdout


def probe_stream_dimensions(
    config: VaapiConfig,
    *,
    runner: ProbeRunner = run_ffprobe,
) -> StreamDimensions:
    try:
        stdout = runner(ffprobe_args(config), config.open_timeout_ms / 1000.0)
    except (OSError, subprocess.SubprocessError) as error:
        raise sanitized_vaapi_error("ffprobe failed", error) from None
    try:
        stream = _FFprobeDocument.model_validate_json(stdout).streams[0]
    except ValidationError as error:
        raise sanitized_vaapi_error("ffprobe metadata unusable", error) from None
    return StreamDimensions(stream.width, stream.height, stream.codec_name)


@dataclass(frozen=True, slots=True)
class VaapiCapability:
    available: bool
    reason: str


def run_ffmpeg_query(args: tuple[str, ...], timeout_sec: float) -> str:
    """Run a bounded ffmpeg capability query, returning combined stdout+stderr."""
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
    return f"{completed.stdout}\n{completed.stderr}"


def probe_vaapi_capability(
    ffmpeg_bin: str = "ffmpeg",
    *,
    render_device: str = "/dev/dri/renderD128",
    timeout_sec: float = 3.0,
    runner: ProbeRunner = run_ffmpeg_query,
    device_exists: Callable[[str], bool] = os.path.exists,
) -> VaapiCapability:
    """Real-driver capability probe for the ``vaapi`` decode backend.

    Deliberately does **not** stop at ``ffmpeg -hwaccels`` reporting ``vaapi``.
    On the target edge node, ffmpeg's build lists ``vaapi`` (the hwaccel API is
    compiled in) even though the iHD VAAPI *driver package*
    (``intel-media-va-driver-non-free``) is not installed -- a build-capability
    check alone would report "available" on a host that will actually fail the
    moment a camera tries to open, producing zero frames with no clear signal.
    That is exactly the silent-failure shape issues #191/#194 called out.

    So this probe:

    1. Fails closed immediately, without touching ffmpeg, if ``render_device``
       (``/dev/dri/renderD128`` by default) does not exist -- the cheapest and
       clearest of the three documented failure modes (no ``/dev/dri``).
    2. Actually asks ffmpeg to initialise a VAAPI hardware device against that
       render node (``-init_hw_device vaapi=va:<device>``) against a tiny
       synthetic ``lavfi`` source, rather than only grepping ``-hwaccels``
       text. Initialising the device is what forces libva to load and probe
       the iHD/i965 driver -- a missing driver package fails exactly here,
       with a clear non-zero exit and stderr naming the failure, which
       ``-hwaccels`` output would never surface.

    Returns ``available=False`` -- never a guess -- when the render device is
    absent, the binary is missing, the probe times out, or device init exits
    non-zero, so a host with ffmpeg's VAAPI API compiled in but no iHD driver
    (this repo's documented target-hardware state before the Dockerfile change
    in this PR) fails closed instead of reporting a false positive.
    """
    if not device_exists(render_device):
        return VaapiCapability(False, f"VAAPI render device not found: {render_device}")

    init_args = (
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-init_hw_device",
        f"vaapi=va:{render_device}",
        "-f",
        "lavfi",
        "-i",
        "nullsrc=s=2x2:d=1",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    )
    try:
        _ = runner(init_args, timeout_sec)
    except FileNotFoundError:
        return VaapiCapability(False, "ffmpeg missing")
    except subprocess.TimeoutExpired:
        return VaapiCapability(False, "VAAPI device init probe timed out")
    except subprocess.CalledProcessError as error:
        return VaapiCapability(
            False,
            f"VAAPI device init failed with exit code {error.returncode} "
            "(iHD/i965 driver likely missing)",
        )
    except Exception as error:  # noqa: BLE001 - decode probe must never break startup
        return VaapiCapability(False, f"VAAPI device init probe failed: {type(error).__name__}")

    return VaapiCapability(True, f"VAAPI device init succeeded on {render_device}")


__all__ = [
    "VaapiCapability",
    "ProbeRunner",
    "ffprobe_args",
    "ffprobe_binary",
    "probe_stream_dimensions",
    "probe_vaapi_capability",
    "run_ffmpeg_query",
    "run_ffprobe",
]
