from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worker.adapters.decode.nvdec_cuvid.errors import (
    UnsupportedCodecError,
    sanitized_nvdec_error,
)
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig, StreamMetadata

# Ported verbatim from edge/runtime/profile/registry.py:118 (``_nvdec_capability``,
# pre-migration reference, read-only). ffmpeg reports "cuda", "cuvid", or
# "nvdec" in ``-hwaccels`` output depending on build; any one is a positive
# signal.
_NVDEC_HWACCEL_MARKERS: Final[tuple[str, ...]] = ("cuda", "cuvid", "nvdec")

# Ported verbatim from edge/runtime/profile/registry.py:142 -- matches decoder
# tokens like ``h264_cuvid``/``hevc_cuvid`` in ``-decoders`` output without
# false-matching unrelated substrings.
_CUVID_DECODER_PATTERN: Final = re.compile(r"\b[\w-]+_cuvid\b", re.IGNORECASE)

_CUVID_DECODER_BY_CODEC: Final[Mapping[str, str]] = MappingProxyType(
    {
        "hevc": "hevc_cuvid",
        "h265": "hevc_cuvid",
        "h264": "h264_cuvid",
        "avc": "h264_cuvid",
        "av1": "av1_cuvid",
        "vp9": "vp9_cuvid",
        "vp8": "vp8_cuvid",
        "mjpeg": "mjpeg_cuvid",
    }
)


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


def ffprobe_args(config: NvdecCuvidConfig) -> tuple[str, ...]:
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


def probe_stream_metadata(
    config: NvdecCuvidConfig,
    *,
    runner: ProbeRunner = run_ffprobe,
) -> StreamMetadata:
    try:
        stdout = runner(ffprobe_args(config), config.open_timeout_ms / 1000.0)
    except (OSError, subprocess.SubprocessError) as error:
        raise sanitized_nvdec_error("ffprobe failed", error) from None
    try:
        stream = _FFprobeDocument.model_validate_json(stdout).streams[0]
    except ValidationError as error:
        raise sanitized_nvdec_error("ffprobe metadata unusable", error) from None
    return StreamMetadata(stream.width, stream.height, stream.codec_name)


def cuvid_decoder_for(codec_name: str) -> str:
    normalized = codec_name.strip().lower()
    try:
        return _CUVID_DECODER_BY_CODEC[normalized]
    except KeyError:
        raise UnsupportedCodecError(normalized) from None


@dataclass(frozen=True, slots=True)
class NvdecCapability:
    available: bool
    reason: str


def run_ffmpeg_query(args: tuple[str, ...], timeout_sec: float) -> str:
    """Run a bounded ffmpeg capability query, returning combined stdout+stderr.

    Ported verbatim from edge/runtime/profile/registry.py:83-92
    (``_run_ffmpeg``, pre-migration reference, read-only): ffmpeg reports
    hwaccel/decoder capability lists on stdout in modern builds but on
    stderr in some older builds, so both streams are combined rather than
    relying on just one.
    """
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
    return f"{completed.stdout}\n{completed.stderr}"


def probe_nvdec_cuvid_capability(
    ffmpeg_bin: str = "ffmpeg",
    *,
    timeout_sec: float = 3.0,
    runner: ProbeRunner = run_ffmpeg_query,
) -> NvdecCapability:
    """Real ffmpeg-build capability probe for the ``nvdec`` decode backend.

    This is a faithful port of ``_nvdec_capability`` in
    ``edge/runtime/profile/registry.py:95-149`` (pre-migration reference,
    read-only -- ``edge/`` is scheduled for deletion once migration
    completes), including its timeout handling, exit-code branch, and
    per-exception-type messages. ``timeout_sec`` defaults to ``3.0`` seconds
    to match the reference's hardcoded ``timeout=3``.

    ``NvdecCuvidAdapter`` (``worker/adapters/decode/nvdec_cuvid/adapter.py``)
    spawns ``ffmpeg -hwaccel cuda -c:v <codec>_cuvid ...`` as a subprocess
    (``ffmpeg_decode_args``). This checks, against the actual ``ffmpeg``
    binary resolvable on ``ffmpeg_bin``, that the pieces that call depends on
    were really compiled in:

    1. ``ffmpeg -hide_banner -hwaccels`` mentions ``cuda``, ``cuvid``, or
       ``nvdec`` -- any one is a positive signal that the ``-hwaccel cuda``
       flag is not a no-op or hard failure. Checked first and short-circuits
       to ``True`` on a match, exactly as the reference does.
    2. Only if step 1 finds nothing: ``ffmpeg -hide_banner -decoders`` lists
       at least one ``*_cuvid`` decoder token (matched with a word-boundary
       regex, not a bare substring, to avoid matching unrelated text) -- the
       specific codec (``h264_cuvid``, ``hevc_cuvid``, ...) is only known once
       a stream's codec is probed per-camera (:func:`cuvid_decoder_for`), so
       this is a build-capability floor, not a per-stream guarantee.

    Neither check touches a camera, a GPU device, or an RTSP URL: this is
    purely "does this ffmpeg build support NVDEC/CUVID at all". GPU device
    presence is verified separately by the ``profile_device`` bootstrap stage
    (``worker/runtime/profile/device.py``: ``probe_cuda``), which always runs
    before ``decode_capability`` in the bootstrap sequence
    (``worker/runtime/bootstrap.py``) -- so, unlike the reference (which
    re-checks CUDA device availability inline before its ffmpeg query), this
    port does not duplicate that device check here; it would be dead code by
    construction, since bootstrap already aborts before reaching this stage
    if the device check failed.

    Returns ``available=False`` -- never a guess -- when the binary is
    missing, times out, exits non-zero, or neither marker is present, so an
    environment without an NVIDIA/CUVID-enabled ffmpeg build (e.g. this
    repo's macOS dev machines, confirmed via real ``ffmpeg -hwaccels``/
    ``-decoders`` output showing neither) fails closed instead of reporting a
    false positive.
    """
    try:
        hwaccels = runner((ffmpeg_bin, "-hide_banner", "-hwaccels"), timeout_sec)
    except FileNotFoundError:
        return NvdecCapability(False, "ffmpeg missing")
    except subprocess.TimeoutExpired:
        return NvdecCapability(False, "ffmpeg capability probe timed out")
    except subprocess.CalledProcessError as error:
        return NvdecCapability(
            False, f"ffmpeg capability probe failed with exit code {error.returncode}"
        )
    except Exception as error:  # noqa: BLE001 - decode probe must never break startup
        return NvdecCapability(False, f"ffmpeg capability probe failed: {type(error).__name__}")

    if any(marker in hwaccels.lower() for marker in _NVDEC_HWACCEL_MARKERS):
        return NvdecCapability(True, "ffmpeg CUDA/CUVID hwaccel is available")

    try:
        decoders = runner((ffmpeg_bin, "-hide_banner", "-decoders"), timeout_sec)
    except FileNotFoundError:
        return NvdecCapability(False, "ffmpeg missing")
    except subprocess.TimeoutExpired:
        return NvdecCapability(False, "ffmpeg decoder probe timed out")
    except subprocess.CalledProcessError as error:
        return NvdecCapability(
            False, f"ffmpeg decoder probe failed with exit code {error.returncode}"
        )
    except Exception as error:  # noqa: BLE001 - decode probe must never break startup
        return NvdecCapability(False, f"ffmpeg decoder probe failed: {type(error).__name__}")

    if _CUVID_DECODER_PATTERN.search(decoders):
        return NvdecCapability(True, "ffmpeg CUVID decoder is available")
    return NvdecCapability(False, "ffmpeg has no CUDA/CUVID/NVDEC hwaccel or *_cuvid decoder")


__all__ = [
    "NvdecCapability",
    "ProbeRunner",
    "cuvid_decoder_for",
    "ffprobe_args",
    "ffprobe_binary",
    "probe_nvdec_cuvid_capability",
    "probe_stream_metadata",
    "run_ffmpeg_query",
    "run_ffprobe",
]
