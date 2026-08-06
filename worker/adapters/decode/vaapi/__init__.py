"""VAAPI (Intel iGPU) decode adapter -- decode only, CPU inference stays unchanged."""

from __future__ import annotations

from worker.adapters.decode.vaapi.adapter import (
    VaapiAdapter,
    VaapiSession,
    ffmpeg_decode_args,
)
from worker.adapters.decode.vaapi.errors import (
    VaapiConfigError,
    VaapiReadError,
    VaapiUnavailableError,
    sanitized_vaapi_error,
)
from worker.adapters.decode.vaapi.models import StreamDimensions, VaapiConfig
from worker.adapters.decode.vaapi.probe import (
    VaapiCapability,
    ffprobe_args,
    ffprobe_binary,
    probe_stream_dimensions,
    probe_vaapi_capability,
)

__all__ = [
    "StreamDimensions",
    "VaapiAdapter",
    "VaapiCapability",
    "VaapiConfig",
    "VaapiConfigError",
    "VaapiReadError",
    "VaapiSession",
    "VaapiUnavailableError",
    "ffmpeg_decode_args",
    "ffprobe_args",
    "ffprobe_binary",
    "probe_stream_dimensions",
    "probe_vaapi_capability",
    "sanitized_vaapi_error",
]
