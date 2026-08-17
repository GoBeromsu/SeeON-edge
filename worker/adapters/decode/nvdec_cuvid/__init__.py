"""Fail-loud NVDEC/CUVID decode adapter."""

from __future__ import annotations

from worker.adapters.decode.nvdec_cuvid.adapter import (
    NvdecCuvidAdapter,
    NvdecCuvidSession,
    ffmpeg_decode_args,
    ffmpeg_rtsp_decode_args,
)
from worker.adapters.decode.nvdec_cuvid.errors import (
    NvdecConfigError,
    NvdecReadError,
    NvdecUnavailableError,
    UnsupportedCodecError,
    sanitized_nvdec_error,
)
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig, StreamMetadata
from worker.adapters.decode.nvdec_cuvid.probe import (
    NvdecCapability,
    cuvid_decoder_for,
    ffprobe_args,
    ffprobe_binary,
    probe_nvdec_cuvid_capability,
    probe_stream_metadata,
)

__all__ = [
    "NvdecCapability",
    "NvdecConfigError",
    "NvdecCuvidAdapter",
    "NvdecCuvidConfig",
    "NvdecCuvidSession",
    "NvdecReadError",
    "NvdecUnavailableError",
    "StreamMetadata",
    "UnsupportedCodecError",
    "cuvid_decoder_for",
    "ffmpeg_decode_args",
    "ffmpeg_rtsp_decode_args",
    "ffprobe_args",
    "ffprobe_binary",
    "probe_nvdec_cuvid_capability",
    "probe_stream_metadata",
    "sanitized_nvdec_error",
]
