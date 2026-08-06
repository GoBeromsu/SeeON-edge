from __future__ import annotations

from dataclasses import dataclass

from worker.adapters.decode.vaapi.errors import VaapiConfigError

# Matches the target edge node (Intel Arrow Lake): the first (and, on a
# single-iGPU node, only) DRM render node exposed by the i915 kernel driver.
_DEFAULT_RENDER_DEVICE = "/dev/dri/renderD128"


@dataclass(frozen=True, slots=True)
class VaapiConfig:
    camera_id: str
    url: str
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    ffmpeg_bin: str = "ffmpeg"
    render_device: str = _DEFAULT_RENDER_DEVICE

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise VaapiConfigError("camera id must not be blank")
        if not self.url.strip():
            raise VaapiConfigError("RTSP URL must not be blank")
        if self.open_timeout_ms <= 0:
            raise VaapiConfigError("open timeout must be positive")
        if self.read_timeout_ms <= 0:
            raise VaapiConfigError("read timeout must be positive")
        if not self.ffmpeg_bin.strip():
            raise VaapiConfigError("ffmpeg binary must not be blank")
        if not self.render_device.strip():
            raise VaapiConfigError("VAAPI render device must not be blank")


@dataclass(frozen=True, slots=True)
class StreamDimensions:
    width: int
    height: int
    codec_name: str


__all__ = ["StreamDimensions", "VaapiConfig"]
