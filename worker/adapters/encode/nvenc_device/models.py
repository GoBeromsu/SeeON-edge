"""Declared codec/container/profile candidates for the device-input NVENC seam.

A small closed vocabulary, not a guessed SDK enum: every value here names
only what ``worker.adapters.encode.ffmpeg_segment_encoder``'s existing
``h264_nvenc``/``libx264`` policy and this repo's evidence pipeline already
produce (H.264 in an MP4 container). Extending this to HEVC/AV1 or a
fragmented-MP4/MKV container is a later, separately reviewed change -- this
prototype declares the candidate list explicitly so a caller's request and
the encoder's truthful selection are both checkable values, never free text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DeviceEncoderCodec(StrEnum):
    H264 = "h264"


class DeviceEncoderContainer(StrEnum):
    MP4 = "mp4"


class DeviceEncoderProfile(StrEnum):
    """NVENC-documented H.264 profile names (NVIDIA Video Codec SDK)."""

    BASELINE = "baseline"
    MAIN = "main"
    HIGH = "high"


DEFAULT_CODEC_CANDIDATES: tuple[DeviceEncoderCodec, ...] = (DeviceEncoderCodec.H264,)
DEFAULT_CONTAINER_CANDIDATES: tuple[DeviceEncoderContainer, ...] = (DeviceEncoderContainer.MP4,)
DEFAULT_PROFILE_CANDIDATES: tuple[DeviceEncoderProfile, ...] = (
    DeviceEncoderProfile.HIGH,
    DeviceEncoderProfile.MAIN,
    DeviceEncoderProfile.BASELINE,
)


@dataclass(frozen=True, slots=True)
class DeviceEncoderPoolConfig:
    """Bounded configuration for one camera's device-input NVENC session pool.

    Mirrors ``worker.adapters.decode.nvdec_device.models.DeviceResidentPoolConfig``:
    ``capacity`` bounds outstanding in-flight encode submissions
    (backpressure), never an unbounded queue.
    """

    camera_id: str
    capacity: int
    width: int
    height: int
    codec_candidates: tuple[DeviceEncoderCodec, ...] = DEFAULT_CODEC_CANDIDATES
    container_candidates: tuple[DeviceEncoderContainer, ...] = DEFAULT_CONTAINER_CANDIDATES
    profile_candidates: tuple[DeviceEncoderProfile, ...] = field(
        default_factory=lambda: DEFAULT_PROFILE_CANDIDATES
    )

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("device encoder pool config requires a camera id")
        if self.capacity <= 0:
            raise ValueError("device encoder pool capacity must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("device encoder pool frame dimensions must be positive")
        if not self.codec_candidates:
            raise ValueError("device encoder pool requires at least one codec candidate")
        if not self.container_candidates:
            raise ValueError("device encoder pool requires at least one container candidate")
        if not self.profile_candidates:
            raise ValueError("device encoder pool requires at least one profile candidate")


@dataclass(frozen=True, slots=True)
class DeviceEncoderSelection:
    """Truthful encoder choice: what was requested vs. what was actually opened."""

    requested_codec: DeviceEncoderCodec
    requested_container: DeviceEncoderContainer
    requested_profile: DeviceEncoderProfile
    selected_codec: DeviceEncoderCodec | None
    selected_container: DeviceEncoderContainer | None
    selected_profile: DeviceEncoderProfile | None
    device_resident: bool
    reason: str


__all__ = [
    "DEFAULT_CODEC_CANDIDATES",
    "DEFAULT_CONTAINER_CANDIDATES",
    "DEFAULT_PROFILE_CANDIDATES",
    "DeviceEncoderCodec",
    "DeviceEncoderContainer",
    "DeviceEncoderPoolConfig",
    "DeviceEncoderProfile",
    "DeviceEncoderSelection",
]
