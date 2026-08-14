from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Literal

MediaType = Literal["video", "audio"]


class PacketTruncationReason(StrEnum):
    HISTORY_UNAVAILABLE = "HISTORY_UNAVAILABLE"
    KEYFRAME_UNAVAILABLE = "KEYFRAME_UNAVAILABLE"
    FUTURE_UNAVAILABLE = "FUTURE_UNAVAILABLE"
    CONFIGURATION_CHANGED = "CONFIGURATION_CHANGED"
    TIMESTAMP_DISCONTINUITY = "TIMESTAMP_DISCONTINUITY"
    RING_CLOSED = "RING_CLOSED"


@dataclass(frozen=True, slots=True)
class StreamEpoch:
    worker_boot_id: str
    camera_id: str
    stream_epoch: int

    def __post_init__(self) -> None:
        if not self.worker_boot_id or not self.camera_id or self.stream_epoch <= 0:
            raise ValueError("stream epoch identity must be complete")


@dataclass(frozen=True, slots=True)
class SourceStreamDescriptor:
    index: int
    media_type: MediaType
    codec_name: str
    codec_tag: str
    time_base: Fraction
    extradata: bytes
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    profile: str | None = None
    level: int | None = None

    def __post_init__(self) -> None:
        if self.index < 0 or not self.codec_name or self.time_base <= 0:
            raise ValueError("source stream descriptor is invalid")
        if self.media_type == "video" and (
            self.width is None or self.height is None or self.width <= 0 or self.height <= 0
        ):
            raise ValueError("video stream dimensions must be positive")
        if self.media_type == "audio" and (self.sample_rate is None or self.sample_rate <= 0):
            raise ValueError("audio sample rate must be positive")


@dataclass(frozen=True, slots=True)
class SourceStreamConfiguration:
    streams: tuple[SourceStreamDescriptor, ...]
    configuration_id: str
    mux_template: bytes = b""

    @classmethod
    def from_streams(
        cls,
        streams: list[SourceStreamDescriptor] | tuple[SourceStreamDescriptor, ...],
        *,
        mux_template: bytes = b"",
    ) -> SourceStreamConfiguration:
        ordered = tuple(sorted(streams, key=lambda stream: stream.index))
        if not ordered or not any(stream.media_type == "video" for stream in ordered):
            raise ValueError("source configuration requires a video stream")
        if len({stream.index for stream in ordered}) != len(ordered):
            raise ValueError("source stream indexes must be unique")
        payload = [
            {
                "index": stream.index,
                "media_type": stream.media_type,
                "codec_name": stream.codec_name,
                "codec_tag": stream.codec_tag,
                "time_base": [stream.time_base.numerator, stream.time_base.denominator],
                "extradata_sha256": hashlib.sha256(stream.extradata).hexdigest(),
                "width": stream.width,
                "height": stream.height,
                "sample_rate": stream.sample_rate,
                "channels": stream.channels,
                "profile": stream.profile,
                "level": stream.level,
            }
            for stream in ordered
        ]
        digest = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        return cls(ordered, digest, mux_template)

    def stream(self, index: int) -> SourceStreamDescriptor:
        for stream in self.streams:
            if stream.index == index:
                return stream
        raise ValueError(f"stream index {index} is absent from source configuration")

    @property
    def video_streams(self) -> tuple[SourceStreamDescriptor, ...]:
        return tuple(stream for stream in self.streams if stream.media_type == "video")

    @property
    def audio_streams(self) -> tuple[SourceStreamDescriptor, ...]:
        return tuple(stream for stream in self.streams if stream.media_type == "audio")


@dataclass(frozen=True, slots=True)
class SourcePacket:
    epoch: StreamEpoch
    configuration: SourceStreamConfiguration
    stream_index: int
    pts: int | None
    dts: int | None
    duration: int | None
    is_keyframe: bool
    payload: bytes
    arrival_index: int
    discontinuity: str | None = None

    def __post_init__(self) -> None:
        if not self.payload:
            raise ValueError("source packet payload must not be empty")
        if self.arrival_index < 0:
            raise ValueError("packet arrival index must not be negative")
        _ = self.stream

    @property
    def stream(self) -> SourceStreamDescriptor:
        return self.configuration.stream(self.stream_index)

    @property
    def presentation_time(self) -> Fraction:
        if self.pts is None:
            raise ValueError("source packet has no PTS")
        return self.pts * self.stream.time_base

    @property
    def decode_time(self) -> Fraction:
        if self.dts is None:
            raise ValueError("source packet has no DTS")
        return self.dts * self.stream.time_base

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass(slots=True)
class PacketSelectionError(Exception):
    reason: PacketTruncationReason
    detail: str

    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}"


__all__ = [
    "MediaType",
    "PacketSelectionError",
    "PacketTruncationReason",
    "SourcePacket",
    "SourceStreamConfiguration",
    "SourceStreamDescriptor",
    "StreamEpoch",
]
