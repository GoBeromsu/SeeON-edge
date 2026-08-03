from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, final

from worker.adapters.encode.adapter_errors import EncoderPolicyError

EncodePolicy: TypeAlias = Literal["h264_nvenc", "libx264"]


@dataclass(frozen=True, slots=True)
class EncoderGeometry:
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise EncoderPolicyError("encoder dimensions must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise EncoderPolicyError("encoder fps must be finite and positive")


@dataclass(frozen=True, slots=True)
class SegmentEncoderConfig:
    store_dir: Path
    segment_seconds: float = 2.0
    ffmpeg_bin: str = "ffmpeg"

    def __post_init__(self) -> None:
        if not math.isfinite(self.segment_seconds) or self.segment_seconds <= 0:
            raise EncoderPolicyError("segment duration must be finite and positive")
        if not self.ffmpeg_bin.strip():
            raise EncoderPolicyError("ffmpeg binary must not be blank")


@final
class EncoderMetrics:
    __slots__: tuple[str, ...] = (
        "active_sessions",
        "encode_fallbacks",
        "failures",
        "finalized_segments",
        "process_starts",
        "recreates",
    )

    process_starts: int
    recreates: int
    failures: int
    active_sessions: int
    finalized_segments: int
    # Count of cameras that were demoted from h264_nvenc to libx264 after a
    # failed session open (#53). Distinct from `failures`, which counts
    # sessions that never started at all.
    encode_fallbacks: int

    def __init__(self) -> None:
        self.process_starts = 0
        self.recreates = 0
        self.failures = 0
        self.active_sessions = 0
        self.finalized_segments = 0
        self.encode_fallbacks = 0


@dataclass(frozen=True, slots=True)
class ClipArtifact:
    path: Path
    generation: int
    segment_count: int
    duration_s: float


__all__ = [
    "ClipArtifact",
    "EncodePolicy",
    "EncoderGeometry",
    "EncoderMetrics",
    "SegmentEncoderConfig",
]
