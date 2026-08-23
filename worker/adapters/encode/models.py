from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
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
    allow_runtime_encode_fallback: bool = False
    """Permit demoting a failed ``h264_nvenc`` session open to CPU ``libx264``.

    Defaults to ``False`` so an NVENC failure fails loud and closed, per ADR 0002.
    ADR 0003 prohibits *implicit* fallback and permits *explicit* fallback, so a
    deployment that consciously accepts CPU encode cost opts in here rather than
    inheriting the demotion automatically.
    """

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
class RemuxStreamFact:
    index: int
    media_type: str
    codec_name: str
    codec_tag: str
    time_base: Fraction
    extradata_sha256: str
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    packet_count: int = 0
    timestamp_translation_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class ClipArtifact:
    path: Path
    generation: int
    segment_count: int
    duration_s: float
    worker_boot_id: str = ""
    camera_id: str = ""
    stream_epoch: int = 0
    media_origin_pts_sec: float | None = None
    selected_start_pts_sec: float | None = None
    selected_end_pts_sec: float | None = None
    packet_count: int | None = None
    configuration_id: str | None = None
    streams: tuple[RemuxStreamFact, ...] = ()
    remux_method: str | None = None
    remux_version: str | None = None
    timestamp_translation_seconds: Fraction = Fraction(0)
    truncation_reasons: tuple[str, ...] = ()


__all__ = [
    "ClipArtifact",
    "EncodePolicy",
    "EncoderGeometry",
    "EncoderMetrics",
    "RemuxStreamFact",
    "SegmentEncoderConfig",
]
