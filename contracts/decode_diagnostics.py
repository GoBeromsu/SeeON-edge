from __future__ import annotations

from dataclasses import dataclass

DECODE_BACKENDS = ("auto", "nvdec", "opencv", "cpu", "vaapi")

DECODE_FALLBACK_REASONS = (
    "ffprobe_unavailable",
    "unsupported_codec",
    "spawn_failed",
    "first_frame_failed",
    "vaapi_probe_failed",
)


@dataclass(frozen=True, slots=True)
class DecodeSelection:
    requested: str
    selected: str | None
    fallback_count: int
    last_reason: str | None
    updated_at_sec: float

    def __post_init__(self) -> None:
        if self.requested not in DECODE_BACKENDS:
            raise ValueError(f"unsupported decode backend: {self.requested}")
        if self.selected is not None and self.selected not in DECODE_BACKENDS:
            raise ValueError(f"unsupported decode backend: {self.selected}")


__all__ = ["DECODE_BACKENDS", "DECODE_FALLBACK_REASONS", "DecodeSelection"]
