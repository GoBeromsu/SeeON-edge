from __future__ import annotations

from dataclasses import dataclass

# Mirrors DECODE_BACKENDS (contracts/decode_diagnostics.py): the only two clip
# encoders the worker ever selects (worker/runtime/profile/registry.py:
# PROFILE_REGISTRY).
ENCODE_BACKENDS = ("h264_nvenc", "libx264")

ENCODE_FALLBACK_REASONS = (
    "nvenc_probe_failed",
    "session_open_failed",
)


@dataclass(frozen=True, slots=True)
class EncodeSelection:
    """Requested-vs-selected encode backend, mirroring `DecodeSelection`.

    Unlike decode's fail-fast preflight, a `libx264` fallback here is an
    accepted, intentionally noisy degradation (#53): NVENC and libx264 emit
    the same H.264 content, so silently trading GPU for CPU cost never
    contaminates what a clip records -- it only needs to stay loud (WARNING
    log + this selection surfaced to local diagnostics) so an operator can see
    a camera or the whole worker is running the software encoder.
    """

    requested: str
    selected: str | None
    fallback_count: int
    last_reason: str | None
    updated_at_sec: float

    def __post_init__(self) -> None:
        if self.requested not in ENCODE_BACKENDS:
            raise ValueError(f"unsupported encode backend: {self.requested}")
        if self.selected is not None and self.selected not in ENCODE_BACKENDS:
            raise ValueError(f"unsupported encode backend: {self.selected}")


__all__ = ["ENCODE_BACKENDS", "ENCODE_FALLBACK_REASONS", "EncodeSelection"]
