"""Explicit owner of ingest fps and per-domain decision rates.

Pipeline defaults and the composition-root schedule must compute from this
object. The current production identity is ``ingest_fps=15.0``: target fps
15.0, pose every ingested frame, bed every 90 frames (1/6 Hz).

Design B (todo 13): TemporalProfile is authoritative for ingest fps.
A relay-declared ``CameraRuntimeConfig.fps`` is a recorded hint and is
never the CapturePolicy owner. Rejected design A (relay as per-camera
override) because raising this profile for a 15fps measurement would
then leave relay-configured cameras at their declared rate and the
capacity run would measure nothing. Pose extractor cadence remains
``camera.frame_stride`` until a later todo re-denominates it from
``pose_fps``; that split is explicit, not a silent fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Final


class TemporalProfileError(ValueError):
    """Closed config error for a malformed temporal profile."""


# Identity of today's shipped cadence. Raising ingest_fps re-denominates the
# bed interval in the same edit so the bed decision rate stays 1/6 Hz: the
# frame count is the derived value, the Hz is the invariant. 90 frames at
# 15fps is the same 6 seconds of wall clock as 30 frames at 5fps, so bed-exit
# decision cadence is unchanged by this raise.
_CURRENT_INGEST_FPS: Final = 15.0
_CURRENT_BED_INTERVAL_FRAMES: Final = 90
_CURRENT_BED_DECISION_HZ: Final = _CURRENT_INGEST_FPS / _CURRENT_BED_INTERVAL_FRAMES


def _require_positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalProfileError(f"{name} must be a finite number > 0")
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise TemporalProfileError(f"{name} must be a finite number > 0")
    return numeric


def _default_decision_hz() -> dict[str, float]:
    return {"bed": _CURRENT_BED_DECISION_HZ}


@dataclass(frozen=True, slots=True)
class TemporalProfile:
    """Owns ingest fps, pose fps, and per-domain decision Hz."""

    ingest_fps: float
    pose_fps: float | None = None
    decision_hz: Mapping[str, float] = field(default_factory=_default_decision_hz)

    def __post_init__(self) -> None:
        ingest_fps = _require_positive_finite("ingest_fps", self.ingest_fps)
        object.__setattr__(self, "ingest_fps", ingest_fps)
        if self.pose_fps is None:
            object.__setattr__(self, "pose_fps", ingest_fps)
        else:
            object.__setattr__(
                self,
                "pose_fps",
                _require_positive_finite("pose_fps", self.pose_fps),
            )
        normalized: dict[str, float] = {}
        for name, hz in self.decision_hz.items():
            if not name:
                raise TemporalProfileError("decision_hz keys must be non-empty")
            normalized[name] = _require_positive_finite(f"decision_hz[{name!r}]", hz)
        object.__setattr__(self, "decision_hz", MappingProxyType(normalized))

    @property
    def target_fps(self) -> float:
        return self.ingest_fps

    @property
    def frame_interval_sec(self) -> float:
        return 1.0 / self.ingest_fps

    def pose_interval_frames(self, *, frame_stride: int = 1) -> int:
        if type(frame_stride) is not int or frame_stride <= 0:
            raise TemporalProfileError("frame_stride must be an integer > 0")
        pose_fps = self.ingest_fps if self.pose_fps is None else self.pose_fps
        return max(1, round(self.ingest_fps / pose_fps)) * frame_stride

    def decision_interval_frames(self, domain: str) -> int:
        try:
            hz = self.decision_hz[domain]
        except KeyError as exc:
            raise TemporalProfileError(f"no decision_hz declared for {domain!r}") from exc
        return max(1, round(self.ingest_fps / hz))

    def task_intervals(self, *, frame_stride: int = 1) -> dict[str, int]:
        intervals = {"pose": self.pose_interval_frames(frame_stride=frame_stride)}
        for domain in self.decision_hz:
            intervals[domain] = self.decision_interval_frames(domain)
        return intervals

    def frames_for_seconds(self, seconds: float) -> int:
        """Convert a dwell/hysteresis duration into whole frames at ingest fps.

        Seconds are the durable unit for bed-exit dwell policy; the frame count
        is derived, so a future ingest-fps change re-denominates dwell windows
        automatically instead of silently changing how long they last.
        """
        numeric = _require_positive_finite("seconds", seconds)
        return max(1, round(numeric * self.ingest_fps))


CURRENT_TEMPORAL_PROFILE: Final = TemporalProfile(ingest_fps=_CURRENT_INGEST_FPS)


__all__ = [
    "CURRENT_TEMPORAL_PROFILE",
    "TemporalProfile",
    "TemporalProfileError",
]
