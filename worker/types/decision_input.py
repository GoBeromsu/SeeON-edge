from __future__ import annotations

from dataclasses import dataclass, field

from contracts.observation import BedRegionDebugSnapshot, FrameObservation
from worker.types.bed_pose_features import (
    EMPTY_FRAME_BED_POSE_FEATURES,
    FrameBedPoseFeatures,
)


@dataclass(frozen=True, slots=True)
class DecisionInput:
    observation: FrameObservation = field(hash=False)
    frame_width: int
    frame_height: int
    live_track_ids: tuple[int, ...]
    time_sec: float | None
    frame_index: int
    bed_region: BedRegionDebugSnapshot
    bed_pose_features: FrameBedPoseFeatures = EMPTY_FRAME_BED_POSE_FEATURES


__all__ = ["DecisionInput"]
