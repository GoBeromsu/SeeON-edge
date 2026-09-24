from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from contracts.observation import BedRegionDebugSnapshot, BoundingBox
from shared.detection_policies import BED_EXIT_POLICY_V1_DEFAULT
from worker.domains.bed_exit.night_window import NightWindow

BedOccupancy = Literal["empty", "occupied", "exit", "covered", "unknown"]


class _BedExitConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BedExitConfig:
    camera_id: str
    facility_id: str
    min_containment: float = BED_EXIT_POLICY_V1_DEFAULT.min_containment
    hold_frames: int = BED_EXIT_POLICY_V1_DEFAULT.hold_frames
    # `grace_frames` is retained only for `BedExitPolicyV1` wire compatibility
    # (shared.detection_policies is a wire vocabulary; changing its field
    # names/types has backend blast radius). It no longer gates bed-exit
    # emission timing -- `outside_dwell_sec` below does, measured in wall/PTS
    # seconds so behavior is invariant to ingest fps. Domain-only, analogous
    # to `night_window`.
    grace_frames: int = BED_EXIT_POLICY_V1_DEFAULT.grace_frames
    night_window: NightWindow | None = None
    # Continuous, posture-confirmed, spatially-contained dwell in a resident's
    # own bed required before an exit can be armed (positive evidence of
    # having been in bed at all).
    in_bed_dwell_sec: float = 3.0
    # Continuous spatial non-containment required, once armed, before a
    # bed-exit event fires. Absence (track loss) never satisfies this --
    # only live, positively-observed frames advance it.
    outside_dwell_sec: float = 2.0

    def __post_init__(self) -> None:
        if self.min_containment <= 0.0 or self.min_containment > 1.0:
            raise _BedExitConfigurationError("min_containment must be in (0, 1]")
        if self.hold_frames < 1:
            raise _BedExitConfigurationError("hold_frames must be >= 1")
        if self.grace_frames < 0:
            raise _BedExitConfigurationError("grace_frames must be >= 0")
        if self.in_bed_dwell_sec <= 0.0:
            raise _BedExitConfigurationError("in_bed_dwell_sec must be > 0")
        if self.outside_dwell_sec <= 0.0:
            raise _BedExitConfigurationError("outside_dwell_sec must be > 0")


@dataclass(frozen=True, slots=True)
class BedStatus:
    bed_id: int
    box: BoundingBox
    occupancy: BedOccupancy
    person_id: int | None = None


@dataclass(frozen=True, slots=True)
class BedExitEvent:
    person_id: int
    bed_id: int


@dataclass(frozen=True, slots=True)
class BedExitFrame:
    statuses: tuple[BedStatus, ...]
    events: tuple[BedExitEvent, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BedExitDebugSnapshot:
    frame_index: int | None
    person_boxes: tuple[BoundingBox, ...]
    bed_boxes: tuple[BoundingBox, ...]
    statuses: tuple[BedStatus, ...]
    events: tuple[BedExitEvent, ...] = field(default_factory=tuple)
    bed_region: BedRegionDebugSnapshot | None = None
    stale: bool = False
    observation_age_sec: float | None = None


__all__ = [
    "BedExitConfig",
    "BedExitDebugSnapshot",
    "BedExitEvent",
    "BedExitFrame",
    "BedOccupancy",
    "BedStatus",
]
