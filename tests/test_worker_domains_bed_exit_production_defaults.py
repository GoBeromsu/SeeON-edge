"""Pin shipped bed-exit defaults and the containment geometries they fail on.

Every other bed-exit test overrides ``min_containment`` / ``hold_frames`` /
``grace_frames``, so the production values from
``BED_EXIT_POLICY_V1_DEFAULT`` (0.35 / 2 / 3) never ran through the state
machine in CI. These tests are the BEFORE side of the redesign: they must
keep failing the same way until the containment rule is replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains import bed_exit
from worker.domains.bed_exit.geometry import containment_ratio
from worker.types import DecisionInput

CAMERA_ID: Final = "camera-bed-exit-defaults"
FACILITY_ID: Final = "facility-bed-exit-defaults"
PERSON_ID: Final = 7

# Worked AABB from the redesign plan. No polygon: this is the axis-aligned
# fallback in geometry.py:_aabb_containment_ratio.
BED: Final = BoundingBox(200, 300, 700, 560, 0.99)
LYING: Final = BoundingBox(240, 320, 660, 520, 0.95)
SITTING_UP: Final = BoundingBox(380, 150, 620, 540, 0.95)
EDGE_SITTING: Final = BoundingBox(400, 180, 640, 700, 0.95)
STANDING_BESIDE: Final = BoundingBox(720, 120, 860, 620, 0.95)
HEAD_ONLY_OCCLUDED: Final = BoundingBox(420, 300, 500, 370, 0.95)
CAREGIVER_LEANING: Final = BoundingBox(300, 120, 560, 600, 0.95)

# Exact person∩bed / person-area from the real formula. The plan rounded
# sitting-up to 0.615 and caregiver-leaning to 0.542; those miss 1e-6.
CONTAINMENT_BY_POSE: Final = (
    ("lying", LYING, 1.0),
    ("sitting-up", SITTING_UP, 57_600 / 93_600),
    ("edge-sitting", EDGE_SITTING, 0.5),
    ("standing-beside", STANDING_BESIDE, 0.0),
    ("head-only-occluded", HEAD_ONLY_OCCLUDED, 1.0),
    ("caregiver-leaning", CAREGIVER_LEANING, 67_600 / 124_800),
)


def _clock_at(hour: int = 22, minute: int = 0) -> Callable[[], datetime]:
    fixed = datetime(2026, 7, 31, hour, minute, tzinfo=ZoneInfo("Asia/Seoul"))
    return lambda: fixed


def _monitor() -> bed_exit.BedExitMonitor:
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
        ),
        clock=_clock_at(),
        boot_id="boot-bed-exit-defaults",
        stream_epoch="epoch-bed-exit-defaults",
        source_generation=0,
    )


def _input(person: BoundingBox, frame_index: int) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            regions=((BED,), ()),
            track_ids=(PERSON_ID,),
        ),
        frame_width=1000,
        frame_height=800,
        live_track_ids=(PERSON_ID,),
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def test_bed_exit_production_defaults_match_shipped_policy() -> None:
    # Given / When
    config = bed_exit.BedExitConfig(camera_id=CAMERA_ID, facility_id=FACILITY_ID)

    # Then
    assert config.min_containment == 0.35
    assert config.hold_frames == 2
    assert config.grace_frames == 3
    monitor = _monitor()
    assert monitor.config.min_containment == 0.35
    assert monitor.config.hold_frames == 2
    assert monitor.config.grace_frames == 3


def test_bed_exit_production_defaults_containment_failure_geometry() -> None:
    # Given / When / Then
    for name, person, expected in CONTAINMENT_BY_POSE:
        ratio = containment_ratio(person, BED)
        assert abs(ratio - expected) < 1e-6, name


def test_bed_exit_production_defaults_edge_sitting_emits_zero_events_failure_mode_f1() -> None:
    """Failure mode F1: edge-sitting never leaves the in-bed band.

    ``own_ratio`` for EDGE_SITTING is 0.5, which is still ``>= min_containment``
    (0.35), so every frame resets ``grace_frames`` and the live path never
    triggers. This is today's shipped behaviour, not a desired one.
    """
    # Given
    monitor = _monitor()

    # When: two lying frames satisfy hold_frames=2 and assign the bed
    assigned = tuple(monitor.update(_input(LYING, frame_index)) for frame_index in (0, 1))
    # Then
    assert assigned == ((), ())
    assert monitor.last_debug_snapshot is not None
    assert tuple(status.occupancy for status in monitor.last_debug_snapshot.statuses) == (
        "occupied",
    )

    # When: ten edge-sitting frames — more than grace_frames=3
    outputs = tuple(
        monitor.update(_input(EDGE_SITTING, frame_index)) for frame_index in range(2, 12)
    )

    # Then: zero events; still occupied because containment stays above 0.35
    assert outputs == ((), (), (), (), (), (), (), (), (), ())
    assert monitor.last_debug_snapshot is not None
    assert tuple(status.occupancy for status in monitor.last_debug_snapshot.statuses) == (
        "occupied",
    )
    assert monitor.last_debug_snapshot.events == ()
