"""Regression test for #219, wired through `BedExitMonitor` (the real code
path), not `containment_ratio` in isolation.

Prior art on why this matters: #204 shipped broken because a fix was
verified at a layer that didn't reach the object actually used at decision
time (see #217). This test asserts on `BedExitMonitor.last_debug_snapshot`
/ emitted events -- what `detector.py` actually uses -- not on the geometry
helper directly.
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains import bed_exit
from worker.pipeline.perception.scene_state import SceneState
from worker.types import DecisionInput

PERSON_ID: Final = 1

# A square rotated 45 degrees, centered at (50, 50), inscribed in the AABB
# [0, 100] x [0, 100] -- diamond area 5,000 vs AABB area 10,000 (2.0x
# inflation), matching the #219 measured range (1.57x-2.07x, mean 1.94x).
DIAMOND_BED: Final = BoundingBox(
    x1=0,
    y1=0,
    x2=100,
    y2=100,
    confidence=1.0,
    polygon=((50, 0), (100, 50), (50, 100), (0, 50)),
)

# Fully inside the diamond: all 4 corners satisfy |x-50| + |y-50| <= 50.
IN_BED: Final = BoundingBox(40, 40, 60, 60, 0.95)

# The #219 regression case: fully outside the diamond (every point has
# x+y >= 160, past the diamond's x+y<=150 edge in this quadrant) but fully
# inside the bed's AABB. Pre-fix, `containment_ratio` measured this against
# x1..y2 alone and returned 1.0 (full containment) -- this box is entirely
# within [0,100]x[0,100] -- so `own_ratio >= min_containment` would hold
# forever and grace_frames would never advance; the bed-exit event that
# this test expects would never fire.
BESIDE_BED_IN_AABB: Final = BoundingBox(80, 80, 95, 95, 0.9)


def _monitor(*, min_containment: float = 0.5, grace_frames: int = 1) -> bed_exit.BedExitMonitor:
    fixed = datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id="camera-polygon",
            facility_id="facility-polygon",
            min_containment=min_containment,
            hold_frames=1,
            grace_frames=grace_frames,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=lambda: fixed,
        boot_id="test-boot",
        stream_epoch="test-epoch",
        source_generation=0,
    )


def _input(person: BoundingBox, frame_index: int) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            regions=((DIAMOND_BED,), ()),
            track_ids=(PERSON_ID,),
        ),
        frame_width=100,
        frame_height=100,
        live_track_ids=(PERSON_ID,),
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def test_person_beside_polygon_bed_but_inside_aabb_eventually_exits() -> None:
    # Given: the person is assigned to the bed while genuinely inside the
    # diamond-shaped polygon.
    monitor = _monitor(grace_frames=1)
    assert monitor.update(_input(IN_BED, 0)) == ()
    assert monitor.last_debug_snapshot is not None
    assert monitor.last_debug_snapshot.statuses[0].occupancy == "occupied"

    # When: the person moves to a spot that is outside the real polygon but
    # still inside the bed's AABB, and stays there past the grace period.
    frame_1 = monitor.update(_input(BESIDE_BED_IN_AABB, 1))
    frame_2 = monitor.update(_input(BESIDE_BED_IN_AABB, 2))

    # Then: grace_frames must actually advance (own_ratio must read as 0,
    # not 1.0) -- no event yet after the first off-polygon frame, but the
    # bed-exit event fires once grace_frames exceeds the configured
    # threshold. Pre-fix, this would never happen: own_ratio would read 1.0
    # against the AABB and grace_frames would reset to 0 every frame.
    assert frame_1 == ()
    assert len(frame_2) == 1
    assert frame_2[0].bed_id == 0
    assert frame_2[0].person_id == PERSON_ID


def test_person_beside_polygon_bed_reads_as_not_occupied_immediately() -> None:
    # Given
    monitor = _monitor(grace_frames=1)
    assert monitor.update(_input(IN_BED, 0)) == ()

    # When: the person is beside (not in) the bed, one frame after entry.
    _ = monitor.update(_input(BESIDE_BED_IN_AABB, 1))

    # Then: the debug snapshot -- what the dashboard/overlay actually reads
    # -- must not report "occupied" for a box that never entered the real
    # bed polygon, even though it sits inside the bed's AABB.
    assert monitor.last_debug_snapshot is not None
    assert monitor.last_debug_snapshot.statuses[0].occupancy != "occupied"


def test_bed_exit_input_can_only_receive_persisted_polygons() -> None:
    """A segmentation row can never become the bed ID used by an alert."""
    scene = SceneState("camera-polygon")
    observation, _ = scene.resolve_bed_regions(
        FrameObservation(regions=((DIAMOND_BED,), ())),
        frame_index=0,
        bed_scheduled=True,
        bed_interval=1,
    )
    assert observation.bed_boxes == ()
    assert scene.bed_polygon_source == "none"

    tree = ast.parse(Path("worker/pipeline/perception/scene_state.py").read_text())
    segmentation_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "bed_boxes"
        and isinstance(node.value, ast.Name)
        and node.value.id == "observation"
    ]
    assert segmentation_reads == []
