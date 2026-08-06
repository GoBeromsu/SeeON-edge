from __future__ import annotations

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
from worker.types import DecisionInput

PERSON_ID: Final = 7
BED: Final = BoundingBox(0, 0, 80, 100, 0.99)
IN_BED: Final = BoundingBox(10, 10, 70, 90, 0.95)
OUTSIDE_BED: Final = BoundingBox(100, 10, 160, 90, 0.94)


def _monitor(
    *, camera_id: str, hold_frames: int = 1, grace_frames: int = 0
) -> bed_exit.BedExitMonitor:
    fixed = datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=camera_id,
            facility_id="facility-bed-exit",
            min_containment=0.5,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=lambda: fixed,
    )


def _input(
    person: BoundingBox,
    beds: tuple[BoundingBox, ...],
    frame_index: int,
    *,
    live_track_ids: tuple[int, ...] = (PERSON_ID,),
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            regions=(beds, ()),
            track_ids=(PERSON_ID,),
        ),
        frame_width=180,
        frame_height=120,
        live_track_ids=live_track_ids,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def test_hold_and_containment_tie_assign_the_lowest_bed_id() -> None:
    # Given
    monitor = _monitor(camera_id="camera-tie", hold_frames=2)
    overlapping_beds = (BED, BED)

    # When
    first = monitor.update(_input(IN_BED, overlapping_beds, 0))
    second = monitor.update(_input(IN_BED, overlapping_beds, 1))

    # Then
    assert first == second == ()
    assert monitor.last_debug_snapshot is not None
    assert tuple(status.occupancy for status in monitor.last_debug_snapshot.statuses) == (
        "occupied",
        "empty",
    )


def test_detector_instances_keep_camera_assignments_isolated() -> None:
    # Given
    camera_a = _monitor(camera_id="camera-a")
    camera_b = _monitor(camera_id="camera-b")
    _ = camera_a.update(_input(IN_BED, (BED,), 0))

    # When
    camera_b_events = camera_b.update(_input(OUTSIDE_BED, (BED,), 1))
    camera_a_events = camera_a.update(_input(OUTSIDE_BED, (BED,), 1))

    # Then
    assert camera_b_events == ()
    assert len(camera_a_events) == 1
    assert camera_a_events[0].camera_id == "camera-a"


def test_track_lost_mid_exit_still_fires_the_event() -> None:
    # Given
    monitor = _monitor(camera_id="camera-lost-mid-exit", hold_frames=1, grace_frames=2)
    _ = monitor.update(_input(IN_BED, (BED,), 0))
    _ = monitor.update(_input(OUTSIDE_BED, (BED,), 1))

    # When
    events = monitor.update(_input(OUTSIDE_BED, (BED,), 2, live_track_ids=()))

    # Then
    assert events, (
        "track lost while mid-exit (grace_frames=1 at loss) produced no event -- "
        "a real departure that happens to coincide with track loss is currently "
        "indistinguishable from a track that never left"
    )
    assert events[0].person_id == PERSON_ID
    assert events[0].bed_id == 0


def test_single_sub_threshold_frame_before_track_loss_currently_fires() -> None:
    """Characterization test, not a desired-behavior test.

    A single sub-threshold containment frame (pose/detection jitter, a
    one-frame blanket/caregiver occlusion -- not necessarily a real
    departure) immediately followed by track death currently DOES fire a
    bed-exit event, because the stale-track gate is `grace_frames > 0`.
    This is a known, deliberately-accepted false positive: the user chose
    sensitivity over precision for tonight, since bed_exit was producing
    zero events in production and a missed exit is invisible while a false
    positive is checkable against footage. See #246 for the reproductions,
    the trade-off analysis, and the prepared remedy (`>= 2`) for when
    precision becomes the priority -- adopting it requires first extending
    #218's regression test past its current 1-frame script, since these
    two tests collide at exactly one grace frame.

    If this test starts failing, someone changed the gate -- check #246
    before "fixing" it back.
    """
    # Given: a person's bed assignment sees exactly ONE frame of
    # sub-threshold containment right before their track is lost.
    monitor = _monitor(camera_id="camera-single-frame-jitter", hold_frames=1, grace_frames=3)
    _ = monitor.update(_input(IN_BED, (BED,), 0))
    _ = monitor.update(_input(OUTSIDE_BED, (BED,), 1))

    # When: the track dies immediately after that one sub-threshold frame
    # (grace_frames=1 at the moment of loss).
    events = monitor.update(_input(OUTSIDE_BED, (BED,), 2, live_track_ids=()))

    # Then: current behavior is that it fires.
    assert len(events) == 1
    assert events[0].person_id == PERSON_ID
    assert events[0].bed_id == 0


def test_dead_observed_track_cannot_emit_after_identity_reuse() -> None:
    # Given
    monitor = _monitor(camera_id="camera-reused-track")
    _ = monitor.update(_input(IN_BED, (BED,), 0))

    # When
    dead_track = monitor.update(_input(IN_BED, (BED,), 1, live_track_ids=()))
    reused_identity = monitor.update(_input(OUTSIDE_BED, (BED,), 2))

    # Then
    assert dead_track == reused_identity == ()
