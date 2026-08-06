from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains import bed_exit
from worker.types import DecisionInput


def _decision_input(person: BoundingBox, bed: BoundingBox, frame_index: int) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            regions=((bed,), ()),
            track_ids=(11,),
        ),
        frame_width=160,
        frame_height=120,
        live_track_ids=(11,),
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


@pytest.mark.parametrize(
    ("hour", "minute", "expected_count"),
    ((22, 0, 1), (4, 59, 1), (13, 0, 0), (5, 0, 0)),
)
def test_cross_midnight_and_daytime_gate_use_injected_clock(
    hour: int,
    minute: int,
    expected_count: int,
) -> None:
    # Given
    fixed = datetime(2026, 7, 31, hour, minute, tzinfo=ZoneInfo("Asia/Seoul"))
    clock_calls = 0

    def clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        return fixed

    monitor = bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id="camera-night-window",
            facility_id="facility-night-window",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=0,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=clock,
    )
    bed = BoundingBox(0, 0, 80, 100, 0.99)
    in_bed = BoundingBox(10, 10, 70, 90, 0.95)
    outside = BoundingBox(90, 10, 150, 90, 0.95)
    _ = monitor.update(_decision_input(in_bed, bed, 0))

    # When
    events = monitor.update(_decision_input(outside, bed, 1))

    # Then
    assert len(events) == expected_count
    assert clock_calls > 0


def test_night_window_rejects_naive_wall_clock() -> None:
    # Given
    window = bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul")

    # When / Then
    with pytest.raises(ValueError, match="timezone-aware"):
        _ = window.contains(datetime(2026, 7, 31, 22, 0))


def test_bed_exit_latch_returns_only_rising_edges() -> None:
    # Given
    latch = bed_exit.BedExitLatch()
    event = bed_exit.BedExitEvent(person_id=11, bed_id=2)

    # When
    first = latch.update((event,), time_sec=1.0)
    repeated = latch.update((event,), time_sec=2.0)
    cleared = latch.update((), time_sec=3.0)
    second = latch.update((event,), time_sec=4.0)

    # Then
    assert first == (event,)
    assert repeated == cleared == ()
    assert second == (event,)
    assert latch.event_count == 2
    assert latch.first_event_sec == 1.0
