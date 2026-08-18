from __future__ import annotations

from datetime import UTC, datetime

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor
from worker.domains.fall import FallEventLatch
from worker.types import DecisionInput


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _bed_input(person: BoundingBox, bed: BoundingBox) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            regions=((bed,), ()),
            track_ids=(7,),
        ),
        frame_width=200,
        frame_height=200,
        live_track_ids=(7,),
        time_sec=10.0,
        frame_index=0,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def test_three_second_inference_outage_marks_latches_stale_without_event_or_empty() -> None:
    clock = _Clock()
    fall = FallEventLatch(
        None,
        camera_id="camera-stale",
        facility_id="facility-1",
        clock=clock,
    )
    assert fall.update_signal(True, time_sec=10.0) is True

    bed = BoundingBox(0, 0, 100, 100, 0.9)
    person = BoundingBox(10, 10, 90, 90, 0.9)
    monitor = BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-stale",
            facility_id="facility-1",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=0,
        ),
        clock=lambda: datetime(2026, 8, 18, tzinfo=UTC),
        staleness_clock=clock,
    )
    assert monitor.update(_bed_input(person, bed)) == ()

    clock.now = 3.0
    assert fall.coast() == ()
    assert monitor.coast(frame_index=15) == ()

    assert fall.status_snapshot.is_fall is True
    assert fall.status_snapshot.stale is True
    snapshot = monitor.last_debug_snapshot
    assert snapshot is not None
    assert snapshot.stale is True
    assert snapshot.observation_age_sec == 3.0
    assert tuple(status.occupancy for status in snapshot.statuses) == ("covered",)
    assert snapshot.events == ()


def test_freshness_uses_wall_clock_not_source_timestamp_or_tick_count() -> None:
    clock = _Clock()
    latch = FallEventLatch(
        None,
        camera_id="camera-source-time",
        facility_id="facility-1",
        clock=clock,
    )
    assert latch.update_signal(False, time_sec=10_000.0) is False

    for _ in range(100):
        assert latch.coast() == ()
    assert latch.status_snapshot.stale is False

    clock.now = 3.0
    assert latch.status_snapshot.stale is True
    assert latch.status_snapshot.observation_age_sec == 3.0
