from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains import bed_exit
from worker.types import BusinessEvent, DecisionInput

NIGHT_CAMERA_ID: Final = "camera-night-window"
NIGHT_FACILITY_ID: Final = "facility-night-window"
BOOT_ID: Final = "boot-night-window"
STREAM_EPOCH: Final = "epoch-night-window"
SOURCE_GENERATION: Final = 0
PERSON_ID: Final = 11
BED: Final = BoundingBox(0, 0, 80, 100, 0.99)
IN_BED: Final = BoundingBox(10, 10, 70, 90, 0.95)
OUTSIDE: Final = BoundingBox(90, 10, 150, 90, 0.95)
SEOUL: Final = ZoneInfo("Asia/Seoul")
NIGHT_WINDOW: Final = bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul")


def _clock_at(hour: int, minute: int = 0) -> Callable[[], datetime]:
    fixed = datetime(2026, 7, 31, hour, minute, tzinfo=SEOUL)
    return lambda: fixed


def _decision_input(person: BoundingBox, bed: BoundingBox, frame_index: int) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=((person,), ()),
            regions=((bed,), ()),
            track_ids=(PERSON_ID,),
        ),
        frame_width=160,
        frame_height=120,
        live_track_ids=(PERSON_ID,),
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def _night_monitor(
    clock: Callable[[], datetime],
    *,
    night_window: bed_exit.NightWindow | None = NIGHT_WINDOW,
) -> bed_exit.BedExitMonitor:
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=NIGHT_CAMERA_ID,
            facility_id=NIGHT_FACILITY_ID,
            min_containment=0.5,
            hold_frames=1,
            grace_frames=0,
            night_window=night_window,
        ),
        clock=clock,
        boot_id=BOOT_ID,
        stream_epoch=STREAM_EPOCH,
        source_generation=SOURCE_GENERATION,
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

    monitor = _night_monitor(clock)
    _ = monitor.update(_decision_input(IN_BED, BED, 0))

    # When
    events = monitor.update(_decision_input(OUTSIDE, BED, 1))

    # Then
    assert len(events) == expected_count
    assert clock_calls > 0


def test_night_window_rejects_naive_wall_clock() -> None:
    # Given
    window = bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul")

    # When / Then
    with pytest.raises(ValueError, match="timezone-aware"):
        _ = window.contains(datetime(2026, 7, 31, 22, 0))


def test_bed_exit_latch_tracks_observation_freshness() -> None:
    # Given
    now = 0.0
    latch = bed_exit.BedExitLatch(clock=lambda: now, stale_after_sec=3.0)

    # When
    initially = latch.status_snapshot
    latch.update()
    now = 2.0
    fresh = latch.status_snapshot
    latch.coast()
    now = 3.0
    stale = latch.status_snapshot

    # Then
    assert initially.stale is True
    assert fresh.stale is False
    assert fresh.observation_age_sec == 2.0
    assert stale.stale is True
    assert stale.observation_age_sec == 3.0


def test_night_window_outside_still_populates_debug_snapshot_for_overlay() -> None:
    """Overlay must keep rendering ``bed:exit`` outside the night window.

    Source of truth is failure class ④ in
    ``docs/runbooks/post-redeploy-event-readout.md``: ``last_debug_snapshot``
    is filled *before* the night-window gate, and the overlay draws
    ``statuses[].occupancy`` with no extra gate, so ``bed:exit`` still
    flashes when no ``BusinessEvent`` is emitted. ``DESIGN.md`` only says
    the overlay label is occupancy from the debug snapshot; it does not
    record this gate-ordering.
    """
    # Given
    monitor = _night_monitor(_clock_at(13))
    assert monitor.update(_decision_input(IN_BED, BED, 0)) == ()

    # When
    events = monitor.update(_decision_input(OUTSIDE, BED, 1))

    # Then
    assert events == ()
    snapshot = monitor.last_debug_snapshot
    assert snapshot is not None
    assert snapshot.events == (bed_exit.BedExitEvent(PERSON_ID, 0),)
    assert tuple(status.occupancy for status in snapshot.statuses) == ("exit",)


def test_night_window_does_not_consume_a_daytime_episode_onset() -> None:
    """A boundary exit must remain available once the clock enters the window.

    Suppression must happen before the episode authority receives the onset.
    """
    # Given
    now = datetime(2026, 7, 31, 13, 0, tzinfo=SEOUL)

    def clock() -> datetime:
        return now

    monitor = _night_monitor(clock)
    assert monitor.update(_decision_input(IN_BED, BED, 0)) == ()

    # When: confirmed exit outside the window
    daytime_events = monitor.update(_decision_input(OUTSIDE, BED, 1))

    # Then: suppressed, and no episode onset has been emitted
    assert daytime_events == ()

    # When: the same person/bed pair exits again inside the window
    now = datetime(2026, 7, 31, 22, 0, tzinfo=SEOUL)
    assert monitor.update(_decision_input(IN_BED, BED, 2)) == ()
    night_events = monitor.update(_decision_input(OUTSIDE, BED, 3))

    # Then
    assert night_events == (
        BusinessEvent(
            domain="bed_exit",
            event_type="bed-exit",
            identity="boot-night-window:epoch-night-window:bed-exit:0:11:0:0:1",
            camera_id=NIGHT_CAMERA_ID,
            facility_id=NIGHT_FACILITY_ID,
            time_sec=3.0,
            probability=1.0,
            person_id=PERSON_ID,
            bed_id=0,
        ),
    )


def test_night_window_suppressed_onset_does_not_poison_later_in_window_exit() -> None:
    """A gated onset must not leave the authority holding the episode open."""
    # Given
    now = datetime(2026, 7, 31, 13, 0, tzinfo=SEOUL)

    def clock() -> datetime:
        return now

    monitor = _night_monitor(clock)
    assert monitor.update(_decision_input(IN_BED, BED, 0)) == ()
    assert monitor.update(_decision_input(OUTSIDE, BED, 1)) == ()

    # When: re-assign and exit after the window opens
    now = datetime(2026, 7, 31, 22, 0, tzinfo=SEOUL)
    assert monitor.update(_decision_input(IN_BED, BED, 2)) == ()
    events = monitor.update(_decision_input(OUTSIDE, BED, 3))

    # Then
    assert len(events) == 1
    assert events[0].event_type == "bed-exit"
    assert events[0].person_id == PERSON_ID
    assert events[0].bed_id == 0
