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


def test_night_window_does_not_let_latch_consume_a_daytime_onset() -> None:
    """A boundary exit must remain available once the clock enters the window.

    The latch is rising-edge only and keyed ``(person_id, bed_id)``. If a
    daytime onset is fed to it, ``event_count`` advances and ``_active_exits``
    can swallow the same pair when the window opens. Suppression must happen
    before the latch consumes the onset.
    """
    # Given
    now = datetime(2026, 7, 31, 13, 0, tzinfo=SEOUL)

    def clock() -> datetime:
        return now

    monitor = _night_monitor(clock)
    assert monitor.update(_decision_input(IN_BED, BED, 0)) == ()

    # When: confirmed exit outside the window
    daytime_events = monitor.update(_decision_input(OUTSIDE, BED, 1))

    # Then: suppressed, and the latch must not have consumed the onset
    assert daytime_events == ()
    assert monitor._latch.event_count == 0
    assert monitor._latch.status_snapshot.active_exits == ()

    # When: the same person/bed pair exits again inside the window
    now = datetime(2026, 7, 31, 22, 0, tzinfo=SEOUL)
    assert monitor.update(_decision_input(IN_BED, BED, 2)) == ()
    night_events = monitor.update(_decision_input(OUTSIDE, BED, 3))

    # Then
    assert night_events == (
        BusinessEvent(
            domain="bed_exit",
            event_type="bed-exit",
            identity="11:0",
            camera_id=NIGHT_CAMERA_ID,
            facility_id=NIGHT_FACILITY_ID,
            time_sec=3.0,
            probability=1.0,
            person_id=PERSON_ID,
            bed_id=0,
        ),
    )
    assert monitor._latch.event_count == 1


def test_night_window_suppressed_onset_does_not_poison_later_in_window_exit() -> None:
    """stale_state: a gated onset must not leave `_active_exits` holding the key.

    After the daytime trigger the assignment is cleared, so the next in-window
    fire requires a fresh in-bed hold. The latch key must not still be set
    from the suppressed onset, or that second cycle is swallowed.
    """
    # Given
    now = datetime(2026, 7, 31, 13, 0, tzinfo=SEOUL)

    def clock() -> datetime:
        return now

    monitor = _night_monitor(clock)
    assert monitor.update(_decision_input(IN_BED, BED, 0)) == ()
    assert monitor.update(_decision_input(OUTSIDE, BED, 1)) == ()
    assert monitor._latch.status_snapshot.active_exits == ()

    # When: re-assign and exit after the window opens
    now = datetime(2026, 7, 31, 22, 0, tzinfo=SEOUL)
    assert monitor.update(_decision_input(IN_BED, BED, 2)) == ()
    events = monitor.update(_decision_input(OUTSIDE, BED, 3))

    # Then
    assert len(events) == 1
    assert events[0].event_type == "bed-exit"
    assert events[0].person_id == PERSON_ID
    assert events[0].bed_id == 0
