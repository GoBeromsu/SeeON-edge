from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains.bed_exit.detector import BedExitMonitor
from worker.domains.bed_exit.night_window import NightWindow
from worker.domains.bed_exit.schema import BedExitConfig, BedExitEvent, BedExitFrame, BedStatus
from worker.types import DecisionInput

# Supersession notes (edge assertions not ported here, behavior verified
# elsewhere against the current BedExitMonitor(config=..., clock=...) API):
# - test_own_bed_exit_after_grace_emits_once_and_unassigns ->
#   tests/test_worker_domains_bed_exit.py:77-133
#   (test_own_bed_exit_emits_once_after_grace_period)
# - test_cross_bed_movement_does_not_false_positive_own_bed_exit ->
#   tests/test_worker_domains_bed_exit.py:136-167
#   (test_cross_bed_movement_never_emits_or_reassigns)
# - test_overlapping_beds_choose_best_containment_with_lowest_id_tiebreak ->
#   tests/test_worker_domains_bed_exit_assignment.py:59-74
#   (test_hold_and_containment_tie_assign_the_lowest_bed_id); both fixtures
#   construct a full-containment tie (ratio 1.0 on both beds) and assert the
#   same lowest-bed-id tiebreak.
# - test_runtime_observation_update_returns_domain_event_tuple -> the dict
#   payload shape is retired; the equivalent BusinessEvent-tuple return is
#   covered by tests/test_worker_domains_bed_exit.py:77-133.


def box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


_DEFAULT_NIGHT_WINDOW = NightWindow(start="21:00", end="05:00", tz="Asia/Seoul")


def _input(
    *,
    person_boxes: tuple[BoundingBox, ...],
    bed_boxes: tuple[BoundingBox, ...],
    frame_index: int,
    time_sec: float | None = None,
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(detections=(person_boxes, ()), regions=(bed_boxes, ())),
        frame_width=200,
        frame_height=200,
        live_track_ids=(),
        time_sec=float(frame_index) if time_sec is None else time_sec,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def _monitor(
    *,
    clock: Callable[[], datetime],
    night_window: NightWindow | None = _DEFAULT_NIGHT_WINDOW,
    hold_frames: int = 1,
    grace_frames: int = 0,
    min_containment: float = 0.5,
    camera_id: str = "camera-bed-exit-schema",
    facility_id: str = "facility-bed-exit-schema",
) -> BedExitMonitor:
    return BedExitMonitor(
        config=BedExitConfig(
            camera_id=camera_id,
            facility_id=facility_id,
            min_containment=min_containment,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            night_window=night_window,
        ),
        clock=clock,
        boot_id="boot-bed-exit-schema",
        stream_epoch="epoch-bed-exit-schema",
        source_generation=0,
    )


def _own_bed_exit_events(
    monitor: BedExitMonitor,
    bed: BoundingBox,
    *,
    exit_time_sec: float | None = None,
) -> tuple[object, ...]:
    monitor.update(_input(person_boxes=(box(10, 10, 70, 90),), bed_boxes=(bed,), frame_index=0))
    return monitor.update(
        _input(
            person_boxes=(box(90, 10, 150, 90),),
            bed_boxes=(bed,),
            frame_index=1,
            time_sec=exit_time_sec,
        )
    )


def test_schema_exports_bed_exit_frame_statuses_and_events() -> None:
    bed = box(0, 0, 100, 100)
    status = BedStatus(bed_id=0, box=bed, occupancy="exit", person_id=7)
    event = BedExitEvent(person_id=7, bed_id=0)
    assert BedExitFrame(statuses=(status,), events=(event,)).statuses == (status,)


def test_hold_frames_prevent_jitter_assignment_until_stable() -> None:
    # Candidate-bed switching between frames must reset the hold-frame
    # counter (worker/domains/bed_exit/detector.py:_Assignment.update_candidate);
    # only a stable, repeated candidate reaches hold_frames and gets assigned.
    beds = (box(0, 0, 100, 100), box(120, 0, 220, 100))
    monitor = _monitor(
        clock=lambda: datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC")),
        night_window=None,
        hold_frames=2,
        grace_frames=1,
    )

    monitor.update(_input(person_boxes=(box(10, 10, 60, 60),), bed_boxes=beds, frame_index=0))
    first = monitor.last_debug_snapshot
    assert first is not None
    assert [s.occupancy for s in first.statuses] == ["empty", "empty"]

    monitor.update(_input(person_boxes=(box(130, 10, 180, 60),), bed_boxes=beds, frame_index=1))
    jitter = monitor.last_debug_snapshot
    assert jitter is not None
    assert [s.occupancy for s in jitter.statuses] == ["empty", "empty"]

    monitor.update(_input(person_boxes=(box(130, 10, 180, 60),), bed_boxes=beds, frame_index=2))
    held_once = monitor.last_debug_snapshot
    assert held_once is not None
    assert [s.occupancy for s in held_once.statuses] == ["empty", "occupied"]


def test_bed_exit_config_rejects_out_of_range_min_containment() -> None:
    with pytest.raises(ValueError, match="min_containment"):
        BedExitConfig(camera_id="c", facility_id="f", min_containment=0.0)
    with pytest.raises(ValueError, match="min_containment"):
        BedExitConfig(camera_id="c", facility_id="f", min_containment=1.5)


def test_bed_exit_config_accepts_min_containment_upper_bound() -> None:
    BedExitConfig(camera_id="c", facility_id="f", min_containment=1.0)


def test_bed_exit_config_rejects_non_positive_hold_frames() -> None:
    with pytest.raises(ValueError, match="hold_frames"):
        BedExitConfig(camera_id="c", facility_id="f", hold_frames=0)


def test_bed_exit_config_accepts_hold_frames_lower_bound() -> None:
    BedExitConfig(camera_id="c", facility_id="f", hold_frames=1)


def test_bed_exit_config_rejects_negative_grace_frames() -> None:
    with pytest.raises(ValueError, match="grace_frames"):
        BedExitConfig(camera_id="c", facility_id="f", grace_frames=-1)


def test_bed_exit_config_accepts_grace_frames_lower_bound() -> None:
    BedExitConfig(camera_id="c", facility_id="f", grace_frames=0)


@pytest.mark.parametrize(
    ("hour", "minute", "second", "expected_count"),
    (
        (21, 0, 0, 1),  # exactly at the inclusive start boundary -> inside
        (4, 59, 59, 1),  # one second before the exclusive end -> inside
        (20, 59, 59, 0),  # one second before the start -> outside
        (5, 0, 0, 0),  # exactly at the exclusive end boundary -> outside
    ),
)
def test_night_window_exact_boundary_seconds_gate_the_runtime_path(
    hour: int,
    minute: int,
    second: int,
    expected_count: int,
) -> None:
    fixed = datetime(2026, 1, 1, hour, minute, second, tzinfo=ZoneInfo("Asia/Seoul"))
    monitor = _monitor(clock=lambda: fixed)
    bed = box(0, 0, 80, 100)

    events = _own_bed_exit_events(monitor, bed)

    assert len(events) == expected_count


def test_night_window_uses_injected_clock_not_monotonic_time_sec() -> None:
    # Clock says daytime (13:00, outside the window); time_sec is a large
    # monotonic value (23*3600s) that would look like 23:00 "time of day" if
    # ever mistaken for a wall-clock reading. Only the injected clock may
    # gate the window.
    fixed = datetime(2026, 1, 1, 13, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    monitor = _monitor(clock=lambda: fixed)
    bed = box(0, 0, 80, 100)

    events = _own_bed_exit_events(monitor, bed, exit_time_sec=23 * 3600)

    assert events == ()


def test_night_window_rejects_naive_clock_datetime_through_the_runtime_path() -> None:
    # Unit-level rejection is covered by
    # tests/test_worker_domains_bed_exit_time.py:76-82
    # (test_night_window_rejects_naive_wall_clock), which calls
    # NightWindow.contains() directly. This ports the production call path
    # (BedExitMonitor.update() -> self._clock()) as cheap insurance that the
    # same guarantee holds end to end.
    monitor = _monitor(clock=lambda: datetime(2026, 1, 1, 22, 0, 0))
    bed = box(0, 0, 80, 100)

    with pytest.raises(ValueError, match="timezone-aware"):
        _own_bed_exit_events(monitor, bed)


def test_without_night_window_emits_regardless_of_clock() -> None:
    fixed = datetime(2026, 1, 1, 13, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    monitor = _monitor(clock=lambda: fixed, night_window=None)
    bed = box(0, 0, 80, 100)

    events = _own_bed_exit_events(monitor, bed)

    assert len(events) == 1


def test_a_failing_shadow_evaluation_does_not_discard_the_bed_exit_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shadow path is non-authoritative and must not be able to destroy one.

    `_record_shadow` never replaces the containment decision and never carries
    triggered=True, yet it sat between the events being computed and the return
    that delivers them. A shadow failure therefore discarded a real bed-exit
    event -- a capability that cannot decide anything destroying a decision.
    """
    fixed = datetime(2026, 1, 1, 13, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    monitor = _monitor(clock=lambda: fixed, night_window=None)
    bed = box(0, 0, 80, 100)

    def _explode(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise RuntimeError("shadow evaluation failed")

    monkeypatch.setattr(type(monitor), "_record_shadow", _explode, raising=True)

    events = _own_bed_exit_events(monitor, bed)

    assert len(events) == 1, (
        "the bed-exit event was discarded because a non-authoritative shadow evaluation raised"
    )


def test_bed_exit_rearms_only_after_confirmed_recovery() -> None:
    """Repeated exits are suppressed until a contained recovery is confirmed."""
    fixed = datetime(2026, 1, 1, 13, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    monitor = _monitor(clock=lambda: fixed, night_window=None, grace_frames=0)
    bed = box(0, 0, 80, 100)

    monitor.update(_input(person_boxes=(box(10, 10, 70, 90),), bed_boxes=(bed,), frame_index=0))
    exited = monitor.update(
        _input(person_boxes=(box(90, 10, 150, 90),), bed_boxes=(bed,), frame_index=1)
    )
    assert len(exited) == 1, "the exit was never reported"

    repeated = monitor.update(
        _input(person_boxes=(box(90, 10, 150, 90),), bed_boxes=(bed,), frame_index=2)
    )
    recovered = monitor.update(
        _input(person_boxes=(box(10, 10, 70, 90),), bed_boxes=(bed,), frame_index=3)
    )
    reexited = monitor.update(
        _input(person_boxes=(box(90, 10, 150, 90),), bed_boxes=(bed,), frame_index=4)
    )

    assert repeated == recovered == ()
    assert len(reexited) == 1
    assert reexited[0].identity == "boot-bed-exit-schema:epoch-bed-exit-schema:bed-exit:0:0:0:0:2"


def test_release_reopens_a_failed_bed_exit_for_one_retry() -> None:
    fixed = datetime(2026, 1, 1, 13, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    monitor = _monitor(clock=lambda: fixed, night_window=None, grace_frames=0)
    bed = box(0, 0, 80, 100)

    failed = _own_bed_exit_events(monitor, bed)[0]
    monitor.release_onset(failed)
    retried = monitor.update(
        _input(
            person_boxes=(box(90, 10, 150, 90),),
            bed_boxes=(bed,),
            frame_index=2,
        )
    )

    assert len(retried) == 1
    assert retried[0].identity != failed.identity
    assert (
        monitor.update(
            _input(
                person_boxes=(box(90, 10, 150, 90),),
                bed_boxes=(bed,),
                frame_index=3,
            )
        )
        == ()
    )
