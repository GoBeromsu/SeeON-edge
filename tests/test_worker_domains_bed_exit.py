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
from tests_support.bed_pose_fixtures import frame_pose_features, lying_in_bed, standing
from worker.domains import bed_exit
from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot
from worker.types.bed_pose_features import EMPTY_FRAME_BED_POSE_FEATURES, FrameBedPoseFeatures

CAMERA_ID: Final = "camera-bed-exit"
FACILITY_ID: Final = "facility-bed-exit"
PERSON_ID: Final = 7
BED_A: Final = BoundingBox(0, 0, 80, 100, 0.99)
BED_B: Final = BoundingBox(100, 0, 180, 100, 0.98)
IN_BED_A: Final = BoundingBox(10, 10, 70, 90, 0.95)
IN_BED_B: Final = BoundingBox(110, 10, 170, 90, 0.95)
OUTSIDE_BEDS: Final = BoundingBox(40, 120, 100, 190, 0.94)


def _clock_at(hour: int = 22, minute: int = 0) -> Callable[[], datetime]:
    fixed = datetime(2026, 7, 31, hour, minute, tzinfo=ZoneInfo("Asia/Seoul"))
    return lambda: fixed


def _monitor(
    *,
    camera_id: str = CAMERA_ID,
    hold_frames: int = 1,
    grace_frames: int = 2,
    in_bed_dwell_sec: float = 1.0,
    outside_dwell_sec: float = 1.0,
) -> bed_exit.BedExitMonitor:
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=camera_id,
            facility_id=FACILITY_ID,
            min_containment=0.5,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            in_bed_dwell_sec=in_bed_dwell_sec,
            outside_dwell_sec=outside_dwell_sec,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=_clock_at(),
        boot_id="test-boot",
        stream_epoch="test-epoch",
        source_generation=0,
    )


def _input(
    *,
    person_boxes: tuple[BoundingBox, ...],
    bed_boxes: tuple[BoundingBox, ...],
    track_ids: tuple[int | None, ...],
    frame_index: int,
    source: BedRegionCacheState = BedRegionCacheState.FRESH,
    time_sec: float | None = None,
    bed_pose_features: FrameBedPoseFeatures = EMPTY_FRAME_BED_POSE_FEATURES,
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=(person_boxes, ()),
            regions=(bed_boxes, ()),
            track_ids=track_ids,
        ),
        frame_width=200,
        frame_height=200,
        live_track_ids=tuple(track_id for track_id in track_ids if track_id is not None),
        time_sec=float(frame_index) if time_sec is None else time_sec,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=source),
        bed_pose_features=bed_pose_features,
    )


def _lying_pose(track_id: int = PERSON_ID, bed_id: int = 0) -> FrameBedPoseFeatures:
    return frame_pose_features(lying_in_bed(track_id=track_id, bed_id=bed_id))


def _standing_pose(track_id: int = PERSON_ID, bed_id: int = 0) -> FrameBedPoseFeatures:
    return frame_pose_features(standing(track_id=track_id, bed_id=bed_id))


def test_own_bed_exit_emits_once_after_grace_period() -> None:
    """A track must arm (posture-confirmed in-bed dwell) before an outside
    dwell can ever fire; `outside_dwell_sec=3.0` keeps two non-firing
    outside frames before the third crosses the threshold, mirroring the
    old frame-count "grace period" shape under the new PTS-dwell model."""
    # Given
    monitor = _monitor(grace_frames=2, outside_dwell_sec=3.0)
    assert (
        monitor.update(
            _input(
                person_boxes=(IN_BED_A,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=0,
            )
        )
        == ()
    )
    assert (
        monitor.update(
            _input(
                person_boxes=(IN_BED_A,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=1,
                bed_pose_features=_lying_pose(),
            )
        )
        == ()
    )

    # When
    before_grace = tuple(
        monitor.update(
            _input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
            )
        )
        for frame_index in (2, 3)
    )
    onset = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=4,
        )
    )
    repeated = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=5,
        )
    )

    # Then
    assert before_grace == ((), ())
    assert onset == (
        BusinessEvent(
            domain="bed_exit",
            event_type="bed-exit",
            identity="test-boot:test-epoch:bed-exit:0:7:0:0:1",
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            time_sec=4.0,
            probability=1.0,
            person_id=PERSON_ID,
            bed_id=0,
        ),
    )
    assert repeated == ()


def test_dwell_outcome_is_identical_at_15fps_and_30fps() -> None:
    """Dwell is measured via `input_value.time_sec` (PTS seconds), never frame
    counts, so the same real-time scenario must produce the same outcome
    whether frames arrive at 15fps or 30fps -- twice as many frames covering
    the same wall-clock span at 30fps must not double-count dwell, and half
    as many must not starve it either."""

    def _run(step_sec: float) -> tuple[BusinessEvent, ...]:
        monitor = _monitor(in_bed_dwell_sec=3.0, outside_dwell_sec=2.0)
        events: list[BusinessEvent] = []
        time_sec = 0.0
        frame_index = 0
        while time_sec <= 3.2:  # > in_bed_dwell_sec of real time, in bed
            events.extend(
                monitor.update(
                    _input(
                        person_boxes=(IN_BED_A,),
                        bed_boxes=(BED_A,),
                        track_ids=(PERSON_ID,),
                        frame_index=frame_index,
                        time_sec=time_sec,
                        bed_pose_features=_lying_pose(),
                    )
                )
            )
            time_sec += step_sec
            frame_index += 1
        outside_start = time_sec
        while time_sec - outside_start <= 2.2:  # > outside_dwell_sec, outside
            events.extend(
                monitor.update(
                    _input(
                        person_boxes=(OUTSIDE_BEDS,),
                        bed_boxes=(BED_A,),
                        track_ids=(PERSON_ID,),
                        frame_index=frame_index,
                        time_sec=time_sec,
                    )
                )
            )
            time_sec += step_sec
            frame_index += 1
        return tuple(events)

    at_15fps = _run(1.0 / 15.0)
    at_30fps = _run(1.0 / 30.0)

    assert len(at_15fps) == 1
    assert len(at_30fps) == 1
    assert at_15fps[0].event_type == "bed-exit"
    assert at_30fps[0].event_type == "bed-exit"
    assert at_15fps[0].person_id == at_30fps[0].person_id
    assert at_15fps[0].bed_id == at_30fps[0].bed_id


def test_standing_beside_bed_for_ten_seconds_then_walking_away_never_emits() -> None:
    """Review of real bed-exit clips found caregiver activity, not a genuine
    exit, in 3 of 6 cases: a standing caregiver's bbox easily reaches
    containment >= min_containment while they lean over the bed. `IN_BED_A`
    fully contains against `BED_A` (own_ratio 1.0), so geometry alone would
    arm here -- but `standing()`'s hip_depth sits below
    `_MIN_IN_BED_HIP_DEPTH`, so `posture_confirms_in_bed` is never true and
    `in_bed_dwell_sec` never accumulates. Ten seconds of standing overlap
    (more than 3x `in_bed_dwell_sec`) must never arm the track, and walking
    away afterward must not emit either, since it was never armed."""
    monitor = _monitor(in_bed_dwell_sec=3.0, outside_dwell_sec=2.0)
    events: list[BusinessEvent] = []
    time_sec = 0.0
    frame_index = 0
    while time_sec <= 10.0:  # far past in_bed_dwell_sec, standing at the bed
        events.extend(
            monitor.update(
                _input(
                    person_boxes=(IN_BED_A,),
                    bed_boxes=(BED_A,),
                    track_ids=(PERSON_ID,),
                    frame_index=frame_index,
                    time_sec=time_sec,
                    bed_pose_features=_standing_pose(),
                )
            )
        )
        time_sec += 1.0 / 15.0
        frame_index += 1
    walk_away_start = time_sec
    while time_sec - walk_away_start <= 2.2:  # walks away, past outside_dwell_sec
        events.extend(
            monitor.update(
                _input(
                    person_boxes=(OUTSIDE_BEDS,),
                    bed_boxes=(BED_A,),
                    track_ids=(PERSON_ID,),
                    frame_index=frame_index,
                    time_sec=time_sec,
                )
            )
        )
        time_sec += 1.0 / 15.0
        frame_index += 1

    assert events == []


def test_release_reopens_a_failed_bed_exit_for_one_retry() -> None:
    """`release_onset` reopens episode bookkeeping only -- not the detector's
    own arm latch (hysteresis, addendum #1).

    Supersedes `test_release_reopens_a_failed_stale_exit_after_the_track_
    reassociates`: that test's premise -- a vanishing track (empty
    `person_boxes`/`track_ids`) "failing" a bed-exit onset -- is now
    architecturally impossible, since absence never emits under the dwell
    model. The retry scenario that still makes sense is a downstream
    failure after a genuine, armed exit: the retry must still require a
    fresh, positively-observed in-bed dwell before it can re-fire.
    """
    monitor = _monitor(grace_frames=1)
    assert (
        monitor.update(
            _input(
                person_boxes=(IN_BED_A,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=0,
            )
        )
        == ()
    )
    assert (
        monitor.update(
            _input(
                person_boxes=(IN_BED_A,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=1,
                bed_pose_features=_lying_pose(),
            )
        )
        == ()
    )
    failed = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=2,
        )
    )[0]

    monitor.release_onset(failed)

    bare_repeat = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=3,
        )
    )
    assert bare_repeat == ()

    assert (
        monitor.update(
            _input(
                person_boxes=(IN_BED_A,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=4,
                bed_pose_features=_lying_pose(),
            )
        )
        == ()
    )
    retried = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=5,
        )
    )

    assert len(retried) == 1
    assert retried[0].identity != failed.identity


def test_cross_bed_movement_never_emits_or_reassigns() -> None:
    # Given
    monitor = _monitor(grace_frames=1)
    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A, BED_B),
            track_ids=(PERSON_ID,),
            frame_index=0,
        )
    )

    # When
    outputs = tuple(
        monitor.update(
            _input(
                person_boxes=(IN_BED_B,),
                bed_boxes=(BED_A, BED_B),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
            )
        )
        for frame_index in range(1, 5)
    )

    # Then
    assert outputs == ((), (), (), ())
    assert monitor.last_debug_snapshot is not None
    assert tuple(status.occupancy for status in monitor.last_debug_snapshot.statuses) == (
        "empty",
        "empty",
    )


def test_expired_cached_roi_does_not_advance_grace_or_emit() -> None:
    """An expired ROI must neither emit nor permanently wedge the camera.

    The name says only the first half. The second half is the one that matters
    more in a ward: an expiry must not leave the camera in a state where real
    bed exits stop alerting. So this asserts both directions --

    * while the cached ROI is ``EXPIRED``, grace does not advance and nothing
      is emitted (false positives), and
    * once a fresh ROI returns, a genuine bed exit still emits after grace
      (false negatives).

    The original repository split these across
    ``test_expired_cached_roi_cannot_fabricate_bed_exit`` and
    ``test_expired_cached_roi_cannot_suppress_fresh_bed_exit``. Both properties
    live here now; auditing this file by test name alone will miss the recovery
    half.
    """
    # Given
    monitor = _monitor(grace_frames=2, outside_dwell_sec=3.0)
    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
        )
    )
    # Arms at t=1.0 (dt=1.0 >= in_bed_dwell_sec=1.0). `update()` returns
    # early for an unusable region *before* touching assignment state at
    # all (worker/domains/bed_exit/detector.py:222), so the dwell clock
    # freezes at this frame's `last_time_sec` for the whole EXPIRED run
    # below and only resumes spanning the gap once FRESH returns.
    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=1,
            bed_pose_features=_lying_pose(),
        )
    )

    # When
    expired_outputs = tuple(
        monitor.update(
            _input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
                source=BedRegionCacheState.EXPIRED,
            )
        )
        for frame_index in range(2, 6)
    )
    fresh_outputs = tuple(
        monitor.update(
            _input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
                time_sec=time_sec,
            )
        )
        for frame_index, time_sec in ((6, 1.5), (7, 2.0), (8, 4.5))
    )

    # Then
    assert expired_outputs == ((), (), (), ())
    assert fresh_outputs[:2] == ((), ())
    assert len(fresh_outputs[2]) == 1


def test_missing_bed_roi_produces_zero_alerts_and_preserves_assignment() -> None:
    # Given
    monitor = _monitor(grace_frames=0)
    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
        )
    )
    # Arms at t=1.0 (dt=1.0 >= in_bed_dwell_sec=1.0).
    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=1,
            bed_pose_features=_lying_pose(),
        )
    )

    # When
    missing_roi = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(),
            track_ids=(PERSON_ID,),
            frame_index=2,
        )
    )
    fresh_roi = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=3,
            time_sec=2.5,
        )
    )

    # Then
    assert missing_roi == ()
    assert len(fresh_roi) == 1
    assert monitor.last_debug_snapshot is not None
    assert monitor.last_debug_snapshot.events == (bed_exit.BedExitEvent(PERSON_ID, 0),)


def test_early_return_reports_zero_shadow_snapshots_as_authoritative() -> None:
    """bed_exit has no shadow decision path (the shadow state machine that
    used to record separate "shadow" rows -- including a
    ``bed-polygon-invalid`` row -- was deleted entirely, see
    worker/domains/bed_exit/detector.py's ``last_shadow_trace_count``
    docstring). Every trace snapshot this monitor produces is authoritative,
    so the count must always read zero -- including on the frame where the
    bed region itself is unusable and ``update()`` returns early. A stale
    nonzero shadow count here would mislabel that single "bed unavailable"
    row as a shadow row, which the runbook tells operators to ignore."""
    from worker.pipeline.decision import EventAggregator
    from worker.pipeline.decision.incident_manager import IncidentManager

    monitor = _monitor()
    person = BoundingBox(10, 10, 30, 40, 0.9)

    # An ordinary contained frame first, to prove the count is zero on the
    # common path too, not only vacuously on the very first call.
    monitor.update(
        _input(person_boxes=(person,), bed_boxes=(BED_A,), track_ids=(1,), frame_index=0)
    )
    assert monitor.last_shadow_trace_count == 0

    # Frame with no bed boxes: early return. Must still report zero, or the
    # single authoritative unavailability row would be labelled shadow.
    monitor.update(_input(person_boxes=(person,), bed_boxes=(), track_ids=(1,), frame_index=1))
    assert monitor.last_shadow_trace_count == 0
    assert len(monitor.last_trace_snapshots) == 1
    aggregator = EventAggregator(deciders=(monitor,), incidents=IncidentManager())
    (attributed,) = aggregator.attributed_trace_snapshots()
    assert attributed.authority == "authoritative"
    assert attributed.snapshot.reason in ("bed-region-unavailable", "bed-observation-missing")


def _drive_to_onset(monitor: bed_exit.BedExitMonitor) -> tuple[BusinessEvent, ...]:
    """Assign, arm via a posture-confirmed in-bed dwell, then exit.

    Matches `_monitor()`'s default `in_bed_dwell_sec=outside_dwell_sec=1.0`:
    frame 0 assigns, frame 1 (1s later, lying/sitting posture) arms, frame 2
    (another 1s later, outside) fires exactly one event.
    """
    monitor.update(
        _input(person_boxes=(IN_BED_A,), bed_boxes=(BED_A,), track_ids=(PERSON_ID,), frame_index=0)
    )
    monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=1,
            bed_pose_features=_lying_pose(),
        )
    )
    return monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,), bed_boxes=(BED_A,), track_ids=(PERSON_ID,), frame_index=2
        )
    )


def _authoritative_rows(monitor: bed_exit.BedExitMonitor) -> tuple[DecisionTraceSnapshot, ...]:
    cut = len(monitor.last_trace_snapshots) - monitor.last_shadow_trace_count
    return monitor.last_trace_snapshots[:cut]


def test_onset_repeat_after_exit_is_explained_by_hysteresis_not_silent() -> None:
    """TC3-02 (superseded): firing an exit clears the armed latch (hysteresis,
    addendum #1) -- the very next frame at the same outside position can
    never recompute a second trigger without a fresh, positively-observed
    in-bed dwell, so it never even reaches the episode authority. The row
    must still carry triggered=False and name the hysteresis reason
    (outside-not-armed), not read as a silently dropped repeat."""
    monitor = _monitor(grace_frames=2)
    onset = _drive_to_onset(monitor)
    assert len(onset) == 1
    fired = [r for r in _authoritative_rows(monitor) if r.track_id == PERSON_ID]
    assert any(r.triggered for r in fired), "the onset frame itself keeps triggered=True"

    repeated = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,), bed_boxes=(BED_A,), track_ids=(PERSON_ID,), frame_index=3
        )
    )
    assert repeated == ()
    rows = [r for r in _authoritative_rows(monitor) if r.track_id == PERSON_ID]
    assert rows
    for row in rows:
        assert row.triggered is False
        assert row.reason == "outside-not-armed"


def test_onset_outside_night_window_is_an_explicit_non_event() -> None:
    """TC3-03: with the clock outside the internal NightWindow the monitor
    computes the onset but emits nothing. The row must say
    outside-detection-window with triggered=False."""
    monitor = _monitor(grace_frames=2)
    monitor._clock = _clock_at(hour=12)  # noqa: SLF001 - outside 21:00-05:00
    onset = _drive_to_onset(monitor)
    assert onset == ()
    rows = [r for r in _authoritative_rows(monitor) if r.track_id == PERSON_ID]
    assert rows
    assert all(r.triggered is False for r in rows)
    assert any(r.reason == "outside-detection-window" for r in rows)
