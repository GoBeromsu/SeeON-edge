"""``BedExitMonitor``'s optional scoring recorder (issue #238).

Mirrors ``test_worker_composite_bed_region_diagnostics.py``'s shape: `update()`
hands the monitor's own cumulative-since-boot scoring state to an injected
`scoring_recorder` (a structural match for
`WorkerDiagnostics.record_bed_exit_scoring`), when one is present. Without one
(`scoring_recorder=None`, the default) nothing changes. This closes the gap
#224 left open -- `BedRegionDiagnostics` only says whether the bed region was
usable, never what `BedExitMonitor` did with it once it was, so a
zero-bed_exit-events night was indistinguishable between (b) "person never
scored inside the polygon" and (c) "scored inside, but the exit counter never
crossed the grace threshold".
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
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.types import DecisionInput

CAMERA_ID: Final = "camera-bed-exit-scoring"
FACILITY_ID: Final = "facility-bed-exit-scoring"
PERSON_ID: Final = 3
BED_A: Final = BoundingBox(0, 0, 80, 100, 0.99)
IN_BED_A: Final = BoundingBox(10, 10, 70, 90, 0.95)
OUTSIDE_BEDS: Final = BoundingBox(40, 120, 100, 190, 0.94)


def _clock_at(hour: int = 22, minute: int = 0) -> Callable[[], datetime]:
    fixed = datetime(2026, 7, 31, hour, minute, tzinfo=ZoneInfo("Asia/Seoul"))
    return lambda: fixed


def _monitor(
    *,
    hold_frames: int = 1,
    grace_frames: int = 2,
    scoring_recorder: bed_exit.BedExitScoringRecorder | None = None,
) -> bed_exit.BedExitMonitor:
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            min_containment=0.5,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=_clock_at(),
        scoring_recorder=scoring_recorder,
    )


def _input(
    *,
    person_boxes: tuple[BoundingBox, ...],
    bed_boxes: tuple[BoundingBox, ...],
    track_ids: tuple[int | None, ...],
    frame_index: int,
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
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def test_update_without_a_recorder_does_not_crash() -> None:
    """The default (``scoring_recorder=None``) composition is unchanged."""
    monitor = _monitor()

    result = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
        )
    )

    assert result == ()


def test_never_near_a_bed_reports_near_zero_containment_and_no_assignment() -> None:
    """Signal (b): max containment stays at 0, nothing is ever assigned."""
    diagnostics = WorkerDiagnostics()
    monitor = _monitor(scoring_recorder=diagnostics)

    for frame_index in range(3):
        _ = monitor.update(
            _input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
            )
        )

    scoring = diagnostics.bed_exit_scoring_selection(CAMERA_ID)
    assert scoring is not None
    assert scoring.max_containment_observed == 0.0
    assert scoring.assignments_made == 0
    assert scoring.grace_positive_transitions == 0


def test_assignment_and_exit_are_both_reflected_cumulatively() -> None:
    """Signal (c): scored inside, assigned, then a full grace window recorded.

    Mirrors ``test_own_bed_exit_emits_once_after_grace_period`` in
    tests/test_worker_domains_bed_exit.py's frame sequence, but reads the
    scoring recorder instead of the returned events.
    """
    diagnostics = WorkerDiagnostics()
    monitor = _monitor(grace_frames=2, scoring_recorder=diagnostics)

    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
        )
    )
    after_assignment = diagnostics.bed_exit_scoring_selection(CAMERA_ID)
    assert after_assignment is not None
    assert after_assignment.max_containment_observed == 1.0
    assert after_assignment.assignments_made == 1
    assert after_assignment.grace_positive_transitions == 0

    for frame_index in (1, 2, 3):
        _ = monitor.update(
            _input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
            )
        )

    after_exit = diagnostics.bed_exit_scoring_selection(CAMERA_ID)
    assert after_exit is not None
    # Cumulative-since-boot (matches `BedRegionCacheCounterSnapshot`'s
    # precedent): the max containment from frame 0 is still visible even
    # though the person has since left, and `assignments_made` does not
    # reset just because the assignment was cleared after the exit fired.
    assert after_exit.max_containment_observed == 1.0
    assert after_exit.assignments_made == 1
    # One 0 -> 1 entry into the grace window, not one per off-bed frame.
    assert after_exit.grace_positive_transitions == 1
