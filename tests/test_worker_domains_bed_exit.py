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
from worker.types import BusinessEvent, DecisionInput

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
) -> bed_exit.BedExitMonitor:
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=camera_id,
            facility_id=FACILITY_ID,
            min_containment=0.5,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=_clock_at(),
    )


def _input(
    *,
    person_boxes: tuple[BoundingBox, ...],
    bed_boxes: tuple[BoundingBox, ...],
    track_ids: tuple[int | None, ...],
    frame_index: int,
    source: BedRegionCacheState = BedRegionCacheState.FRESH,
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
        bed_region=BedRegionDebugSnapshot(
            source=source,
            age_frames=0 if source is BedRegionCacheState.FRESH else 99,
        ),
    )


def test_own_bed_exit_emits_once_after_grace_period() -> None:
    # Given
    monitor = _monitor(grace_frames=2)
    assert monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
        )
    ) == ()

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
        for frame_index in (1, 2)
    )
    onset = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=3,
        )
    )
    repeated = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=4,
        )
    )

    # Then
    assert before_grace == ((), ())
    assert onset == (
        BusinessEvent(
            domain="bed_exit",
            event_type="bed-exit",
            identity="7:0",
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            time_sec=3.0,
            probability=1.0,
            person_id=PERSON_ID,
            bed_id=0,
        ),
    )
    assert repeated == ()


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
    monitor = _monitor(grace_frames=2)
    _ = monitor.update(
        _input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=0,
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
        for frame_index in range(1, 5)
    )
    fresh_outputs = tuple(
        monitor.update(
            _input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(PERSON_ID,),
                frame_index=frame_index,
            )
        )
        for frame_index in range(5, 8)
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

    # When
    missing_roi = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(),
            track_ids=(PERSON_ID,),
            frame_index=1,
        )
    )
    fresh_roi = monitor.update(
        _input(
            person_boxes=(OUTSIDE_BEDS,),
            bed_boxes=(BED_A,),
            track_ids=(PERSON_ID,),
            frame_index=2,
        )
    )

    # Then
    assert missing_roi == ()
    assert len(fresh_roi) == 1
    assert monitor.last_debug_snapshot is not None
    assert monitor.last_debug_snapshot.events == (bed_exit.BedExitEvent(PERSON_ID, 0),)
