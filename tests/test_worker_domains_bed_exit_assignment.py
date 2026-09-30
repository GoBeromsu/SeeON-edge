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
from tests_support.bed_pose_fixtures import frame_pose_features, lying_in_bed
from worker.domains import bed_exit
from worker.types import DecisionInput
from worker.types.bed_pose_features import EMPTY_FRAME_BED_POSE_FEATURES, FrameBedPoseFeatures

PERSON_ID: Final = 7
BED: Final = BoundingBox(0, 0, 80, 100, 0.99)
IN_BED: Final = BoundingBox(10, 10, 70, 90, 0.95)
OUTSIDE_BED: Final = BoundingBox(100, 10, 160, 90, 0.94)


def _monitor(
    *,
    camera_id: str,
    hold_frames: int = 1,
    grace_frames: int = 0,
    in_bed_dwell_sec: float = 1.0,
    outside_dwell_sec: float = 1.0,
) -> bed_exit.BedExitMonitor:
    fixed = datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    return bed_exit.BedExitMonitor(
        config=bed_exit.BedExitConfig(
            camera_id=camera_id,
            facility_id="facility-bed-exit",
            min_containment=0.5,
            hold_frames=hold_frames,
            grace_frames=grace_frames,
            in_bed_dwell_sec=in_bed_dwell_sec,
            outside_dwell_sec=outside_dwell_sec,
            night_window=bed_exit.NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=lambda: fixed,
        boot_id=f"boot-{camera_id}",
        stream_epoch=f"epoch-{camera_id}",
        source_generation=0,
    )


def _input(
    person: BoundingBox,
    beds: tuple[BoundingBox, ...],
    frame_index: int,
    *,
    live_track_ids: tuple[int, ...] = (PERSON_ID,),
    bed_pose_features: FrameBedPoseFeatures = EMPTY_FRAME_BED_POSE_FEATURES,
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
        bed_pose_features=bed_pose_features,
    )


def _lying_pose() -> FrameBedPoseFeatures:
    return frame_pose_features(lying_in_bed(track_id=PERSON_ID, bed_id=0))


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
    _ = camera_a.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose()))

    # When
    camera_b_events = camera_b.update(_input(OUTSIDE_BED, (BED,), 2))
    camera_a_events = camera_a.update(_input(OUTSIDE_BED, (BED,), 2))

    # Then
    assert camera_b_events == ()
    assert len(camera_a_events) == 1
    assert camera_a_events[0].camera_id == "camera-a"


def test_track_lost_mid_grace_never_emits_on_absence() -> None:
    """Absence must never emit (the firehose's dominant cause, issue #246).

    Supersedes `test_track_lost_mid_exit_still_fires_the_event`, which
    asserted the old absence-emit bug -- a track disappearing mid-departure
    firing a bed-exit event -- as desired behavior. Under the dwell model a
    track that simply vanishes, however far its dwell timers had climbed,
    must retire silently: zero events, no matter how many grace frames are
    configured.
    """
    # Given
    monitor = _monitor(camera_id="camera-lost-mid-exit", hold_frames=1, grace_frames=2)
    _ = monitor.update(_input(IN_BED, (BED,), 0))
    _ = monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose()))
    _ = monitor.update(_input(OUTSIDE_BED, (BED,), 2))

    # When
    events = monitor.update(_input(OUTSIDE_BED, (BED,), 3, live_track_ids=()))

    # Then
    assert events == ()


def test_single_sub_threshold_frame_before_track_loss_no_longer_fires() -> None:
    """Supersedes `test_single_sub_threshold_frame_before_track_loss_currently_fires`.

    That test named a deliberately-accepted false positive (see #246) --
    a single sub-threshold containment frame immediately followed by track
    death firing a bed-exit event, because the stale-track gate was
    `grace_frames > 0` -- as desired behavior. That trade-off no longer
    exists: absence never emits, so a single sub-threshold frame followed
    by track death must produce zero events regardless of grace_frames.
    """
    # Given: a person's bed assignment sees exactly ONE frame of
    # sub-threshold containment right before their track is lost.
    monitor = _monitor(camera_id="camera-single-frame-jitter", hold_frames=1, grace_frames=3)
    _ = monitor.update(_input(IN_BED, (BED,), 0))
    _ = monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose()))
    _ = monitor.update(_input(OUTSIDE_BED, (BED,), 2))

    # When: the track dies immediately after that one sub-threshold frame.
    events = monitor.update(_input(OUTSIDE_BED, (BED,), 3, live_track_ids=()))

    # Then: absence never emits.
    assert events == ()


def test_dead_observed_track_cannot_emit_after_identity_reuse() -> None:
    # Given
    monitor = _monitor(camera_id="camera-reused-track")
    _ = monitor.update(_input(IN_BED, (BED,), 0))

    # When
    dead_track = monitor.update(_input(IN_BED, (BED,), 1, live_track_ids=()))
    reused_identity = monitor.update(_input(OUTSIDE_BED, (BED,), 2))

    # Then
    assert dead_track == reused_identity == ()


def test_a_released_bed_exit_does_not_rearm_on_track_loss() -> None:
    """Track loss must never resurrect a released bed-exit episode.

    Supersedes `test_a_released_stale_track_exit_does_not_rearm_on_track_
    loss`: that test's premise -- a vanishing track "firing" the exit it
    then released -- is now architecturally impossible (absence never
    emits). The property worth keeping is real: a genuine, posture-armed
    exit that gets released for retry must not be reopened by the same
    track simply disappearing afterward.
    """
    from worker.pipeline.decision.event_aggregator import EventAggregator
    from worker.pipeline.decision.incident_manager import IncidentManager

    monitor = _monitor(camera_id="camera-lost-mid-exit", hold_frames=1, grace_frames=2)
    aggregator = EventAggregator(deciders=(monitor,), incidents=IncidentManager(cooldown_sec=300.0))
    aggregator.update(_input(IN_BED, (BED,), 0))
    aggregator.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose()))

    exited = aggregator.update(_input(OUTSIDE_BED, (BED,), 2))
    assert len(exited) == 1, "the armed exit was never reported"

    # The envelope failed to reach durable storage.
    aggregator.release(exited[0])

    # The track then disappears entirely -- absence must not resurrect it.
    repeated = aggregator.update(_input(OUTSIDE_BED, (BED,), 3, live_track_ids=()))

    assert repeated == ()


def test_track_id_churn_during_continuous_occupancy_preserves_arm_progress() -> None:
    """A track ID churning mid-occupancy must not restart the dwell clock.

    NvDCF runs without ReID; median track lifetime measures well under a
    typical `in_bed_dwell_sec` on several cameras. If dwell state were purely
    track-ID-keyed, a resident lying continuously still could rarely
    accumulate enough dwell under any single ID to ever arm. When a track
    goes stale while a brand-new, never-before-assigned track is this same
    frame independently re-confirmed (containment + posture) inside the
    identical bed, that is occupancy evidence of whoever is in the polygon,
    not of a specific track ID -- the hand-off carries the armed/in-bed-dwell
    progress forward instead of discarding it. The hand-off never grants
    exit evidence by itself: the new track still has to independently
    re-earn the posture gate every frame, and outside-dwell progress is
    never carried across the identity gap.
    """
    SUCCESSOR_ID: Final = 8

    # Given: PERSON_ID arms by lying in bed for a full in_bed_dwell_sec.
    monitor = _monitor(camera_id="camera-churn", hold_frames=1)
    assert monitor.update(_input(IN_BED, (BED,), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose())) == ()
    assert monitor._assignments[PERSON_ID].armed is True  # noqa: SLF001

    # When: PERSON_ID's track vanishes and a brand-new ID appears this same
    # frame, still lying in the identical bed -- a pure identity swap, no
    # new evidence of an exit.
    churned = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((IN_BED,), ()),
                regions=((BED,), ()),
                track_ids=(SUCCESSOR_ID,),
            ),
            frame_width=180,
            frame_height=120,
            live_track_ids=(SUCCESSOR_ID,),
            time_sec=2.0,
            frame_index=2,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
            bed_pose_features=frame_pose_features(lying_in_bed(track_id=SUCCESSOR_ID, bed_id=0)),
        )
    )

    # Then: the churn itself never fires, and the successor inherits the
    # armed latch instead of starting a fresh dwell cycle from zero.
    assert churned == ()
    assert PERSON_ID not in monitor._assignments  # noqa: SLF001
    assert monitor._assignments[SUCCESSOR_ID].armed is True  # noqa: SLF001

    # When: the successor is then observed outside for a single dwell period.
    exited = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((OUTSIDE_BED,), ()),
                regions=((BED,), ()),
                track_ids=(SUCCESSOR_ID,),
            ),
            frame_width=180,
            frame_height=120,
            live_track_ids=(SUCCESSOR_ID,),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    # Then: one dwell period suffices -- the successor never had to re-earn
    # its own full in_bed_dwell_sec from zero, proving continuity survived
    # the identity churn.
    assert len(exited) == 1
    assert exited[0].person_id == SUCCESSOR_ID
    assert exited[0].bed_id == 0


def test_track_id_churn_during_outside_dwell_carries_progress_and_fires_once() -> None:
    """An ID switch mid-exit must carry outside-dwell progress forward.

    Finding #2 (PR #588 review): NvDCF's median track life is well under a
    typical `outside_dwell_sec` on several cameras, so a resident who is
    already outside when their track ID churns would otherwise lose all
    accumulated outside-dwell evidence and never complete the exit. A
    never-before-assigned live track that, this same frame, is contained in
    neither the vacated bed nor any other bed is evidence that whoever left
    is still out, so it inherits the stale assignment's `armed` state and
    `outside_dwell_sec` instead of starting over from zero. This supersedes
    the old `test_new_track_appearing_beside_an_unrelated_stale_exit_does_not_complete_it`,
    whose premise -- that a bare "new ID" at the vacated position can never
    be exit evidence -- is exactly the gap this fix closes.
    """
    monitor = _monitor(camera_id="camera-unrelated-churn", hold_frames=1, outside_dwell_sec=2.0)
    assert monitor.update(_input(IN_BED, (BED,), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose())) == ()

    # When: PERSON_ID walks outside (armed, only halfway through
    # outside_dwell_sec), then vanishes mid-departure while a new track ID
    # appears this same frame at the same outside position.
    assert monitor.update(_input(OUTSIDE_BED, (BED,), 2)) == ()
    successor_id = 9
    events = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((OUTSIDE_BED,), ()),
                regions=((BED,), ()),
                track_ids=(successor_id,),
            ),
            frame_width=180,
            frame_height=120,
            live_track_ids=(successor_id,),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    # Then: 1.0s of outside evidence before the churn plus 1.0s after crosses
    # the 2.0s threshold and fires exactly once, for the successor.
    assert len(events) == 1
    assert events[0].person_id == successor_id
    assert events[0].bed_id == 0
    assert PERSON_ID not in monitor._assignments  # noqa: SLF001
    assert monitor._assignments[successor_id].armed is False  # noqa: SLF001
    assert monitor._assignments[successor_id].outside_dwell_sec == 0.0  # noqa: SLF001


def test_new_track_contained_in_a_different_bed_does_not_inherit_outside_dwell() -> None:
    """A new track accounted for by a different bed must not inherit.

    The outside-dwell hand-off only fires for a track positively unaccounted
    for by any bed. A new track that is instead contained in a second,
    different bed is evidence of that bed's own occupant, not of the
    vacating resident continuing their exit, so it must start with no
    inherited armed/dwell state.
    """
    bed_two: Final = BoundingBox(200, 10, 260, 90, 0.94)
    monitor = _monitor(camera_id="camera-two-bed-churn", hold_frames=1, outside_dwell_sec=2.0)
    assert monitor.update(_input(IN_BED, (BED, bed_two), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED, bed_two), 1, bed_pose_features=_lying_pose())) == ()
    assert monitor.update(_input(OUTSIDE_BED, (BED, bed_two), 2)) == ()

    other_id = 9
    events = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((bed_two,), ()),
                regions=((BED, bed_two), ()),
                track_ids=(other_id,),
            ),
            frame_width=300,
            frame_height=120,
            live_track_ids=(other_id,),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    assert events == ()
    assert PERSON_ID not in monitor._assignments  # noqa: SLF001
    assert monitor._assignments[other_id].armed is False  # noqa: SLF001
    assert monitor._assignments[other_id].outside_dwell_sec == 0.0  # noqa: SLF001


def test_unrelated_new_track_far_away_does_not_inherit_outside_dwell() -> None:
    """A new ID appearing elsewhere in frame must never, by itself, inherit.

    Restores (correctly scoped) the negative case dropped when the
    outside-dwell hand-off was added: a caregiver arriving at the door while
    the resident is genuinely still outside is another live, bed-unclaimed
    body, but it never overlaps where the resident was actually last seen.
    Finding (independent review, PR #588): the hand-off previously picked
    the first unclaimed, bed-unaccounted-for track with no regard for
    position, so a caregiver merely being "the only other body in frame"
    was enough to inherit the resident's outside-dwell progress and fire a
    false exit. The hand-off now also requires spatial overlap with the
    vacating track's last observed box.
    """
    door: Final = BoundingBox(300, 10, 360, 90, 0.9)
    monitor = _monitor(camera_id="camera-caregiver-at-door", hold_frames=1, outside_dwell_sec=2.0)
    assert monitor.update(_input(IN_BED, (BED,), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose())) == ()

    # When: PERSON_ID walks outside (armed, only halfway through
    # outside_dwell_sec) then vanishes entirely, while an unrelated
    # caregiver ID appears at the door -- far from where PERSON_ID was last
    # seen and not contained in any bed.
    assert monitor.update(_input(OUTSIDE_BED, (BED,), 2)) == ()
    caregiver_id = 9
    events = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((door,), ()),
                regions=((BED,), ()),
                track_ids=(caregiver_id,),
            ),
            frame_width=400,
            frame_height=120,
            live_track_ids=(caregiver_id,),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    # Then: no event, and the caregiver starts with no inherited state.
    assert events == ()
    assert PERSON_ID not in monitor._assignments  # noqa: SLF001
    assert monitor._assignments[caregiver_id].armed is False  # noqa: SLF001
    assert monitor._assignments[caregiver_id].outside_dwell_sec == 0.0  # noqa: SLF001


def test_outside_handoff_blocked_when_the_vacated_bed_is_reoccupied() -> None:
    """A track re-occupying the vacated bed must block any hand-off elsewhere.

    Finding (independent review, PR #588): reproduced as a caregiver
    entering at the door inheriting the resident's outside-dwell while the
    resident is actually back in bed under a new, posture-unconfirmed ID --
    the bed-exit alert fired anyway. Even before the reentrant track has
    held containment long enough to arm (posture not yet confirmed this
    frame), the bed being physically occupied again is itself proof the
    departure never became an exit, so no one else in frame -- including an
    unrelated caregiver at the door -- may be handed the vacating resident's
    progress.
    """
    door: Final = BoundingBox(300, 10, 360, 90, 0.9)
    monitor = _monitor(
        camera_id="camera-reentry-blocks-handoff", hold_frames=1, outside_dwell_sec=2.0
    )
    assert monitor.update(_input(IN_BED, (BED,), 0)) == ()
    assert monitor.update(_input(IN_BED, (BED,), 1, bed_pose_features=_lying_pose())) == ()
    assert monitor.update(_input(OUTSIDE_BED, (BED,), 2)) == ()

    # When: PERSON_ID vanishes mid-departure while, this same frame, a new ID
    # re-occupies the bed (posture not yet confirmed) and an unrelated
    # caregiver ID appears at the door.
    reentry_id = 10
    door_id = 11
    events = monitor.update(
        DecisionInput(
            observation=FrameObservation(
                detections=((IN_BED, door), ()),
                regions=((BED,), ()),
                track_ids=(reentry_id, door_id),
            ),
            frame_width=400,
            frame_height=120,
            live_track_ids=(reentry_id, door_id),
            time_sec=3.0,
            frame_index=3,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )
    )

    # Then: no event, and neither new track inherits armed/outside state.
    assert events == ()
    assert PERSON_ID not in monitor._assignments  # noqa: SLF001
    assert monitor._assignments[reentry_id].armed is False  # noqa: SLF001
    assert monitor._assignments[door_id].armed is False  # noqa: SLF001
    assert monitor._assignments[door_id].outside_dwell_sec == 0.0  # noqa: SLF001
