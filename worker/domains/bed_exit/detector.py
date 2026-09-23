from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from time import monotonic
from typing import Protocol

from contracts.observation import BedRegionCacheState
from worker.domains.bed_exit.geometry import best_bed_id, containment_ratio
from worker.domains.bed_exit.latch import BedExitLatch
from worker.domains.bed_exit.night_window import NightWindow
from worker.domains.bed_exit.schema import (
    BedExitConfig,
    BedExitDebugSnapshot,
    BedExitEvent,
    BedExitFrame,
    BedStatus,
)
from worker.domains.episode import EpisodeAuthority, EpisodeProposal, suppression_reason
from worker.domains.staleness import DEFAULT_STALE_AFTER_SEC
from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot
from worker.types.bed_pose_features import BedPoseFeatures
from worker.types.trace import DecisionTraceReason

_LOGGER = logging.getLogger(__name__)

# Anti-occlusion guard before trusting posture fields (ported from the
# deleted shadow state machine, worker/domains/bed_exit/state_machine.py,
# which measured this threshold against real camera views).
_MIN_OBSERVABILITY = 0.35

# hip_depth sign/magnitude convention measured by the deleted shadow state
# machine: positive = lying/sitting weight on the mattress (IN_BED +0.257,
# SITTING_UP +0.236, EDGE_SITTING +0.043); negative = upright/standing
# (OUT_OF_BED -0.289). 0.10 sits strictly between EDGE_SITTING and
# SITTING_UP -- rim-perched is excluded (the state machine treated
# EDGE_SITTING as exit-eligible, never arming), genuine lying/sitting is
# included, and a standing caregiver's negative hip_depth is well clear of
# either side. torso_angle is NOT used: the deleted state machine's own
# measurements showed it reads ~pi/2 for every posture and does not
# discriminate.
_MIN_IN_BED_HIP_DEPTH = 0.10


class BedExitScoringRecorder(Protocol):
    """Structural view of ``WorkerDiagnostics.record_bed_exit_scoring()``.

    Kept narrow for the same layering reason as
    ``worker/pipeline/analytics/composite.py``'s ``BedRegionRecorder``
    (issue #238): the domain layer depends on this shape instead of
    importing ``worker.runtime.telemetry.runtime_diagnostics.WorkerDiagnostics``
    directly.
    """

    def record_bed_exit_scoring(
        self,
        camera_id: str,
        max_containment_observed: float,
        grace_positive_transitions: int,
        assignments_made: int,
    ) -> None: ...


class _Assignment:
    __slots__: tuple[str, ...] = (
        "armed",
        "bed_id",
        "candidate_bed_id",
        "candidate_frames",
        "in_bed_dwell_sec",
        "last_time_sec",
        "outside_dwell_sec",
    )

    def __init__(self) -> None:
        self.bed_id: int | None = None
        self.candidate_bed_id: int | None = None
        self.candidate_frames: int = 0
        # One-way-per-cycle latch: only a continuous, posture-confirmed,
        # spatially-contained dwell in this bed sets it True (arms an exit).
        # Cleared only by firing an exit or by track loss -- never by a mere
        # dip below dwell -- so a resident must fully re-earn "was in bed"
        # before the same track can exit again (hysteresis; prevents
        # oscillating containment from re-firing, issue addendum #1).
        self.in_bed_dwell_sec: float = 0.0
        self.outside_dwell_sec: float = 0.0
        self.armed: bool = False
        self.last_time_sec: float | None = None

    def update_candidate(self, bed_id: int | None) -> None:
        if bed_id is None:
            self.candidate_bed_id = None
            self.candidate_frames = 0
            return
        if self.candidate_bed_id == bed_id:
            self.candidate_frames += 1
            return
        self.candidate_bed_id = bed_id
        self.candidate_frames = 1


class BedExitMonitor:
    """Interpret numeric observations with camera-local bed assignment state.

    Exit requires positive evidence, never absence: a track must be observed
    lying/sitting in its own bed for ``config.in_bed_dwell_sec`` (posture-
    confirmed via pose keypoints, not bbox containment alone -- a standing
    caregiver's bbox can reach containment without ever lying down) before
    it is armed, and then observed outside that bed's polygon continuously
    for ``config.outside_dwell_sec`` on live frames before firing. A track
    disappearing (occlusion, tracker drop, frame edge) never emits by
    itself; dwell is measured via ``DecisionInput.time_sec`` (PTS), never
    frame counts, so behavior is invariant to ingest fps.
    """

    def __init__(
        self,
        *,
        config: BedExitConfig,
        clock: Callable[[], datetime],
        scoring_recorder: BedExitScoringRecorder | None = None,
        staleness_clock: Callable[[], float] = monotonic,
        stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
        boot_id: str,
        stream_epoch: str,
        source_generation: int,
    ) -> None:
        self._config: BedExitConfig = config
        self._clock: Callable[[], datetime] = clock
        self._night_window: NightWindow | None = config.night_window
        self._assignments: dict[int, _Assignment] = {}
        self._latch = BedExitLatch(
            clock=staleness_clock,
            stale_after_sec=stale_after_sec,
        )
        if (
            not boot_id
            or not stream_epoch
            or isinstance(source_generation, bool)
            or source_generation < 0
        ):
            raise ValueError("bed-exit event identities must name a boot and source epoch")
        self._episodes = EpisodeAuthority(
            boot_id=boot_id,
            stream_epoch=stream_epoch,
            source_generation=source_generation,
        )
        self._recovery_events: list[BedExitEvent] = []
        self._lost_track_ids: list[int] = []
        self.last_debug_snapshot: BedExitDebugSnapshot | None = None
        self.last_trace_snapshots: tuple[DecisionTraceSnapshot, ...] = ()
        # bed_exit has no shadow decision path (the shadow state machine was
        # deleted: an equally-bad absence bug, and this containment path now
        # carries the posture gate directly). Always zero -- generic infra
        # (worker/interfaces/decision.py's `ShadowTraceProvider`, consumed by
        # worker/pipeline/decision/event_aggregator.py) still duck-types this
        # attribute, so it must keep existing and truthfully report "no
        # shadow path" rather than being removed.
        self.last_shadow_trace_count: int = 0
        self._scoring_recorder = scoring_recorder
        # Cumulative-since-boot, matching `StageTimingAccumulator.max_sec` and
        # `BedRegionCacheCounterSnapshot`'s precedent elsewhere in this
        # codebase -- never reset per `RuntimeStatusSender` tick. Distinguishes
        # (b) "never scored inside the polygon" from (c) "scored inside, but
        # the exit counter never crossed the grace threshold" when bed_exit
        # fires zero events overnight (issue #238); #224's `BedRegionDiagnostics`
        # only covers whether the region itself was usable, not what this
        # monitor did with it once it was.
        self._max_containment_observed: float = 0.0
        self._grace_positive_transitions: int = 0
        self._assignments_made: int = 0

    @property
    def track_id_switch_absorbed_total(self) -> int:
        """Re-associations the episode authority absorbed instead of re-alerting."""
        return self._episodes.track_id_switch_absorbed_total

    @property
    def config(self) -> BedExitConfig:
        return self._config

    def update_night_window(self, night_window: NightWindow | None) -> None:
        self._night_window = night_window

    def release_onset(self, event: BusinessEvent) -> None:
        """Reopen only the exact bed-exit onset that failed durable staging."""
        self._episodes.release(event)

    def coast(self, *, frame_index: int | None = None) -> tuple[BusinessEvent, ...]:
        """Hold assignment/latch state when no person inference was made."""
        self._latch.coast()
        freshness = self._latch.status_snapshot
        previous = self.last_debug_snapshot
        statuses = (
            ()
            if previous is None
            else tuple(
                BedStatus(
                    bed_id=status.bed_id,
                    box=status.box,
                    occupancy="covered",
                    person_id=status.person_id,
                )
                for status in previous.statuses
            )
        )
        self.last_debug_snapshot = BedExitDebugSnapshot(
            frame_index=frame_index,
            person_boxes=() if previous is None else previous.person_boxes,
            bed_boxes=() if previous is None else previous.bed_boxes,
            statuses=statuses,
            events=(),
            bed_region=None if previous is None else previous.bed_region,
            stale=freshness.stale,
            observation_age_sec=freshness.observation_age_sec,
        )
        return ()

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        observation = input_value.observation
        if not _bed_region_is_usable(input_value.bed_region.source) or not observation.bed_boxes:
            reason = (
                "bed-region-unavailable"
                if not _bed_region_is_usable(input_value.bed_region.source)
                else "bed-observation-missing"
            )
            self.last_trace_snapshots = (
                DecisionTraceSnapshot(
                    reason=reason,
                    previous_state="unknown",
                    current_state="no-decision",
                    triggered=False,
                    track_id=None,
                    bed_id=None,
                    missing_values={
                        "containment_ratio": reason,
                        "bed_id": reason,
                    },
                ),
            )
            # No shadow path exists at all: every snapshot here is
            # authoritative, so the trailing-shadow count must be zero.
            self.last_shadow_trace_count = 0
            self.last_debug_snapshot = BedExitDebugSnapshot(
                frame_index=input_value.frame_index,
                person_boxes=observation.boxes,
                bed_boxes=(),
                statuses=(),
                events=(),
                bed_region=input_value.bed_region,
            )
            return ()

        frame = self._update_frame(input_value)
        if self._scoring_recorder is not None:
            # Same discipline as `record_bed_region` (#207/#224): this only
            # overwrites an in-memory value on the existing per-frame call
            # path -- no new thread, timer, or per-frame I/O. Actual emission
            # is on `log_snapshot()`'s ~5s `RuntimeStatusSender` cadence.
            # Telemetry, never detection. This call sits directly before the
            # onset latch, so a raising recorder discarded the bed-exit event
            # itself. Five other auxiliary capabilities in this runtime were
            # found holding that same power over a resident alert.
            try:
                self._scoring_recorder.record_bed_exit_scoring(
                    self._config.camera_id,
                    self._max_containment_observed,
                    self._grace_positive_transitions,
                    self._assignments_made,
                )
            except Exception:  # noqa: BLE001 - telemetry never blocks detection
                _LOGGER.warning(
                    "bed-exit scoring recorder failed for camera %s; detection continues",
                    self._config.camera_id,
                    exc_info=True,
                )
        event_time = 0.0 if input_value.time_sec is None else input_value.time_sec
        # The night window gates proposals, not freshness. The snapshot still
        # uses the real frame so the overlay renders `bed:exit`.
        in_window = self._night_window is None or self._night_window.contains(self._clock())
        self._latch.update()
        freshness = self._latch.status_snapshot
        self.last_debug_snapshot = BedExitDebugSnapshot(
            frame_index=input_value.frame_index,
            person_boxes=observation.boxes,
            bed_boxes=observation.bed_boxes,
            statuses=frame.statuses,
            events=frame.events,
            bed_region=input_value.bed_region,
            stale=freshness.stale,
            observation_age_sec=freshness.observation_age_sec,
        )
        if not in_window:
            # Onsets computed this frame are suppressed by the detection window.
            # The trace must say so: a triggered=True row with no event and no
            # reason would read as a delivery failure.
            self._mark_suppressed(
                track_ids={event.person_id for event in frame.events},
                reason=str(DecisionTraceReason.OUTSIDE_DETECTION_WINDOW),
            )
            return ()
        self._episodes.expire(frame_index=input_value.frame_index, time_sec=event_time)
        emitted: list[BusinessEvent] = []
        for event in frame.events:
            produced = self._episodes.propose(
                EpisodeProposal(
                    camera_id=self._config.camera_id,
                    facility_id=self._config.facility_id,
                    event_type="bed-exit",
                    track_id=event.person_id,
                    bed_id=event.bed_id,
                    frame_index=input_value.frame_index,
                    time_sec=event_time,
                    qualifying=True,
                    probability=1.0,
                    domain="bed_exit",
                    confirmation_votes=1,
                    confirmation_window=1,
                )
            )
            emitted.extend(produced)
            if not produced:
                # The episode authority declined this onset; name why. A None
                # mapping means it was not a suppression (cannot follow a
                # qualifying onset today) and the row keeps its own reason.
                reason = suppression_reason(self._episodes.last_disposition)
                if reason is not None:
                    self._mark_suppressed(track_ids={event.person_id}, reason=reason)
        # Track loss is decoupled from emission: absence never emits (a
        # stale/disappeared track produced no event above), but the episode
        # authority still needs to know an assigned identity is gone so an
        # already-OPEN episode can move to UNKNOWN for possible
        # re-association or eventual expiry, instead of staying silently
        # bound to a dead track id forever.
        for lost_id in self._lost_track_ids:
            self._episodes.track_lost(
                camera_id=self._config.camera_id,
                frame_index=input_value.frame_index,
                time_sec=event_time,
                track_id=lost_id,
            )
        for event in self._recovery_events:
            _ = self._episodes.propose(
                EpisodeProposal(
                    camera_id=self._config.camera_id,
                    facility_id=self._config.facility_id,
                    event_type="bed-exit",
                    track_id=event.person_id,
                    bed_id=event.bed_id,
                    frame_index=input_value.frame_index,
                    time_sec=event_time,
                    qualifying=False,
                    confirmed_recovery=True,
                    probability=1.0,
                    domain="bed_exit",
                )
            )
        return tuple(emitted)

    def _mark_suppressed(self, *, track_ids: set[int | None], reason: str) -> None:
        """Rewrite this frame's triggered snapshots for ``track_ids`` as suppressed.

        The row keeps its states and values; triggered flips to False and the
        reason names the suppression, so the non-event is explained rather
        than looking like a lost delivery.
        """
        if not track_ids:
            return
        rewritten: list[DecisionTraceSnapshot] = []
        for snapshot in self.last_trace_snapshots:
            if snapshot.triggered and snapshot.track_id in track_ids:
                rewritten.append(
                    DecisionTraceSnapshot(
                        reason=str(reason),
                        previous_state=snapshot.previous_state,
                        current_state=snapshot.current_state,
                        triggered=False,
                        track_id=snapshot.track_id,
                        bed_id=snapshot.bed_id,
                        values=dict(snapshot.values),
                        missing_values=dict(snapshot.missing_values),
                    )
                )
            else:
                rewritten.append(snapshot)
        self.last_trace_snapshots = tuple(rewritten)

    def _update_frame(self, input_value: DecisionInput) -> BedExitFrame:
        self._recovery_events = []
        self._lost_track_ids = []
        observation = input_value.observation
        has_track_ids = bool(observation.track_ids)
        if has_track_ids:
            person_ids = observation.track_ids
            live_ids = set(input_value.live_track_ids)
        else:
            person_ids = tuple(range(len(observation.boxes)))
            live_ids = set(person_ids)
        events: list[BedExitEvent] = []
        traces: list[DecisionTraceSnapshot] = []
        occupied: dict[int, int] = {}
        exit_beds: set[int] = set()
        by_track: dict[int, BedPoseFeatures] = {
            item.track_id: item for item in input_value.bed_pose_features.items
        }

        # NvDCF runs without ReID: median track lifetime measures well under
        # a typical in_bed_dwell_sec (and outside_dwell_sec) on several
        # cameras, so state keyed purely by track ID would rarely accumulate
        # enough dwell under any single ID to arm, or to complete an exit
        # already in progress. A never-before-assigned live track that, this
        # same frame, independently re-satisfies the identical bed's
        # containment+posture gate is occupancy evidence of whoever is in
        # the polygon, not of a specific track ID, so its armed/in-bed-dwell
        # progress hands off instead of resetting to zero. Symmetrically, a
        # never-before-assigned live track that is *not* contained in that
        # same bed (and not contained in any other bed either) is evidence
        # that whoever vacated it is still out, so an armed assignment's
        # outside_dwell_sec hands off the same way instead of being dropped
        # on the identity switch. Either hand-off requires the stale
        # assignment to have owned a bed already: a bare "new ID appeared"
        # is never by itself exit evidence.
        unclaimed_live = [
            (pid, box)
            for pid, box in zip(person_ids, observation.boxes, strict=True)
            if pid is not None and pid in live_ids and pid not in self._assignments
        ]

        def _handoff_recipient(bed_id: int) -> int | None:
            if bed_id >= len(observation.bed_boxes):
                return None
            bed_box = observation.bed_boxes[bed_id]
            best_pid: int | None = None
            best_ratio = self._config.min_containment
            for pid, box in unclaimed_live:
                features = by_track.get(pid)
                if (
                    features is None
                    or not features.bed_polygon_valid
                    or features.observability < _MIN_OBSERVABILITY
                    or features.hip_depth < _MIN_IN_BED_HIP_DEPTH
                ):
                    continue
                ratio = containment_ratio(box, bed_box)
                if ratio >= best_ratio:
                    best_ratio = ratio
                    best_pid = pid
            return best_pid

        def _outside_handoff_recipient(bed_id: int) -> int | None:
            if bed_id >= len(observation.bed_boxes):
                return None
            bed_box = observation.bed_boxes[bed_id]
            for pid, box in unclaimed_live:
                if containment_ratio(box, bed_box) >= self._config.min_containment:
                    continue
                if any(
                    other_id != bed_id
                    and containment_ratio(box, other_box) >= self._config.min_containment
                    for other_id, other_box in enumerate(observation.bed_boxes)
                ):
                    continue
                return pid
            return None

        # A track can vanish mid-exit (occlusion, tracker drop, walking out of
        # frame). Absence must never emit: an assignment that disappears is
        # simply retired, with no event, regardless of how far its dwell
        # timers had climbed. This is the fix for the firehose's dominant
        # cause -- the previous stale-track path fired on `grace_frames > 0`,
        # i.e. on any departure merely "in progress", including a single
        # noisy sub-threshold frame immediately followed by track death
        # (issue #246) and a resident simply lying still while briefly
        # unmatched by the tracker.
        for stale_id in sorted(set(self._assignments) - live_ids):
            assignment = self._assignments[stale_id]
            had_assignment = assignment.bed_id is not None
            recipient = _handoff_recipient(assignment.bed_id) if had_assignment else None
            carries_in_bed_dwell = recipient is not None
            if (
                recipient is None
                and had_assignment
                and assignment.armed
                and assignment.outside_dwell_sec > 0.0
            ):
                assert assignment.bed_id is not None
                recipient = _outside_handoff_recipient(assignment.bed_id)
            if recipient is not None:
                assert assignment.bed_id is not None
                successor = _Assignment()
                successor.bed_id = assignment.bed_id
                successor.candidate_bed_id = assignment.bed_id
                successor.candidate_frames = self._config.hold_frames
                successor.armed = assignment.armed
                successor.in_bed_dwell_sec = (
                    assignment.in_bed_dwell_sec if carries_in_bed_dwell else 0.0
                )
                successor.outside_dwell_sec = (
                    0.0 if carries_in_bed_dwell else assignment.outside_dwell_sec
                )
                successor.last_time_sec = assignment.last_time_sec
                self._assignments[recipient] = successor
                unclaimed_live = [
                    (pid, box) for pid, box in unclaimed_live if pid != recipient
                ]
                self._episodes.reassociate_bed_exit(
                    EpisodeProposal(
                        camera_id=self._config.camera_id,
                        facility_id=self._config.facility_id,
                        event_type="bed-exit",
                        track_id=recipient,
                        bed_id=successor.bed_id,
                        frame_index=input_value.frame_index,
                        time_sec=(
                            0.0 if input_value.time_sec is None else input_value.time_sec
                        ),
                        qualifying=False,
                        probability=1.0,
                        domain="bed_exit",
                    )
                )
                traces.append(
                    DecisionTraceSnapshot(
                        reason="identity-handoff",
                        previous_state="armed" if assignment.armed else "arming",
                        current_state="armed" if assignment.armed else "arming",
                        triggered=False,
                        track_id=recipient,
                        bed_id=successor.bed_id,
                        values={
                            "in_bed_dwell_sec": successor.in_bed_dwell_sec,
                            "outside_dwell_sec": successor.outside_dwell_sec,
                        },
                    )
                )
                del self._assignments[stale_id]
                continue
            traces.append(
                DecisionTraceSnapshot(
                    reason="stale-track-clear",
                    previous_state="armed" if assignment.armed else "arming",
                    current_state="retired",
                    triggered=False,
                    track_id=stale_id,
                    bed_id=assignment.bed_id,
                    values={
                        "in_bed_dwell_sec": assignment.in_bed_dwell_sec,
                        "outside_dwell_sec": assignment.outside_dwell_sec,
                    },
                    missing_values={
                        "containment_ratio": "track-no-longer-live",
                    },
                )
            )
            if had_assignment:
                self._lost_track_ids.append(stale_id)
            del self._assignments[stale_id]
        for person_id, person_box in zip(person_ids, observation.boxes, strict=True):
            if person_id is None or person_id not in live_ids:
                continue
            assignment = self._assignments.setdefault(person_id, _Assignment())
            containments = tuple(
                containment_ratio(person_box, bed_box) for bed_box in observation.bed_boxes
            )
            # `observation.bed_boxes` is non-empty here -- `update()` returns
            # early otherwise -- so `containments` always has at least one
            # value (#238: this is signal (b), "was anyone ever scored close
            # to a bed at all", independent of whether an assignment formed).
            self._max_containment_observed = max(self._max_containment_observed, *containments)
            candidate_bed_id = best_bed_id(containments, self._config.min_containment)
            if assignment.bed_id is None:
                assignment.update_candidate(candidate_bed_id)
                if assignment.candidate_frames >= self._config.hold_frames:
                    assignment.bed_id = assignment.candidate_bed_id
                    assignment.armed = False
                    assignment.in_bed_dwell_sec = 0.0
                    assignment.outside_dwell_sec = 0.0
                    assignment.last_time_sec = input_value.time_sec
                    self._assignments_made += 1
                    assert assignment.bed_id is not None
                    self._episodes.reassociate_bed_exit(
                        EpisodeProposal(
                            camera_id=self._config.camera_id,
                            facility_id=self._config.facility_id,
                            event_type="bed-exit",
                            track_id=person_id,
                            bed_id=assignment.bed_id,
                            frame_index=input_value.frame_index,
                            time_sec=(
                                0.0 if input_value.time_sec is None else input_value.time_sec
                            ),
                            qualifying=False,
                            probability=1.0,
                            domain="bed_exit",
                        )
                    )
                if assignment.bed_id is not None:
                    occupied[assignment.bed_id] = person_id
                traces.append(
                    DecisionTraceSnapshot(
                        reason=(
                            "assigned"
                            if assignment.bed_id is not None
                            else "assignment-hold"
                            if candidate_bed_id is not None
                            else "below-containment"
                        ),
                        previous_state="unassigned",
                        current_state=(
                            "contained" if assignment.bed_id is not None else "unassigned"
                        ),
                        triggered=False,
                        track_id=person_id,
                        bed_id=(
                            assignment.bed_id if assignment.bed_id is not None else candidate_bed_id
                        ),
                        values={
                            "containment_ratio": max(containments),
                            "min_containment": self._config.min_containment,
                            "candidate_frames": assignment.candidate_frames,
                            "hold_frames_threshold": self._config.hold_frames,
                        },
                    )
                )
                continue

            # Real elapsed PTS time since this assignment's last observed
            # frame. A frame with no `time_sec` never fabricates 0.0 into an
            # absolute timestamp here -- it simply contributes zero dwell
            # this frame (the anchor is left unmoved so the next real
            # timestamp spans correctly across the gap) and the trace
            # records that time was missing rather than silently proceeding.
            time_missing = input_value.time_sec is None
            if time_missing:
                dt = 0.0
            else:
                assert input_value.time_sec is not None
                dt = (
                    0.0
                    if assignment.last_time_sec is None
                    else max(0.0, input_value.time_sec - assignment.last_time_sec)
                )
                assignment.last_time_sec = input_value.time_sec

            features = by_track.get(person_id)
            posture_confirms_in_bed = (
                features is not None
                and features.bed_polygon_valid
                and features.observability >= _MIN_OBSERVABILITY
                and features.hip_depth >= _MIN_IN_BED_HIP_DEPTH
            )

            own_bed_id = assignment.bed_id
            own_ratio = containments[own_bed_id] if own_bed_id < len(containments) else 0.0
            if own_ratio >= self._config.min_containment:
                assignment.outside_dwell_sec = 0.0
                if posture_confirms_in_bed:
                    assignment.in_bed_dwell_sec += dt
                else:
                    assignment.in_bed_dwell_sec = 0.0
                if not assignment.armed and (
                    assignment.in_bed_dwell_sec >= self._config.in_bed_dwell_sec
                ):
                    assignment.armed = True
                    # This is the sole positive-evidence transition: a track
                    # just earned "confirmed in bed" (posture + containment
                    # sustained for in_bed_dwell_sec). It feeds two cumulative
                    # signals from the same site -- (1) telemetry's "did
                    # anything ever climb toward exit-eligible" counter
                    # (issue #238; name kept from the deleted grace-frame
                    # model for wire/dashboard continuity), and (2) the
                    # episode authority's sole re-arm signal: a resident
                    # confirmed back in bed after a prior exit must clear
                    # that episode's OPEN/RESOLVED hold before the same
                    # track+bed can exit-alert again.
                    self._grace_positive_transitions += 1
                    self._recovery_events.append(
                        BedExitEvent(person_id=person_id, bed_id=own_bed_id)
                    )
                occupied[own_bed_id] = person_id
                missing_values: dict[str, str] = {}
                if features is None:
                    missing_values["hip_depth"] = "no-pose-evidence"
                if time_missing:
                    missing_values["time_sec"] = "time-not-provided"
                traces.append(
                    DecisionTraceSnapshot(
                        reason=(
                            "contained"
                            if posture_confirms_in_bed
                            else "contained-posture-unconfirmed"
                        ),
                        previous_state="armed" if assignment.armed else "arming",
                        current_state="armed" if assignment.armed else "arming",
                        triggered=False,
                        track_id=person_id,
                        bed_id=own_bed_id,
                        values={
                            "containment_ratio": own_ratio,
                            "min_containment": self._config.min_containment,
                            "in_bed_dwell_sec": assignment.in_bed_dwell_sec,
                            "in_bed_dwell_threshold_sec": self._config.in_bed_dwell_sec,
                        },
                        missing_values=missing_values,
                    )
                )
                continue
            if any(
                bed_id != own_bed_id and ratio >= self._config.min_containment
                for bed_id, ratio in enumerate(containments)
            ):
                # Moved to a different bed's containment: neutral, not a
                # departure from *this* bed -- resets the arming climb (a
                # visit elsewhere earns no partial credit) but never fires.
                assignment.outside_dwell_sec = 0.0
                assignment.in_bed_dwell_sec = 0.0
                traces.append(
                    DecisionTraceSnapshot(
                        reason="contained-in-other-bed",
                        previous_state="armed" if assignment.armed else "arming",
                        current_state="other-bed",
                        triggered=False,
                        track_id=person_id,
                        bed_id=own_bed_id,
                        values={
                            "containment_ratio": own_ratio,
                            "max_other_containment_ratio": max(
                                ratio
                                for bed_id, ratio in enumerate(containments)
                                if bed_id != own_bed_id
                            ),
                            "min_containment": self._config.min_containment,
                        },
                    )
                )
                continue

            assignment.outside_dwell_sec += dt
            triggered = assignment.armed and (
                assignment.outside_dwell_sec >= self._config.outside_dwell_sec
            )
            if assignment.armed:
                reason = "outside-dwell-exit" if triggered else "outside-dwell"
            else:
                reason = "outside-not-armed"
            traces.append(
                DecisionTraceSnapshot(
                    reason=reason,
                    previous_state="armed" if assignment.armed else "arming",
                    current_state=(
                        "triggered"
                        if triggered
                        else "armed"
                        if assignment.armed
                        else "arming"
                    ),
                    triggered=triggered,
                    track_id=person_id,
                    bed_id=own_bed_id,
                    values={
                        "containment_ratio": own_ratio,
                        "min_containment": self._config.min_containment,
                        "outside_dwell_sec": assignment.outside_dwell_sec,
                        "outside_dwell_threshold_sec": self._config.outside_dwell_sec,
                    },
                )
            )
            if triggered:
                events.append(BedExitEvent(person_id=person_id, bed_id=own_bed_id))
                exit_beds.add(own_bed_id)
                # One-way latch clears on firing: the same track+bed must be
                # positively re-observed in bed for the full in_bed_dwell_sec
                # before it can arm (and therefore exit) again.
                assignment.armed = False
                assignment.in_bed_dwell_sec = 0.0
                assignment.outside_dwell_sec = 0.0

        statuses = tuple(
            BedStatus(
                bed_id=bed_id,
                box=bed_box,
                occupancy=(
                    "exit" if bed_id in exit_beds else "occupied" if bed_id in occupied else "empty"
                ),
                person_id=occupied.get(bed_id),
            )
            for bed_id, bed_box in enumerate(observation.bed_boxes)
        )
        if not traces:
            traces.append(
                DecisionTraceSnapshot(
                    reason="person-observation-missing",
                    previous_state="unknown",
                    current_state="no-decision",
                    triggered=False,
                    track_id=None,
                    bed_id=None,
                    missing_values={"containment_ratio": "no-observed-person"},
                )
            )
        self.last_trace_snapshots = tuple(traces)
        self.last_shadow_trace_count = 0
        return BedExitFrame(statuses=statuses, events=tuple(events))


def _bed_region_is_usable(source: BedRegionCacheState) -> bool:
    return source in (BedRegionCacheState.FRESH, BedRegionCacheState.CACHED)


__all__ = ["BedExitMonitor", "BedExitScoringRecorder"]
