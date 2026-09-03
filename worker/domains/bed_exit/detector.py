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
from worker.domains.bed_exit.state_machine import (
    BedExitStateDecision,
    BedExitStateMachine,
)
from worker.domains.episode import EpisodeAuthority, EpisodeProposal
from worker.domains.staleness import DEFAULT_STALE_AFTER_SEC
from worker.types import (
    BusinessEvent,
    DecisionInput,
    DecisionTraceSnapshot,
    TemporalProfile,
)

_LOGGER = logging.getLogger(__name__)


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
        "bed_id",
        "candidate_bed_id",
        "candidate_frames",
        "grace_frames",
        "recovery_frames",
    )

    def __init__(self) -> None:
        self.bed_id: int | None = None
        self.candidate_bed_id: int | None = None
        self.candidate_frames: int = 0
        self.grace_frames: int = 0
        self.recovery_frames: int = 0

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

    def clear_after_exit(self) -> None:
        self.bed_id = None
        self.candidate_bed_id = None
        self.candidate_frames = 0
        self.grace_frames = 0
        self.recovery_frames = 0


class BedExitMonitor:
    """Interpret numeric observations with camera-local bed assignment state."""

    def __init__(
        self,
        *,
        config: BedExitConfig,
        clock: Callable[[], datetime],
        scoring_recorder: BedExitScoringRecorder | None = None,
        staleness_clock: Callable[[], float] = monotonic,
        stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
        temporal_profile: TemporalProfile | None = None,
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
        self._state_machine = BedExitStateMachine(temporal_profile=temporal_profile)
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
        self.last_shadow_trace_snapshots: tuple[DecisionTraceSnapshot, ...] = ()
        self.last_shadow_decisions: tuple[BedExitStateDecision, ...] = ()
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

    @property
    def state_machine(self) -> BedExitStateMachine:
        return self._state_machine

    def coast(self, *, frame_index: int | None = None) -> tuple[BusinessEvent, ...]:
        """Hold assignment/latch/shadow-machine state when no person inference was made."""
        self._latch.coast()
        _ = self._state_machine.coast()
        freshness = self._latch.status_snapshot
        previous = self.last_debug_snapshot
        statuses = () if previous is None else tuple(
            BedStatus(
                bed_id=status.bed_id,
                box=status.box,
                occupancy="covered",
                person_id=status.person_id,
            )
            for status in previous.statuses
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
            self.last_shadow_trace_snapshots = ()
            self.last_shadow_decisions = ()
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
            return ()
        self._episodes.expire(frame_index=input_value.frame_index, time_sec=event_time)
        emitted: list[BusinessEvent] = []
        for event in frame.events:
            emitted.extend(
                self._episodes.propose(
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
            )
            if event.person_id in self._lost_track_ids:
                self._episodes.track_lost(
                    camera_id=self._config.camera_id,
                    frame_index=input_value.frame_index,
                    time_sec=event_time,
                    track_id=event.person_id,
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

        # A track can vanish mid-exit: `GreedyIouTracker` already tolerates
        # up to `max_misses` (30, ~6s at 5fps) of failed re-matching before
        # dropping an id from `live_track_ids`, so a `stale_id` here isn't
        # reacting to a one-frame blink -- the tracker's own occlusion
        # tolerance already ran out. Fire only when `grace_frames > 0`: that
        # means the last live frame already showed the person outside their
        # own bed's containment, i.e. a departure already in progress before
        # the id died. A track that was still solidly contained
        # (`grace_frames == 0`) when it disappeared does not fire -- that
        # guarantee is what
        # `test_dead_observed_track_cannot_emit_after_identity_reuse` locks
        # in, and firing unconditionally here would break it (issue #218).
        # This is deliberately narrower than "any track loss while
        # assigned": a resident who gets up and leaves frame in one motion,
        # with the last live frame still showing containment, is still
        # swallowed -- see the residual-gap note on the PR.
        #
        # `> 0` is intentional, not unexamined: it also fires on a single
        # noisy sub-threshold frame (pose/occlusion jitter) that isn't a
        # real departure -- reproduced and documented in #246. Sensitivity
        # is chosen over precision for now, deliberately: bed_exit has
        # produced zero events in production, and a false positive is
        # visible and checkable against footage while a missed exit is
        # invisible and indistinguishable from the failure being diagnosed.
        # This trade-off applies only to this track-loss path; the live
        # path a few lines down still requires the full configured
        # `grace_frames` (3 by default) before firing, untouched. When
        # precision becomes the priority, #246 has the prepared remedy
        # (`>= 2`) and the caveat it requires first extending #218's
        # regression test past its current 1-frame script.
        for stale_id in sorted(set(self._assignments) - live_ids):
            assignment = self._assignments[stale_id]
            triggered = assignment.bed_id is not None and assignment.grace_frames > 0
            traces.append(
                DecisionTraceSnapshot(
                    reason="stale-track-exit" if triggered else "stale-track-clear",
                    previous_state=("live-grace" if assignment.grace_frames > 0 else "contained"),
                    current_state="triggered" if triggered else "retired",
                    triggered=triggered,
                    track_id=stale_id,
                    bed_id=assignment.bed_id,
                    values={
                        "grace_frames_before": assignment.grace_frames,
                        "grace_threshold": self._config.grace_frames,
                        "min_containment": self._config.min_containment,
                    },
                    missing_values={
                        "containment_ratio": "track-no-longer-live",
                    },
                )
            )
            if triggered:
                assert assignment.bed_id is not None
                events.append(BedExitEvent(person_id=stale_id, bed_id=assignment.bed_id))
                exit_beds.add(assignment.bed_id)
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
                    assignment.grace_frames = 0
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
                                0.0
                                if input_value.time_sec is None
                                else input_value.time_sec
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

            own_bed_id = assignment.bed_id
            own_ratio = containments[own_bed_id] if own_bed_id < len(containments) else 0.0
            if own_ratio >= self._config.min_containment:
                previous_grace = assignment.grace_frames
                assignment.grace_frames = 0
                assignment.recovery_frames += 1
                if assignment.recovery_frames > self._config.grace_frames:
                    self._recovery_events.append(
                        BedExitEvent(person_id=person_id, bed_id=own_bed_id)
                    )
                occupied[own_bed_id] = person_id
                traces.append(
                    DecisionTraceSnapshot(
                        reason="contained",
                        previous_state="live-grace" if previous_grace > 0 else "contained",
                        current_state="contained",
                        triggered=False,
                        track_id=person_id,
                        bed_id=own_bed_id,
                        values={
                            "containment_ratio": own_ratio,
                            "min_containment": self._config.min_containment,
                            "grace_frames_before": previous_grace,
                            "grace_frames_after": 0,
                            "grace_threshold": self._config.grace_frames,
                        },
                    )
                )
                continue
            if any(
                bed_id != own_bed_id and ratio >= self._config.min_containment
                for bed_id, ratio in enumerate(containments)
            ):
                previous_grace = assignment.grace_frames
                assignment.grace_frames = 0
                assignment.recovery_frames = 0
                traces.append(
                    DecisionTraceSnapshot(
                        reason="contained-in-other-bed",
                        previous_state="live-grace" if previous_grace > 0 else "contained",
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
                            "grace_frames_before": previous_grace,
                            "grace_frames_after": 0,
                            "grace_threshold": self._config.grace_frames,
                        },
                    )
                )
                continue

            grace_before = assignment.grace_frames
            assignment.recovery_frames = 0
            was_off_bed_start = grace_before == 0
            assignment.grace_frames += 1
            if was_off_bed_start:
                # #238 signal (c): counts each *entry* into the grace window
                # (0 -> 1), not distinct tracks -- a track that re-enters
                # grace multiple times (e.g. brief re-containment resets it
                # to 0, then it drifts off again) counts again each time.
                # Deliberately a plain counter, not a set of track ids, to
                # stay O(1) in memory for a full night across 13 cameras.
                self._grace_positive_transitions += 1
            triggered = assignment.grace_frames > self._config.grace_frames
            traces.append(
                DecisionTraceSnapshot(
                    reason="live-grace-exit" if triggered else "live-grace",
                    previous_state="contained" if grace_before == 0 else "live-grace",
                    current_state="triggered" if triggered else "live-grace",
                    triggered=triggered,
                    track_id=person_id,
                    bed_id=own_bed_id,
                    values={
                        "containment_ratio": own_ratio,
                        "min_containment": self._config.min_containment,
                        "grace_frames_before": grace_before,
                        "grace_frames_after": assignment.grace_frames,
                        "grace_threshold": self._config.grace_frames,
                    },
                )
            )
            if triggered:
                events.append(BedExitEvent(person_id=person_id, bed_id=own_bed_id))
                exit_beds.add(own_bed_id)

        statuses = tuple(
            BedStatus(
                bed_id=bed_id,
                box=bed_box,
                occupancy=(
                    "exit"
                    if bed_id in exit_beds
                    else "occupied"
                    if bed_id in occupied
                    else "empty"
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
        # The shadow path is explicitly non-authoritative: it never replaces the
        # containment decision and never carries triggered=True. It nonetheless
        # sat between the events being computed and the return that delivers
        # them, so a shadow failure discarded a real bed-exit event. A capability
        # that cannot decide anything must not be able to destroy a decision.
        try:
            shadow_traces = self._record_shadow(input_value, live_ids)
        except Exception:  # noqa: BLE001 - shadow evaluation never blocks detection
            shadow_traces = ()
            _LOGGER.warning(
                "shadow bed-exit evaluation failed for camera %s; the legacy "
                "decision and its events are unaffected",
                self._config.camera_id,
                exc_info=True,
            )
        # Legacy snapshots stay first so existing [0] assertions keep working.
        # Shadow snapshots are appended, never replace the containment path,
        # and never carry triggered=True in this todo.
        self.last_trace_snapshots = tuple(traces) + shadow_traces
        return BedExitFrame(statuses=statuses, events=tuple(events))

    def _record_shadow(
        self, input_value: DecisionInput, live_ids: set[int]
    ) -> tuple[DecisionTraceSnapshot, ...]:
        decisions: list[BedExitStateDecision] = []
        extra: list[DecisionTraceSnapshot] = []
        by_track = {item.track_id: item for item in input_value.bed_pose_features.items}
        for stale_id in sorted(set(self._state_machine.known_track_ids()) - live_ids):
            decision = self._state_machine.mark_absent(stale_id)
            if decision is not None:
                decisions.append(decision)
        for track_id in sorted(live_ids):
            features = by_track.get(track_id)
            if features is None:
                continue
            if not features.bed_polygon_valid:
                previous = self._state_machine.track_state(track_id)
                extra.append(
                    DecisionTraceSnapshot(
                        reason="bed-polygon-invalid",
                        previous_state=previous.value,
                        current_state="no-decision",
                        triggered=False,
                        track_id=track_id,
                        bed_id=features.bed_id,
                        missing_values={
                            "torso_in_frac": "bed-polygon-invalid",
                            "lower_in_frac": "bed-polygon-invalid",
                            "hip_depth": "bed-polygon-invalid",
                        },
                    )
                )
                continue
            decision = self._state_machine.observe(features)
            if decision is not None:
                decisions.append(decision)
        self.last_shadow_decisions = tuple(decisions)
        snapshots = tuple(item.snapshot for item in decisions) + tuple(extra)
        self.last_shadow_trace_snapshots = snapshots
        return snapshots

def _bed_region_is_usable(source: BedRegionCacheState) -> bool:
    return source in (BedRegionCacheState.FRESH, BedRegionCacheState.CACHED)


__all__ = ["BedExitMonitor", "BedExitScoringRecorder"]
