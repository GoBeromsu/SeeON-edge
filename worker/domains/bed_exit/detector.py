from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
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
from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot


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
    )

    def __init__(self) -> None:
        self.bed_id: int | None = None
        self.candidate_bed_id: int | None = None
        self.candidate_frames: int = 0
        self.grace_frames: int = 0

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


class BedExitMonitor:
    """Interpret numeric observations with camera-local bed assignment state."""

    def __init__(
        self,
        *,
        config: BedExitConfig,
        clock: Callable[[], datetime],
        scoring_recorder: BedExitScoringRecorder | None = None,
    ) -> None:
        self._config: BedExitConfig = config
        self._clock: Callable[[], datetime] = clock
        self._night_window: NightWindow | None = config.night_window
        self._assignments: dict[int, _Assignment] = {}
        self._latch: BedExitLatch = BedExitLatch()
        self.last_debug_snapshot: BedExitDebugSnapshot | None = None
        self.last_trace_snapshots: tuple[DecisionTraceSnapshot, ...] = ()
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
    def config(self) -> BedExitConfig:
        return self._config

    def update_night_window(self, night_window: NightWindow | None) -> None:
        self._night_window = night_window

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
            self._scoring_recorder.record_bed_exit_scoring(
                self._config.camera_id,
                self._max_containment_observed,
                self._grace_positive_transitions,
                self._assignments_made,
            )
        self.last_debug_snapshot = BedExitDebugSnapshot(
            frame_index=input_value.frame_index,
            person_boxes=observation.boxes,
            bed_boxes=observation.bed_boxes,
            statuses=frame.statuses,
            events=frame.events,
            bed_region=input_value.bed_region,
        )
        event_time = 0.0 if input_value.time_sec is None else input_value.time_sec
        onset_events = self._latch.update(frame.events, event_time)
        if self._night_window is not None and not self._night_window.contains(self._clock()):
            return ()
        return tuple(self._business_event(event, event_time) for event in onset_events)

    def _update_frame(self, input_value: DecisionInput) -> BedExitFrame:
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
                assignment.clear_after_exit()

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
        self.last_trace_snapshots = tuple(traces)
        return BedExitFrame(statuses=statuses, events=tuple(events))

    def _business_event(self, event: BedExitEvent, time_sec: float) -> BusinessEvent:
        return BusinessEvent(
            domain="bed_exit",
            event_type="bed-exit",
            identity=f"{event.person_id}:{event.bed_id}",
            camera_id=self._config.camera_id,
            facility_id=self._config.facility_id,
            time_sec=time_sec,
            probability=1.0,
            person_id=event.person_id,
            bed_id=event.bed_id,
        )


def _bed_region_is_usable(source: BedRegionCacheState) -> bool:
    return source in (BedRegionCacheState.FRESH, BedRegionCacheState.CACHED)


__all__ = ["BedExitMonitor", "BedExitScoringRecorder"]
