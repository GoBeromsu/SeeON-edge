"""Pure temporal policy for the unregistered three-class fall candidate."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from shared.detection_policies import FallPolicyV2
from worker.domains.fall.classifier_v2 import FallV2Probabilities
from worker.domains.fall.pose_bbox56 import PoseBbox56Track, pose_bbox56_tracks
from worker.pipeline.perception.pts_resample import PtsResampler, ResampledRow
from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot


@dataclass(slots=True)
class _TrackState:
    generation: int
    last_seen_frame: int
    transition_votes: deque[bool] = field(default_factory=lambda: deque(maxlen=5))
    fallen_streak: int = 0
    recovery_streak: int = 0
    fallen: bool = False
    initialized: bool = False
    alert_event: BusinessEvent | None = None
    alert_frame: int | None = None
    transition_sequence: int | None = None


def _trace_state(state: _TrackState | None) -> str:
    if state is None:
        return "unknown"
    if state.alert_event is not None:
        return "transition-confirmed"
    if state.fallen:
        return "fallen"
    if any(state.transition_votes):
        return "transition-candidate"
    return "clear"


def _missing_score_snapshot(track_id: int, state: _TrackState | None) -> DecisionTraceSnapshot:
    current = _trace_state(state)
    return DecisionTraceSnapshot(
        reason="score-missing",
        previous_state=current,
        current_state=current,
        triggered=False,
        track_id=track_id,
        bed_id=None,
        missing_values={"fall_transition_probability": "no-live-classified-track"},
    )


@dataclass(slots=True)
class FallPolicyDeciderV2:
    """Camera-local lifecycle and alert policy for V2 model probabilities.

    The caller creates one instance per camera.  State is keyed by track id and
    monotonically increasing generation so a reused id is never deduplicated
    with an evicted resident.
    """

    camera_id: str
    facility_id: str
    boot_id: str
    stream_epoch: str
    source_generation: int
    policy: FallPolicyV2 = field(default_factory=FallPolicyV2)
    _states: dict[int, _TrackState] = field(default_factory=dict, init=False)
    _next_generations: dict[int, int] = field(default_factory=dict, init=False)
    _next_transition_sequence: int = field(default=1, init=False)
    last_trace_snapshots: tuple[DecisionTraceSnapshot, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if (
            not self.boot_id
            or not self.stream_epoch
            or isinstance(self.source_generation, bool)
            or self.source_generation < 0
        ):
            raise ValueError("fall event identities must name a boot and source epoch")

    def update(
        self,
        probabilities_by_track: Mapping[int, FallV2Probabilities],
        live_track_ids: Iterable[int],
        *,
        frame_index: int,
        time_sec: float,
    ) -> tuple[BusinessEvent, ...]:
        """Advance live tracks and emit at most one deterministic camera alert."""
        live_ids = frozenset(live_track_ids)
        self._evict_stale(live_ids, frame_index)
        candidates: list[tuple[float, int, BusinessEvent]] = []
        snapshots: list[DecisionTraceSnapshot] = []
        for track_id in sorted(live_ids):
            existing_state = self._states.get(track_id)
            if existing_state is not None:
                # Classifier warming/stride gaps are still a live tracker
                # observation. They must not turn into a synthetic reconnect.
                existing_state.last_seen_frame = frame_index
            probability = probabilities_by_track.get(track_id)
            if probability is None:
                snapshots.append(_missing_score_snapshot(track_id, existing_state))
                continue
            state = self._state_for(track_id, frame_index)
            previous_state = _trace_state(state)
            event = self._advance(track_id, state, probability, frame_index, time_sec)
            snapshots.append(
                self._trace_snapshot(track_id, state, previous_state, probability, event)
            )
            if event is not None:
                candidates.append((probability.fall_transition, track_id, event))
        self.last_trace_snapshots = tuple(snapshots)
        if not candidates:
            return ()
        # The camera OR is resolved only within this tick.  An equal score has a
        # stable, resident-independent tie break.
        _, _, winner = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
        return (winner,)

    def coast(self) -> tuple[BusinessEvent, ...]:
        """A classifier gap never changes temporal counters or emits an event."""
        return ()

    def release_onset(self, event: object | None = None) -> None:
        """Reopen only the exact undelivered onset that this policy emitted."""
        if not isinstance(event, BusinessEvent) or event.domain != "fall":
            return
        if event.camera_id != self.camera_id or event.facility_id != self.facility_id:
            return
        if event.person_id is None:
            return
        state = self._states.get(event.person_id)
        if state is None or state.alert_event != event:
            return
        state.alert_event = None
        state.alert_frame = None

    def generation_for(self, track_id: int) -> int | None:
        state = self._states.get(track_id)
        return None if state is None else state.generation

    def is_fallen(self, track_id: int) -> bool:
        state = self._states.get(track_id)
        return state is not None and state.fallen

    def _state_for(self, track_id: int, frame_index: int) -> _TrackState:
        state = self._states.get(track_id)
        if state is not None:
            state.last_seen_frame = frame_index
            return state
        generation = self._next_generations.get(track_id, 0)
        self._next_generations[track_id] = generation + 1
        state = _TrackState(generation=generation, last_seen_frame=frame_index)
        self._states[track_id] = state
        return state

    def _evict_stale(self, live_ids: frozenset[int], frame_index: int) -> None:
        for track_id, state in tuple(self._states.items()):
            if track_id in live_ids:
                continue
            if frame_index - state.last_seen_frame >= self.policy.track_ttl_frames:
                del self._states[track_id]

    def _advance(
        self,
        track_id: int,
        state: _TrackState,
        probability: FallV2Probabilities,
        frame_index: int,
        time_sec: float,
    ) -> BusinessEvent | None:
        # A person first observed already fallen has internal state, but no
        # synthetic transition alert.
        if not state.initialized:
            state.initialized = True
            if probability.fallen >= self.policy.fallen_threshold:
                state.fallen = True
                return None

        transition = probability.fall_transition >= self.policy.transition_threshold
        state.transition_votes.append(transition)
        if probability.fallen >= self.policy.fallen_threshold:
            state.fallen_streak += 1
        else:
            state.fallen_streak = 0
        if state.fallen_streak >= self.policy.fallen_consecutive:
            state.fallen = True

        recovering = (
            probability.fall_transition < self.policy.recovery_transition_max
            and probability.fallen < self.policy.recovery_fallen_max
        )
        state.recovery_streak = state.recovery_streak + 1 if recovering else 0
        if state.recovery_streak >= self.policy.recovery_consecutive:
            state.fallen = False
            state.fallen_streak = 0
            state.transition_votes.clear()

        if sum(state.transition_votes) < self.policy.transition_votes:
            return None
        if state.alert_event is not None:
            return None
        if (
            state.alert_frame is not None
            and frame_index - state.alert_frame < self.policy.cooldown_frames
        ):
            return None
        event = BusinessEvent(
            domain="fall",
            event_type="fall",
            identity=(
                f"{self.boot_id}:{self.stream_epoch}:{track_id}:"
                f"{self.source_generation}:{state.generation}:{self._sequence_for(state)}"
            ),
            camera_id=self.camera_id,
            facility_id=self.facility_id,
            time_sec=time_sec,
            probability=probability.fall_transition,
            person_id=track_id,
        )
        state.alert_event = event
        state.alert_frame = frame_index
        return event

    def _trace_snapshot(
        self,
        track_id: int,
        state: _TrackState,
        previous_state: str,
        probability: FallV2Probabilities,
        event: BusinessEvent | None,
    ) -> DecisionTraceSnapshot:
        current_state = _trace_state(state)
        if event is not None:
            reason = "transition-confirmed"
        elif state.fallen and previous_state == "fallen":
            reason = "fall-active"
        elif not state.fallen and previous_state == "fallen":
            reason = "fall-recovered"
        elif any(state.transition_votes):
            reason = "transition-candidate"
        else:
            reason = "below-threshold"
        return DecisionTraceSnapshot(
            reason=reason,
            previous_state=previous_state,
            current_state=current_state,
            triggered=event is not None,
            track_id=track_id,
            bed_id=None,
            values={
                "fall_transition_probability": probability.fall_transition,
                "fallen_probability": probability.fallen,
                "transition_threshold": self.policy.transition_threshold,
                "transition_votes": self.policy.transition_votes,
                "transition_window": self.policy.transition_window,
            },
        )

    def _sequence_for(self, state: _TrackState) -> int:
        sequence = state.transition_sequence
        if sequence is not None:
            return sequence
        sequence = self._next_transition_sequence
        self._next_transition_sequence += 1
        state.transition_sequence = sequence
        return sequence


@dataclass(slots=True)
class FallV2DomainDecider:
    """Adapt the V2 row classifier and temporal policy to the domain port."""

    classifier: object
    policy: FallPolicyDeciderV2
    _resampler: PtsResampler[dict[int, tuple[float, ...]]] = field(
        default_factory=PtsResampler, init=False
    )
    resample_gap_rows_total: int = field(default=0, init=False)

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        observation = input_value.observation
        tracks = (
            PoseBbox56Track(
                track_id,
                observation.keypoints[index],
                (box.x1, box.y1, box.x2, box.y2),
            )
            for index, (track_id, box) in enumerate(
                zip(observation.track_ids, observation.boxes, strict=False)
            )
            if track_id is not None and index < len(observation.keypoints)
        )
        rows = dict(pose_bbox56_tracks(tracks, input_value.frame_width, input_value.frame_height))
        classifier = self.classifier
        if not hasattr(classifier, "update"):
            raise TypeError("fall.v2 classifier is invalid")
        seconds = 0.0 if input_value.time_sec is None else input_value.time_sec
        pts_ns = int(seconds * 1_000_000_000)
        resampled = self._resample(pts_ns, rows)
        if not resampled:
            return self.policy.coast()
        probabilities = {}
        for row in resampled:
            if row.valid:
                probabilities = classifier.update(row.value, input_value.live_track_ids)
                continue
            zero_rows = dict.fromkeys(input_value.live_track_ids, (0.0,) * 56)
            classifier.update(zero_rows, input_value.live_track_ids)
            self.resample_gap_rows_total += 1
        return self.policy.update(
            probabilities,
            input_value.live_track_ids,
            frame_index=input_value.frame_index,
            time_sec=0.0 if input_value.time_sec is None else input_value.time_sec,
        )

    def coast(self) -> tuple[BusinessEvent, ...]:
        return self.policy.coast()

    @property
    def last_trace_snapshots(self) -> tuple[DecisionTraceSnapshot, ...]:
        return self.policy.last_trace_snapshots

    def _resample(
        self,
        pts_ns: int,
        rows: dict[int, tuple[float, ...]],
    ) -> tuple[ResampledRow[dict[int, tuple[float, ...]]], ...]:
        """Use the shared PTS contract as the sole fall-input resampling owner.

        NativePolicyPump deliberately forwards accepted native metadata at its
        original PTS.  This adapter turns that stream into exactly one
        66,666,667ns cadence row (or ``valid=0`` zero rows for skipped
        buckets) before the 30-row classifier window sees it.  A decider is
        rebuilt at every stream epoch, so the resampler origin is epoch-local.
        """
        return self._resampler.push(pts_ns, rows)


__all__ = ["FallPolicyDeciderV2", "FallPolicyV2", "FallV2DomainDecider"]
