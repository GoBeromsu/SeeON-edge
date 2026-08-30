"""Pure temporal policy for the unregistered three-class fall candidate."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from worker.domains.fall.classifier_v2 import FallV2Probabilities
from worker.types import BusinessEvent


@dataclass(frozen=True, slots=True)
class FallPolicyV2:
    """Frozen V2 temporal thresholds; this is not a deploy-time configuration."""

    transition_threshold: float = 0.7
    transition_votes: int = 3
    transition_window: int = 5
    fallen_threshold: float = 0.8
    fallen_consecutive: int = 3
    recovery_transition_max: float = 0.4
    recovery_fallen_max: float = 0.5
    recovery_consecutive: int = 5
    track_ttl_frames: int = 45
    cooldown_frames: int = 90


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


@dataclass(slots=True)
class FallPolicyDeciderV2:
    """Camera-local lifecycle and alert policy for V2 model probabilities.

    The caller creates one instance per camera.  State is keyed by track id and
    monotonically increasing generation so a reused id is never deduplicated
    with an evicted resident.
    """

    camera_id: str
    facility_id: str
    policy: FallPolicyV2 = field(default_factory=FallPolicyV2)
    _states: dict[int, _TrackState] = field(default_factory=dict, init=False)
    _next_generations: dict[int, int] = field(default_factory=dict, init=False)

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
        for track_id in sorted(live_ids):
            existing_state = self._states.get(track_id)
            if existing_state is not None:
                # Classifier warming/stride gaps are still a live tracker
                # observation. They must not turn into a synthetic reconnect.
                existing_state.last_seen_frame = frame_index
            probability = probabilities_by_track.get(track_id)
            if probability is None:
                continue
            state = self._state_for(track_id, frame_index)
            event = self._advance(track_id, state, probability, frame_index, time_sec)
            if event is not None:
                candidates.append((probability.fall_transition, track_id, event))
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
            identity=f"{track_id}:{state.generation}",
            camera_id=self.camera_id,
            facility_id=self.facility_id,
            time_sec=time_sec,
            probability=probability.fall_transition,
            person_id=track_id,
        )
        state.alert_event = event
        state.alert_frame = frame_index
        return event


__all__ = ["FallPolicyDeciderV2", "FallPolicyV2"]
