"""V2 fall scoring and proposal policy.

V2 retains classifier votes, fallen/recovery streaks, and trace snapshots.
It deliberately surrenders event emission, de-duplication, and identity
minting to :mod:`worker.domains.episode`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from shared.detection_policies import FallPolicyV2
from worker.domains.episode import EpisodeAuthority, EpisodeProposal
from worker.domains.fall.classifier_v2 import FallV2Probabilities, FallWindowClassifierV2
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


def _trace_state(state: _TrackState | None, episode_state: str | None = None) -> str:
    """Name the lifecycle state the episode authority actually holds.

    The authority owns promotion, so an OPEN episode must trace as confirmed
    even though this decider's own vote deque is only the proposal input.
    """
    if state is None:
        return "unknown"
    if episode_state == "open":
        return "transition-confirmed"
    if state.fallen:
        return "fallen"
    if episode_state == "candidate" or any(state.transition_votes):
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
    _episodes: EpisodeAuthority = field(init=False)
    last_trace_snapshots: tuple[DecisionTraceSnapshot, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if (
            not self.boot_id
            or not self.stream_epoch
            or isinstance(self.source_generation, bool)
            or self.source_generation < 0
        ):
            raise ValueError("fall event identities must name a boot and source epoch")
        self._episodes = EpisodeAuthority(
            boot_id=self.boot_id,
            stream_epoch=self.stream_epoch,
            source_generation=self.source_generation,
        )

    def update(
        self,
        probabilities_by_track: Mapping[int, FallV2Probabilities],
        live_track_ids: Iterable[int],
        *,
        frame_index: int,
        time_sec: float,
    ) -> tuple[BusinessEvent, ...]:
        """Advance live tracks and emit every newly opened episode in track order."""
        live_ids = frozenset(live_track_ids)
        self._evict_stale(live_ids, frame_index, time_sec)
        emitted: list[BusinessEvent] = []
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
            previous_state = _trace_state(state, self._episode_state(track_id))
            event = self._advance(track_id, state, probability, frame_index, time_sec)
            snapshots.append(
                self._trace_snapshot(track_id, state, previous_state, probability, event)
            )
            if event is not None:
                emitted.append(event)
        self.last_trace_snapshots = tuple(snapshots)
        return tuple(emitted)

    def coast(self) -> tuple[BusinessEvent, ...]:
        """A classifier gap never changes temporal counters or emits an event."""
        return ()

    def release_onset(self, event: BusinessEvent) -> None:
        """Reopen only the exact undelivered onset that this policy emitted."""
        self._episodes.release(event)

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

    def _evict_stale(
        self, live_ids: frozenset[int], frame_index: int, time_sec: float
    ) -> None:
        for track_id, state in tuple(self._states.items()):
            if track_id in live_ids:
                continue
            if frame_index - state.last_seen_frame >= self.policy.track_ttl_frames:
                self._episodes.track_lost(
                    camera_id=self.camera_id,
                    frame_index=frame_index,
                    time_sec=time_sec,
                    track_id=track_id,
                )
                del self._states[track_id]

    def _advance(
        self,
        track_id: int,
        state: _TrackState,
        probability: FallV2Probabilities,
        frame_index: int,
        time_sec: float,
    ) -> BusinessEvent | None:
        proposal = EpisodeProposal(
            camera_id=self.camera_id,
            facility_id=self.facility_id,
            event_type="fall",
            track_id=track_id,
            bed_id=None,
            frame_index=frame_index,
            time_sec=time_sec,
            qualifying=probability.fall_transition >= self.policy.transition_threshold,
            confirmed_recovery=False,
            probability=probability.fall_transition,
            domain="fall",
            generation=state.generation,
            confirmation_votes=self.policy.transition_votes,
            confirmation_window=self.policy.transition_window,
        )
        if not state.initialized:
            _ = self._episodes.reassociate_fall(proposal)
        # A person first observed already fallen has internal state, but no
        # synthetic transition alert.
        if not state.initialized:
            state.initialized = True
            if probability.fallen >= self.policy.fallen_threshold:
                state.fallen = True
                return None

        transition = proposal.qualifying
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

        proposal = replace(
            proposal,
            confirmed_recovery=(
                recovering and state.recovery_streak >= self.policy.recovery_consecutive
            ),
        )
        return next(iter(self._episodes.propose(proposal)), None)

    @property
    def track_id_switch_absorbed_total(self) -> int:
        """Re-associations the episode authority absorbed instead of re-alerting."""
        return self._episodes.track_id_switch_absorbed_total

    def _episode_state(self, track_id: int) -> str:
        return str(
            self._episodes.state_for(
                camera_id=self.camera_id, event_type="fall", bed_id=None, track_id=track_id
            )
        )

    def _trace_snapshot(
        self,
        track_id: int,
        state: _TrackState,
        previous_state: str,
        probability: FallV2Probabilities,
        event: BusinessEvent | None,
    ) -> DecisionTraceSnapshot:
        current_state = _trace_state(state, self._episode_state(track_id))
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

@dataclass(slots=True)
class FallV2DomainDecider:
    """Adapt the V2 row classifier and temporal policy to the domain port."""

    classifier: object
    policy: FallPolicyDeciderV2
    _resampler: PtsResampler[dict[int, tuple[float, ...]]] = field(
        default_factory=PtsResampler, init=False
    )
    _last_pts_ns: int | None = field(default=None, init=False)
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
        seconds = 0.0 if input_value.time_sec is None else input_value.time_sec
        pts_ns = int(seconds * 1_000_000_000)
        self._reset_on_pts_rollback(pts_ns)
        classifier = self.classifier
        if not hasattr(classifier, "update"):
            raise TypeError("fall.v2 classifier is invalid")
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

    @property
    def track_id_switch_absorbed_total(self) -> int:
        return self.policy.track_id_switch_absorbed_total

    def _reset_on_pts_rollback(self, pts_ns: int) -> None:
        if self._last_pts_ns is not None and pts_ns < self._last_pts_ns:
            if not isinstance(self.classifier, FallWindowClassifierV2):
                raise TypeError("fall.v2 classifier must support stream-epoch reset")
            self._resampler = PtsResampler()
            self.classifier = FallWindowClassifierV2(self.classifier.model)
        self._last_pts_ns = pts_ns

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
        reset at a PTS rollback, so the host path cannot retain an old epoch.
        """
        return self._resampler.push(pts_ns, rows)


__all__ = ["FallPolicyDeciderV2", "FallPolicyV2", "FallV2DomainDecider"]
