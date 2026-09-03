"""Episode lifecycle and exact-once event authority for fall and bed exit."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from worker.types import BusinessEvent


class EpisodeState(StrEnum):
    NORMAL = "normal"
    CANDIDATE = "candidate"
    OPEN = "open"
    UNKNOWN = "unknown"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class EpisodeProposal:
    camera_id: str
    facility_id: str
    event_type: str
    track_id: int
    bed_id: int | None
    frame_index: int
    time_sec: float
    qualifying: bool
    confirmed_recovery: bool = False
    probability: float | None = None
    domain: str | None = None
    generation: int = 0
    confirmation_votes: int = 3
    confirmation_window: int = 5


@dataclass(slots=True)
class _Episode:
    state: EpisodeState = EpisodeState.NORMAL
    votes: deque[bool] = field(default_factory=deque)
    unknown_frame: int | None = None
    unknown_time: float | None = None
    sequence: int | None = None
    emitted_identity: str | None = None


class EpisodeAuthority:
    """Per-camera authority for event lifecycle, identity, and re-association.

    Track loss deliberately moves an open episode to UNKNOWN rather than
    resolving it. Only a scored confirmed recovery can re-arm RESOLVED.
    """

    def __init__(self, *, boot_id: str, stream_epoch: str, source_generation: int) -> None:
        self._boot_id = boot_id
        self._stream_epoch = stream_epoch
        self._source_generation = source_generation
        self._episodes: dict[tuple[str, str, int | None, int], _Episode] = {}
        self._next_sequence = 1
        self.track_id_switch_absorbed_total = 0

    def state_for(
        self, *, camera_id: str, event_type: str, bed_id: int | None, track_id: int
    ) -> EpisodeState:
        episode = self._episodes.get((camera_id, event_type, bed_id, track_id))
        return EpisodeState.NORMAL if episode is None else episode.state

    def propose(self, proposal: EpisodeProposal) -> tuple[BusinessEvent, ...]:
        if proposal.confirmation_votes < 1:
            raise ValueError("confirmation_votes must be positive")
        if proposal.confirmation_window < proposal.confirmation_votes:
            raise ValueError("confirmation_window must cover confirmation_votes")
        key = (proposal.camera_id, proposal.event_type, proposal.bed_id, proposal.track_id)
        episode = self._episodes.setdefault(key, _Episode())
        if episode.state is EpisodeState.UNKNOWN:
            # Same id returning is re-association, never a second onset.
            if self._within_reassociation(episode, proposal):
                episode.state = EpisodeState.OPEN
                return ()
            episode.state = EpisodeState.RESOLVED
        if episode.state is EpisodeState.RESOLVED:
            if proposal.confirmed_recovery:
                episode.state = EpisodeState.NORMAL
                episode.votes.clear()
            else:
                return ()
        if episode.state is EpisodeState.OPEN:
            if proposal.confirmed_recovery:
                # A recovery is the sole re-arm signal.  RESOLVED is retained
                # for unresolved loss/timeout, which must not re-arm.
                episode.state = EpisodeState.NORMAL
                episode.votes.clear()
            return ()
        if not proposal.qualifying:
            episode.state = EpisodeState.NORMAL
            episode.votes.clear()
            return ()
        if episode.state is EpisodeState.NORMAL:
            episode.state = EpisodeState.CANDIDATE
            episode.votes = deque(maxlen=proposal.confirmation_window)
        episode.votes.append(True)
        promoted = sum(episode.votes) >= proposal.confirmation_votes
        if not promoted:
            return ()
        episode.state = EpisodeState.OPEN
        sequence = self._next_sequence
        self._next_sequence += 1
        episode.sequence = sequence
        domain = proposal.domain or (
            "bed_exit" if proposal.event_type == "bed-exit" else proposal.event_type
        )
        identity = (
            f"{self._boot_id}:{self._stream_epoch}:{proposal.event_type}:"
            f"{'none' if proposal.bed_id is None else proposal.bed_id}:{proposal.track_id}:"
            f"{self._source_generation}:{proposal.generation}:{sequence}"
        )
        episode.emitted_identity = identity
        return (
            BusinessEvent(
                domain=domain,
                event_type=proposal.event_type,
                identity=identity,
                camera_id=proposal.camera_id,
                facility_id=proposal.facility_id,
                time_sec=proposal.time_sec,
                probability=proposal.probability,
                person_id=proposal.track_id,
                bed_id=proposal.bed_id,
            ),
        )

    def release(self, event: BusinessEvent) -> None:
        """Reopen exactly the episode whose emitted event failed durable staging."""
        for episode in self._episodes.values():
            if episode.emitted_identity != event.identity:
                continue
            episode.state = EpisodeState.NORMAL
            episode.votes.clear()
            episode.emitted_identity = None
            return

    def track_lost(
        self,
        *,
        camera_id: str,
        frame_index: int,
        time_sec: float,
        track_id: int | None = None,
    ) -> None:
        for (episode_camera, _event, _bed, episode_track), episode in self._episodes.items():
            if (
                episode_camera == camera_id
                and (track_id is None or episode_track == track_id)
                and episode.state is EpisodeState.OPEN
            ):
                episode.state = EpisodeState.UNKNOWN
                episode.unknown_frame = frame_index
                episode.unknown_time = time_sec

    def reassociate(self, proposal: EpisodeProposal, previous_track_id: int) -> bool:
        old_key = (proposal.camera_id, proposal.event_type, proposal.bed_id, previous_track_id)
        episode = self._episodes.get(old_key)
        if (
            episode is None
            or episode.state is not EpisodeState.UNKNOWN
            or not self._within_reassociation(episode, proposal)
        ):
            return False
        new_key = (proposal.camera_id, proposal.event_type, proposal.bed_id, proposal.track_id)
        self._episodes.pop(old_key)
        self._episodes[new_key] = episode
        episode.state = EpisodeState.OPEN
        self.track_id_switch_absorbed_total += 1
        return True

    def reassociate_bed_exit(self, proposal: EpisodeProposal) -> bool:
        """Absorb one unknown bed-exit episode for the same bed into a new track."""
        candidates = sorted(
            track_id
            for (camera_id, event_type, bed_id, track_id), episode in self._episodes.items()
            if (
                camera_id == proposal.camera_id
                and event_type == "bed-exit"
                and bed_id == proposal.bed_id
                and episode.state is EpisodeState.UNKNOWN
                and self._within_reassociation(episode, proposal)
            )
        )
        return bool(candidates) and self.reassociate(proposal, candidates[0])

    def reassociate_fall(self, proposal: EpisodeProposal) -> bool:
        """Absorb one unknown fall episode into a newly observed track."""
        candidates = sorted(
            track_id
            for (camera_id, event_type, bed_id, track_id), episode in self._episodes.items()
            if (
                camera_id == proposal.camera_id
                and event_type == "fall"
                and bed_id is None
                and episode.state is EpisodeState.UNKNOWN
                and self._within_reassociation(episode, proposal)
            )
        )
        return bool(candidates) and self.reassociate(proposal, candidates[0])

    def expire(self, *, frame_index: int, time_sec: float) -> None:
        for episode in self._episodes.values():
            if episode.state is EpisodeState.UNKNOWN and not self._within_values(
                episode, frame_index, time_sec
            ):
                episode.state = EpisodeState.RESOLVED

    @staticmethod
    def _within_values(episode: _Episode, frame_index: int, time_sec: float) -> bool:
        return (
            episode.unknown_frame is not None
            and frame_index - episode.unknown_frame <= 75
            and episode.unknown_time is not None
            and time_sec - episode.unknown_time <= 5.0
        )

    def _within_reassociation(self, episode: _Episode, proposal: EpisodeProposal) -> bool:
        return self._within_values(episode, proposal.frame_index, proposal.time_sec)
