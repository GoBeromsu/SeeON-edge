from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import ClassVar

from contracts.observation import FrameObservation
from shared.detection_policies import FALL_POLICY_V1_DEFAULT
from worker.domains.fall.classifier import (
    FallModelProtocol,
    FallWindowClassifier,
)
from worker.domains.fall.schema import FallEvent
from worker.domains.staleness import DEFAULT_STALE_AFTER_SEC, ObservationFreshness
from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot


@dataclass(frozen=True, slots=True)
class FallLatchStatus:
    is_fall: bool
    stale: bool
    observation_age_sec: float | None


class FallEventLatch:
    enabled: ClassVar[bool] = True
    classifier: FallWindowClassifier
    camera_id: str
    facility_id: str
    event_count: int
    first_event_sec: float | None
    _previous_fall: bool

    def __init__(
        self,
        model: FallModelProtocol,
        *,
        camera_id: str,
        facility_id: str,
        operating_threshold: float = FALL_POLICY_V1_DEFAULT.operating_threshold,
        clock: Callable[[], float] = monotonic,
        stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
    ) -> None:
        self.classifier = FallWindowClassifier(model, operating_threshold)
        self.camera_id = camera_id
        self.facility_id = facility_id
        self.event_count = 0
        self.first_event_sec = None
        self._previous_fall = False
        self._freshness = ObservationFreshness(
            clock=clock,
            stale_after_sec=stale_after_sec,
        )
        self.last_trace_snapshots: tuple[DecisionTraceSnapshot, ...] = ()

    @property
    def status_snapshot(self) -> FallLatchStatus:
        freshness = self._freshness.snapshot()
        return FallLatchStatus(
            is_fall=self._previous_fall,
            stale=freshness.stale,
            observation_age_sec=freshness.observation_age_sec,
        )

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        observation = self.classifier.classify(input_value)
        time_sec = 0.0 if input_value.time_sec is None else input_value.time_sec
        previous_fall = self._previous_fall
        is_fall = _is_fall(observation)
        event = self.update_event(is_fall, time_sec)
        probability, track_id = _fall_probability_and_track(
            observation, frozenset(input_value.live_track_ids)
        )
        reason = (
            "score-missing"
            if probability is None
            else "fall-onset"
            if event is not None
            else "fall-active"
            if is_fall
            else "below-threshold"
        )
        values: dict[str, int | float] = {
            "operating_threshold": self.classifier.operating_threshold,
            "window_frames": self.classifier.model.metadata.window,
        }
        missing_values: dict[str, str] = {}
        if probability is None:
            missing_values["fall_probability"] = "no-live-classified-track"
        else:
            values["fall_probability"] = probability
        self.last_trace_snapshots = (
            DecisionTraceSnapshot(
                reason=reason,
                previous_state="fall" if previous_fall else "clear",
                current_state="fall" if is_fall else "clear",
                triggered=event is not None,
                track_id=track_id,
                bed_id=None,
                values=values,
                missing_values=missing_values,
            ),
        )
        if event is None:
            return ()
        return (
            BusinessEvent(
                domain="fall",
                event_type="fall",
                identity=event.event_count,
                camera_id=self.camera_id,
                facility_id=self.facility_id,
                time_sec=event.onset_sec,
                probability=1.0 if probability is None else probability,
            ),
        )

    def coast(self) -> tuple[BusinessEvent, ...]:
        """Emit nothing and hold the last-known fall state during a gap."""
        return ()

    def update_signal(self, is_fall: bool, time_sec: float | None = None) -> bool:
        self._freshness.observe()
        onset_sec = 0.0 if time_sec is None else time_sec
        onset = is_fall and not self._previous_fall
        if onset:
            self.event_count += 1
            if self.first_event_sec is None:
                self.first_event_sec = onset_sec
        self._previous_fall = is_fall
        return onset


    def release_onset(self, event: object | None = None) -> None:
        """Re-open the rising edge for an onset that was never made durable.

        `update_signal` consumes the edge by setting `_previous_fall`, so once an
        onset has been reported the next positive frame is merely "still
        falling" and emits nothing. If the envelope for that onset never reached
        durable storage, leaving the edge consumed destroys the fall outright:
        the decision cannot be re-derived from any later frame.

        The exact state the onset advanced is undone -- the edge, the count, and
        the first-event timestamp if this onset set it -- so a subsequent
        positive frame reports the same fall rather than a different one.
        """
        # Fail closed on ownership. The caller is expected to route this only
        # to the decider that produced the event, but an earlier version of
        # that routing was a permissive `getattr` default that matched every
        # decider, so a bed-exit failure re-opened a delivered fall and the next
        # frame would report it twice. Checking here means a future caller
        # cannot make that mistake silently: an event this latch did not produce
        # is refused, and so is one whose ownership cannot be established.
        if getattr(event, "domain", None) != "fall":
            return
        if getattr(event, "camera_id", None) != self.camera_id:
            return
        if not self._previous_fall:
            return
        self._previous_fall = False
        if self.event_count > 0:
            self.event_count -= 1
        if self.event_count == 0:
            self.first_event_sec = None

    def update_event(self, is_fall: bool, time_sec: float) -> FallEvent | None:
        if not self.update_signal(is_fall, time_sec):
            return None
        first_event_sec = self.first_event_sec
        if first_event_sec is None:
            first_event_sec = time_sec
        return FallEvent(
            event_count=self.event_count,
            onset_sec=time_sec,
            first_event_sec=first_event_sec,
        )


def _is_fall(observation: FrameObservation) -> bool:
    return any(label.is_fall for label in observation.labels)


def _fall_probability_and_track(
    observation: FrameObservation,
    live_track_ids: frozenset[int],
) -> tuple[float | None, int | None]:
    track_ids = tuple(
        track_id
        for track_id in observation.track_ids
        if track_id is not None and track_id in live_track_ids
    )
    pairs = tuple(
        (label.confidence, track_id)
        for label, track_id in zip(observation.labels, track_ids, strict=True)
    )
    if not pairs:
        return None, None
    return max(pairs, key=lambda item: (item[0], -item[1]))


__all__ = ["FallEventLatch", "FallLatchStatus"]
