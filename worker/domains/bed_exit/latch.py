from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from worker.domains.bed_exit.schema import BedExitEvent
from worker.domains.staleness import DEFAULT_STALE_AFTER_SEC, ObservationFreshness


@dataclass(frozen=True, slots=True)
class BedExitLatchStatus:
    active_exits: tuple[tuple[int, int], ...]
    stale: bool
    observation_age_sec: float | None


class BedExitLatch:
    """Return rising edges while retaining state across inference gaps."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
    ) -> None:
        self.event_count: int = 0
        self.first_event_sec: float | None = None
        self._active_exits: set[tuple[int, int]] = set()
        self._freshness = ObservationFreshness(
            clock=clock,
            stale_after_sec=stale_after_sec,
        )

    @property
    def status_snapshot(self) -> BedExitLatchStatus:
        freshness = self._freshness.snapshot()
        return BedExitLatchStatus(
            active_exits=tuple(sorted(self._active_exits)),
            stale=freshness.stale,
            observation_age_sec=freshness.observation_age_sec,
        )

    def update(
        self,
        events: tuple[BedExitEvent, ...],
        time_sec: float,
    ) -> tuple[BedExitEvent, ...]:
        self._freshness.observe()
        event_keys = {(event.person_id, event.bed_id) for event in events}
        onset_events = tuple(
            event for event in events if (event.person_id, event.bed_id) not in self._active_exits
        )
        if onset_events:
            self.event_count += len(onset_events)
            if self.first_event_sec is None:
                self.first_event_sec = time_sec
        self._active_exits = event_keys
        return onset_events

    def coast(self) -> tuple[BedExitEvent, ...]:
        """Emit nothing and retain the last-known latch state during a gap."""
        return ()


__all__ = ["BedExitLatch", "BedExitLatchStatus"]
