from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from worker.domains.staleness import DEFAULT_STALE_AFTER_SEC, ObservationFreshness


@dataclass(frozen=True, slots=True)
class BedExitLatchStatus:
    stale: bool
    observation_age_sec: float | None


class BedExitLatch:
    """Retain only bed-observation freshness across inference gaps."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
    ) -> None:
        self._freshness = ObservationFreshness(
            clock=clock,
            stale_after_sec=stale_after_sec,
        )

    @property
    def status_snapshot(self) -> BedExitLatchStatus:
        freshness = self._freshness.snapshot()
        return BedExitLatchStatus(
            stale=freshness.stale,
            observation_age_sec=freshness.observation_age_sec,
        )

    def update(self) -> None:
        self._freshness.observe()
    def coast(self) -> None:
        """Retain the last freshness observation during a gap."""


__all__ = ["BedExitLatch", "BedExitLatchStatus"]
