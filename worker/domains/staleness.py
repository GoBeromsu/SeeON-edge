from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

DEFAULT_STALE_AFTER_SEC = 3.0


@dataclass(frozen=True, slots=True)
class FreshnessSnapshot:
    stale: bool
    observation_age_sec: float | None


class ObservationFreshness:
    """Wall-clock freshness for a latch's last actual observation."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
    ) -> None:
        if stale_after_sec <= 0:
            raise ValueError("stale_after_sec must be > 0")
        self._clock = clock
        self._stale_after_sec = stale_after_sec
        self._last_observed_at: float | None = None

    def observe(self) -> None:
        self._last_observed_at = self._clock()

    def snapshot(self) -> FreshnessSnapshot:
        if self._last_observed_at is None:
            return FreshnessSnapshot(stale=True, observation_age_sec=None)
        age = max(0.0, self._clock() - self._last_observed_at)
        return FreshnessSnapshot(
            stale=age >= self._stale_after_sec,
            observation_age_sec=age,
        )


__all__ = [
    "DEFAULT_STALE_AFTER_SEC",
    "FreshnessSnapshot",
    "ObservationFreshness",
]
