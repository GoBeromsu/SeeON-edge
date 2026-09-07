from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from contracts.worker_config import PulledWorkerConfig

PullWorkerConfig: TypeAlias = Callable[[str, str | None], PulledWorkerConfig | None]


@dataclass(frozen=True, order=True, slots=True)
class RestartDirective:
    """Monotonic restart identity ordered by generation, config, then registry."""

    generation: int
    version: int
    registry: int = 0

    @classmethod
    def from_pulled(cls, config: PulledWorkerConfig) -> RestartDirective:
        return cls(
            generation=config.restart_epoch,
            version=config.config_version,
            registry=config.registry_version,
        )


class RestartDirectiveTracker:
    def __init__(self, initial: RestartDirective) -> None:
        self._current: RestartDirective = initial
        self._lock: threading.Lock = threading.Lock()

    @property
    def current(self) -> RestartDirective:
        with self._lock:
            return self._current

    def observe(self, candidate: RestartDirective) -> bool:
        with self._lock:
            if candidate <= self._current:
                return False
            self._current = candidate
            return True


def make_restart_check(
    relay_url: str,
    relay_token: str | None,
    boot_directive: RestartDirective,
    *,
    poll_interval_sec: float = 60.0,
    monotonic: Callable[[], float] = time.monotonic,
    pull_config: PullWorkerConfig,
) -> Callable[[], bool]:
    tracker = RestartDirectiveTracker(boot_directive)
    state_lock = threading.Lock()
    last_checked = -poll_interval_sec

    def check() -> bool:
        nonlocal last_checked
        with state_lock:
            now = monotonic()
            if now - last_checked < poll_interval_sec:
                return False
            last_checked = now
            try:
                pulled = pull_config(relay_url, relay_token)
            except (OSError, TimeoutError):
                return False
            if pulled is None:
                return False
            return tracker.observe(RestartDirective.from_pulled(pulled))

    return check


__all__ = [
    "PullWorkerConfig",
    "RestartDirective",
    "RestartDirectiveTracker",
    "make_restart_check",
]
