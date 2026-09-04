"""Status-tick supervision for the Flow media plane."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol, final


class _Frame(Protocol):
    native_publish_sequence: int


class _Metadata(Protocol):
    def peek(self, camera_id: str) -> _Frame | None: ...


class _Status(Protocol):
    fatal_error: str | None


class _Plane(Protocol):
    metadata: _Metadata

    def status(self) -> _Status: ...

    def source_failure(self, camera_id: str, category: str) -> object: ...

    def clear_preview(self, camera_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class FlowLifecycleCounters:
    outages: int = 0
    recoveries: int = 0


@dataclass(slots=True)
class _CameraState:
    sequence: int | None = None
    accepted_at: float | None = None
    ready: bool = False
    outage: bool = False
    counters: FlowLifecycleCounters = FlowLifecycleCounters()


@final
class FlowLifecycleSupervisor:
    """Rotate stalled Flow sources from the same accepted-frame slot as pumps.

    Thirty seconds permits six ``nvurisrcbin`` five-second reconnect attempts
    before declaring an outage, while keeping the old stream epoch from
    surviving an extended reconnect.
    """

    DEFAULT_SILENCE_TIMEOUT_SEC = 30.0

    def __init__(
        self,
        plane: _Plane,
        camera_ids: Iterable[str],
        *,
        on_ready: Callable[[str], None],
        on_unready: Callable[[str], None],
        on_fatal: Callable[[str], None],
        silence_timeout_sec: float = DEFAULT_SILENCE_TIMEOUT_SEC,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if silence_timeout_sec <= 0:
            raise ValueError("Flow silence timeout must be positive")
        self._plane = plane
        self._states = {camera_id: _CameraState() for camera_id in camera_ids}
        self._on_ready = on_ready
        self._on_unready = on_unready
        self._on_fatal = on_fatal
        self._silence_timeout_sec = silence_timeout_sec
        self._clock = clock
        self._fatal = False

    def tick(self) -> None:
        status = self._plane.status()
        fatal_error = status.fatal_error
        if fatal_error is not None:
            if not self._fatal:
                self._fatal = True
                self._on_fatal(str(fatal_error))
            return

        now = self._clock()
        for camera_id, state in self._states.items():
            frame = self._plane.metadata.peek(camera_id)
            sequence = None if frame is None else frame.native_publish_sequence
            if sequence is not None and not isinstance(sequence, int):
                raise TypeError("accepted Flow metadata sequence must be an integer")
            if sequence is not None and sequence != state.sequence:
                state.sequence = sequence
                state.accepted_at = now
                if not state.ready:
                    state.ready = True
                if state.outage:
                    state.outage = False
                    state.counters = FlowLifecycleCounters(
                        outages=state.counters.outages,
                        recoveries=state.counters.recoveries + 1,
                    )
                    self._on_ready(camera_id)
                continue
            if state.accepted_at is None:
                state.accepted_at = now
                continue
            if not state.outage and now - state.accepted_at >= self._silence_timeout_sec:
                self._plane.source_failure(camera_id, "metadata_silence")
                self._plane.clear_preview(camera_id)
                self._on_unready(camera_id)
                state.ready = False
                state.outage = True
                state.sequence = None
                state.accepted_at = now
                state.counters = FlowLifecycleCounters(
                    outages=state.counters.outages + 1,
                    recoveries=state.counters.recoveries,
                )

    def counters(self, camera_id: str) -> FlowLifecycleCounters:
        return self._states[camera_id].counters


__all__ = ["FlowLifecycleCounters", "FlowLifecycleSupervisor"]
