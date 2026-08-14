from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Generic, TypeVar
from uuid import uuid4

from worker.interfaces.decode import (
    DecodeAdapter,
    DecodeSession,
    StreamIdentityDecodeSession,
)
from worker.types import FramePacket

_DecodeConfigT = TypeVar("_DecodeConfigT")
_LivenessCallback = Callable[[str], None]
_OpenFailureCallback = Callable[[str], None]
_StopPredicate = Callable[[], bool]
_DEFAULT_PROCESSED_FPS = 5.0
_PROCESS_BOOT_ID = uuid4().hex


class RTSPSource(Generic[_DecodeConfigT]):
    def __init__(
        self,
        config: _DecodeConfigT,
        decoder: DecodeAdapter[_DecodeConfigT],
        max_failures: int = 30,
        reconnect_initial_backoff_sec: float = 0.25,
        reconnect_max_backoff_sec: float = 5.0,
        max_total_reconnects: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff_wait: Callable[[float], bool] | None = None,
        stop_requested: _StopPredicate | None = None,
        target_fps: float = _DEFAULT_PROCESSED_FPS,
        clock: Callable[[], float] = time.monotonic,
        pace_wait: Callable[[float], bool] | None = None,
        on_open_failure: _OpenFailureCallback | None = None,
        worker_boot_id: str = _PROCESS_BOOT_ID,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self._config = config
        self._decoder = decoder
        self._max_failures = max(1, max_failures)
        self._reconnect_initial_backoff_sec = max(0.0, reconnect_initial_backoff_sec)
        self._reconnect_max_backoff_sec = max(
            self._reconnect_initial_backoff_sec,
            reconnect_max_backoff_sec,
        )
        self._max_total_reconnects = (
            None if max_total_reconnects is None else max(0, max_total_reconnects)
        )
        self._sleep = sleep
        self._backoff_wait = backoff_wait
        self._stop_requested = stop_requested
        self._frame_interval_sec = 1.0 / target_fps
        self._clock = clock
        self._pace_wait = pace_wait
        self._on_open_failure = on_open_failure
        self._worker_boot_id = worker_boot_id
        self._next_stream_epoch = 1
        self._on_reconnecting: _LivenessCallback | None = None
        self._on_recovered: _LivenessCallback | None = None

    def set_liveness_callbacks(
        self,
        *,
        on_reconnecting: _LivenessCallback | None = None,
        on_recovered: _LivenessCallback | None = None,
        on_open_failure: _OpenFailureCallback | None = None,
    ) -> None:
        self._on_reconnecting = on_reconnecting
        self._on_recovered = on_recovered
        if on_open_failure is not None:
            self._on_open_failure = on_open_failure

    def __iter__(self) -> Iterator[FramePacket]:
        session: DecodeSession | None = None
        consecutive_failures = 0
        reconnects = 0
        reconnecting = False
        session_epoch: int | None = None
        try:
            while not self._stop_is_requested():
                if session is None:
                    try:
                        session = self._decoder.open(self._config)
                        session_epoch = self._next_stream_epoch
                        self._next_stream_epoch += 1
                        if isinstance(session, StreamIdentityDecodeSession):
                            session.set_stream_identity(
                                self._worker_boot_id,
                                session_epoch,
                            )
                    except (OSError, RuntimeError):
                        self._notify_open_failure("spawn_failed")
                        if not reconnecting:
                            reconnecting = True
                            self._notify_reconnecting("open_failure")
                        if self._reconnect_budget_exhausted(reconnects):
                            break
                        reconnects += 1
                        if self._wait_for_backoff_or_stop(self._backoff_delay(reconnects)):
                            break
                        continue

                packet = session.read()
                if packet is None:
                    consecutive_failures += 1
                    if consecutive_failures >= self._max_failures:
                        if not reconnecting:
                            reconnecting = True
                            self._notify_reconnecting("read_failure")
                        session.close()
                        session = None
                        session_epoch = None
                        if self._reconnect_budget_exhausted(reconnects):
                            break
                        reconnects += 1
                        if self._wait_for_backoff_or_stop(self._backoff_delay(reconnects)):
                            break
                    continue

                consecutive_failures = 0
                if session_epoch is None:  # pragma: no cover - open assigns every session
                    raise RuntimeError("decode session epoch was not assigned")
                packet = replace(
                    packet,
                    worker_boot_id=self._worker_boot_id,
                    stream_epoch=session_epoch,
                )
                if reconnecting:
                    reconnecting = False
                    self._notify_recovered("read_recovered")
                started_at = self._clock()
                yield packet
                remaining = max(0.0, self._frame_interval_sec - (self._clock() - started_at))
                if remaining > 0 and self._wait_for_pace_or_stop(remaining):
                    break
        finally:
            if session is not None:
                session.close()

    def _reconnect_budget_exhausted(self, reconnects: int) -> bool:
        return self._max_total_reconnects is not None and reconnects >= self._max_total_reconnects

    def _backoff_delay(self, reconnects: int) -> float:
        if reconnects <= 0:
            return 0.0
        delay = self._reconnect_initial_backoff_sec * (2.0 ** min(reconnects - 1, 32))
        return min(delay, self._reconnect_max_backoff_sec)

    def _stop_is_requested(self) -> bool:
        return self._stop_requested is not None and self._stop_requested()

    def _wait_for_backoff_or_stop(self, delay_sec: float) -> bool:
        if self._stop_is_requested():
            return True
        if self._backoff_wait is not None:
            return self._backoff_wait(delay_sec) or self._stop_is_requested()
        self._sleep(delay_sec)
        return self._stop_is_requested()

    def _wait_for_pace_or_stop(self, delay_sec: float) -> bool:
        if self._stop_is_requested():
            return True
        if self._pace_wait is not None:
            return self._pace_wait(delay_sec) or self._stop_is_requested()
        self._sleep(delay_sec)
        return self._stop_is_requested()

    def _notify_reconnecting(self, reason: str) -> None:
        if self._on_reconnecting is not None:
            self._on_reconnecting(reason)

    def _notify_recovered(self, reason: str) -> None:
        if self._on_recovered is not None:
            self._on_recovered(reason)

    def _notify_open_failure(self, reason: str) -> None:
        if self._on_open_failure is not None:
            self._on_open_failure(reason)


__all__ = ["RTSPSource"]
