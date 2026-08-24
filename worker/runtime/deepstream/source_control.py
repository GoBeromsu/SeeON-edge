"""Python-owned serialized source lifecycle and exact native-frame readiness."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, final, override

from worker.native.deepstream.control import ChildControlError
from worker.native.deepstream.ipc import MetadataFrame
from worker.native.deepstream.metadata import AcceptanceToken, SourceBinding


class SourceState(StrEnum):
    ABSENT = "absent"
    ADDING = "adding"
    STARTING = "starting"
    SOURCE_READY = "source_ready"
    REBUILDING = "rebuilding"
    DEGRADED = "degraded"
    REMOVING = "removing"
    TOMBSTONED = "tombstoned"


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    camera_id: str
    state: SourceState
    source_generation: int | None
    stream_epoch: int | None


class SourceReadinessError(TimeoutError):
    def __init__(self, code: str, camera_id: str) -> None:
        super().__init__(code, camera_id)
        self.code: str = code
        self.camera_id: str = camera_id

    @override
    def __str__(self) -> str:
        return f"{self.code}: camera_id={self.camera_id}"


class SourceControl(Protocol):
    def add_source(self, camera_id: str, uri: str) -> SourceBinding: ...
    def remove_source(self, camera_id: str) -> None: ...
    def source_failure(self, camera_id: str, category: str) -> SourceBinding: ...


class MetadataAdmission(Protocol):
    def register_source(self, binding: SourceBinding) -> AcceptanceToken: ...
    def remove_source(self, camera_id: str) -> None: ...
    def wait_accepted(
        self,
        token: AcceptanceToken,
        *,
        timeout_sec: float,
    ) -> MetadataFrame: ...


class FrameReceiver(Protocol):
    def pull_now(self, camera_id: str) -> MetadataFrame | None: ...


@final
class DarkSourceController:
    """Serialize lifecycle commands; only an exact real native frame establishes ready."""

    def __init__(
        self,
        control: SourceControl,
        slot: MetadataAdmission,
        receiver: FrameReceiver,
    ) -> None:
        self._control = control
        self._slot = slot
        self._receiver = receiver
        self._state_lock = threading.Lock()
        self._lifecycle = threading.Condition()
        self._lifecycle_active = False
        self._states: dict[str, SourceSnapshot] = {}

    def snapshot(self, camera_id: str) -> SourceSnapshot:
        with self._state_lock:
            return self._states.get(
                camera_id,
                SourceSnapshot(camera_id, SourceState.ABSENT, None, None),
            )

    def _await_ready(self, binding: SourceBinding) -> SourceSnapshot:
        token = self._slot.register_source(binding)
        _ = self._receiver.pull_now(binding.camera_id)
        try:
            _ = self._slot.wait_accepted(token, timeout_sec=2.0)
        except TimeoutError as error:
            raise SourceReadinessError("source_ready_timeout", binding.camera_id) from error
        ready = _snapshot(binding, SourceState.SOURCE_READY)
        with self._state_lock:
            current = self._states.get(binding.camera_id)
            if current != _snapshot(binding, SourceState.STARTING):
                raise SourceReadinessError("source_binding_changed", binding.camera_id)
            self._states[binding.camera_id] = ready
        return ready

    def add(self, camera_id: str, uri: str) -> SourceSnapshot:
        self._begin_lifecycle()
        try:
            current = self.snapshot(camera_id)
            self._set(_transition(current, SourceState.ADDING))
            try:
                binding = self._control.add_source(camera_id, uri)
            except ChildControlError:
                self._slot.remove_source(camera_id)
                self._set(_transition(current, SourceState.TOMBSTONED))
                raise
            self._set(_snapshot(binding, SourceState.STARTING))
            try:
                return self._await_ready(binding)
            except SourceReadinessError:
                rollback_error = self._rollback_native(camera_id)
                self._set(_transition(current, SourceState.TOMBSTONED))
                if rollback_error is not None:
                    raise SourceReadinessError(
                        "source_rollback_failed", camera_id
                    ) from rollback_error
                raise
        finally:
            self._end_lifecycle()

    def rebuild(self, camera_id: str, category: str) -> SourceSnapshot:
        self._begin_lifecycle()
        try:
            current = self.snapshot(camera_id)
            self._set(_transition(current, SourceState.REBUILDING))
            try:
                binding = self._control.source_failure(camera_id, category)
            except ChildControlError:
                self._set(_transition(current, SourceState.DEGRADED))
                raise
            self._set(_snapshot(binding, SourceState.STARTING))
            try:
                return self._await_ready(binding)
            except SourceReadinessError:
                rollback_error = self._rollback_native(camera_id)
                self._set(_transition(current, SourceState.DEGRADED))
                if rollback_error is not None:
                    raise SourceReadinessError(
                        "source_rollback_failed", camera_id
                    ) from rollback_error
                raise
        finally:
            self._end_lifecycle()

    def remove(self, camera_id: str) -> SourceSnapshot:
        self._begin_lifecycle()
        try:
            current = self.snapshot(camera_id)
            self._set(_transition(current, SourceState.REMOVING))
            self._control.remove_source(camera_id)
            self._slot.remove_source(camera_id)
            removed = _transition(current, SourceState.TOMBSTONED)
            self._set(removed)
            return removed
        finally:
            self._end_lifecycle()

    def _rollback_native(self, camera_id: str) -> ChildControlError | None:
        try:
            self._control.remove_source(camera_id)
        except ChildControlError as error:
            return error
        finally:
            self._slot.remove_source(camera_id)
        return None

    def _begin_lifecycle(self) -> None:
        with self._lifecycle:
            _ = self._lifecycle.wait_for(lambda: not self._lifecycle_active)
            self._lifecycle_active = True

    def _end_lifecycle(self) -> None:
        with self._lifecycle:
            self._lifecycle_active = False
            self._lifecycle.notify()

    def _set(self, snapshot: SourceSnapshot) -> None:
        with self._state_lock:
            self._states[snapshot.camera_id] = snapshot


def _snapshot(binding: SourceBinding, state: SourceState) -> SourceSnapshot:
    return SourceSnapshot(
        binding.camera_id,
        state,
        binding.source_generation,
        binding.stream_epoch,
    )


def _transition(snapshot: SourceSnapshot, state: SourceState) -> SourceSnapshot:
    return SourceSnapshot(
        snapshot.camera_id,
        state,
        snapshot.source_generation,
        snapshot.stream_epoch,
    )


__all__ = [
    "DarkSourceController",
    "SourceReadinessError",
    "SourceSnapshot",
    "SourceState",
]
