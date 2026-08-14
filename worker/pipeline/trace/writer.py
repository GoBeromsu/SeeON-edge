from __future__ import annotations

import queue
import sqlite3
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from worker.pipeline.trace.models import (
    RecoveredCameraTrace,
    TraceContractError,
    TraceFrame,
    TracePersistenceError,
    TraceWriterStats,
    trace_frame_row_count,
    trace_frame_size_bytes,
)
from worker.pipeline.trace.store import TraceStore


@dataclass(frozen=True, slots=True)
class TraceRetentionPolicy:
    """Explicit handoff, shape, row, byte, camera, and stream-time bounds."""

    max_frames_per_camera: int
    max_age_seconds: float
    max_pending_frames: int
    max_batch_size: int
    max_numeric_values_per_decision: int
    persistence_timeout_seconds: float = 6.0
    max_persistence_attempts: int = 3
    max_cameras: int = 64
    max_persons_per_frame: int = 128
    max_beds_per_frame: int = 32
    max_components_per_frame: int = 64
    max_decisions_per_frame: int = 64
    max_rows_per_frame: int = 1_024
    max_bytes_per_frame: int = 262_144
    max_total_frames: int = 150_000
    max_total_rows: int = 1_000_000
    max_total_bytes: int = 268_435_456

    def __post_init__(self) -> None:
        positive = (
            self.max_frames_per_camera,
            self.max_age_seconds,
            self.max_pending_frames,
            self.max_batch_size,
            self.max_numeric_values_per_decision,
            self.persistence_timeout_seconds,
            self.max_persistence_attempts,
            self.max_cameras,
            self.max_persons_per_frame,
            self.max_beds_per_frame,
            self.max_components_per_frame,
            self.max_decisions_per_frame,
            self.max_rows_per_frame,
            self.max_bytes_per_frame,
            self.max_total_frames,
            self.max_total_rows,
            self.max_total_bytes,
        )
        if any(value <= 0 for value in positive) or self.max_batch_size > self.max_pending_frames:
            raise ValueError("trace retention bounds must be positive and internally ordered")
        if self.max_total_rows < self.max_rows_per_frame:
            raise ValueError("global trace row bound must cover one frame")
        if self.max_total_bytes < self.max_bytes_per_frame:
            raise ValueError("global trace byte bound must cover one frame")

    @classmethod
    def testing(cls) -> TraceRetentionPolicy:
        return cls(
            max_frames_per_camera=32,
            max_age_seconds=300.0,
            max_pending_frames=8,
            max_batch_size=4,
            max_numeric_values_per_decision=32,
            max_total_frames=64,
            max_total_rows=2_048,
            max_total_bytes=1_048_576,
        )


DEFAULT_TRACE_RETENTION_POLICY = TraceRetentionPolicy(
    max_frames_per_camera=3_000,
    max_age_seconds=600.0,
    max_pending_frames=64,
    max_batch_size=16,
    max_numeric_values_per_decision=64,
)


@dataclass(slots=True)
class _WriteRequest:
    frame: TraceFrame
    completed: threading.Event | None
    error: BaseException | None = None


@dataclass(slots=True)
class _Barrier:
    completed: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


class _Lifecycle(StrEnum):
    NEW = "new"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


_QueueItem = _WriteRequest | _Barrier | _Stop


class BoundedTraceWriter:
    """Single worker-owned writer with bounded backpressure and durable event waits."""

    def __init__(
        self,
        database_path: Path,
        policy: TraceRetentionPolicy = DEFAULT_TRACE_RETENTION_POLICY,
    ) -> None:
        self.database_path = database_path
        self.policy = policy
        self._store = TraceStore(database_path)
        self._queue: queue.Queue[_QueueItem] = queue.Queue(policy.max_pending_frames + 1)
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state = _Lifecycle.NEW
        self._pending_frames = 0
        self._dropped_by_camera: dict[str, int] = {}
        self._failed_by_camera: dict[str, int] = {}
        self._handoff_dropped = 0
        self._persisted = 0
        self._failed_batches = 0
        self._persistence_failed_frames = 0
        self._unobserved_failed_frames = 0
        self._retry_attempts = 0
        self._rejected_frames = 0
        self._duplicate_frames = 0
        self._terminal_error: BaseException | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._state is _Lifecycle.RUNNING:
                return
            if self._state is not _Lifecycle.NEW:
                raise TracePersistenceError("trace writer cannot restart after it has stopped")
            self._start_locked()

    def _start_locked(self) -> None:
        thread = threading.Thread(
            target=self._run,
            name="bounded-analysis-trace-writer",
            daemon=True,
        )
        self._thread = thread
        self._state = _Lifecycle.RUNNING
        thread.start()

    def submit(self, frame: TraceFrame, *, require_persisted: bool = False) -> bool:
        try:
            self._validate_frame(frame)
        except TraceContractError:
            with self._lock:
                self._rejected_frames += 1
            raise
        completed = threading.Event() if require_persisted else None
        request = _WriteRequest(frame, completed)
        with self._lifecycle_lock:
            if self._state in {_Lifecycle.STOPPING, _Lifecycle.STOPPED}:
                with self._lock:
                    self._rejected_frames += 1
                raise TracePersistenceError("trace writer is stopped")
            if require_persisted and self._state is not _Lifecycle.RUNNING:
                with self._lock:
                    self._rejected_frames += 1
                raise TracePersistenceError("persisted trace submission requires a started writer")
            if self._pending_frames >= self.policy.max_pending_frames:
                camera_id = frame.analysis.frame_key[1]
                with self._lock:
                    self._handoff_dropped += 1
                    self._dropped_by_camera[camera_id] = (
                        self._dropped_by_camera.get(camera_id, 0) + 1
                    )
                return False
            self._pending_frames += 1
            self._queue.put_nowait(request)
        if completed is None:
            return True
        if not completed.wait(self.policy.persistence_timeout_seconds):
            raise TracePersistenceError("trace persistence exceeded its bounded timeout")
        if request.error is not None:
            raise TracePersistenceError(
                f"trace persistence failed: {request.error}"
            ) from request.error
        return True

    def _validate_frame(self, frame: TraceFrame) -> None:
        analysis = frame.analysis
        limits = (
            (len(analysis.persons), self.policy.max_persons_per_frame, "max_persons_per_frame"),
            (len(analysis.beds), self.policy.max_beds_per_frame, "max_beds_per_frame"),
            (
                len(analysis.components),
                self.policy.max_components_per_frame,
                "max_components_per_frame",
            ),
            (len(frame.decisions), self.policy.max_decisions_per_frame, "max_decisions_per_frame"),
            (trace_frame_row_count(frame), self.policy.max_rows_per_frame, "max_rows_per_frame"),
            (trace_frame_size_bytes(frame), self.policy.max_bytes_per_frame, "max_bytes_per_frame"),
        )
        for actual, maximum, name in limits:
            if actual > maximum:
                raise TraceContractError(f"decision trace exceeds {name}")
        for decision in frame.decisions:
            if decision.analysis_trace_id != analysis.trace_id:
                raise TraceContractError("decision trace references another analysis trace")
            count = len(decision.snapshot.values) + len(decision.snapshot.missing_values)
            if count > self.policy.max_numeric_values_per_decision:
                raise TraceContractError("decision trace exceeds max_numeric_values_per_decision")

    def flush(self) -> None:
        if not self._control_lock.acquire(timeout=self.policy.persistence_timeout_seconds):
            raise TracePersistenceError("trace flush could not acquire lifecycle control")
        try:
            with self._lifecycle_lock:
                if self._state is _Lifecycle.NEW:
                    return
                if self._state is not _Lifecycle.RUNNING:
                    raise TracePersistenceError("trace writer is stopped")
                barrier = _Barrier()
                self._queue.put_nowait(barrier)
            if not barrier.completed.wait(self.policy.persistence_timeout_seconds):
                raise TracePersistenceError("trace flush exceeded its bounded timeout")
            if barrier.error is not None:
                raise TracePersistenceError("trace flush failed") from barrier.error
        finally:
            self._control_lock.release()

    def stop(self) -> None:
        if not self._control_lock.acquire(timeout=self.policy.persistence_timeout_seconds):
            raise TracePersistenceError("trace stop could not acquire lifecycle control")
        try:
            with self._lifecycle_lock:
                if self._state is _Lifecycle.STOPPED:
                    return
                if self._state is _Lifecycle.STOPPING:
                    raise TracePersistenceError("trace writer stop is already in progress")
                if self._state is _Lifecycle.NEW:
                    self._start_locked()
                thread = self._thread
                assert thread is not None
                self._state = _Lifecycle.STOPPING
                self._queue.put_nowait(_Stop())
            thread.join(self.policy.persistence_timeout_seconds)
            if thread.is_alive():
                raise TracePersistenceError("trace writer did not stop within its bounded timeout")
            with self._lifecycle_lock:
                self._thread = None
                self._state = _Lifecycle.STOPPED
            with self._lock:
                failures = self._unobserved_failed_frames
                terminal_error = self._terminal_error
            if failures:
                raise TracePersistenceError(
                    f"trace writer failed to persist {failures} accepted trace frame(s)"
                ) from terminal_error
        finally:
            self._control_lock.release()

    def stats(self) -> TraceWriterStats:
        with self._lock:
            return TraceWriterStats(
                handoff_dropped_frames=self._handoff_dropped,
                persisted_frames=self._persisted,
                failed_batches=self._failed_batches,
                persistence_failed_frames=self._persistence_failed_frames,
                retry_attempts=self._retry_attempts,
                rejected_frames=self._rejected_frames,
                duplicate_frames=self._duplicate_frames,
            )

    def recover_camera(self, camera_id: str) -> RecoveredCameraTrace:
        return self._store.recover_camera(camera_id)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            requests: list[_WriteRequest] = []
            control: _Barrier | _Stop | None = None
            if isinstance(item, _WriteRequest):
                requests.append(item)
                with self._lifecycle_lock:
                    self._pending_frames -= 1
                while len(requests) < self.policy.max_batch_size:
                    try:
                        following = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(following, _WriteRequest):
                        requests.append(following)
                        with self._lifecycle_lock:
                            self._pending_frames -= 1
                    else:
                        control = following
                        break
            else:
                control = item
            error = self._persist(requests)
            for request in requests:
                request.error = error
                if request.completed is not None:
                    request.completed.set()
            if isinstance(control, _Barrier):
                with self._lock:
                    control.error = self._terminal_error
                control.completed.set()
            elif isinstance(control, _Stop):
                return

    def _persist(self, requests: list[_WriteRequest]) -> BaseException | None:
        with self._lock:
            dropped = dict(self._dropped_by_camera)
            previously_failed = dict(self._failed_by_camera)
        if not requests and not dropped and not previously_failed:
            return None
        error: BaseException | None = None
        for attempt in range(self.policy.max_persistence_attempts):
            try:
                persisted = self._store.persist_batch(
                    [request.frame for request in requests],
                    max_frames_per_camera=self.policy.max_frames_per_camera,
                    max_age_seconds=self.policy.max_age_seconds,
                    max_cameras=self.policy.max_cameras,
                    max_total_frames=self.policy.max_total_frames,
                    max_total_rows=self.policy.max_total_rows,
                    max_total_bytes=self.policy.max_total_bytes,
                    dropped_by_camera=dropped,
                    failed_by_camera=previously_failed,
                )
            except BaseException as caught:  # noqa: BLE001 - delivered to exact submitter
                error = caught
                if (
                    attempt + 1 < self.policy.max_persistence_attempts
                    and isinstance(caught, sqlite3.OperationalError)
                    and any(token in str(caught).lower() for token in ("busy", "locked"))
                ):
                    with self._lock:
                        self._retry_attempts += 1
                    continue
                break
            with self._lock:
                self._persisted += persisted
                self._duplicate_frames += len(requests) - persisted
                self._subtract_counts(self._dropped_by_camera, dropped)
                self._subtract_counts(self._failed_by_camera, previously_failed)
            return None
        with self._lock:
            self._failed_batches += 1
            self._persistence_failed_frames += len(requests)
            self._unobserved_failed_frames += sum(request.completed is None for request in requests)
            self._terminal_error = error
            for request in requests:
                camera_id = request.frame.analysis.frame_key[1]
                self._failed_by_camera[camera_id] = self._failed_by_camera.get(camera_id, 0) + 1
        return error

    @staticmethod
    def _subtract_counts(target: dict[str, int], persisted: dict[str, int]) -> None:
        for camera_id, count in persisted.items():
            remaining = target.get(camera_id, 0) - count
            if remaining > 0:
                target[camera_id] = remaining
            else:
                target.pop(camera_id, None)


__all__ = [
    "BoundedTraceWriter",
    "DEFAULT_TRACE_RETENTION_POLICY",
    "TraceRetentionPolicy",
]
