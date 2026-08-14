from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    AnnotatedDerivativeLimits,
    BoundedDerivativeQueue,
    CpuAnnotatedStillRenderer,
    CpuAnnotatedVideoRenderer,
    DerivativeArtifact,
    DerivativeCancelled,
    DerivativeKind,
    DerivativeRenderError,
    DerivativeUnavailableReason,
)
from worker.pipeline.output.evidence.derivative_artifact_store import (
    DerivativeArtifactCapacityError,
    DerivativeArtifactConflictError,
    DerivativeArtifactStore,
)
from worker.pipeline.output.evidence.derivative_job_source import (
    CentralDerivativeJobSource,
    DerivativeSourceUnavailable,
)
from worker.pipeline.output.evidence.derivative_job_store import (
    DerivativeJobRecord,
    DerivativeJobState,
    DerivativeJobStore,
)

LOGGER = logging.getLogger(__name__)


class DerivativeRenderer(Protocol):
    def render(
        self,
        job: AnnotatedDerivativeJob,
        destination: Path,
        *,
        cancelled: threading.Event | None = None,
    ) -> DerivativeArtifact: ...


@dataclass(frozen=True, slots=True)
class DerivativeRuntimeStatus:
    incident_id: str
    derivative_kind: DerivativeKind
    request_id: str
    state: DerivativeJobState
    reason: str | None
    attempt_count: int


class DerivativeControlService:
    """JSON-safe authenticated HTTP adapter over the production supervisor."""

    def __init__(self, runtime: DerivativeProductionRuntime) -> None:
        self.runtime = runtime

    def request(self, clip_id: str, kind: str) -> dict[str, object]:
        return _status_payload(self.runtime.request(clip_id, kind))

    def cancel(self, clip_id: str, kind: str) -> dict[str, object] | None:
        status = self.runtime.cancel(clip_id, kind)
        return None if status is None else _status_payload(status)

    def status(self, clip_id: str, kind: str) -> dict[str, object] | None:
        status = self.runtime.status(clip_id, kind)
        return None if status is None else _status_payload(status)


class DerivativeProductionRuntime:
    """One bounded CPU derivative worker with durable restart semantics."""

    def __init__(
        self,
        database_path: Path,
        store_root: Path,
        *,
        limits: AnnotatedDerivativeLimits | None = None,
        still_renderer: DerivativeRenderer | None = None,
        video_renderer: DerivativeRenderer | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.database_path = database_path
        self.store_root = store_root
        self.limits = limits or AnnotatedDerivativeLimits()
        self.source = CentralDerivativeJobSource(database_path, store_root)
        self.jobs = DerivativeJobStore(database_path)
        self.artifacts = DerivativeArtifactStore(
            database_path,
            store_root,
            max_disk_bytes=self.limits.max_disk_bytes,
        )
        self.still_renderer = still_renderer or CpuAnnotatedStillRenderer(limits=self.limits)
        self.video_renderer = video_renderer or CpuAnnotatedVideoRenderer(limits=self.limits)
        self._clock = clock or _utc_now
        self._queue: BoundedDerivativeQueue | None = None
        self._thread: threading.Thread | None = None
        self._running: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._stopping = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
        now = self._clock()
        self.store_root.mkdir(parents=True, exist_ok=True)
        _ = self.artifacts.reconcile(updated_at=now)
        _ = self.jobs.reset_running(updated_at=now)
        queue = BoundedDerivativeQueue(self.limits)
        for record in self.jobs.recoverable():
            if record.cancel_requested:
                self._cancel_recovered(record)
                continue
            try:
                queue.submit(self.source.for_incident(record.incident_id, record.derivative_kind))
            except DerivativeSourceUnavailable as error:
                self.jobs.mark_unavailable_record(record, error.reason, updated_at=self._clock())
            except OverflowError:
                self.jobs.mark_unavailable_record(
                    record,
                    DerivativeUnavailableReason.RESOURCE_LIMIT,
                    updated_at=self._clock(),
                )
        thread = threading.Thread(target=self._run, name="derivative-production", daemon=True)
        with self._lock:
            self._queue = queue
            self._thread = thread
        thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            self._stopping = True
            queue = self._queue
            running = tuple(self._running.values())
            thread = self._thread
        if queue is not None:
            _ = queue.close()
        for cancellation in running:
            cancellation.set()
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError("derivative runtime did not stop within its bound")
        with self._lock:
            self._queue = None
            self._thread = None
            self._running.clear()
            self._changed.notify_all()

    def request(self, clip_id: str, kind: DerivativeKind | str) -> DerivativeRuntimeStatus:
        resolved_kind = DerivativeKind(kind)
        job = self.source.for_clip(clip_id, resolved_kind)
        result = self.jobs.request(job, updated_at=self._clock())
        if result.scheduled:
            with self._lock:
                queue = self._queue
            if queue is None:
                raise RuntimeError("derivative runtime is not started")
            try:
                queue.submit(job)
            except OverflowError:
                _ = self.jobs.mark_unavailable(
                    job,
                    DerivativeUnavailableReason.RESOURCE_LIMIT,
                    updated_at=self._clock(),
                )
        self._notify_changed()
        record = self.jobs.get(job.incident_id, resolved_kind)
        assert record is not None
        return _status(record)

    def cancel(self, clip_id: str, kind: DerivativeKind | str) -> DerivativeRuntimeStatus | None:
        resolved_kind = DerivativeKind(kind)
        try:
            job = self.source.for_clip(clip_id, resolved_kind)
        except DerivativeSourceUnavailable:
            return None
        accepted = self.jobs.request_cancel(
            job.incident_id, resolved_kind, updated_at=self._clock()
        )
        if not accepted:
            # Publication (or another terminal transition) already committed.
            record = self.jobs.get(job.incident_id, resolved_kind)
            return None if record is None else _status(record)
        with self._lock:
            queue = self._queue
            cancellation = self._running.get(job.identity)
        if queue is not None:
            queue.cancel_request(job.identity)
        if cancellation is not None:
            cancellation.set()
        record = self.jobs.get(job.incident_id, resolved_kind)
        self._notify_changed()
        return None if record is None else _status(record)

    def status(self, clip_id: str, kind: DerivativeKind | str) -> DerivativeRuntimeStatus | None:
        resolved_kind = DerivativeKind(kind)
        try:
            job = self.source.for_clip(clip_id, resolved_kind)
        except DerivativeSourceUnavailable:
            return None
        record = self.jobs.get(job.incident_id, resolved_kind)
        return None if record is None else _status(record)

    def wait_for_terminal(
        self, incident_id: str, kind: DerivativeKind, *, timeout: float
    ) -> DerivativeRuntimeStatus | None:
        terminal = {
            DerivativeJobState.AVAILABLE,
            DerivativeJobState.UNAVAILABLE,
            DerivativeJobState.CORRUPT,
            DerivativeJobState.CANCELLED,
        }
        with self._changed:
            ready = self._changed.wait_for(
                lambda: (
                    (record := self.jobs.get(incident_id, kind)) is not None
                    and record.state in terminal
                ),
                timeout=timeout,
            )
        record = self.jobs.get(incident_id, kind)
        return None if not ready or record is None else _status(record)

    def _run(self) -> None:
        while True:
            with self._lock:
                queue = self._queue
            if queue is None:
                return
            try:
                job = queue.take_wait()
            except DerivativeCancelled as error:
                if error.job is not None:
                    _ = self.jobs.mark_cancelled(error.job, updated_at=self._clock())
                    self._notify_changed()
                continue
            if job is None:
                return
            self._execute(job)

    def _execute(self, job: AnnotatedDerivativeJob) -> None:
        if not self.jobs.mark_running(job, updated_at=self._clock()):
            record = self.jobs.get(job.incident_id, job.derivative_kind)
            if record is not None and record.cancel_requested:
                _ = self.jobs.mark_cancelled(job, updated_at=self._clock())
                self._notify_changed()
            return
        cancellation = threading.Event()
        with self._lock:
            self._running[job.identity] = cancellation
        filename = f"{job.identity}{job.derivative_kind.extension}"
        work = self.store_root / ".derivative-work" / filename
        try:
            renderer = (
                self.still_renderer
                if job.derivative_kind is DerivativeKind.STILL
                else self.video_renderer
            )
            artifact = renderer.render(job, work, cancelled=cancellation)
            _ = self.artifacts.publish(job, artifact, updated_at=self._clock())
        except DerivativeCancelled:
            # Operator cancel is durable across stop/interrupt. Prefer CANCELLED
            # whenever the DB/request flag already won; only uncancelled stop
            # paths return work to PENDING for restart recovery.
            record = self.jobs.get(job.incident_id, job.derivative_kind)
            cancel_won = record is not None and record.cancel_requested
            with self._lock:
                stopping = self._stopping
            if cancel_won or not stopping:
                _ = self.jobs.mark_cancelled(job, updated_at=self._clock())
            elif not self.jobs.mark_interrupted(job, updated_at=self._clock()):
                latest = self.jobs.get(job.incident_id, job.derivative_kind)
                if latest is not None and latest.cancel_requested:
                    _ = self.jobs.mark_cancelled(job, updated_at=self._clock())
        except DerivativeArtifactCapacityError:
            _ = self.jobs.mark_unavailable(
                job,
                DerivativeUnavailableReason.RESOURCE_LIMIT,
                updated_at=self._clock(),
            )
        except (DerivativeRenderError, DerivativeArtifactConflictError, OSError, ValueError):
            LOGGER.exception(
                "derivative render failed incident_id=%s kind=%s",
                job.incident_id,
                job.derivative_kind.value,
            )
            _ = self.jobs.mark_unavailable(
                job,
                DerivativeUnavailableReason.RENDER_FAILED,
                updated_at=self._clock(),
            )
        finally:
            work.unlink(missing_ok=True)
            with self._lock:
                self._running.pop(job.identity, None)
            self._notify_changed()

    def _cancel_recovered(self, record: DerivativeJobRecord) -> None:
        try:
            job = self.source.for_incident(record.incident_id, record.derivative_kind)
        except DerivativeSourceUnavailable as error:
            _ = self.jobs.mark_unavailable_record(record, error.reason, updated_at=self._clock())
            return
        _ = self.jobs.mark_cancelled(job, updated_at=self._clock())

    def _notify_changed(self) -> None:
        with self._changed:
            self._changed.notify_all()


def _status(record: DerivativeJobRecord) -> DerivativeRuntimeStatus:
    return DerivativeRuntimeStatus(
        record.incident_id,
        record.derivative_kind,
        record.request_id,
        record.state,
        record.reason,
        record.attempt_count,
    )


def _status_payload(status: DerivativeRuntimeStatus) -> dict[str, object]:
    return {
        "incident_id": status.incident_id,
        "kind": status.derivative_kind.value,
        "request_id": status.request_id,
        "state": status.state.value,
        "reason": status.reason,
        "attempt_count": status.attempt_count,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "DerivativeControlService",
    "DerivativeProductionRuntime",
    "DerivativeRuntimeStatus",
]
