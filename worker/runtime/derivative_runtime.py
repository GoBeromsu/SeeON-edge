from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from worker.pipeline.output.annotated_derivative import (
    AnnotatedDerivativeJob,
    AnnotatedDerivativeLimits,
    CpuAnnotatedStillRenderer,
    DerivativeArtifact,
    DerivativeKind,
    DerivativeRenderError,
)
from worker.pipeline.output.evidence.derivative_artifact_store import (
    DerivativeArtifactCapacityError,
    DerivativeArtifactConflictError,
    DerivativeArtifactStore,
    StoredDerivativeArtifact,
)
from worker.pipeline.output.native_annotated_derivative import (
    NativeGpuAnnotatedVideoRenderer,
)
from worker.pipeline.trace.models import DetailUnavailableReason

LOGGER = logging.getLogger(__name__)


class DerivativeRenderer(Protocol):
    def render(
        self,
        job: AnnotatedDerivativeJob,
        destination: Path,
        *,
        cancelled: threading.Event | None = None,
    ) -> DerivativeArtifact: ...


class DerivativeOutcome(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class DerivativeCommand:
    """Backend-owned rendering intent, including any trace-detail disposition."""

    job: AnnotatedDerivativeJob
    detail_unavailable_reason: DetailUnavailableReason | None = None


@dataclass(frozen=True, slots=True)
class DerivativeReceipt:
    command_id: str
    incident_id: str
    derivative_kind: DerivativeKind
    outcome: DerivativeOutcome
    reason: str | None
    artifact: StoredDerivativeArtifact | None = None


class DerivativeCommandExecutor:
    """Stateless command executor; artifacts are the idempotency witness."""

    def __init__(
        self,
        store_root: Path,
        *,
        limits: AnnotatedDerivativeLimits | None = None,
        still_renderer: DerivativeRenderer | None = None,
        video_renderer: DerivativeRenderer | None = None,
    ) -> None:
        self.store_root = store_root
        self.limits = limits or AnnotatedDerivativeLimits()
        self.artifacts = DerivativeArtifactStore(
            store_root, max_disk_bytes=self.limits.max_disk_bytes
        )
        self.still_renderer = still_renderer or CpuAnnotatedStillRenderer(limits=self.limits)
        self.video_renderer = video_renderer or NativeGpuAnnotatedVideoRenderer(
            limits=self.limits
        )

    def execute(self, command: DerivativeCommand) -> DerivativeReceipt:
        job = command.job
        if command.detail_unavailable_reason is not None:
            return DerivativeReceipt(
                job.identity,
                job.incident_id,
                job.derivative_kind,
                DerivativeOutcome.UNAVAILABLE,
                command.detail_unavailable_reason.value,
            )
        existing = self.artifacts.receipt_for(job)
        if existing is not None:
            return _available_receipt(job, existing)
        work = (
            self.store_root / ".derivative-work" / f"{job.identity}{job.derivative_kind.extension}"
        )
        try:
            renderer = (
                self.still_renderer
                if job.derivative_kind is DerivativeKind.STILL
                else self.video_renderer
            )
            artifact = renderer.render(job, work)
            stored = self.artifacts.publish(job, artifact)
            return _available_receipt(job, stored)
        except DerivativeArtifactCapacityError:
            return DerivativeReceipt(
                job.identity,
                job.incident_id,
                job.derivative_kind,
                DerivativeOutcome.UNAVAILABLE,
                "resource-limit",
            )
        except (DerivativeRenderError, DerivativeArtifactConflictError, OSError, ValueError):
            LOGGER.exception(
                "derivative render failed incident_id=%s kind=%s",
                job.incident_id,
                job.derivative_kind.value,
            )
            return DerivativeReceipt(
                job.identity,
                job.incident_id,
                job.derivative_kind,
                DerivativeOutcome.UNAVAILABLE,
                "render-failed",
            )
        finally:
            work.unlink(missing_ok=True)


def _available_receipt(
    job: AnnotatedDerivativeJob, artifact: StoredDerivativeArtifact
) -> DerivativeReceipt:
    """Receipt facts are the durable command result, not renderer diagnostics."""
    return DerivativeReceipt(
        job.identity,
        job.incident_id,
        job.derivative_kind,
        DerivativeOutcome.AVAILABLE,
        None,
        StoredDerivativeArtifact(
            artifact.incident_id,
            artifact.derivative_kind,
            artifact.derivative_id,
            artifact.media_relpath,
            artifact.sha256,
            artifact.size_bytes,
            artifact.mime_type,
            0,
            0,
            0,
            0,
        ),
    )


class DerivativeFuture:
    def __init__(self) -> None:
        self._done = threading.Event()
        self._receipt: DerivativeReceipt | None = None

    def complete(self, receipt: DerivativeReceipt) -> None:
        self._receipt = receipt
        self._done.set()

    def result(self, timeout: float | None = None) -> DerivativeReceipt:
        if not self._done.wait(timeout) or self._receipt is None:
            raise TimeoutError("derivative command did not complete")
        return self._receipt


class DerivativeControlService:
    """HTTP adapter for command delivery; clip-only requests are explicitly unavailable."""

    def __init__(self, executor: DerivativeCommandExecutor) -> None:
        self.executor = executor
        self._pending = threading.BoundedSemaphore(executor.limits.max_pending_jobs)

    def submit(self, command: DerivativeCommand) -> DerivativeFuture:
        if not self._pending.acquire(blocking=False):
            raise OverflowError("bounded derivative command capacity exceeded")
        future = DerivativeFuture()

        def execute() -> None:
            try:
                future.complete(self.executor.execute(command))
            finally:
                self._pending.release()

        threading.Thread(
            target=execute, name="native-derivative-render", daemon=True
        ).start()
        return future

    def execute(self, command: DerivativeCommand) -> DerivativeReceipt:
        return self.executor.execute(command)

    def request(self, clip_id: str, kind: str) -> dict[str, object]:
        resolved_kind = DerivativeKind(kind)
        return {
            "clip_id": clip_id,
            "kind": resolved_kind.value,
            "outcome": DerivativeOutcome.UNAVAILABLE.value,
            "reason": "command-payload-required",
        }

    def cancel(self, clip_id: str, kind: str) -> dict[str, object] | None:
        del clip_id, kind
        return None

    def status(self, clip_id: str, kind: str) -> dict[str, object] | None:
        del clip_id, kind
        return None


__all__ = [
    "DerivativeCommand",
    "DerivativeCommandExecutor",
    "DerivativeControlService",
    "DerivativeFuture",
    "DerivativeOutcome",
    "DerivativeReceipt",
]
