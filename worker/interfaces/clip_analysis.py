"""Stored-clip analysis control seam used by the relay HTTP server."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

Admission = Literal[
    "queued", "already_queued", "already_running", "available", "queue_full", "stopped", "rejected"
]


class ClipAnalysisDisabledError(RuntimeError):
    """Stored-clip analysis is deliberately disabled for this deployment."""


class ClipAnalysisStatus(Protocol):
    state: Literal["idle", "queued", "running", "available", "failed"]
    reason: str | None


class ClipAnalysisSupervisor(Protocol):
    def status(self, clip_id: str) -> ClipAnalysisStatus: ...

    def trigger(
        self,
        clip_id: str,
        clip_path: Path,
        clip_sha256: str,
        *,
        size_bytes: int,
        duration_ms: int,
        width: int,
        height: int,
    ) -> Admission: ...

    def enqueue(
        self,
        clip_id: str,
        clip_path: Path,
        clip_sha256: str,
        *,
        size_bytes: int,
        duration_ms: int,
        width: int,
        height: int,
        front: bool = False,
    ) -> Admission: ...

    def wait_for_capacity(self, timeout: float) -> bool: ...

    def notify(
        self, clip_id: str, clip_path: Path, clip_sha256: str, *, size_bytes: int, duration_ms: int
    ) -> None: ...

    def cancel(self, clip_id: str) -> bool: ...

    def shutdown(self) -> None: ...


__all__ = ["ClipAnalysisDisabledError", "ClipAnalysisStatus", "ClipAnalysisSupervisor"]
