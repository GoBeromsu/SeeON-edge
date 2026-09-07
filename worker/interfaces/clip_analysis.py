"""Stored-clip analysis control seam used by the relay HTTP server."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol


class ClipAnalysisDisabledError(RuntimeError):
    """Stored-clip analysis is deliberately disabled for this deployment."""


class ClipAnalysisStatus(Protocol):
    state: Literal["idle", "running", "available", "failed"]
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
    ) -> bool: ...

    def cancel(self, clip_id: str) -> bool: ...

    def shutdown(self) -> None: ...


__all__ = ["ClipAnalysisDisabledError", "ClipAnalysisStatus", "ClipAnalysisSupervisor"]
