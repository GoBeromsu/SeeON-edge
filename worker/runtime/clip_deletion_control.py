"""Authenticated worker command adapter for owned clip-byte deletion."""

from __future__ import annotations

import threading
from collections.abc import Callable

from worker.pipeline.output.evidence.evidence_retention import PurgeResult

DeleteClip = Callable[[str], PurgeResult]
PreflightClip = Callable[[str], PurgeResult | None]
LOCK_STRIPES = 64


class ClipDeletionControlService:
    """Serialize non-destructive checks and destructive commands per clip."""

    def __init__(
        self,
        *,
        preflight_clip: PreflightClip,
        delete_clip: DeleteClip,
    ) -> None:
        self._preflight_clip = preflight_clip
        self._delete_clip = delete_clip
        self._locks = tuple(threading.Lock() for _ in range(LOCK_STRIPES))

    def preflight(self, clip_id: str) -> dict[str, object]:
        with self._lock_for(clip_id):
            result = self._preflight_clip(clip_id)
            status = "READY" if result is None else result.value
            return {"clip_id": clip_id, "status": status}

    def delete(self, clip_id: str) -> dict[str, object]:
        with self._lock_for(clip_id):
            result = self._delete_clip(clip_id)
            return {"clip_id": clip_id, "status": result.value}

    def _lock_for(self, clip_id: str) -> threading.Lock:
        return self._locks[hash(clip_id) % LOCK_STRIPES]


__all__ = ["ClipDeletionControlService", "LOCK_STRIPES"]
