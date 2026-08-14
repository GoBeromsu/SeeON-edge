"""Composition-root adapter: authenticated operator clip deletion over HTTP.

Mirrors ``worker.runtime.derivative_runtime.DerivativeControlService`` --
a thin JSON-safe wrapper the shared control HTTP server calls -- but for the
one-shot "delete this primary clip" command instead of derivative rendering.
The actual deletion mechanics (hold checks, manifest/containment
verification, filesystem removal, DB tombstone) all live in
``ClipMaintenance``/``EvidenceRetention``; this class only adds the
idempotent-short-circuit and per-clip serialization an HTTP handler needs.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from worker.pipeline.output.evidence.evidence_retention import PurgeResult

DeleteClip = Callable[[str], PurgeResult]
ClipRetentionState = Callable[[str], "str | None"]
CompletePendingPurge = Callable[[str], None]
LOCK_STRIPES = 64


class ClipDeletionControlService:
    def __init__(
        self,
        *,
        delete_clip: DeleteClip,
        retention_state: ClipRetentionState,
        complete_pending_purge: CompletePendingPurge | None = None,
    ) -> None:
        self._delete_clip = delete_clip
        self._retention_state = retention_state
        self._complete_pending_purge = complete_pending_purge
        self._locks = tuple(threading.Lock() for _ in range(LOCK_STRIPES))

    def delete(self, clip_id: str) -> dict[str, object]:
        """Return ``{"clip_id": ..., "status": ...}`` -- always, never ``None``.

        A clip already tombstoned ``PURGED`` short-circuits before touching
        the filesystem again: calling ``purge_clip`` a second time against an
        already-absent directory would (correctly) report ``MISSING``, which
        would misreport a genuinely completed deletion as a failure. Every
        other state (never staged, ``PENDING``, or ``FAILED``) retries the
        real purge, which is itself idempotent-safe (``begin_clip_retention``
        no-ops on an existing ``PENDING``/``PURGED`` row).
        """
        with self._lock_for(clip_id):
            retention_state = self._retention_state(clip_id)
            if retention_state == "PURGED":
                return {"clip_id": clip_id, "status": PurgeResult.PURGED.value}
            result = self._delete_clip(clip_id)
            # A crash can occur after rmtree() but before complete_purge(). A
            # same-process operator retry must converge exactly like startup
            # reconciliation, rather than turning the durable pending intent
            # into a misleading MISSING/FAILED tombstone.
            if (
                retention_state == "PENDING"
                and result is PurgeResult.MISSING
                and self._complete_pending_purge is not None
            ):
                self._complete_pending_purge(clip_id)
                result = PurgeResult.PURGED
            return {"clip_id": clip_id, "status": result.value}

    def _lock_for(self, clip_id: str) -> threading.Lock:
        return self._locks[hash(clip_id) % LOCK_STRIPES]


__all__ = ["ClipDeletionControlService", "LOCK_STRIPES"]
