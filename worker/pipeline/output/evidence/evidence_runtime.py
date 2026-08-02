"""Production lifecycle for durable evidence reconciliation and sending."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from worker.pipeline.output.evidence.evidence_outbox import ClipId, EvidenceOutbox
from worker.pipeline.output.evidence.evidence_reconciliation import (
    reconcile_event_evidence,
    reconcile_finalized_clip,
)
from worker.pipeline.output.evidence.evidence_sender import (
    EvidenceSender,
    SenderConfig,
    SenderStep,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager

EXPORT_ENABLED_ENV = "ML_WORKER_EVENT_CLIP_EXPORT_ENABLED"


class SenderProtocol(Protocol):
    def run_once(self) -> SenderStep: ...


@dataclass(slots=True)
class EvidenceExportRuntime:
    store_dir: Path
    database_path: Path
    sender: SenderProtocol
    on_open: Callable[[], None] | None = None
    on_reconciled: Callable[[], None] | None = None
    on_holds_loaded: Callable[[], None] | None = None
    _initialized: bool = field(default=False, init=False)
    _held_at_startup: frozenset[ClipId] = field(default_factory=frozenset, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _pending_finalized: set[ClipId] = field(default_factory=set, init=False)
    _pending_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _wake_sender: threading.Event = field(default_factory=threading.Event, init=False)

    @classmethod
    def from_environment(
        cls,
        *,
        store_dir: Path,
        relay_url: str,
        relay_token: str | None,
        probe_camera_id: str,
        database_path: Path | None = None,
    ) -> EvidenceExportRuntime | None:
        if not _enabled(os.environ.get(EXPORT_ENABLED_ENV)):
            return None
        token = "" if relay_token is None else relay_token.strip()
        if not relay_url.strip() or not token or not probe_camera_id.strip():
            raise ValueError("evidence export requires relay URL, token, and camera")
        if database_path is None:
            database_path = store_dir / "worker-state.sqlite3"
        config = SenderConfig(
            relay_url=relay_url,
            relay_token=token,
            probe_camera_id=probe_camera_id,
            enabled=True,
        )
        return cls(
            store_dir=store_dir,
            database_path=database_path,
            sender=EvidenceSender(database_path, config),
        )

    def initialize_under_lock(self) -> None:
        self._initialized = False
        self._held_at_startup = frozenset()
        with EvidenceOutbox.open(self.database_path) as outbox:
            if self.on_open is not None:
                self.on_open()
            reconcile_event_evidence(self.store_dir, outbox)
            if self.on_reconciled is not None:
                self.on_reconciled()
            self._held_at_startup = frozenset(outbox.held_clip_ids())
            if self.on_holds_loaded is not None:
                self.on_holds_loaded()
        self._initialized = True

    def is_clip_held(self, clip_id: str) -> bool:
        if not self._initialized:
            return True
        try:
            with EvidenceOutbox.open(self.database_path) as outbox:
                return outbox.is_clip_held(ClipId(clip_id))
        except Exception:  # noqa: BLE001 - retention must fail closed
            return True

    def stager(
        self,
        *,
        camera_id: str,
        facility_id: str,
        resident_id: str | None,
        config_version: int,
    ) -> DurableEvidenceStager:
        return DurableEvidenceStager(
            self.database_path,
            camera_id=camera_id,
            facility_id=facility_id,
            resident_id=resident_id,
            config_version=config_version,
            clock=time.time,
        )

    def notify_clip_finalized(self, clip_id: ClipId) -> None:
        with self._pending_lock:
            self._pending_finalized.add(clip_id)
        self._wake_sender.set()

    def start_sender(self) -> None:
        if not self._initialized:
            raise RuntimeError("evidence runtime must initialize before sender start")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_sender,
            name="evidence-sender",
            daemon=True,
        )
        self._thread.start()

    def stop_sender(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._wake_sender.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._thread is None or not self._thread.is_alive():
            self._thread = None

    def _run_sender(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._drain_finalized_clips()
                step = self.sender.run_once()
            except Exception:  # noqa: BLE001 - durable rows remain retryable
                step = SenderStep.RETRY_SCHEDULED
            if step not in {SenderStep.EVENT_ACKED, SenderStep.CLIP_ACKED}:
                self._wake_sender.wait(1.0)
                self._wake_sender.clear()

    def _drain_finalized_clips(self) -> None:
        with self._pending_lock:
            pending = tuple(self._pending_finalized)
        for clip_id in pending:
            with EvidenceOutbox.open(self.database_path) as outbox:
                reconcile_finalized_clip(self.store_dir, clip_id, outbox)
            with self._pending_lock:
                self._pending_finalized.remove(clip_id)


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def export_enabled() -> bool:
    """Whether evidence export is switched on for this process.

    Public so the composition root can ask *before* building collaborators that
    only export needs. ``from_environment`` answers the same question
    internally, but it requires a ``store_dir`` argument -- and building that
    config reads clip-only environment variables that can themselves be
    malformed. Asking first stops a clip-only misconfiguration from failing a
    worker which has both clip recording and export switched off.
    """
    return _enabled(os.environ.get(EXPORT_ENABLED_ENV))


__all__ = ["EvidenceExportRuntime", "export_enabled"]
