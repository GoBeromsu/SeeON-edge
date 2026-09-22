"""Runtime-owned drain that batches lanes and POSTs them to the Backend."""

from __future__ import annotations

import logging
import threading

from shared.events.evidence_export_contract import DeliveryFailure
from shared.events.execution_records import WireBatch, WireBatchReceipt, WireProvenance
from shared.events.execution_records_client import ExecutionRecordsClient
from worker.pipeline.diagnostics.lanes import DrainedLane, ExecutionRecordLanes

LOGGER = logging.getLogger(__name__)


class ExecutionRecordExporter:
    """Drain thread. Export failure drops the batch and records an export-failed gap."""

    def __init__(
        self,
        *,
        lanes: ExecutionRecordLanes,
        client: ExecutionRecordsClient,
        provenance: WireProvenance,
        batch_max: int,
        flush_ms: int,
    ) -> None:
        if batch_max < 1 or flush_ms < 1:
            raise ValueError("batch_max and flush_ms must be positive")
        self._lanes = lanes
        self._client = client
        self._provenance = provenance
        self._batch_max = batch_max
        self._flush_sec = flush_ms / 1000.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._receipts: list[WireBatchReceipt] = []
        self._failures: list[DeliveryFailure] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        thread = threading.Thread(target=self._run, name="execution-records-export", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        if thread is None or not thread.is_alive():
            self._thread = None

    def receipts(self) -> tuple[WireBatchReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def failures(self) -> tuple[DeliveryFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    def flush_once(self) -> None:
        for camera_id, worker_boot_id in self._lanes.cameras_with_work():
            drained = self._lanes.drain_for(camera_id, worker_boot_id, limit=self._batch_max)
            if drained is None:
                continue
            self._post(drained)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._lanes.wait_for_work(timeout_sec=self._flush_sec, batch_max=self._batch_max)
            self.flush_once()

    def _post(self, drained: DrainedLane) -> None:
        try:
            batch = WireBatch(
                drained.camera_id,
                drained.worker_boot_id,
                self._provenance,
                drained.records,
                drained.gaps,
            )
        except Exception:  # noqa: BLE001 - a bad batch must not stall the drain
            LOGGER.exception("execution-record batch could not be built")
            self._lanes.note_export_failure(drained)
            return
        result = self._client.post_batch(batch)
        if isinstance(result, DeliveryFailure):
            with self._lock:
                self._failures.append(result)
            self._lanes.note_export_failure(drained)
            return
        with self._lock:
            self._receipts.append(result)


__all__ = ["ExecutionRecordExporter"]
