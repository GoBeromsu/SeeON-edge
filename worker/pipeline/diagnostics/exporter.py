"""Runtime-owned byte-bounded diagnostic drain; every rejected item is accounted."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterable
from itertools import chain

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.execution_records import (
    MAX_EXECUTION_RECORD_BODY_BYTES,
    WireBatch,
    WireBatchReceipt,
    WireGap,
    WireProvenance,
    WireRecord,
    canonical_json,
)
from shared.events.execution_records_client import ExecutionRecordsClient
from worker.pipeline.diagnostics.lanes import (
    DrainedLane,
    ExecutionRecordLanes,
    account_unsendable_records,
)

LOGGER = logging.getLogger(__name__)
EXPORT_HISTORY_LIMIT = 256
_FAILURE_BACKOFF_SEC = 0.05


class ExecutionRecordExporter:
    """Drain thread. A failed chunk becomes explicit loss, never a model retry."""

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
        self._receipts: deque[WireBatchReceipt] = deque(maxlen=EXPORT_HISTORY_LIMIT)
        self._failures: deque[DeliveryFailure] = deque(maxlen=EXPORT_HISTORY_LIMIT)
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._had_failure = False

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
        # One owner holds drained records through delivery or restoration.
        # This lock is separate from the producers' short append-or-drop lock.
        with self._flush_lock:
            self._had_failure = False
            for camera_id, worker_boot_id in self._lanes.cameras_with_work():
                drained = self._lanes.drain_for(camera_id, worker_boot_id, limit=self._batch_max)
                if drained is not None:
                    self._post(drained)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._lanes.wait_for_work(timeout_sec=self._flush_sec, batch_max=self._batch_max)
            self.flush_once()
            if self._had_failure:
                # A gap alone wakes the lane immediately: back off fast failures.
                self._stop.wait(max(_FAILURE_BACKOFF_SEC, self._flush_sec))

    def _post(self, drained: DrainedLane) -> None:
        try:
            # Derive the envelope from its owner, not a second wire definition.
            body = WireBatch(
                drained.camera_id,
                drained.worker_boot_id,
                self._provenance,
                (),
                (WireGap("exporter", 0, 0, 0, 0, 1, "export-failed"),),
            ).to_json()
            body["records"] = []
            body["gaps"] = []
            envelope_bytes = len(canonical_json(body).encode())
        except Exception:  # noqa: BLE001 - malformed diagnostics remain accounted
            # No record has been selected for an attempted chunk yet.
            self._failed(
                DrainedLane(drained.camera_id, drained.worker_boot_id, (), drained.gaps),
                DeliveryFailure(DeliveryDisposition.PERMANENT, "ENCODING_ERROR"),
            )
            self._restore_unattempted(drained, drained.records)
            return

        records: list[WireRecord] = []
        gaps: list[WireGap] = []
        encoded_bytes = envelope_bytes
        # Encode each member once for sizing. Arrays have independent commas;
        # the SHA-256 batch identity always occupies the same 64 bytes.
        # Commit already-known loss before records can advance lane watermarks.
        items = chain(drained.gaps, drained.records)
        for item in items:
            is_record = isinstance(item, WireRecord)
            single = DrainedLane(
                drained.camera_id,
                drained.worker_boot_id,
                (item,) if is_record else (),
                () if is_record else (item,),
            )
            try:
                item_bytes = len(canonical_json(item.to_json()).encode())
            except Exception:  # noqa: BLE001 - one invalid payload cannot stop later items
                if not is_record:
                    self._failed(
                        single, DeliveryFailure(DeliveryDisposition.PERMANENT, "ENCODING_ERROR")
                    )
                    self._restore_unattempted(drained, chain(gaps, records, items))
                    return
                item = self._invalid_record(single, "ENCODING_ERROR")
                is_record = False
                item_bytes = len(canonical_json(item.to_json()).encode())
                single = DrainedLane(drained.camera_id, drained.worker_boot_id, (), (item,))
            if envelope_bytes + item_bytes > MAX_EXECUTION_RECORD_BODY_BYTES:
                if is_record:
                    item = self._invalid_record(single, "OVERSIZE")
                    is_record = False
                    item_bytes = len(canonical_json(item.to_json()).encode())
                    single = DrainedLane(drained.camera_id, drained.worker_boot_id, (), (item,))
                if envelope_bytes + item_bytes > MAX_EXECUTION_RECORD_BODY_BYTES:
                    self._failed(single, DeliveryFailure(DeliveryDisposition.PERMANENT, "OVERSIZE"))
                    self._restore_unattempted(drained, chain(gaps, records, items))
                    return
            comma_bytes = int(bool(records if is_record else gaps))
            if encoded_bytes + item_bytes + comma_bytes > MAX_EXECUTION_RECORD_BODY_BYTES:
                if not self._send(
                    DrainedLane(
                        drained.camera_id, drained.worker_boot_id, tuple(records), tuple(gaps)
                    )
                ):
                    # The current item may already be a record-invalid gap.
                    # Only the failed chunk was attempted; never replay it.
                    self._restore_unattempted(drained, chain((item,), items))
                    return
                records.clear()
                gaps.clear()
                encoded_bytes = envelope_bytes
                comma_bytes = 0
            if isinstance(item, WireRecord):
                records.append(item)
            else:
                gaps.append(item)
            encoded_bytes += item_bytes + comma_bytes
        if records or gaps:
            self._send(
                DrainedLane(drained.camera_id, drained.worker_boot_id, tuple(records), tuple(gaps))
            )

    def _restore_unattempted(
        self, drained: DrainedLane, items: Iterable[WireRecord | WireGap]
    ) -> None:
        records: list[WireRecord] = []
        gaps: list[WireGap] = []
        for item in items:
            if isinstance(item, WireRecord):
                records.append(item)
            else:
                gaps.append(item)
        self._lanes.restore_unattempted(
            DrainedLane(drained.camera_id, drained.worker_boot_id, tuple(records), tuple(gaps))
        )

    def _send(self, drained: DrainedLane) -> bool:
        try:
            batch = WireBatch(
                drained.camera_id,
                drained.worker_boot_id,
                self._provenance,
                drained.records,
                drained.gaps,
            )
            if len(batch.encode()) > MAX_EXECUTION_RECORD_BODY_BYTES:
                self._failed(drained, DeliveryFailure(DeliveryDisposition.PERMANENT, "OVERSIZE"))
                return False
        except Exception:  # noqa: BLE001 - final serialization validates nested values
            self._failed(drained, DeliveryFailure(DeliveryDisposition.PERMANENT, "ENCODING_ERROR"))
            return False
        try:
            result = self._client.post_batch(batch)
        except Exception:  # noqa: BLE001 - transport exceptions cannot kill the drain thread
            self._failed(drained, DeliveryFailure(DeliveryDisposition.RETRY, "TRANSPORT_EXCEPTION"))
            return False
        if isinstance(result, DeliveryFailure):
            self._failed(drained, result)
            return False
        if result.storage_state != "committed":
            self._failed(drained, DeliveryFailure(DeliveryDisposition.RETRY, "STORAGE_UNAVAILABLE"))
            return False
        with self._lock:
            self._receipts.append(result)
        return True

    def _invalid_record(self, single: DrainedLane, code: str) -> WireGap:
        with self._lock:
            self._failures.append(DeliveryFailure(DeliveryDisposition.PERMANENT, code))
        LOGGER.warning(
            "execution-record export for camera %s: %s "
            "(one record cannot be encoded within the byte cap; recorded as record-invalid)",
            single.camera_id,
            code,
        )
        return account_unsendable_records(single, single.records).gaps[0]

    def _failed(self, drained: DrainedLane, failure: DeliveryFailure) -> None:
        self._had_failure = True
        with self._lock:
            self._failures.append(failure)
        LOGGER.warning(
            "execution-record export failed for camera %s: %s %s "
            "(%d records recorded as export-failed; %d existing gaps retained; "
            "unsendable envelope loss is retained, not silently discarded)",
            drained.camera_id,
            failure.disposition.name,
            failure.code,
            len(drained.records),
            len(drained.gaps),
        )
        self._lanes.note_export_failure(drained)


__all__ = ["EXPORT_HISTORY_LIMIT", "ExecutionRecordExporter"]
