"""Backend-owned execution-record persistence (schema 19)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import psycopg

from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.features.diagnostics.coverage import UNSCOPED_GAP_CAUSE, insert_coverage
from backend.app.features.diagnostics.ingest import (
    insert_record,
    payload_text_and_bytes,
    upsert_provenance,
)
from backend.app.features.diagnostics.prune import coarsen_coverage
from backend.app.features.diagnostics.query import (
    QueryResult,
    execute_query,
)
from backend.app.features.diagnostics.records import (
    BatchReceipt,
    CoverageKind,
    IngestBatch,
    StorageState,
    canonical_json,
)
from backend.app.features.diagnostics.retention import RetentionBudget, UsageMeter, enforce_budget


def _receipt_json(receipt: BatchReceipt) -> str:
    return canonical_json(
        {
            "accepted": receipt.accepted,
            "batch_id": receipt.batch_id,
            "committed_at_ns": receipt.committed_at_ns,
            "duplicates": receipt.duplicates,
            "rejected": [[record_id, reason] for record_id, reason in receipt.rejected],
            "storage_state": str(receipt.storage_state),
        }
    )


def _receipt_from_json(text: str) -> BatchReceipt:
    raw = json.loads(text)
    return BatchReceipt(
        batch_id=str(raw["batch_id"]),
        accepted=int(raw["accepted"]),
        duplicates=int(raw["duplicates"]),
        rejected=tuple((str(pair[0]), str(pair[1])) for pair in raw["rejected"]),
        storage_state=StorageState(raw["storage_state"]),
        committed_at_ns=int(raw["committed_at_ns"]),
    )


class ExecutionRecordStore:
    """Idempotent batch ingest and query over the six execution_* tables."""

    def __init__(
        self,
        database: PostgresDatabase,
        budget: RetentionBudget,
        clock: Callable[[], int] = time.time_ns,
    ) -> None:
        self._database = database
        self.budget = budget
        self._clock = clock
        self._meter = UsageMeter()

    def ingest_batch(self, batch: IngestBatch) -> BatchReceipt:
        now_ns = self._clock()
        return self._database.transact(lambda connection: self._ingest(connection, batch, now_ns))

    def query(
        self,
        camera_id: str,
        from_ns: int,
        to_ns: int,
        limit: int,
        cursor: str | None = None,
    ) -> QueryResult:
        return self._database.read_snapshot(
            lambda connection: execute_query(
                connection,
                camera_id=camera_id,
                from_ns=from_ns,
                to_ns=to_ns,
                limit=limit,
                cursor=cursor,
            )
        )

    def _ingest(
        self, connection: psycopg.Connection, batch: IngestBatch, now_ns: int
    ) -> BatchReceipt:
        # One writer at a time: the batch-id
        # lookup, retention measurement and prune below all assume no
        # concurrent ingest. The lock conflicts only with itself and
        # stronger modes, so snapshot readers never wait on it.
        connection.execute("LOCK TABLE execution_batches IN SHARE ROW EXCLUSIVE MODE")
        existing = connection.execute(
            "SELECT receipt FROM execution_batches WHERE batch_id = %s",
            (batch.batch_id,),
        ).fetchone()
        if existing is not None:
            prior_receipt = _receipt_from_json(str(existing[0]))
            if prior_receipt.storage_state is not StorageState.STORAGE_UNAVAILABLE:
                return prior_receipt
            # Only a refusal is retryable. Remove it before the savepoint so
            # either outcome can replace it under the same id. The enclosing
            # write transaction restores it if ingestion raises.
            connection.execute(
                "DELETE FROM execution_batches WHERE batch_id = %s", (batch.batch_id,)
            )
        connection.execute("SAVEPOINT ingest")
        provenance_id = upsert_provenance(connection, batch.provenance, now_ns)
        accepted = 0
        written_bytes = 0
        duplicates = 0
        rejected: list[tuple[str, str]] = []
        epoch_ns = _batch_epoch(batch, now_ns)
        for record in batch.records:
            payload_text, payload_bytes = payload_text_and_bytes(record)
            if payload_bytes > self.budget.max_record_bytes:
                rejected.append((record.record_id, "oversize"))
                insert_coverage(
                    connection,
                    camera_id=record.camera_id,
                    worker_boot_id=record.worker_boot_id,
                    source_generation=record.source_generation,
                    stream_epoch=record.stream_epoch,
                    kind=CoverageKind.REJECTED_OVERSIZE,
                    producer=record.producer,
                    from_sequence=record.producer_sequence,
                    to_sequence=record.producer_sequence,
                    from_ns=record.observed_at_ns,
                    to_ns=record.observed_at_ns,
                    record_count=1,
                    exact=True,
                    cause="oversize",
                    recorded_at_ns=now_ns,
                )
                continue
            disposition = insert_record(
                connection,
                record,
                payload_text,
                payload_bytes,
                provenance_id,
                self.budget,
                batch.batch_id,
                now_ns,
            )
            if disposition is None:
                accepted += 1
                written_bytes += payload_bytes
            elif disposition == "duplicate":
                duplicates += 1
            else:
                rejected.append((record.record_id, disposition))
        gap_lanes: set[tuple[str, str, int, int]] = set()
        for gap in batch.gaps:
            scoped = gap.source_generation is not None and gap.stream_epoch is not None
            # (0, 0) is only a storage bucket for legacy unresolved loss.
            # Its UNKNOWN marker explicitly applies across this camera/boot;
            # no batch neighbour establishes the lost observation's scope.
            generation = gap.source_generation if scoped else 0
            epoch = gap.stream_epoch if scoped else 0
            insert_coverage(
                connection,
                camera_id=batch.camera_id,
                worker_boot_id=batch.worker_boot_id,
                source_generation=generation,
                stream_epoch=epoch,
                kind=CoverageKind.MISSING_NOT_RECORDED if scoped else CoverageKind.UNKNOWN,
                producer=gap.producer,
                from_sequence=gap.from_sequence,
                to_sequence=gap.to_sequence,
                from_ns=gap.from_ns,
                to_ns=gap.to_ns,
                record_count=gap.record_count,
                exact=scoped,
                cause=gap.cause if scoped else UNSCOPED_GAP_CAUSE,
                recorded_at_ns=now_ns,
            )
            gap_lanes.add((batch.camera_id, batch.worker_boot_id, generation, epoch))
        if batch.gaps:
            coarsen_coverage(
                connection,
                self.budget.coverage_rows_per_epoch,
                now_ns,
                gap_lanes,
            )
        receipt = BatchReceipt(
            batch_id=batch.batch_id,
            accepted=accepted,
            duplicates=duplicates,
            rejected=tuple(rejected),
            storage_state=StorageState.COMMITTED,
            committed_at_ns=now_ns,
        )
        # The receipt row is part of the control envelope, so it must exist
        # before the budget is enforced; otherwise every commit lands a few
        # hundred bytes over the line it was just checked against.
        _write_batch_row(connection, batch, receipt, now_ns)
        self._meter.accrue(written_bytes)
        if not enforce_budget(connection, self.budget, now_ns, meter=self._meter):
            connection.execute("ROLLBACK TO ingest")
            insert_coverage(
                connection,
                camera_id=batch.camera_id,
                worker_boot_id=batch.worker_boot_id,
                source_generation=epoch_ns[0],
                stream_epoch=epoch_ns[1],
                kind=CoverageKind.STORAGE_UNAVAILABLE,
                producer=None,
                from_sequence=None,
                to_sequence=None,
                from_ns=now_ns,
                to_ns=now_ns,
                record_count=0,
                exact=False,
                cause="capacity",
                recorded_at_ns=now_ns,
            )
            # Budget enforcement's coarsening rolled back with ingest.
            coarsen_coverage(
                connection,
                self.budget.coverage_rows_per_epoch,
                now_ns,
                {(batch.camera_id, batch.worker_boot_id, epoch_ns[0], epoch_ns[1])},
            )
            receipt = BatchReceipt(
                batch_id=batch.batch_id,
                accepted=0,
                duplicates=0,
                rejected=(),
                storage_state=StorageState.STORAGE_UNAVAILABLE,
                committed_at_ns=now_ns,
            )
            _write_batch_row(connection, batch, receipt, now_ns)
            return receipt
        still_recorded = connection.execute(
            "SELECT 1 FROM execution_batches WHERE batch_id = %s", (batch.batch_id,)
        ).fetchone()
        if still_recorded is None:
            # Capacity pruned every record this batch contributed, which also
            # dropped its receipt. Keep the receipt anyway: it is the truthful
            # answer to a retry of this batch id, and the prune is already
            # visible as DELETED_BY_CAPACITY coverage.
            _write_batch_row(connection, batch, receipt, now_ns)
        return receipt


def _batch_epoch(batch: IngestBatch, now_ns: int) -> tuple[int, int]:
    del now_ns
    if not batch.records:
        return 0, 0
    first = batch.records[0]
    return first.source_generation, first.stream_epoch


def _write_batch_row(
    connection: psycopg.Connection,
    batch: IngestBatch,
    receipt: BatchReceipt,
    now_ns: int,
) -> None:
    connection.execute(
        """
        INSERT INTO execution_batches (
            batch_id, camera_id, worker_boot_id, received_at_ns,
            accepted_records, duplicate_records, rejected_records, receipt
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            batch.batch_id,
            batch.camera_id,
            batch.worker_boot_id,
            now_ns,
            receipt.accepted,
            receipt.duplicates,
            len(receipt.rejected),
            _receipt_json(receipt),
        ),
    )


__all__ = ["ExecutionRecordStore"]
