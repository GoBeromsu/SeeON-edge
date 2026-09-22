"""Convert the frozen worker wire contract into Backend store types.

WireProvenance has no backend_build_revision; the Backend stamps its own
revision at ingest so provenance_id is the Backend's identity of the batch.
"""

from __future__ import annotations

from backend.app.features.diagnostics.records import (
    BatchReceipt,
    ExecutionRecordInput,
    GapReport,
    IngestBatch,
    Provenance,
    RecordKind,
)
from shared.events.execution_records import WireBatch, WireBatchReceipt, WireRecord


def ingest_batch_from_wire(batch: WireBatch, *, backend_build_revision: str) -> IngestBatch:
    return IngestBatch(
        batch_id=batch.batch_id,
        camera_id=batch.camera_id,
        worker_boot_id=batch.worker_boot_id,
        provenance=Provenance(
            worker_build_revision=batch.provenance.worker_build_revision,
            worker_image_digest=batch.provenance.worker_image_digest,
            model_digest=batch.provenance.model_digest,
            calibration_digest=batch.provenance.calibration_digest,
            preprocessing_identity=batch.provenance.preprocessing_identity,
            config_digest=batch.provenance.config_digest,
            policy_identity=batch.provenance.policy_identity,
            backend_build_revision=backend_build_revision,
        ),
        records=tuple(_record_from_wire(record) for record in batch.records),
        gaps=tuple(
            GapReport(
                producer=gap.producer,
                from_sequence=gap.from_sequence,
                to_sequence=gap.to_sequence,
                from_ns=gap.from_ns,
                to_ns=gap.to_ns,
                record_count=gap.record_count,
                cause=gap.cause,
            )
            for gap in batch.gaps
        ),
    )


def _record_from_wire(record: WireRecord) -> ExecutionRecordInput:
    return ExecutionRecordInput(
        record_id=record.record_id,
        record_kind=RecordKind(record.record_kind),
        camera_id=record.camera_id,
        worker_boot_id=record.worker_boot_id,
        source_generation=record.source_generation,
        stream_epoch=record.stream_epoch,
        producer=record.producer,
        producer_sequence=record.producer_sequence,
        observed_at_ns=record.observed_at_ns,
        time_quality=record.time_quality,
        causal_unit_id=record.causal_unit_id,
        outcome=record.outcome,
        payload=record.payload,
        frame_seq=record.frame_seq,
        source_pts_ns=record.source_pts_ns,
        parent_record_id=record.parent_record_id,
        reason=record.reason,
    )


def wire_receipt_from_store(receipt: BatchReceipt) -> WireBatchReceipt:
    return WireBatchReceipt(
        batch_id=receipt.batch_id,
        accepted=receipt.accepted,
        duplicates=receipt.duplicates,
        rejected=receipt.rejected,
        storage_state=str(receipt.storage_state),
        committed_at_ns=receipt.committed_at_ns,
    )


__all__ = ["ingest_batch_from_wire", "wire_receipt_from_store"]
