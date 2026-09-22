"""Exporter batches lanes and reports export-failed gaps after a drop."""

from __future__ import annotations

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.execution_records import WireBatchReceipt, WireProvenance, WireRecord
from worker.pipeline.diagnostics.exporter import ExecutionRecordExporter
from worker.pipeline.diagnostics.lanes import EXPORT_FAILED_CAUSE, ExecutionRecordLanes

_PROVENANCE = WireProvenance(
    worker_build_revision="abc123",
    worker_image_digest="sha256:deadbeef",
    model_digest="model-1",
    calibration_digest="cal-1",
    preprocessing_identity="pose-bbox56/v1",
    config_digest="cfg-1",
    policy_identity="fall.policy:2",
)


def _record(seq: int = 0) -> WireRecord:
    return WireRecord(
        record_kind="policy.consume",
        camera_id="cam-1",
        worker_boot_id="boot-1",
        source_generation=0,
        stream_epoch=1,
        producer="policy",
        producer_sequence=seq,
        observed_at_ns=2_000 + seq,
        time_quality="monotonic",
        causal_unit_id="cam-1:boot-1:1:frame:0",
        outcome="consumed",
        payload={"processed_count": seq},
    )


class _Client:
    def __init__(self) -> None:
        self.posted: list[object] = []
        self.fail_next = False

    def post_batch(self, batch: object) -> WireBatchReceipt | DeliveryFailure:
        self.posted.append(batch)
        if self.fail_next:
            self.fail_next = False
            return DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK")
        return WireBatchReceipt(batch.batch_id, 1, 0, (), "committed", 9)


def test_exporter_posts_a_batch_and_keeps_a_receipt() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=8)
    client = _Client()
    exporter = ExecutionRecordExporter(
        lanes=lanes,
        client=client,  # type: ignore[arg-type]
        provenance=_PROVENANCE,
        batch_max=2,
        flush_ms=50,
    )
    assert lanes.try_emit(_record()) is True
    exporter.flush_once()
    assert len(client.posted) == 1
    assert len(exporter.receipts()) == 1


def test_failed_export_is_reported_as_gap_on_the_next_batch() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=8)
    client = _Client()
    exporter = ExecutionRecordExporter(
        lanes=lanes,
        client=client,  # type: ignore[arg-type]
        provenance=_PROVENANCE,
        batch_max=8,
        flush_ms=50,
    )
    assert lanes.try_emit(_record()) is True
    client.fail_next = True
    exporter.flush_once()
    assert exporter.failures()
    assert lanes.try_emit(_record()) is True
    exporter.flush_once()
    second = client.posted[-1]
    assert [gap.cause for gap in second.gaps] == [EXPORT_FAILED_CAUSE]
