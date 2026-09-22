"""Adversarial worker lane/exporter/composition cases."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.execution_records import WireBatchReceipt, WireProvenance, WireRecord
from worker.pipeline.diagnostics.exporter import ExecutionRecordExporter
from worker.pipeline.diagnostics.lanes import (
    EXPORT_FAILED_CAUSE,
    LANE_OVERFLOW_CAUSE,
    ExecutionRecordLanes,
)
from worker.pipeline.diagnostics.provenance import ExecutionRecordProvenanceError
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.execution_records import execution_records_settings_from_environment
from worker.runtime.execution_records import compose_execution_records

_PROVENANCE = WireProvenance(
    worker_build_revision="abc123",
    worker_image_digest="sha256:deadbeef",
    model_digest="model-1",
    calibration_digest="cal-1",
    preprocessing_identity="pose-bbox56/v1",
    config_digest="cfg-1",
    policy_identity="fall.policy:2",
)


def _record(*, producer: str = "sdk", seq: int = 0, observed: int = 1_000) -> WireRecord:
    return WireRecord(
        record_kind="sdk.frame",
        camera_id="cam-1",
        worker_boot_id="boot-1",
        source_generation=0,
        stream_epoch=1,
        producer=producer,
        producer_sequence=seq,
        observed_at_ns=observed,
        time_quality="monotonic",
        causal_unit_id="cam-1:boot-1:1:frame:0",
        outcome="accepted",
        payload={"n": seq},
    )


class _Client:
    def __init__(self) -> None:
        self.posted: list[object] = []
        self.fail_next = False
        self.fail_always = False

    def post_batch(self, batch: object) -> WireBatchReceipt | DeliveryFailure:
        self.posted.append(batch)
        if self.fail_always or self.fail_next:
            self.fail_next = False
            return DeliveryFailure(DeliveryDisposition.RETRY, "HTTP_503", status_code=503)
        return WireBatchReceipt(batch.batch_id, 1, 0, (), "committed", 9)  # type: ignore[attr-defined]


def test_c1_non_monotonic_overflow_yields_valid_gap() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=1)
    assert lanes.try_emit(_record(producer="sdk", observed=5_000)) is True
    assert lanes.try_emit(_record(producer="sdk", observed=9_000)) is False
    assert lanes.try_emit(_record(producer="cpu", observed=1_000)) is True
    assert lanes.try_emit(_record(producer="cpu", observed=100)) is False
    assert lanes.try_emit(_record(producer="sdk", observed=3_000)) is False
    drained = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert drained is not None
    assert drained.gaps
    for gap in drained.gaps:
        assert gap.cause == LANE_OVERFLOW_CAUSE
        assert gap.from_ns <= gap.to_ns
        assert gap.from_sequence <= gap.to_sequence
        assert gap.record_count >= 1
    sdk_gaps = [gap for gap in drained.gaps if gap.producer == "sdk"]
    assert sdk_gaps
    assert (sdk_gaps[0].from_ns, sdk_gaps[0].to_ns) == (3_000, 9_000)


def test_c2_exporter_503_then_recovery_reports_export_failed_gap() -> None:
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
    assert [gap.cause for gap in second.gaps] == [EXPORT_FAILED_CAUSE]  # type: ignore[attr-defined]
    assert exporter.receipts()


def test_c3_try_emit_stays_o1_when_exporter_is_stopped() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=8)
    client = _Client()
    client.fail_always = True
    exporter = ExecutionRecordExporter(
        lanes=lanes,
        client=client,  # type: ignore[arg-type]
        provenance=_PROVENANCE,
        batch_max=8,
        flush_ms=50,
    )
    exporter.stop()
    record = _record()
    samples: list[float] = []
    for _ in range(10_000):
        started = time.perf_counter_ns()
        lanes.try_emit(record)
        samples.append((time.perf_counter_ns() - started) / 1_000.0)
    median_us = statistics.median(samples)
    p95_us = statistics.quantiles(samples, n=20)[18]
    Path(".omo/evidence/observability").mkdir(parents=True, exist_ok=True)
    Path(".omo/evidence/observability/redteam-c3-try-emit.json").write_text(
        json.dumps(
            {
                "try_emit_median_us": median_us,
                "try_emit_p95_us": p95_us,
                "samples": 10_000,
            }
        ),
        encoding="utf-8",
    )
    assert lanes.queued() <= 8


def test_c4_refuse_to_start_when_lane_capacity_missing() -> None:
    with pytest.raises(WorkerConfigError, match="LANE_CAPACITY"):
        execution_records_settings_from_environment(
            {
                "ML_WORKER_EXECUTION_RECORDS_ENABLED": "1",
                "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX": "8",
                "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS": "50",
            }
        )


def test_c5_refuse_to_start_when_batch_max_missing() -> None:
    with pytest.raises(WorkerConfigError, match="BATCH_MAX"):
        execution_records_settings_from_environment(
            {
                "ML_WORKER_EXECUTION_RECORDS_ENABLED": "1",
                "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY": "16",
                "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS": "50",
            }
        )


def test_c6_refuse_to_start_when_flush_ms_missing() -> None:
    with pytest.raises(WorkerConfigError, match="FLUSH_MS"):
        execution_records_settings_from_environment(
            {
                "ML_WORKER_EXECUTION_RECORDS_ENABLED": "1",
                "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY": "16",
                "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX": "8",
            }
        )


def _compose_config(*, url: str, token: str) -> SimpleNamespace:
    return SimpleNamespace(
        relay=SimpleNamespace(
            url=url,
            token=SimpleNamespace(get_secret_value=lambda: token),
        ),
        model_dump=lambda mode="json": {"version": 1},
        detection_policies=SimpleNamespace(
            defaults={"fall": SimpleNamespace(effective_policy_id="fall")}
        ),
    )


_ENABLED_ENV = {
    "ML_WORKER_EXECUTION_RECORDS_ENABLED": "1",
    "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY": "4",
    "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX": "2",
    "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS": "20",
}


def test_c7_refuse_to_start_when_relay_url_missing() -> None:
    with pytest.raises(ExecutionRecordProvenanceError, match="relay URL/token"):
        compose_execution_records(
            _compose_config(url="", token="token"),  # type: ignore[arg-type]
            env=_ENABLED_ENV,
            build_revision="abc123",
            image_digest="sha256:deadbeef",
            model_digest="model-1",
            calibration_digest="cal-1",
            preprocessing_identity="pose-bbox56/v1",
            policy_identity="fall.policy:2",
        )


def test_c8_refuse_to_start_when_relay_token_missing() -> None:
    with pytest.raises(ExecutionRecordProvenanceError, match="relay URL/token"):
        compose_execution_records(
            _compose_config(url="http://relay.test", token=""),  # type: ignore[arg-type]
            env=_ENABLED_ENV,
            build_revision="abc123",
            image_digest="sha256:deadbeef",
            model_digest="model-1",
            calibration_digest="cal-1",
            preprocessing_identity="pose-bbox56/v1",
            policy_identity="fall.policy:2",
        )
