"""Wire-contract tests for worker -> backend execution-record batches."""

from __future__ import annotations

import json

import pytest

from shared.events.execution_records import (
    ExecutionRecordContractError,
    WireBatch,
    WireBatchReceipt,
    WireGap,
    WireProvenance,
    WireRecord,
)

_PROVENANCE = WireProvenance(
    worker_build_revision="abc123",
    worker_image_digest="sha256:deadbeef",
    model_digest="m1",
    calibration_digest="c1",
    preprocessing_identity="pose-bbox56/v1",
    config_digest="cfg1",
    policy_identity="fall.policy:2",
)


def _record(seq: int, **overrides: object) -> WireRecord:
    fields: dict[str, object] = {
        "record_kind": "model.score",
        "camera_id": "cam-1",
        "worker_boot_id": "boot-1",
        "source_generation": 0,
        "stream_epoch": 3,
        "producer": "model",
        "producer_sequence": seq,
        "observed_at_ns": 1_000 + seq,
        "time_quality": "monotonic",
        "causal_unit_id": "unit-1",
        "outcome": "scored",
        "payload": {"raw_logit": -0.25, "temperature": 1.7},
    }
    fields.update(overrides)
    return WireRecord(**fields)  # type: ignore[arg-type]


def test_record_id_is_content_derived_and_stable_across_encoding() -> None:
    first = _record(0)
    second = WireRecord.from_json(json.loads(json.dumps(first.to_json())))
    assert first.record_id == second.record_id
    assert len(first.record_id) == 64
    assert _record(0, payload={"raw_logit": -0.26}).record_id != first.record_id


def test_supplied_record_id_must_match_content() -> None:
    body = _record(0).to_json()
    body["record_id"] = "0" * 64
    with pytest.raises(ExecutionRecordContractError, match="record_id does not match"):
        WireRecord.from_json(body)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("record_kind", "not.a.kind", "record_kind"),
        ("time_quality", "guess", "time_quality"),
        ("producer_sequence", -1, "producer_sequence"),
        ("producer_sequence", True, "producer_sequence"),
        ("camera_id", "", "camera_id"),
        ("camera_id", "x" * 129, "camera_id"),
        ("parent_record_id", "zz", "parent_record_id"),
        ("payload", "not-an-object", "payload"),
    ],
)
def test_record_rejects_invalid_fields(field: str, value: object, message: str) -> None:
    with pytest.raises(ExecutionRecordContractError, match=message):
        _record(0, **{field: value})


def test_record_rejects_non_json_payload() -> None:
    with pytest.raises(ExecutionRecordContractError, match="JSON"):
        _record(0, payload={"bad": object()})


def test_batch_id_depends_on_record_set_and_gaps_not_order() -> None:
    a, b = _record(0), _record(1)
    gap = WireGap(
        producer="model",
        from_sequence=2,
        to_sequence=4,
        from_ns=1_002,
        to_ns=1_004,
        record_count=3,
        cause="lane-overflow",
    )
    forward = WireBatch("cam-1", "boot-1", _PROVENANCE, (a, b), (gap,))
    backward = WireBatch("cam-1", "boot-1", _PROVENANCE, (b, a), (gap,))
    without_gap = WireBatch("cam-1", "boot-1", _PROVENANCE, (a, b))
    assert forward.batch_id == backward.batch_id
    assert forward.batch_id != without_gap.batch_id


def test_batch_round_trips_through_json_and_verifies_batch_id() -> None:
    batch = WireBatch("cam-1", "boot-1", _PROVENANCE, (_record(0), _record(1)))
    decoded = WireBatch.from_json(json.loads(batch.encode()))
    assert decoded.batch_id == batch.batch_id
    assert [r.record_id for r in decoded.records] == [r.record_id for r in batch.records]
    tampered = json.loads(batch.encode())
    tampered["records"][0]["payload"]["raw_logit"] = 9.0
    with pytest.raises(ExecutionRecordContractError, match="record_id does not match"):
        WireBatch.from_json(tampered)


def test_batch_rejects_mismatched_camera_and_empty_body() -> None:
    with pytest.raises(ExecutionRecordContractError, match="does not match batch"):
        WireBatch("cam-2", "boot-1", _PROVENANCE, (_record(0),))
    with pytest.raises(ExecutionRecordContractError, match="no records and no gaps"):
        WireBatch("cam-1", "boot-1", _PROVENANCE, ())


def test_gap_only_batch_is_valid() -> None:
    gap = WireGap("sdk", 10, 12, 5_000, 5_002, 3, "export-failed")
    batch = WireBatch("cam-1", "boot-1", _PROVENANCE, (), (gap,))
    assert WireBatch.from_json(batch.to_json()).gaps == (gap,)


def test_receipt_round_trip_and_storage_state_vocabulary() -> None:
    receipt = WireBatchReceipt("a" * 64, 2, 1, (("b" * 64, "oversize"),), "committed", 7)
    assert WireBatchReceipt.from_json(receipt.to_json()) == receipt
    with pytest.raises(ExecutionRecordContractError, match="storage_state"):
        WireBatchReceipt("a" * 64, 0, 0, (), "hub-accepted", 7)
