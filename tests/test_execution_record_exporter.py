"""Exporter batches lanes and reports export-failed gaps after a drop."""

from __future__ import annotations

import logging
import threading

import pytest

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.execution_records import (
    MAX_EXECUTION_RECORD_BODY_BYTES,
    WireBatch,
    WireBatchReceipt,
    WireGap,
    WireProvenance,
    WireRecord,
)
from worker.pipeline.diagnostics.exporter import ExecutionRecordExporter
from worker.pipeline.diagnostics.lanes import (
    EXPORT_FAILED_CAUSE,
    LANE_OVERFLOW_CAUSE,
    RECORD_INVALID_CAUSE,
    DrainedLane,
    ExecutionRecordLanes,
)

_PROVENANCE = WireProvenance(
    worker_build_revision="abc123",
    worker_image_digest="sha256:deadbeef",
    model_digest="model-1",
    calibration_digest="cal-1",
    preprocessing_identity="pose-bbox56/v1",
    config_digest="cfg-1",
    policy_identity="fall.policy:2",
)


def _record(
    seq: int = 0,
    *,
    producer: str = "policy",
    observed: int | None = None,
    payload: dict[str, object] | None = None,
    camera: str = "cam-1",
    boot: str = "boot-1",
    generation: int = 0,
    epoch: int = 1,
) -> WireRecord:
    return WireRecord(
        record_kind="policy.consume",
        camera_id=camera,
        worker_boot_id=boot,
        source_generation=generation,
        stream_epoch=epoch,
        producer=producer,
        producer_sequence=seq,
        observed_at_ns=2_000 + seq if observed is None else observed,
        time_quality="monotonic",
        causal_unit_id=f"{camera}:{boot}:{epoch}:frame:0",
        outcome="consumed",
        payload={"processed_count": seq} if payload is None else payload,
    )


class _Client:
    def __init__(self) -> None:
        self.posted: list[WireBatch] = []
        self.committed: list[WireBatch] = []
        self.fail_next = False

    def post_batch(self, batch: WireBatch) -> WireBatchReceipt | DeliveryFailure:
        self.posted.append(batch)
        if self.fail_next:
            self.fail_next = False
            return DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK")
        self.committed.append(batch)
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
    assert client.committed == client.posted
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
    assert client.committed == []
    assert lanes.try_emit(_record()) is True
    exporter.flush_once()
    assert len(client.posted) == 2 and len(client.committed) == 1
    second = client.committed[-1]
    assert [gap.cause for gap in second.gaps] == [EXPORT_FAILED_CAUSE]


def _exporter(
    lanes: ExecutionRecordLanes, client: object, *, batch_max: int = 8
) -> ExecutionRecordExporter:
    return ExecutionRecordExporter(
        lanes=lanes,
        client=client,  # type: ignore[arg-type]
        provenance=_PROVENANCE,
        batch_max=batch_max,
        flush_ms=50,
    )


def _batch_len(records: tuple[WireRecord, ...] = (), gaps: tuple[WireGap, ...] = ()) -> int:
    return len(WireBatch("cam-1", "boot-1", _PROVENANCE, records, gaps).encode())


def _receipt(batch: WireBatch) -> WireBatchReceipt:
    return WireBatchReceipt(batch.batch_id, len(batch.records), 0, (), "committed", 9)


def _sequences(batches: list[WireBatch]) -> list[int]:
    return [record.producer_sequence for batch in batches for record in batch.records]


def _exact_cap_record(seq: int) -> WireRecord:
    """One-record batch whose UTF-8 body is exactly the client cap, using multibyte text."""
    unit = "가"
    low, high, best = 0, 400_000, 0
    while low <= high:
        mid = (low + high) // 2
        size = _batch_len((_record(seq, payload={"m": unit * mid}),))
        if size <= MAX_EXECUTION_RECORD_BODY_BYTES:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    pad = 0
    while pad < 4:
        nxt = _record(seq, payload={"m": unit * best + "a" * (pad + 1)})
        if _batch_len((nxt,)) > MAX_EXECUTION_RECORD_BODY_BYTES:
            break
        pad += 1
    record = _record(seq, payload={"m": unit * best + "a" * pad})
    assert _batch_len((record,)) == MAX_EXECUTION_RECORD_BODY_BYTES
    return record


# Issue 598 repros. Select with: pytest -k test_598_
# Failure-before and pass-after are verified separately against exact source.


def test_598_multibyte_payload_near_exact_byte_cap_stays_bounded() -> None:
    capped = _exact_cap_record(0)
    tiny = _record(1, payload={"n": 1})
    assert _batch_len((capped, tiny)) > MAX_EXECUTION_RECORD_BODY_BYTES
    lanes = ExecutionRecordLanes(lane_capacity=4)
    client = _Client()
    exporter = _exporter(lanes, client, batch_max=8)
    assert lanes.try_emit(capped) is True
    assert lanes.try_emit(tiny) is True
    exporter.flush_once()
    assert len(client.posted) == 2
    first, second = client.posted
    body = first.encode()
    assert len(body) == MAX_EXECUTION_RECORD_BODY_BYTES
    assert len(body.decode()) < len(body)
    assert [record.producer_sequence for record in first.records] == [0]
    assert [record.producer_sequence for record in second.records] == [1]
    assert len(second.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES
    assert _sequences(client.posted) == [0, 1]


def test_598_large_batch_posts_each_sequence_once_under_the_cap() -> None:
    records = tuple(_record(seq, payload={"blob": "b" * 200_000}) for seq in range(8))
    assert all(_batch_len((record,)) <= MAX_EXECUTION_RECORD_BODY_BYTES for record in records)
    assert _batch_len(records) > MAX_EXECUTION_RECORD_BODY_BYTES
    lanes = ExecutionRecordLanes(lane_capacity=8)
    client = _Client()
    exporter = _exporter(lanes, client, batch_max=8)
    for record in records:
        assert lanes.try_emit(record) is True
    exporter.flush_once()
    assert len(client.posted) > 1
    assert _sequences(client.posted) == list(range(8))
    assert all(
        batch.camera_id == "cam-1" and batch.worker_boot_id == "boot-1" for batch in client.posted
    )
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_598_oversized_record_is_record_invalid_gap_with_reason(caplog) -> None:
    caplog.set_level(logging.WARNING)
    small_a = _record(0, payload={"n": 0})
    huge = _record(1, payload={"blob": "a" * 1_100_000})
    small_b = _record(2, payload={"n": 2})
    assert _batch_len((huge,)) > MAX_EXECUTION_RECORD_BODY_BYTES
    lanes = ExecutionRecordLanes(lane_capacity=4)
    client = _Client()
    exporter = _exporter(lanes, client, batch_max=8)
    assert lanes.try_emit(small_a) is True
    assert lanes.try_emit(huge) is True
    assert lanes.try_emit(small_b) is True
    exporter.flush_once()
    assert _sequences(client.posted) == [0, 2]
    gaps = [gap for batch in client.posted for gap in batch.gaps]
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.cause == RECORD_INVALID_CAUSE
    assert (gap.from_sequence, gap.to_sequence, gap.record_count) == (1, 1, 1)
    assert (gap.from_ns, gap.to_ns) == (2_001, 2_001)
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "cam-1" in message and "OVERSIZE" in message and "record-invalid" in message
        for message in messages
    )


def test_598_gap_only_batches_split_on_encoded_bytes() -> None:
    sample_a = WireGap(
        producer="0000" + "p" * 124,
        from_sequence=0,
        to_sequence=0,
        from_ns=1_000,
        to_ns=1_000,
        record_count=1,
        cause=EXPORT_FAILED_CAUSE,
    )
    sample_b = WireGap(
        producer="0001" + "p" * 124,
        from_sequence=0,
        to_sequence=0,
        from_ns=1_001,
        to_ns=1_001,
        record_count=1,
        cause=EXPORT_FAILED_CAUSE,
    )
    marginal = _batch_len(gaps=(sample_a, sample_b)) - _batch_len(gaps=(sample_a,))
    count = MAX_EXECUTION_RECORD_BODY_BYTES // marginal + 2
    assert 2 <= count <= 20_000
    lanes = ExecutionRecordLanes(lane_capacity=2)
    held: list[object] = []
    expected: list[tuple[str, int, int]] = []
    for index in range(count):
        producer = f"{index:04d}{'p' * 124}"
        record = _record(0, producer=producer, observed=1_000 + index, payload={"i": index})
        assert lanes.try_emit(record) is True
        drained = lanes.drain_for("cam-1", "boot-1", limit=4)
        assert drained is not None and len(drained.records) == 1 and drained.gaps == ()
        held.append(drained)
        record = drained.records[0]
        expected.append((record.producer, record.producer_sequence, record.observed_at_ns))
    for drained in held:
        lanes.note_export_failure(drained)  # type: ignore[arg-type]
    gaps = tuple(
        WireGap(
            producer=producer,
            from_sequence=sequence,
            to_sequence=sequence,
            from_ns=observed,
            to_ns=observed,
            record_count=1,
            cause=EXPORT_FAILED_CAUSE,
        )
        for producer, sequence, observed in expected
    )
    assert _batch_len(gaps=gaps) > MAX_EXECUTION_RECORD_BODY_BYTES
    client = _Client()
    exporter = _exporter(lanes, client, batch_max=8)
    exporter.flush_once()
    assert len(client.posted) > 1
    assert all(not batch.records for batch in client.posted)
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)
    flat = [gap for batch in client.posted for gap in batch.gaps]
    flat_keys = [
        (
            gap.producer,
            gap.from_sequence,
            gap.to_sequence,
            gap.record_count,
            gap.cause,
            gap.from_ns,
            gap.to_ns,
        )
        for gap in flat
    ]
    assert flat_keys == [
        (producer, sequence, sequence, 1, EXPORT_FAILED_CAUSE, observed, observed)
        for producer, sequence, observed in expected
    ]
    assert sum(gap.record_count for gap in flat) == count


def _two_fat_records() -> tuple[WireRecord, WireRecord]:
    records = tuple(_record(seq, payload={"blob": "c" * 700_000}) for seq in (1, 2))
    assert all(_batch_len((record,)) <= MAX_EXECUTION_RECORD_BODY_BYTES for record in records)
    assert _batch_len(records) > MAX_EXECUTION_RECORD_BODY_BYTES
    return records


@pytest.mark.parametrize("failure_code", ("NETWORK", "TRANSPORT_EXCEPTION", "STORAGE_UNAVAILABLE"))
def test_598_failed_chunk_defers_suffix_until_loss_commits(failure_code) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    assert lanes.try_emit(_record(0, payload={"seed": 1})) is True
    seeded = lanes.drain_for("cam-1", "boot-1", limit=4)
    assert seeded is not None and len(seeded.records) == 1
    lanes.note_export_failure(seeded)
    fat = _two_fat_records()
    assert lanes.try_emit(fat[0]) is True
    assert lanes.try_emit(fat[1]) is True
    client = _FailNth({0}, failure_code=failure_code)
    exporter = _exporter(lanes, client, batch_max=8)
    exporter.flush_once()
    assert len(client.posted) == 1
    assert [record.producer_sequence for record in client.posted[0].records] == [1]
    assert client.posted[0].gaps == (
        WireGap("policy", 0, 0, 2_000, 2_000, 1, EXPORT_FAILED_CAUSE, 0, 1),
    )
    assert client.committed == []
    assert exporter.receipts() == ()
    assert lanes.queued() == 1
    assert [failure.code for failure in exporter.failures()] == [failure_code]
    exporter.flush_once()
    assert len(client.posted) == 2
    assert len(client.committed) == 1
    accounted = client.committed[0]
    assert _sequences(client.committed) == [2]
    assert accounted.records[0].record_id == fat[1].record_id
    assert len(accounted.gaps) == 2
    assert all(gap.cause == EXPORT_FAILED_CAUSE for gap in accounted.gaps)
    assert all((gap.source_generation, gap.stream_epoch) == (0, 1) for gap in accounted.gaps)
    assert sorted(
        (gap.from_sequence, gap.to_sequence, gap.record_count) for gap in accounted.gaps
    ) == [(0, 0, 1), (1, 1, 1)]
    assert len(exporter.failures()) == 1
    assert len(exporter.receipts()) == 1
    assert lanes.cameras_with_work() == ()
    exporter.flush_once()
    assert len(client.posted) == 2
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


class _FailNth:
    def __init__(self, fail_at: set[int], *, failure_code: str = "NETWORK") -> None:
        self.fail_at = fail_at
        self.failure_code = failure_code
        self.posted: list[WireBatch] = []
        self.committed: list[WireBatch] = []

    def post_batch(self, batch: WireBatch) -> WireBatchReceipt | DeliveryFailure:
        index = len(self.posted)
        self.posted.append(batch)
        if index in self.fail_at:
            if self.failure_code == "TRANSPORT_EXCEPTION":
                raise ConnectionError("relay down")
            if self.failure_code == "STORAGE_UNAVAILABLE":
                return WireBatchReceipt(batch.batch_id, 0, 0, (), "STORAGE_UNAVAILABLE", 9)
            return DeliveryFailure(DeliveryDisposition.RETRY, self.failure_code)
        self.committed.append(batch)
        return _receipt(batch)


@pytest.mark.parametrize("failure_code", ("NETWORK", "TRANSPORT_EXCEPTION", "STORAGE_UNAVAILABLE"))
def test_repeated_gap_refusal_keeps_unsent_record_without_double_counting(failure_code) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    assert lanes.try_emit(_record())
    seed = lanes.drain_for("cam-1", "boot-1", limit=4)
    assert seed is not None
    lanes.note_export_failure(seed)
    first = _record(1, payload={"blob": "c" * 700_000})
    tail = _exact_cap_record(2)
    assert lanes.try_emit(first)
    assert lanes.try_emit(tail)
    client = _FailNth({0, 1, 2}, failure_code=failure_code)
    exporter = _exporter(lanes, client)

    for attempt in range(3):
        exporter.flush_once()
        assert len(client.posted) == attempt + 1
        assert client.committed == []
        assert exporter.receipts() == ()
        assert lanes.queued() == 1
        assert _sequences(client.posted) == [1]
        if attempt:
            assert client.posted[-1].records == ()
            assert sorted(
                (gap.from_sequence, gap.to_sequence, gap.record_count)
                for gap in client.posted[-1].gaps
            ) == [(0, 0, 1), (1, 1, 1)]

    exporter.flush_once()
    assert len(client.committed) == 2
    assert client.committed[0].records == ()
    assert _sequences(client.committed) == [2]
    assert client.committed[1].records[0].record_id == tail.record_id
    assert client.committed[1].gaps == ()
    gaps = client.committed[0].gaps
    assert sorted((gap.from_sequence, gap.to_sequence, gap.record_count) for gap in gaps) == [
        (0, 0, 1),
        (1, 1, 1),
    ]
    assert all(gap.cause == EXPORT_FAILED_CAUSE for gap in gaps)
    assert all((gap.source_generation, gap.stream_epoch) == (0, 1) for gap in gaps)
    assert [failure.code for failure in exporter.failures()] == [failure_code] * 3
    assert len(exporter.receipts()) == 2
    assert lanes.cameras_with_work() == ()
    exporter.flush_once()
    assert len(client.posted) == 5
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_failure_after_commit_restores_only_the_unattempted_suffix() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    records = [_record(seq, payload={"blob": "c" * 700_000}) for seq in range(3)]
    for record in records:
        assert lanes.try_emit(record)
    client = _FailNth({1})
    exporter = _exporter(lanes, client)
    exporter.flush_once()
    assert _sequences(client.posted) == [0, 1]
    assert _sequences(client.committed) == [0]
    assert lanes.queued() == 1
    exporter.flush_once()
    assert _sequences(client.posted) == [0, 1, 2]
    assert _sequences(client.committed) == [0, 2]
    assert [record.record_id for batch in client.committed for record in batch.records] == [
        records[0].record_id,
        records[2].record_id,
    ]
    assert client.committed[0].gaps == ()
    assert client.committed[1].gaps == (
        WireGap("policy", 1, 1, 2_001, 2_001, 1, EXPORT_FAILED_CAUSE, 0, 1),
    )
    assert len(exporter.failures()) == 1
    assert len(exporter.receipts()) == 2
    exporter.flush_once()
    assert len(client.posted) == 3
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


@pytest.mark.parametrize("failure_code", ("NETWORK", "STORAGE_UNAVAILABLE"))
def test_failed_camera_boot_does_not_block_other_camera_boots(failure_code) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    first, tail = _two_fat_records()
    assert lanes.try_emit(_record())
    seed = lanes.drain_for("cam-1", "boot-1", limit=4)
    assert seed is not None
    lanes.note_export_failure(seed)
    assert lanes.try_emit(first)
    assert lanes.try_emit(tail)
    other_camera = _record(camera="cam-2")
    other_boot = _record(boot="boot-2")
    assert lanes.try_emit(other_camera)
    assert lanes.try_emit(other_boot)
    client = _FailNth({0}, failure_code=failure_code)
    exporter = _exporter(lanes, client)
    exporter.flush_once()
    assert len(client.posted) == 3
    assert [(batch.camera_id, batch.worker_boot_id) for batch in client.committed] == [
        ("cam-2", "boot-1"),
        ("cam-1", "boot-2"),
    ]
    assert [record.record_id for batch in client.committed for record in batch.records] == [
        other_camera.record_id,
        other_boot.record_id,
    ]
    assert all(not batch.gaps for batch in client.committed)
    assert lanes.cameras_with_work() == (("cam-1", "boot-1"),)
    exporter.flush_once()
    recovered = client.committed[-1]
    assert (recovered.camera_id, recovered.worker_boot_id) == ("cam-1", "boot-1")
    assert [record.record_id for record in recovered.records] == [tail.record_id]
    assert sorted(
        (gap.from_sequence, gap.to_sequence, gap.record_count) for gap in recovered.gaps
    ) == [(0, 0, 1), (1, 1, 1)]
    assert len(exporter.failures()) == 1
    assert lanes.cameras_with_work() == ()
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_gap_only_failure_restores_unattempted_gaps_without_changing_scope(monkeypatch) -> None:
    import worker.pipeline.diagnostics.exporter as exporter_module

    gaps = tuple(
        WireGap(
            f"{index:04d}{'p' * 124}",
            index * 10,
            index * 10 + 4,
            2_000,
            2_004,
            2,
            LANE_OVERFLOW_CAUSE,
            None if index % 2 else 0,
            None if index % 2 else 0,
        )
        for index in range(8)
    )
    cap = _batch_len(gaps=gaps[:4])
    record = _record()
    assert _batch_len((record,)) <= cap < MAX_EXECUTION_RECORD_BODY_BYTES
    monkeypatch.setattr(exporter_module, "MAX_EXECUTION_RECORD_BODY_BYTES", cap)
    lanes = ExecutionRecordLanes(lane_capacity=2)
    lanes.note_export_failure(DrainedLane("cam-1", "boot-1", (), gaps))
    assert lanes.try_emit(record)
    client = _FailNth({0})
    exporter = _exporter(lanes, client)
    exporter.flush_once()
    assert len(client.posted) == 1
    assert client.posted[0].gaps == gaps[:4]
    assert client.posted[0].records == ()
    assert client.committed == []
    assert lanes.queued() == 1
    exporter.flush_once()
    delivered = [gap for batch in client.committed for gap in batch.gaps]
    assert delivered == list(gaps)
    assert all(actual is original for actual, original in zip(delivered, gaps, strict=True))
    assert sum(gap.record_count for gap in delivered) == 16
    assert _sequences(client.committed) == [0]
    assert len(exporter.failures()) == 1
    for batch in client.committed:
        if batch.records:
            assert batch is client.committed[-1]
        assert WireBatch.from_json(batch.to_json()).batch_id == batch.batch_id
    for gap in delivered[1::2]:
        assert gap.source_generation is None and gap.stream_epoch is None
        assert "source_generation" not in gap.to_json()
        assert "stream_epoch" not in gap.to_json()
    assert lanes.cameras_with_work() == ()
    assert all(len(batch.encode()) <= cap for batch in client.posted)


@pytest.mark.parametrize("invalid_code", ("OVERSIZE", "ENCODING_ERROR"))
def test_failed_flush_restores_current_normalized_gap_not_invalid_record(
    monkeypatch, invalid_code
) -> None:
    bad_encodes: list[str] = []
    original = WireRecord.to_json

    def encode(self: WireRecord) -> dict[str, object]:
        if self.producer_sequence == 1:
            bad_encodes.append(self.record_id)
            if invalid_code == "ENCODING_ERROR":
                raise TypeError("cannot encode payload")
        return original(self)

    monkeypatch.setattr(WireRecord, "to_json", encode)
    first = _exact_cap_record(0)
    bad = _record(1, payload={"blob": "a" * 1_100_000}, generation=7, epoch=9)
    tail = _record(2, generation=7, epoch=9)
    lanes = ExecutionRecordLanes(lane_capacity=4)
    for record in (first, bad, tail):
        assert lanes.try_emit(record)
    client = _FailNth({0})
    exporter = _exporter(lanes, client)
    exporter.flush_once()
    assert _sequences(client.posted) == [0]
    assert client.committed == []
    assert lanes.queued() == 1
    assert bad_encodes == [bad.record_id]
    exporter.flush_once()
    assert _sequences(client.committed) == [2]
    assert client.committed[0].records[0].record_id == tail.record_id
    assert client.committed[0].gaps == (
        WireGap("policy", 0, 0, 2_000, 2_000, 1, EXPORT_FAILED_CAUSE, 0, 1),
        WireGap("policy", 1, 1, 2_001, 2_001, 1, RECORD_INVALID_CAUSE, 7, 9),
    )
    assert bad_encodes == [bad.record_id]
    assert [failure.code for failure in exporter.failures()] == [invalid_code, "NETWORK"]
    assert lanes.cameras_with_work() == ()
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_598_client_exception_does_not_drop_later_chunk() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    # Sequences 0 and 1, each alone under the cap and together over it.
    first = _record(0, payload={"blob": "c" * 700_000})
    second = _record(1, payload={"blob": "c" * 700_000})
    assert _batch_len((first,)) <= MAX_EXECUTION_RECORD_BODY_BYTES
    assert _batch_len((first, second)) > MAX_EXECUTION_RECORD_BODY_BYTES
    assert lanes.try_emit(first) is True
    assert lanes.try_emit(second) is True
    client = _RaiseFirst()
    exporter = _exporter(lanes, client, batch_max=8)
    exporter.flush_once()
    assert _sequences(client.posted) == [0]
    assert client.committed == []
    assert lanes.queued() == 1
    exporter.flush_once()
    assert _sequences(client.posted) == [0, 1]
    assert _sequences(client.committed) == [1]
    failed = [
        gap for batch in client.committed for gap in batch.gaps if gap.cause == EXPORT_FAILED_CAUSE
    ]
    assert len(failed) == 1
    assert (failed[0].from_sequence, failed[0].to_sequence, failed[0].record_count) == (0, 0, 1)
    assert len(exporter.failures()) == 1
    assert lanes.cameras_with_work() == ()
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


class _RaiseFirst:
    def __init__(self) -> None:
        self.posted: list[WireBatch] = []
        self.committed: list[WireBatch] = []

    def post_batch(self, batch: WireBatch) -> WireBatchReceipt:
        self.posted.append(batch)
        if len(self.posted) == 1:
            raise ConnectionError("relay down")
        self.committed.append(batch)
        return _receipt(batch)


def test_598_client_exception_does_not_kill_drain_thread() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    client = _RaiseOnce()
    exporter = _exporter(lanes, client)
    assert lanes.try_emit(_record()) is True
    exporter.start()
    try:
        assert client.done.wait(2), "drain thread died after a transport exception"
        assert exporter._thread is not None and exporter._thread.is_alive()
        with client.lock:
            posted = tuple(client.posted)
    finally:
        exporter.stop(timeout=2)
    assert len(posted) >= 2
    assert posted[1].records == ()
    gap = posted[1].gaps[0]
    assert gap.cause == EXPORT_FAILED_CAUSE
    assert gap.record_count == 1
    assert gap.from_sequence == 0


class _RaiseOnce:
    def __init__(self) -> None:
        self.posted: list[WireBatch] = []
        self.done = threading.Event()
        self.lock = threading.Lock()

    def post_batch(self, batch: WireBatch) -> WireBatchReceipt:
        with self.lock:
            self.posted.append(batch)
            count = len(self.posted)
        if count == 1:
            raise ConnectionError("relay down")
        self.done.set()
        return _receipt(batch)


@pytest.mark.parametrize("bad_sequence", (0, 1, 2))
def test_598_encoding_exception_accounts_record_and_posts_neighbors(
    monkeypatch, caplog, bad_sequence
) -> None:
    caplog.set_level(logging.WARNING)
    original = WireRecord.to_json

    def boom(self: WireRecord) -> dict[str, object]:
        if dict(self.payload) == {"boom": 1}:
            raise TypeError("cannot encode payload")
        return original(self)

    monkeypatch.setattr(WireRecord, "to_json", boom)
    lanes = ExecutionRecordLanes(lane_capacity=4)
    for sequence in range(3):
        payload = {"boom": 1} if sequence == bad_sequence else {"ok": sequence}
        assert lanes.try_emit(_record(sequence, payload=payload)) is True
    client = _Client()
    exporter = _exporter(lanes, client, batch_max=8)
    exporter.flush_once()
    assert _sequences(client.posted) == [
        sequence for sequence in range(3) if sequence != bad_sequence
    ]
    gaps = [gap for batch in client.posted for gap in batch.gaps]
    assert len(gaps) == 1
    assert gaps[0].cause == RECORD_INVALID_CAUSE
    assert (gaps[0].from_sequence, gaps[0].to_sequence, gaps[0].record_count) == (
        bad_sequence,
        bad_sequence,
        1,
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "cam-1" in message and "encoded" in message and "record-invalid" in message
        for message in messages
    )


def test_598_unsplittable_envelope_retains_loss_without_unbounded_retry(
    monkeypatch, caplog
) -> None:
    caplog.set_level(logging.WARNING)
    import worker.pipeline.diagnostics.exporter as exporter_module

    if not hasattr(exporter_module, "MAX_EXECUTION_RECORD_BODY_BYTES"):
        raise AssertionError("exporter does not bound encoded body bytes")
    monkeypatch.setattr(exporter_module, "MAX_EXECUTION_RECORD_BODY_BYTES", 32)
    lanes = ExecutionRecordLanes(lane_capacity=4)
    client = _Client()
    exporter = _exporter(lanes, client, batch_max=8)
    assert lanes.try_emit(_record()) is True
    exporter.flush_once()
    matched = [
        record.getMessage() for record in caplog.records if "envelope" in record.getMessage()
    ]
    assert len(matched) == 1
    assert "cam-1" in matched[0]
    assert "retained" in matched[0]
    assert client.posted == []
    assert all(failure.code == "OVERSIZE" for failure in exporter.failures())
    assert ("cam-1", "boot-1") in lanes.cameras_with_work()
    exporter.flush_once()
    assert client.posted == []
    pending = lanes.drain_for("cam-1", "boot-1", limit=4)
    assert pending is not None
    assert pending.records == ()
    assert len(pending.gaps) == 1
    assert pending.gaps[0].record_count == 1
    assert pending.gaps[0].from_sequence == pending.gaps[0].to_sequence == 0


def test_unsendable_gap_envelope_keeps_valid_unattempted_neighbors(monkeypatch) -> None:
    import worker.pipeline.diagnostics.exporter as exporter_module

    lanes = ExecutionRecordLanes(lane_capacity=4)
    records = tuple(_record(seq) for seq in range(3))
    for record in records:
        assert lanes.try_emit(record)
    client = _Client()
    exporter = _exporter(lanes, client)
    monkeypatch.setattr(exporter_module, "MAX_EXECUTION_RECORD_BODY_BYTES", 32)
    for _ in range(2):
        exporter.flush_once()
        assert client.posted == []
        assert lanes.queued() == 2
    monkeypatch.setattr(
        exporter_module, "MAX_EXECUTION_RECORD_BODY_BYTES", MAX_EXECUTION_RECORD_BODY_BYTES
    )
    exporter.flush_once()
    assert _sequences(client.committed) == [1, 2]
    assert [record.record_id for batch in client.committed for record in batch.records] == [
        record.record_id for record in records[1:]
    ]
    assert client.committed[0].gaps == (
        WireGap("policy", 0, 0, 2_000, 2_000, 1, RECORD_INVALID_CAUSE, 0, 1),
    )
    assert lanes.cameras_with_work() == ()
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_envelope_encoding_failure_restores_records_without_export_failed_loss(monkeypatch) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    gap = WireGap("policy", 0, 0, 2_000, 2_000, 1, EXPORT_FAILED_CAUSE, 0, 1)
    assert lanes.try_emit(_record())
    seed = lanes.drain_for("cam-1", "boot-1", limit=4)
    assert seed is not None
    lanes.note_export_failure(DrainedLane("cam-1", "boot-1", (), (gap,)))
    records = (_record(1), _record(2))
    for record in records:
        assert lanes.try_emit(record)
    client = _Client()
    exporter = _exporter(lanes, client)
    original = WireProvenance.to_json

    def encode(_self):
        raise TypeError("cannot encode provenance")

    monkeypatch.setattr(WireProvenance, "to_json", encode)
    exporter.flush_once()
    assert client.posted == []
    assert lanes.queued() == 2
    assert [failure.code for failure in exporter.failures()] == ["ENCODING_ERROR"]
    monkeypatch.setattr(WireProvenance, "to_json", original)
    exporter.flush_once()
    assert _sequences(client.committed) == [1, 2]
    assert [record.record_id for batch in client.committed for record in batch.records] == [
        record.record_id for record in records
    ]
    assert client.committed[0].gaps == (gap,)
    assert client.committed[0].gaps[0] is gap
    assert lanes.cameras_with_work() == ()
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_gap_encoding_failure_restores_buffered_and_remaining_unattempted_items(
    monkeypatch,
) -> None:
    gaps = (
        WireGap("before", 0, 0, 2_000, 2_000, 1, EXPORT_FAILED_CAUSE, 0, 1),
        WireGap("broken", 1, 3, 2_001, 2_003, 2, EXPORT_FAILED_CAUSE),
        WireGap("after", 4, 4, 2_004, 2_004, 1, EXPORT_FAILED_CAUSE, 7, 9),
    )
    original = WireGap.to_json

    def encode(self: WireGap) -> dict[str, object]:
        if self.producer == "broken":
            raise TypeError("cannot encode gap")
        return original(self)

    monkeypatch.setattr(WireGap, "to_json", encode)
    lanes = ExecutionRecordLanes(lane_capacity=2)
    lanes.note_export_failure(DrainedLane("cam-1", "boot-1", (), gaps))
    record = _record()
    assert lanes.try_emit(record)
    client = _Client()
    exporter = _exporter(lanes, client)
    exporter.flush_once()
    assert client.posted == []
    assert lanes.queued() == 1
    assert [failure.code for failure in exporter.failures()] == ["ENCODING_ERROR"]
    monkeypatch.setattr(WireGap, "to_json", original)
    exporter.flush_once()
    assert _sequences(client.committed) == [0]
    assert client.committed[0].records[0].record_id == record.record_id
    delivered = {gap.producer: gap for batch in client.committed for gap in batch.gaps}
    assert len(client.committed[0].gaps) == 3
    assert all(delivered[gap.producer] is gap for gap in gaps)
    assert sum(gap.record_count for gap in delivered.values()) == 4
    assert lanes.cameras_with_work() == ()
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_598_receipt_and_failure_histories_are_finite() -> None:
    try:
        from worker.pipeline.diagnostics.exporter import EXPORT_HISTORY_LIMIT
    except ImportError as error:
        raise AssertionError("exporter history is unbounded") from error
    extra = 5
    total = EXPORT_HISTORY_LIMIT + extra
    lanes = ExecutionRecordLanes(lane_capacity=4)
    client = _Client()
    exporter = _exporter(lanes, client, batch_max=4)
    for seq in range(total):
        assert lanes.try_emit(_record(seq)) is True
        exporter.flush_once()
    assert len(exporter.receipts()) == EXPORT_HISTORY_LIMIT
    assert [item.batch_id for item in exporter.receipts()] == [
        batch.batch_id for batch in client.posted[-EXPORT_HISTORY_LIMIT:]
    ]
    failed_lanes = ExecutionRecordLanes(lane_capacity=4)
    failed_client = _CodedFailure()
    failed = _exporter(failed_lanes, failed_client, batch_max=4)
    for seq in range(total):
        assert failed_lanes.try_emit(_record(seq)) is True
        failed.flush_once()
    assert len(failed.failures()) == EXPORT_HISTORY_LIMIT
    expected = [f"N{index:04d}" for index in range(extra, total)]
    assert [item.code for item in failed.failures()] == expected


class _CodedFailure:
    def __init__(self) -> None:
        self.index = 0
        self.posted: list[WireBatch] = []

    def post_batch(self, batch: WireBatch) -> DeliveryFailure:
        self.posted.append(batch)
        code = f"N{self.index:04d}"
        self.index += 1
        return DeliveryFailure(DeliveryDisposition.RETRY, code)


def test_598_failed_chunk_does_not_double_count_invalid_sequence() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    client = _FailNth({0})
    exporter = _exporter(lanes, client)
    for sequence in range(3):
        payload = {"blob": "a" * 1_100_000} if sequence == 1 else {"ok": sequence}
        assert lanes.try_emit(_record(sequence, payload=payload))
    exporter.flush_once()
    exporter.flush_once()
    delivered = client.committed
    assert len(client.posted) == 2 and len(delivered) == 1
    assert all(not batch.records for batch in delivered)
    gaps = [gap for batch in delivered for gap in batch.gaps]
    assert sorted((gap.from_sequence, gap.to_sequence, gap.record_count) for gap in gaps) == [
        (0, 0, 1),
        (1, 1, 1),
        (2, 2, 1),
    ]
    assert sum(gap.record_count for gap in gaps) == 3


def test_598_http_does_not_hold_producer_lane_lock() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=4)
    entered, release, emitted = threading.Event(), threading.Event(), threading.Event()

    class BlockingClient:
        def post_batch(self, batch):
            entered.set()
            assert release.wait(2)
            return _receipt(batch)

    exporter = _exporter(lanes, BlockingClient())
    assert lanes.try_emit(_record())
    flushing = threading.Thread(target=exporter.flush_once)
    flushing.start()
    producer = threading.Thread(target=lambda: (lanes.try_emit(_record()), emitted.set()))
    try:
        assert entered.wait(2)
        producer.start()
        assert emitted.wait(1), "HTTP stalled the producer append-or-drop lane"
    finally:
        release.set()
        flushing.join(2)
        if producer.ident is not None:
            producer.join(2)
    assert not flushing.is_alive()
    assert lanes.queued() == 1


class _BlockingFirstFailure(_FailNth):
    def __init__(self, *, failure_code: str = "NETWORK") -> None:
        super().__init__({0}, failure_code=failure_code)
        self.entered = threading.Event()
        self.release = threading.Event()

    def post_batch(self, batch: WireBatch) -> WireBatchReceipt | DeliveryFailure:
        result = super().post_batch(batch)
        if len(self.posted) == 1:
            self.entered.set()
            assert self.release.wait(2), "blocked HTTP was not released"
        return result


@pytest.mark.parametrize("failure_code", ("NETWORK", "STORAGE_UNAVAILABLE"))
def test_failed_http_restores_older_suffix_and_counts_only_newer_tail_overflow(
    failure_code,
) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=3)
    original = [_record(seq, payload={"blob": "c" * 700_000}) for seq in range(3)]
    for record in original:
        assert lanes.try_emit(record)
    client = _BlockingFirstFailure(failure_code=failure_code)
    exporter = _exporter(lanes, client)
    emitted = threading.Event()
    admissions: list[bool] = []

    def produce() -> None:
        for seq in range(3, 7):
            admissions.append(lanes.try_emit(_record(seq)))
        emitted.set()

    flushing = threading.Thread(target=exporter.flush_once)
    producer = threading.Thread(target=produce)
    flushing.start()
    try:
        assert client.entered.wait(2)
        producer.start()
        assert emitted.wait(1), "blocked HTTP held the producer lane lock"
        assert lanes.queued() == 3
    finally:
        client.release.set()
        flushing.join(2)
        if producer.ident is not None:
            producer.join(2)
    assert not flushing.is_alive()
    assert not producer.is_alive()
    assert admissions == [True, True, True, False]
    assert lanes.queued() == 3
    assert _sequences(client.posted) == [0]
    assert client.committed == []
    assert [failure.code for failure in exporter.failures()] == [failure_code]

    exporter.flush_once()
    assert _sequences(client.committed) == [1, 2, 3]
    assert [record.record_id for batch in client.committed for record in batch.records] == [
        original[1].record_id,
        original[2].record_id,
        _record(3).record_id,
    ]
    gaps = [gap for batch in client.committed for gap in batch.gaps]
    assert sorted(
        (gap.cause, gap.from_sequence, gap.to_sequence, gap.record_count) for gap in gaps
    ) == [
        (EXPORT_FAILED_CAUSE, 0, 0, 1),
        (LANE_OVERFLOW_CAUSE, 4, 6, 3),
    ]
    assert all((gap.source_generation, gap.stream_epoch) == (0, 1) for gap in gaps)
    assert lanes.cameras_with_work() == ()
    assert lanes.try_emit(_record(7))
    exporter.flush_once()
    assert _sequences(client.committed) == [1, 2, 3, 7]
    assert client.committed[-1].records[0].record_id == _record(7).record_id
    assert client.committed[-1].gaps == ()
    assert len(exporter.failures()) == 1
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)


def test_overlapping_flushes_cannot_drain_past_uncommitted_loss(monkeypatch) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=3)
    for seq in range(2):
        assert lanes.try_emit(_record(seq, payload={"blob": "c" * 700_000}))
    client = _BlockingFirstFailure()
    exporter = _exporter(lanes, client)
    contended = threading.Event()

    class ObservedLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                contended.set()
                assert self.lock.acquire(timeout=2), "flush ownership was not released"
            return self

        def __exit__(self, *_args) -> None:
            self.lock.release()

    monkeypatch.setattr(exporter, "_flush_lock", ObservedLock())
    first = threading.Thread(target=exporter.flush_once)
    second = threading.Thread(target=exporter.flush_once)
    first.start()
    try:
        assert client.entered.wait(2)
        assert lanes.try_emit(_record(2))
        second.start()
        assert contended.wait(1), "concurrent flush did not serialize drain ownership"
        assert _sequences(client.posted) == [0]
        assert client.committed == []
    finally:
        client.release.set()
        first.join(2)
        if second.ident is not None:
            second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert _sequences(client.posted) == [0, 1, 2]
    assert _sequences(client.committed) == [1, 2]
    assert len(client.committed) == 1
    assert client.committed[0].gaps == (
        WireGap("policy", 0, 0, 2_000, 2_000, 1, EXPORT_FAILED_CAUSE, 0, 1),
    )
    assert len(exporter.failures()) == 1
    assert lanes.cameras_with_work() == ()
    assert all(len(batch.encode()) <= MAX_EXECUTION_RECORD_BODY_BYTES for batch in client.posted)
