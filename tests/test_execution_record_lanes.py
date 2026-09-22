"""Bounded in-memory execution-record lanes: overflow is counted as WireGap."""

from __future__ import annotations

from shared.events.execution_records import WireRecord
from worker.pipeline.diagnostics.lanes import (
    EXPORT_FAILED_CAUSE,
    LANE_OVERFLOW_CAUSE,
    RECORD_INVALID_CAUSE,
    ExecutionRecordLanes,
)


def _record(
    *, producer: str = "sdk", seq: int = 0, observed: int = 1_000, boot: str = "boot-1"
) -> WireRecord:
    return WireRecord(
        record_kind="sdk.frame",
        camera_id="cam-1",
        worker_boot_id=boot,
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


def test_try_emit_assigns_monotonic_producer_sequence() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=8)
    assert lanes.try_emit(_record()) is True
    assert lanes.try_emit(_record()) is True
    drained = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert drained is not None
    assert [record.producer_sequence for record in drained.records] == [0, 1]


def test_overflow_is_reported_as_lane_overflow_gap_on_next_drain() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=1)
    assert lanes.try_emit(_record()) is True
    assert lanes.try_emit(_record()) is False
    drained = lanes.drain_for("cam-1", "boot-1", limit=1)
    assert drained is not None
    assert len(drained.records) == 1
    assert len(drained.gaps) == 1
    gap = drained.gaps[0]
    assert gap.cause == LANE_OVERFLOW_CAUSE
    assert gap.from_sequence == 1
    assert gap.to_sequence == 1
    assert gap.record_count == 1


def test_export_failure_is_reported_on_the_next_batch() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=8)
    assert lanes.try_emit(_record()) is True
    first = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert first is not None
    lanes.note_export_failure(first)
    assert lanes.try_emit(_record()) is True
    second = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert second is not None
    assert [gap.cause for gap in second.gaps] == [EXPORT_FAILED_CAUSE]
    assert second.gaps[0].from_sequence == 0
    assert second.records[0].producer_sequence == 1


def test_overflow_gap_time_range_is_min_max_not_arrival_order() -> None:
    """Dropped items' observed_at_ns may arrive out of order (PTS-derived and
    process-monotonic producers, reordered publishes). The gap is a range, so
    its bounds must be min/max; first/last produced from_ns > to_ns, which the
    wire contract rejects and which killed the exporter thread instead of
    reporting the loss."""
    lanes = ExecutionRecordLanes(lane_capacity=1)
    assert lanes.try_emit(_record(observed=5_000)) is True
    # three drops, non-monotonic timestamps
    assert lanes.try_emit(_record(observed=9_000)) is False
    assert lanes.try_emit(_record(observed=3_000)) is False
    assert lanes.try_emit(_record(observed=7_000)) is False
    drained = lanes.drain_for("cam-1", "boot-1", limit=1)
    assert drained is not None
    (gap,) = drained.gaps
    assert gap.cause == LANE_OVERFLOW_CAUSE
    assert gap.record_count == 3
    assert (gap.from_ns, gap.to_ns) == (3_000, 9_000)
    assert gap.from_sequence <= gap.to_sequence


def test_invalid_sequenced_record_is_reported_as_record_invalid_gap() -> None:
    """A contract failure after the sequence was consumed must leave a hole."""

    def _invalid() -> WireRecord:
        template = _record()
        broken = object.__new__(WireRecord)
        for name in WireRecord.__slots__:
            object.__setattr__(broken, name, getattr(template, name))
        object.__setattr__(broken, "time_quality", "not-a-quality")
        return broken

    lanes = ExecutionRecordLanes(lane_capacity=8)
    assert lanes.try_emit(_record()) is True
    assert lanes.try_emit(_invalid()) is False
    assert lanes.try_emit(_record()) is True
    drained = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert drained is not None
    assert [record.producer_sequence for record in drained.records] == [0, 2]
    assert len(drained.gaps) == 1
    gap = drained.gaps[0]
    assert gap.cause == RECORD_INVALID_CAUSE
    assert gap.from_sequence == 1
    assert gap.to_sequence == 1
    assert gap.record_count == 1


def test_pending_loss_is_offered_for_export_without_any_valid_record() -> None:
    """A lane holding only overflow/invalid gaps must still be drained: the
    loss has to reach the Backend even if no further valid record ever
    arrives for that boot, otherwise it hides until a later batch."""
    lanes = ExecutionRecordLanes(lane_capacity=1)
    assert lanes.try_emit(_record(seq=0)) is True
    assert lanes.try_emit(_record(seq=1)) is False  # overflow
    first = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert first is not None and len(first.records) == 1 and len(first.gaps) == 1
    # Now nothing valid remains; produce one more overflow with no valid successor.
    assert lanes.try_emit(_record(seq=2)) is True
    assert lanes.try_emit(_record(seq=3)) is False
    drained = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert drained is not None
    # drain the last valid record; the overflow gap must ride along or stand alone
    assert ("cam-1", "boot-1") in lanes.cameras_with_work() or drained.gaps
    assert any(g.cause == LANE_OVERFLOW_CAUSE for g in drained.gaps)


def test_pending_loss_is_scoped_to_the_boot_that_suffered_it() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=1)
    boot_a = _record(seq=0)
    boot_b = _record(seq=0, boot="boot-2")
    assert lanes.try_emit(boot_a) is True
    assert lanes.try_emit(_record(seq=1)) is False  # boot-1 overflow
    assert lanes.try_emit(boot_b) is True
    # Draining boot-2 must not swallow boot-1's loss.
    other = lanes.drain_for("cam-1", "boot-2", limit=8)
    assert other is not None and len(other.records) == 1 and other.gaps == ()
    mine = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert mine is not None
    assert any(g.cause == LANE_OVERFLOW_CAUSE for g in mine.gaps)
    assert ("cam-1", "boot-1") not in lanes.cameras_with_work()
