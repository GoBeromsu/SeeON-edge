"""Bounded in-memory execution-record lanes: overflow is counted as WireGap."""

from __future__ import annotations

from shared.events.execution_records import WireGap, WireRecord
from worker.pipeline.diagnostics.lanes import (
    EXPORT_FAILED_CAUSE,
    LANE_OVERFLOW_CAUSE,
    RECORD_INVALID_CAUSE,
    DrainedLane,
    ExecutionRecordLanes,
)


def _record(
    *,
    producer: str = "sdk",
    seq: int = 0,
    observed: int = 1_000,
    boot: str = "boot-1",
    camera: str = "cam-1",
    generation: int = 0,
    epoch: int = 1,
) -> WireRecord:
    return WireRecord(
        record_kind="sdk.frame",
        camera_id=camera,
        worker_boot_id=boot,
        source_generation=generation,
        stream_epoch=epoch,
        producer=producer,
        producer_sequence=seq,
        observed_at_ns=observed,
        time_quality="monotonic",
        causal_unit_id=f"{camera}:{boot}:{epoch}:frame:0",
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
    """A lane holding ONLY loss is still offered and drained records-empty.

    Overflow and record-invalid gaps ride with the drain that empties their
    lane, so the records-empty case is the export-failed path: a batch that
    failed to export leaves loss behind with no queued record. That loss must
    reach the Backend even if no further valid record ever arrives.
    """
    lanes = ExecutionRecordLanes(lane_capacity=4)
    assert lanes.try_emit(_record(seq=0)) is True
    drained = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert drained is not None and len(drained.records) == 1 and drained.gaps == ()
    lanes.note_export_failure(drained)

    # Nothing queued, yet the boot has work: the export-failed loss.
    assert lanes.queued() == 0
    assert ("cam-1", "boot-1") in lanes.cameras_with_work()
    only_loss = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert only_loss is not None
    assert only_loss.records == ()
    assert [g.cause for g in only_loss.gaps] == [EXPORT_FAILED_CAUSE]
    assert only_loss.gaps[0].from_sequence == 0 and only_loss.gaps[0].to_sequence == 0
    assert ("cam-1", "boot-1") not in lanes.cameras_with_work()


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


def test_restored_records_keep_original_lanes_identities_and_capacity() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=3)
    for producer in ("sdk", "policy"):
        for seq in range(3):
            assert lanes.try_emit(_record(producer=producer, seq=seq))
    # One full lane plus a prefix of another; the rest of policy stays queued.
    drained = lanes.drain_for("cam-1", "boot-1", limit=4)
    assert drained is not None and len(drained.records) == 4
    for seq in range(3, 6):
        assert lanes.try_emit(_record(seq=seq))
    assert not lanes.try_emit(_record(seq=6))
    assert lanes.try_emit(_record(producer="policy", seq=3))
    assert not lanes.try_emit(_record(producer="policy", seq=4))

    lanes.restore_unattempted(drained)
    assert lanes.queued() == 6
    restored = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert restored is not None
    assert [(record.producer, record.producer_sequence) for record in restored.records] == [
        ("sdk", 0),
        ("sdk", 1),
        ("sdk", 2),
        ("policy", 0),
        ("policy", 1),
        ("policy", 2),
    ]
    assert all(
        actual is original
        for actual, original in zip(restored.records[:4], drained.records, strict=True)
    )
    assert [record.record_id for record in restored.records[:4]] == [
        record.record_id for record in drained.records
    ]
    assert restored.gaps == (
        WireGap("sdk", 3, 6, 1_000, 1_000, 4, LANE_OVERFLOW_CAUSE, 0, 1),
        WireGap("policy", 3, 4, 1_000, 1_000, 2, LANE_OVERFLOW_CAUSE, 0, 1),
    )
    assert lanes.cameras_with_work() == ()
    # Restoration neither consumed sequences nor emitted new producer records.
    assert lanes.try_emit(_record(seq=7))
    assert lanes.try_emit(_record(producer="policy", seq=5))
    later = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert later is not None and later.gaps == ()
    assert [(record.producer, record.producer_sequence) for record in later.records] == [
        ("sdk", 7),
        ("policy", 5),
    ]


def test_restoration_preserves_gap_objects_and_camera_boot_ownership() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=2)
    gaps = (
        WireGap("sdk", 3, 8, 1_000, 2_000, 2, EXPORT_FAILED_CAUSE),
        WireGap("sdk", 10, 10, 3_000, 3_000, 1, RECORD_INVALID_CAUSE, 7, 9),
    )
    lanes.note_export_failure(DrainedLane("cam-1", "boot-1", (), gaps))
    assert lanes.try_emit(_record(generation=7, epoch=9))
    drained = lanes.drain_for("cam-1", "boot-1", limit=2)
    assert drained is not None
    other_camera = _record(camera="cam-2")
    other_boot = _record(boot="boot-2")
    assert lanes.try_emit(other_camera)
    assert lanes.try_emit(other_boot)
    lanes.restore_unattempted(drained)
    for camera, boot, expected in (
        ("cam-2", "boot-1", other_camera),
        ("cam-1", "boot-2", other_boot),
    ):
        other = lanes.drain_for(camera, boot, limit=2)
        assert other is not None
        assert other.records == (expected,)
        assert other.gaps == ()
    mine = lanes.drain_for("cam-1", "boot-1", limit=2)
    assert mine == drained
    assert mine is not None
    assert mine.records[0] is drained.records[0]
    assert all(actual is original for actual, original in zip(mine.gaps, gaps, strict=True))
    assert mine.gaps[0].source_generation is None and mine.gaps[0].stream_epoch is None
    assert (mine.records[0].source_generation, mine.records[0].stream_epoch) == (7, 9)
    assert lanes.cameras_with_work() == ()


def test_restoration_overflow_gaps_do_not_bridge_scope_changes_or_sequence_holes() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=2)
    for seq in range(2):
        assert lanes.try_emit(_record(seq=seq))
    drained = lanes.drain_for("cam-1", "boot-1", limit=2)
    assert drained is not None
    # New arrivals span scopes and their observed times need not be monotonic.
    assert lanes.try_emit(_record(seq=2, generation=7, epoch=9, observed=9_000))
    assert lanes.try_emit(_record(seq=3, generation=7, epoch=10, observed=5_000))
    assert not lanes.try_emit(_record(seq=4, generation=7, epoch=9, observed=3_000))
    assert not lanes.try_emit(_record(seq=5, generation=7, epoch=9, observed=7_000))
    lanes.restore_unattempted(drained)
    restored = lanes.drain_for("cam-1", "boot-1", limit=2)
    assert restored is not None
    assert restored.records == drained.records
    assert restored.gaps == (
        WireGap("sdk", 2, 2, 9_000, 9_000, 1, LANE_OVERFLOW_CAUSE, 7, 9),
        WireGap("sdk", 4, 5, 3_000, 7_000, 2, LANE_OVERFLOW_CAUSE, 7, 9),
        WireGap("sdk", 3, 3, 5_000, 5_000, 1, LANE_OVERFLOW_CAUSE, 7, 10),
    )
    assert sum(gap.record_count for gap in restored.gaps) == 4
    assert lanes.cameras_with_work() == ()


# Issue 598 repro. Select with: pytest -k test_598_
# Fails on the pre-fix lanes and passes once an unsendable drained record is an
# explicit record-invalid gap.


def test_598_unsendable_records_keep_neighbor_sequences_and_gap_counts() -> None:
    try:
        from worker.pipeline.diagnostics.lanes import account_unsendable_records
    except ImportError as error:
        raise AssertionError("unsendable records are not accounted") from error
    lanes = ExecutionRecordLanes(lane_capacity=8)
    assert lanes.try_emit(_record()) is True
    first = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert first is not None
    lanes.note_export_failure(first)
    assert lanes.try_emit(_record()) is True
    assert lanes.try_emit(_record()) is True
    assert lanes.try_emit(_record()) is True
    second = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert second is not None
    assert [record.producer_sequence for record in second.records] == [1, 2, 3]
    accounted = account_unsendable_records(second, (second.records[0], second.records[2]))
    assert [record.producer_sequence for record in accounted.records] == [2]
    assert [gap.cause for gap in accounted.gaps] == [
        EXPORT_FAILED_CAUSE,
        RECORD_INVALID_CAUSE,
        RECORD_INVALID_CAUSE,
    ]
    assert accounted.gaps[0].record_count == 1
    assert accounted.gaps[0].from_sequence == 0
    invalid = [(gap.from_sequence, gap.to_sequence, gap.record_count) for gap in accounted.gaps[1:]]
    assert invalid == [(1, 1, 1), (3, 3, 1)]
    assert accounted.camera_id == "cam-1"
    assert accounted.worker_boot_id == "boot-1"
    assert sum(gap.record_count for gap in accounted.gaps) == 3
