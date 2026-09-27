"""Bounded in-memory execution-record lanes. Overflow is counted, never hidden."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from shared.events.execution_records import ExecutionRecordContractError, WireGap, WireRecord

LANE_OVERFLOW_CAUSE = "lane-overflow"
EXPORT_FAILED_CAUSE = "export-failed"
RECORD_INVALID_CAUSE = "record-invalid"


@dataclass(frozen=True, slots=True)
class _LaneKey:
    """One lane per (camera, boot, producer).

    Boot is part of the key so producer_sequence restarts per boot (the
    Backend indexes sequences per boot) and so pending loss - overflow and
    record-invalid gaps - is attributed to the boot that suffered it, never
    swallowed by a drain for another boot of the same camera.
    """

    camera_id: str
    worker_boot_id: str
    producer: str


@dataclass(slots=True)
class _Lane:
    records: deque[WireRecord]
    next_sequence: int = 0
    overflow: list[WireRecord] | None = None
    invalid_gaps: list[WireGap] | None = None


@dataclass(frozen=True, slots=True)
class DrainedLane:
    camera_id: str
    worker_boot_id: str
    records: tuple[WireRecord, ...]
    gaps: tuple[WireGap, ...]


class ExecutionRecordLanes:
    """Per-(camera, boot, producer) deques. ``try_emit`` is a short-lock append-or-drop."""

    def __init__(self, *, lane_capacity: int) -> None:
        if lane_capacity < 1:
            raise ValueError("lane_capacity must be a positive integer")
        self._capacity = lane_capacity
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._lanes: dict[_LaneKey, _Lane] = {}
        self._export_failed: dict[tuple[str, str], list[WireGap]] = {}

    def try_emit(self, record: object) -> bool:
        if not isinstance(record, WireRecord):
            return False
        with self._condition:
            key = _LaneKey(record.camera_id, record.worker_boot_id, record.producer)
            lane = self._lanes.get(key)
            if lane is None:
                lane = _Lane(deque())
                self._lanes[key] = lane
            sequence = lane.next_sequence
            try:
                queued = _with_sequence(record, sequence)
            except ExecutionRecordContractError:
                lane.next_sequence = sequence + 1
                _note_invalid(lane, record, sequence)
                self._condition.notify_all()
                return False
            lane.next_sequence = sequence + 1
            if len(lane.records) >= self._capacity:
                if lane.overflow is None:
                    lane.overflow = []
                lane.overflow.append(queued)
                self._condition.notify_all()
                return False
            lane.records.append(queued)
            self._condition.notify_all()
        return True

    def wait_for_work(self, *, timeout_sec: float, batch_max: int) -> None:
        with self._condition:
            _ = self._condition.wait_for(
                lambda: self._batch_ready(batch_max) or self._has_work(),
                timeout=timeout_sec,
            )

    def cameras_with_work(self) -> tuple[tuple[str, str], ...]:
        """Every (camera, boot) with queued records OR pending loss to report.

        A lane holding only overflow / record-invalid gaps still has work:
        the loss must reach the Backend even if no further valid record ever
        arrives for that boot, otherwise the gap is hidden until it happens
        to ride along with a later batch.
        """
        with self._lock:
            keys: dict[tuple[str, str], None] = {}
            for key, lane in self._lanes.items():
                if lane.records or lane.overflow or lane.invalid_gaps:
                    keys[(key.camera_id, key.worker_boot_id)] = None
            for key in self._export_failed:
                keys[key] = None
            return tuple(keys)

    def queued(self) -> int:
        with self._lock:
            return sum(len(lane.records) for lane in self._lanes.values())

    def drain_for(self, camera_id: str, worker_boot_id: str, *, limit: int) -> DrainedLane | None:
        """Pop up to ``limit`` records and this boot's pending gaps.

        Encoding and HTTP stay with the caller, after this lock is released.
        """
        if limit < 1:
            raise ValueError("drain limit must be a positive integer")
        records: list[WireRecord] = []
        gaps: list[WireGap] = []
        with self._lock:
            for key, lane in self._lanes.items():
                if key.camera_id != camera_id or key.worker_boot_id != worker_boot_id:
                    continue
                while lane.records and len(records) < limit:
                    records.append(lane.records.popleft())
                # Pending loss for this boot's lane drains with or without records.
                gaps.extend(_take_overflow(lane))
                gaps.extend(_take_invalid(lane))
            if records or gaps:
                gaps.extend(self._export_failed.pop((camera_id, worker_boot_id), ()))
            elif (camera_id, worker_boot_id) in self._export_failed:
                gaps.extend(self._export_failed.pop((camera_id, worker_boot_id)))
        if not records and not gaps:
            return None
        return DrainedLane(camera_id, worker_boot_id, tuple(records), tuple(gaps))

    def restore_unattempted(self, drained: DrainedLane) -> None:
        """Return a serialized drain's unsent suffix without assigning identities.

        Each producer's suffix fits its original capacity. Older restored
        records displace only the newest arrivals, which become lane overflow.
        Gaps are already normalized loss and retain their objects and counts.
        """
        by_lane: dict[_LaneKey, list[WireRecord]] = {}
        for record in drained.records:
            key = _LaneKey(record.camera_id, record.worker_boot_id, record.producer)
            by_lane.setdefault(key, []).append(record)
        with self._condition:
            for key, records in by_lane.items():
                lane = self._lanes[key]
                evicted: list[WireRecord] = []
                while len(lane.records) + len(records) > self._capacity:
                    evicted.append(lane.records.pop())
                lane.records.extendleft(reversed(records))
                if evicted:
                    if lane.overflow is None:
                        lane.overflow = []
                    # Tail pops are descending and precede any arrival-time
                    # overflow, so restore sequence order for gap grouping.
                    lane.overflow[:0] = reversed(evicted)
            if drained.gaps:
                pending = self._export_failed.setdefault(
                    (drained.camera_id, drained.worker_boot_id), []
                )
                pending.extend(drained.gaps)
            self._condition.notify_all()

    def note_export_failure(self, drained: DrainedLane) -> None:
        gaps = [
            *_gaps_for_records(drained.records, EXPORT_FAILED_CAUSE),
            *drained.gaps,
        ]
        if not gaps:
            return
        key = (drained.camera_id, drained.worker_boot_id)
        with self._condition:
            pending = self._export_failed.setdefault(key, [])
            pending.extend(gaps)
            self._condition.notify_all()

    def _has_work(self) -> bool:
        return bool(self._export_failed) or any(
            lane.records or lane.overflow or lane.invalid_gaps for lane in self._lanes.values()
        )

    def _batch_ready(self, batch_max: int) -> bool:
        counts: dict[tuple[str, str], int] = {}
        for lane in self._lanes.values():
            for record in lane.records:
                key = (record.camera_id, record.worker_boot_id)
                counts[key] = counts.get(key, 0) + 1
                if counts[key] >= batch_max:
                    return True
        return False


def account_unsendable_records(
    drained: DrainedLane, unsendable: Sequence[WireRecord]
) -> DrainedLane:
    """Drop ``unsendable`` records and append one ``record-invalid`` gap each.

    Neighbor records keep their order and sequences. Existing gaps stay in
    front, with their ``record_count`` unchanged. One gap per dropped record
    so a hole is not swallowed by a single span.
    """
    if not unsendable:
        return drained
    pending = {record.record_id: record for record in unsendable}
    kept: list[WireRecord] = []
    gaps = list(drained.gaps)
    for record in drained.records:
        key = record.record_id
        if pending.pop(key, None) is None:
            kept.append(record)
            continue
        gaps.append(_single_record_gap(record, RECORD_INVALID_CAUSE, record.producer_sequence))
    if pending:
        raise ValueError("unsendable records must belong to the drained batch")
    return DrainedLane(drained.camera_id, drained.worker_boot_id, tuple(kept), tuple(gaps))


def _take_overflow(lane: _Lane) -> tuple[WireGap, ...]:
    dropped = lane.overflow
    if not dropped:
        return ()
    lane.overflow = None
    return _gaps_for_records(tuple(dropped), LANE_OVERFLOW_CAUSE)


def _take_invalid(lane: _Lane) -> tuple[WireGap, ...]:
    gaps = lane.invalid_gaps
    if not gaps:
        return ()
    lane.invalid_gaps = None
    return tuple(gaps)


def _single_record_gap(record: WireRecord, cause: str, sequence: int) -> WireGap:
    return WireGap(
        producer=record.producer,
        from_sequence=sequence,
        to_sequence=sequence,
        from_ns=record.observed_at_ns,
        to_ns=record.observed_at_ns,
        record_count=1,
        cause=cause,
        source_generation=record.source_generation,
        stream_epoch=record.stream_epoch,
    )


def _note_invalid(lane: _Lane, record: WireRecord, sequence: int) -> None:
    gap = _single_record_gap(record, RECORD_INVALID_CAUSE, sequence)
    if lane.invalid_gaps is None:
        lane.invalid_gaps = []
    lane.invalid_gaps.append(gap)


def _gaps_for_records(
    records: tuple[WireRecord, ...] | list[WireRecord], cause: str
) -> list[WireGap]:
    grouped: dict[tuple[str, int, int], list[list[WireRecord]]] = {}
    for record in records:
        key = (record.producer, record.source_generation, record.stream_epoch)
        runs = grouped.setdefault(key, [])
        if not runs or runs[-1][-1].producer_sequence + 1 != record.producer_sequence:
            runs.append([])
        runs[-1].append(record)
    # A separately accounted invalid record can leave holes in a failed batch.
    # Never describe those holes as part of another exact gap. Timestamps may not
    # be monotonic across the dropped items (PTS-derived vs process-monotonic
    # producers, reordered publishes), so the time range must be min/max: a
    # WireGap with from_ns > to_ns is a contract error that would kill the
    # exporter thread and silently hide the very loss it is meant to report.
    return [
        WireGap(
            producer=producer,
            from_sequence=items[0].producer_sequence,
            to_sequence=items[-1].producer_sequence,
            from_ns=min(item.observed_at_ns for item in items),
            to_ns=max(item.observed_at_ns for item in items),
            record_count=len(items),
            cause=cause,
            source_generation=generation,
            stream_epoch=epoch,
        )
        for (producer, generation, epoch), runs in grouped.items()
        for items in runs
    ]


def _with_sequence(record: WireRecord, sequence: int) -> WireRecord:
    return WireRecord(
        record_kind=record.record_kind,
        camera_id=record.camera_id,
        worker_boot_id=record.worker_boot_id,
        source_generation=record.source_generation,
        stream_epoch=record.stream_epoch,
        producer=record.producer,
        producer_sequence=sequence,
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


__all__ = [
    "EXPORT_FAILED_CAUSE",
    "LANE_OVERFLOW_CAUSE",
    "RECORD_INVALID_CAUSE",
    "DrainedLane",
    "ExecutionRecordLanes",
    "account_unsendable_records",
]
