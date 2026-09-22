"""Bounded in-memory execution-record lanes. Overflow is counted, never hidden."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from shared.events.execution_records import ExecutionRecordContractError, WireGap, WireRecord

LANE_OVERFLOW_CAUSE = "lane-overflow"
EXPORT_FAILED_CAUSE = "export-failed"
RECORD_INVALID_CAUSE = "record-invalid"


@dataclass(frozen=True, slots=True)
class _LaneKey:
    camera_id: str
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
    """Per-(camera, producer) deques. ``try_emit`` is a short-lock append-or-drop."""

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
            key = _LaneKey(record.camera_id, record.producer)
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
                return False
            lane.next_sequence = sequence + 1
            if len(lane.records) >= self._capacity:
                if lane.overflow is None:
                    lane.overflow = []
                lane.overflow.append(queued)
                return False
            lane.records.append(queued)
            self._condition.notify_all()
        return True

    def wait_for_work(self, *, timeout_sec: float, batch_max: int) -> None:
        with self._condition:
            _ = self._condition.wait_for(
                lambda: self._batch_ready(batch_max) or self._has_records(),
                timeout=timeout_sec,
            )

    def cameras_with_work(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            keys: dict[tuple[str, str], None] = {}
            for lane in self._lanes.values():
                if lane.records:
                    record = lane.records[0]
                    keys[(record.camera_id, record.worker_boot_id)] = None
            return tuple(keys)

    def queued(self) -> int:
        with self._lock:
            return sum(len(lane.records) for lane in self._lanes.values())

    def drain_for(self, camera_id: str, worker_boot_id: str, *, limit: int) -> DrainedLane | None:
        if limit < 1:
            raise ValueError("drain limit must be a positive integer")
        records: list[WireRecord] = []
        gaps: list[WireGap] = []
        with self._lock:
            for lane in self._lanes.values():
                while lane.records and len(records) < limit:
                    record = lane.records[0]
                    if record.camera_id != camera_id or record.worker_boot_id != worker_boot_id:
                        break
                    records.append(lane.records.popleft())
                    gaps.extend(_take_overflow(lane))
                    gaps.extend(_take_invalid(lane))
            if not records:
                for key, lane in self._lanes.items():
                    if key.camera_id != camera_id:
                        continue
                    pending = _take_invalid(lane)
                    if pending:
                        gaps.extend(pending)
            if records or gaps:
                gaps.extend(self._export_failed.pop((camera_id, worker_boot_id), ()))
        if not records and not gaps:
            return None
        return DrainedLane(camera_id, worker_boot_id, tuple(records), tuple(gaps))

    def note_export_failure(self, drained: DrainedLane) -> None:
        gaps = [
            *_gaps_for_records(drained.records, EXPORT_FAILED_CAUSE),
            *drained.gaps,
        ]
        if not gaps:
            return
        key = (drained.camera_id, drained.worker_boot_id)
        with self._lock:
            pending = self._export_failed.setdefault(key, [])
            pending.extend(gaps)

    def _has_records(self) -> bool:
        return any(lane.records for lane in self._lanes.values())

    def _batch_ready(self, batch_max: int) -> bool:
        counts: dict[tuple[str, str], int] = {}
        for lane in self._lanes.values():
            for record in lane.records:
                key = (record.camera_id, record.worker_boot_id)
                counts[key] = counts.get(key, 0) + 1
                if counts[key] >= batch_max:
                    return True
        return False


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


def _note_invalid(lane: _Lane, record: WireRecord, sequence: int) -> None:
    gap = WireGap(
        producer=record.producer,
        from_sequence=sequence,
        to_sequence=sequence,
        from_ns=record.observed_at_ns,
        to_ns=record.observed_at_ns,
        record_count=1,
        cause=RECORD_INVALID_CAUSE,
    )
    if lane.invalid_gaps is None:
        lane.invalid_gaps = []
    lane.invalid_gaps.append(gap)


def _gaps_for_records(
    records: tuple[WireRecord, ...] | list[WireRecord], cause: str
) -> list[WireGap]:
    grouped: dict[str, list[WireRecord]] = {}
    for record in records:
        grouped.setdefault(record.producer, []).append(record)
    # Sequence bounds are lane-assigned and monotonic per producer, so
    # first/last are correct. observed_at_ns is producer-supplied and may not
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
        )
        for producer, items in grouped.items()
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
]
