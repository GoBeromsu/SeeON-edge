"""Ingest helpers: provenance and set-based record ingest for one batch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import psycopg

from backend.app.features.diagnostics.coverage import (
    insert_coverage,
    parent_loss_kind,
)
from backend.app.features.diagnostics.records import (
    CoverageKind,
    ExecutionRecordInput,
    Provenance,
    RecordKind,
    UnitCausalState,
    canonical_json,
    late_ack_unit_id,
)
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.segments import SegmentAllocator


def payload_text_and_bytes(record: ExecutionRecordInput) -> tuple[str, int]:
    text = canonical_json(dict(record.payload))
    return text, len(text.encode())


def upsert_provenance(connection: psycopg.Connection, provenance: Provenance, now_ns: int) -> str:
    provenance_id = provenance.provenance_id
    connection.execute(
        """
        INSERT INTO execution_provenance (
            provenance_id, worker_build_revision, worker_image_digest, model_digest,
            calibration_digest, preprocessing_identity, config_digest, policy_identity,
            backend_build_revision, first_seen_ns
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(provenance_id) DO NOTHING
        """,
        (
            provenance_id,
            provenance.worker_build_revision,
            provenance.worker_image_digest,
            provenance.model_digest,
            provenance.calibration_digest,
            provenance.preprocessing_identity,
            provenance.config_digest,
            provenance.policy_identity,
            provenance.backend_build_revision,
            now_ns,
        ),
    )
    return provenance_id


def _late_ack_unit(
    connection: psycopg.Connection,
    record: ExecutionRecordInput,
    batch_id: str,
    now_ns: int,
) -> str:
    parent = parent_loss_kind(connection, record.camera_id, record.observed_at_ns)
    if parent is None:
        return record.causal_unit_id
    deleted = parent is CoverageKind.DELETED_BY_CAPACITY
    kind = (
        CoverageKind.ACK_OBSERVED_PARENT_DELETED
        if deleted
        else CoverageKind.ACK_OBSERVED_PARENT_UNKNOWN_COARSENED
    )
    insert_coverage(
        connection,
        camera_id=record.camera_id,
        worker_boot_id=record.worker_boot_id,
        source_generation=record.source_generation,
        stream_epoch=record.stream_epoch,
        kind=kind,
        producer=record.producer,
        from_sequence=record.producer_sequence,
        to_sequence=record.producer_sequence,
        from_ns=record.observed_at_ns,
        to_ns=record.observed_at_ns,
        record_count=1,
        exact=deleted,
        cause="late-ack",
        recorded_at_ns=now_ns,
    )
    return late_ack_unit_id(record.causal_unit_id, batch_id)


@dataclass(frozen=True, slots=True)
class IngestResult:
    accepted: int
    written_bytes: int
    duplicates: int
    rejected: tuple[tuple[str, str], ...]


def _content_key(record: ExecutionRecordInput, payload_text: str) -> tuple[object, ...]:
    return (
        str(record.record_kind),
        record.camera_id,
        record.worker_boot_id,
        record.source_generation,
        record.stream_epoch,
        record.producer,
        record.producer_sequence,
        record.frame_seq,
        record.source_pts_ns,
        record.observed_at_ns,
        record.time_quality,
        record.causal_unit_id,
        record.parent_record_id,
        record.outcome,
        record.reason,
        payload_text,
    )


def _stored_keys(
    connection: psycopg.Connection, record_ids: list[str]
) -> dict[str, tuple[object, ...]]:
    rows = connection.execute(
        """
        SELECT record_id, record_kind, camera_id, worker_boot_id, source_generation,
               stream_epoch, producer, producer_sequence, frame_seq, source_pts_ns,
               observed_at_ns, time_quality, causal_unit_id, parent_record_id, outcome,
               reason, payload
        FROM execution_records WHERE record_id = ANY(%s)
        """,
        (record_ids,),
    ).fetchall()
    return {
        str(row[0]): (
            str(row[1]),
            str(row[2]),
            str(row[3]),
            int(row[4]),
            int(row[5]),
            str(row[6]),
            int(row[7]),
            None if row[8] is None else int(row[8]),
            None if row[9] is None else int(row[9]),
            int(row[10]),
            str(row[11]),
            str(row[12]),
            None if row[13] is None else str(row[13]),
            str(row[14]),
            None if row[15] is None else str(row[15]),
            str(row[16]),
        )
        for row in rows
    }


def _stored_units(connection: psycopg.Connection, unit_ids: set[str]) -> set[str]:
    rows = connection.execute(
        "SELECT causal_unit_id FROM execution_units WHERE causal_unit_id = ANY(%s)",
        (sorted(unit_ids),),
    ).fetchall()
    return {str(row[0]) for row in rows}


class _BatchWriter:
    """Buffered unit, segment and record writes for one ingest batch.

    Units must exist before the records that reference them, so ``flush``
    writes new units, then unit aggregates, then segment increments, then
    records. Anything that reads or writes these tables outside this writer
    during the batch must ``flush`` first.
    """

    def __init__(self, connection: psycopg.Connection, known_units: set[str]) -> None:
        self._connection = connection
        self._known_units = known_units
        # unit_id -> (camera, boot, generation, epoch) of a unit not yet inserted
        self._new_units: dict[str, tuple[str, str, int, int]] = {}
        # unit_id -> [first_ns, last_ns, record_count, payload_bytes] not yet written
        self._unit_totals: dict[str, list[int]] = {}
        self._records: list[tuple[object, ...]] = []

    def is_known_unit(self, unit_id: str) -> bool:
        return unit_id in self._known_units

    def add_to_unit(self, unit_id: str, record: ExecutionRecordInput, payload_bytes: int) -> None:
        if unit_id not in self._known_units:
            self._known_units.add(unit_id)
            self._new_units[unit_id] = (
                record.camera_id,
                record.worker_boot_id,
                record.source_generation,
                record.stream_epoch,
            )
        observed = record.observed_at_ns
        totals = self._unit_totals.get(unit_id)
        if totals is None:
            self._unit_totals[unit_id] = [observed, observed, 1, payload_bytes]
            return
        totals[0] = min(totals[0], observed)
        totals[1] = max(totals[1], observed)
        totals[2] += 1
        totals[3] += payload_bytes

    def add_record(self, row: tuple[object, ...]) -> None:
        self._records.append(row)

    def flush(self, allocator: SegmentAllocator) -> None:
        inserts = [
            (
                unit_id,
                *scope,
                str(UnitCausalState.INCOMPLETE_UNKNOWN),
                *self._unit_totals[unit_id],
            )
            for unit_id, scope in self._new_units.items()
        ]
        updates = [
            (first_ns, last_ns, count, size, unit_id)
            for unit_id, (first_ns, last_ns, count, size) in self._unit_totals.items()
            if unit_id not in self._new_units
        ]
        with self._connection.cursor() as cursor:
            if inserts:
                cursor.executemany(
                    """
                    INSERT INTO execution_units (
                        causal_unit_id, camera_id, worker_boot_id, source_generation,
                        stream_epoch, causal_state, terminal, first_observed_ns,
                        last_observed_ns, record_count, payload_bytes
                    ) VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
                    """,
                    inserts,
                )
            if updates:
                cursor.executemany(
                    """
                    UPDATE execution_units
                    SET first_observed_ns = LEAST(first_observed_ns, %s),
                        last_observed_ns = GREATEST(last_observed_ns, %s),
                        record_count = record_count + %s,
                        payload_bytes = payload_bytes + %s
                    WHERE causal_unit_id = %s
                    """,
                    updates,
                )
            allocator.flush()
            if self._records:
                cursor.executemany(
                    """
                    INSERT INTO execution_records (
                        record_id, record_kind, camera_id, worker_boot_id, source_generation,
                        stream_epoch, producer, producer_sequence, frame_seq, source_pts_ns,
                        observed_at_ns, time_quality, causal_unit_id, parent_record_id,
                        segment_id, provenance_id, outcome, reason, payload, payload_bytes,
                        committed_at_ns
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    self._records,
                )
        self._new_units.clear()
        self._unit_totals.clear()
        self._records.clear()


def ingest_records(
    connection: psycopg.Connection,
    records: Sequence[ExecutionRecordInput],
    provenance_id: str,
    budget: RetentionBudget,
    batch_id: str,
    now_ns: int,
) -> IngestResult:
    """Store one batch's records with set-based reads and pipelined writes.

    Dispositions match a per-record insert in batch order: an oversize record
    is rejected before any lookup, a stored or earlier-in-batch record_id is a
    duplicate when its stored content is identical and a conflict otherwise,
    and an ACK whose unit is gone is re-homed to its late-ACK unit. The
    buffered rows are flushed before every inline coverage write so it sees
    the same unit state a per-record insert would have left.
    """
    stored = _stored_keys(connection, [record.record_id for record in records])
    unit_ids = {record.causal_unit_id for record in records}
    unit_ids.update(
        late_ack_unit_id(record.causal_unit_id, batch_id)
        for record in records
        if record.record_kind is RecordKind.BACKEND_ACCEPTANCE
    )
    writer = _BatchWriter(connection, _stored_units(connection, unit_ids))
    allocator = SegmentAllocator(connection, budget, now_ns)
    accepted = 0
    written_bytes = 0
    duplicates = 0
    rejected: list[tuple[str, str]] = []
    for record in records:
        payload_text, payload_bytes = payload_text_and_bytes(record)
        if payload_bytes > budget.max_record_bytes:
            writer.flush(allocator)
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
        existing = stored.get(record.record_id)
        if existing is not None:
            if existing == _content_key(record, payload_text):
                duplicates += 1
            else:
                rejected.append((record.record_id, "conflict"))
            continue
        unit_id = record.causal_unit_id
        if (
            not writer.is_known_unit(unit_id)
            and record.record_kind is RecordKind.BACKEND_ACCEPTANCE
        ):
            writer.flush(allocator)
            unit_id = _late_ack_unit(connection, record, batch_id, now_ns)
        writer.add_to_unit(unit_id, record, payload_bytes)
        stored_record = (
            record if unit_id == record.causal_unit_id else replace(record, causal_unit_id=unit_id)
        )
        segment_id = allocator.assign(stored_record, payload_bytes)
        writer.add_record(
            (
                stored_record.record_id,
                str(stored_record.record_kind),
                stored_record.camera_id,
                stored_record.worker_boot_id,
                stored_record.source_generation,
                stored_record.stream_epoch,
                stored_record.producer,
                stored_record.producer_sequence,
                stored_record.frame_seq,
                stored_record.source_pts_ns,
                stored_record.observed_at_ns,
                stored_record.time_quality,
                stored_record.causal_unit_id,
                stored_record.parent_record_id,
                segment_id,
                provenance_id,
                stored_record.outcome,
                stored_record.reason,
                payload_text,
                payload_bytes,
                now_ns,
            )
        )
        stored[record.record_id] = _content_key(stored_record, payload_text)
        accepted += 1
        written_bytes += payload_bytes
    writer.flush(allocator)
    return IngestResult(
        accepted=accepted,
        written_bytes=written_bytes,
        duplicates=duplicates,
        rejected=tuple(rejected),
    )


__all__ = [
    "IngestResult",
    "ingest_records",
    "payload_text_and_bytes",
    "upsert_provenance",
]
