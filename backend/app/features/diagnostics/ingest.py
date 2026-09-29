"""Ingest helpers: provenance, units, and per-record insert."""

from __future__ import annotations

from dataclasses import replace

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
from backend.app.features.diagnostics.segments import assign_segment


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


def upsert_unit(
    connection: psycopg.Connection,
    record: ExecutionRecordInput,
    payload_bytes: int,
    batch_id: str,
    now_ns: int,
) -> str:
    unit_id = record.causal_unit_id
    existing = connection.execute(
        "SELECT 1 FROM execution_units WHERE causal_unit_id = %s", (unit_id,)
    ).fetchone()
    if existing is None and record.record_kind is RecordKind.BACKEND_ACCEPTANCE:
        unit_id = _late_ack_unit(connection, record, batch_id, now_ns)
        existing = connection.execute(
            "SELECT 1 FROM execution_units WHERE causal_unit_id = %s", (unit_id,)
        ).fetchone()
    if existing is None:
        connection.execute(
            """
            INSERT INTO execution_units (
                causal_unit_id, camera_id, worker_boot_id, source_generation,
                stream_epoch, causal_state, terminal, first_observed_ns,
                last_observed_ns, record_count, payload_bytes
            ) VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s, 1, %s)
            """,
            (
                unit_id,
                record.camera_id,
                record.worker_boot_id,
                record.source_generation,
                record.stream_epoch,
                str(UnitCausalState.INCOMPLETE_UNKNOWN),
                record.observed_at_ns,
                record.observed_at_ns,
                payload_bytes,
            ),
        )
        return unit_id
    connection.execute(
        """
        UPDATE execution_units
        SET first_observed_ns = LEAST(first_observed_ns, %s),
            last_observed_ns = GREATEST(last_observed_ns, %s),
            record_count = record_count + 1,
            payload_bytes = payload_bytes + %s
        WHERE causal_unit_id = %s
        """,
        (record.observed_at_ns, record.observed_at_ns, payload_bytes, unit_id),
    )
    return unit_id


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


def insert_record(
    connection: psycopg.Connection,
    record: ExecutionRecordInput,
    payload_text: str,
    payload_bytes: int,
    provenance_id: str,
    budget: RetentionBudget,
    batch_id: str,
    now_ns: int,
) -> str | None:
    existing = connection.execute(
        """
        SELECT record_kind, camera_id, worker_boot_id, source_generation, stream_epoch,
               producer, producer_sequence, frame_seq, source_pts_ns, observed_at_ns,
               time_quality, causal_unit_id, parent_record_id, outcome, reason, payload
        FROM execution_records WHERE record_id = %s
        """,
        (record.record_id,),
    ).fetchone()
    if existing is not None:
        stored = (
            str(existing[0]),
            str(existing[1]),
            str(existing[2]),
            int(existing[3]),
            int(existing[4]),
            str(existing[5]),
            int(existing[6]),
            None if existing[7] is None else int(existing[7]),
            None if existing[8] is None else int(existing[8]),
            int(existing[9]),
            str(existing[10]),
            str(existing[11]),
            None if existing[12] is None else str(existing[12]),
            str(existing[13]),
            None if existing[14] is None else str(existing[14]),
            str(existing[15]),
        )
        if stored == _content_key(record, payload_text):
            return "duplicate"
        return "conflict"
    unit_id = upsert_unit(connection, record, payload_bytes, batch_id, now_ns)
    stored_record = (
        record if unit_id == record.causal_unit_id else replace(record, causal_unit_id=unit_id)
    )
    segment_id = assign_segment(connection, stored_record, payload_bytes, budget, now_ns)
    connection.execute(
        """
        INSERT INTO execution_records (
            record_id, record_kind, camera_id, worker_boot_id, source_generation,
            stream_epoch, producer, producer_sequence, frame_seq, source_pts_ns,
            observed_at_ns, time_quality, causal_unit_id, parent_record_id, segment_id,
            provenance_id, outcome, reason, payload, payload_bytes, committed_at_ns
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """,
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
        ),
    )
    return None


__all__ = [
    "insert_record",
    "payload_text_and_bytes",
    "upsert_provenance",
    "upsert_unit",
]
