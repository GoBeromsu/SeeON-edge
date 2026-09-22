"""CREATE statements for the schema-19 execution-record (diagnostics) tables.

Schema 19 is schema 18 plus these six STRICT tables. They hold the Backend-owned
original-run diagnostics: one immutable record per producer observation, the
physical write segments and logical causal-retention units that bound how those
records are kept and removed together, the execution provenance every record
points at, the bounded coverage summary (known gaps, capacity deletions,
coarsened ranges, late acknowledgements), and the idempotent batch receipts the
worker relay retries against.

Only ``backend.app.edge_db`` may create these tables; the runtime authorizer
denies DDL. The API writer owns every row.
"""

from __future__ import annotations

from typing import Final

EXECUTION_RECORD_TABLES: Final = frozenset(
    {
        "execution_provenance",
        "execution_segments",
        "execution_units",
        "execution_records",
        "execution_coverage",
        "execution_batches",
    }
)

EXECUTION_RECORD_KINDS: Final = (
    "sdk.frame",
    "cpu.projection",
    "handoff.slot",
    "policy.consume",
    "model.score",
    "policy.decision",
    "event.delivery",
    "backend.acceptance",
    "coverage.gap",
)

SEGMENT_STORAGE_STATES: Final = ("OPEN", "SEALED_PENDING", "SEALED_FINAL", "PRUNED_SUMMARY")
UNIT_CAUSAL_STATES: Final = ("COMPLETE", "INCOMPLETE_KNOWN", "INCOMPLETE_UNKNOWN")
COVERAGE_KINDS: Final = (
    "MISSING_NOT_RECORDED",
    "DELETED_BY_CAPACITY",
    "UNKNOWN_COARSENED",
    "UNKNOWN",
    "ACK_OBSERVED_PARENT_DELETED",
    "ACK_OBSERVED_PARENT_UNKNOWN_COARSENED",
    "REJECTED_OVERSIZE",
    "STORAGE_UNAVAILABLE",
)

_SHA256_CHECK = "length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*'"
_IDENTITY_CHECK = "length({column}) BETWEEN 1 AND 128 AND instr({column}, char(0)) = 0"


def _in_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


EXECUTION_RECORD_CREATE_STATEMENTS: Final = (
    f"""
    CREATE TABLE execution_provenance (
        provenance_id TEXT PRIMARY KEY CHECK ({_SHA256_CHECK.format(column="provenance_id")}),
        worker_build_revision TEXT NOT NULL
            CHECK ({_IDENTITY_CHECK.format(column="worker_build_revision")}),
        worker_image_digest TEXT NOT NULL
            CHECK ({_IDENTITY_CHECK.format(column="worker_image_digest")}),
        model_digest TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="model_digest")}),
        calibration_digest TEXT NOT NULL
            CHECK ({_IDENTITY_CHECK.format(column="calibration_digest")}),
        preprocessing_identity TEXT NOT NULL
            CHECK ({_IDENTITY_CHECK.format(column="preprocessing_identity")}),
        config_digest TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="config_digest")}),
        policy_identity TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="policy_identity")}),
        backend_build_revision TEXT NOT NULL
            CHECK ({_IDENTITY_CHECK.format(column="backend_build_revision")}),
        first_seen_ns INTEGER NOT NULL CHECK (first_seen_ns >= 0)
    ) STRICT
    """,
    f"""
    CREATE TABLE execution_segments (
        segment_id INTEGER PRIMARY KEY,
        camera_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="camera_id")}),
        worker_boot_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="worker_boot_id")}),
        source_generation INTEGER NOT NULL CHECK (source_generation >= 0),
        stream_epoch INTEGER NOT NULL CHECK (stream_epoch >= 0),
        segment_ordinal INTEGER NOT NULL CHECK (segment_ordinal >= 0),
        storage_state TEXT NOT NULL
            CHECK (storage_state IN ({_in_list(SEGMENT_STORAGE_STATES)})),
        opened_at_ns INTEGER NOT NULL CHECK (opened_at_ns >= 0),
        sealed_at_ns INTEGER CHECK (sealed_at_ns IS NULL OR sealed_at_ns >= opened_at_ns),
        record_count INTEGER NOT NULL DEFAULT 0 CHECK (record_count >= 0),
        payload_bytes INTEGER NOT NULL DEFAULT 0 CHECK (payload_bytes >= 0),
        UNIQUE (camera_id, worker_boot_id, source_generation, stream_epoch, segment_ordinal)
    ) STRICT
    """,
    f"""
    CREATE TABLE execution_units (
        causal_unit_id TEXT PRIMARY KEY
            CHECK ({_IDENTITY_CHECK.format(column="causal_unit_id")}),
        camera_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="camera_id")}),
        worker_boot_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="worker_boot_id")}),
        source_generation INTEGER NOT NULL CHECK (source_generation >= 0),
        stream_epoch INTEGER NOT NULL CHECK (stream_epoch >= 0),
        causal_state TEXT NOT NULL CHECK (causal_state IN ({_in_list(UNIT_CAUSAL_STATES)})),
        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1)),
        first_observed_ns INTEGER NOT NULL CHECK (first_observed_ns >= 0),
        last_observed_ns INTEGER NOT NULL CHECK (last_observed_ns >= first_observed_ns),
        record_count INTEGER NOT NULL DEFAULT 0 CHECK (record_count >= 0),
        payload_bytes INTEGER NOT NULL DEFAULT 0 CHECK (payload_bytes >= 0)
    ) STRICT
    """,
    f"""
    CREATE TABLE execution_records (
        record_id TEXT PRIMARY KEY CHECK ({_SHA256_CHECK.format(column="record_id")}),
        record_kind TEXT NOT NULL CHECK (record_kind IN ({_in_list(EXECUTION_RECORD_KINDS)})),
        camera_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="camera_id")}),
        worker_boot_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="worker_boot_id")}),
        source_generation INTEGER NOT NULL CHECK (source_generation >= 0),
        stream_epoch INTEGER NOT NULL CHECK (stream_epoch >= 0),
        producer TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="producer")}),
        producer_sequence INTEGER NOT NULL CHECK (producer_sequence >= 0),
        frame_seq INTEGER CHECK (frame_seq IS NULL OR frame_seq >= 0),
        source_pts_ns INTEGER,
        observed_at_ns INTEGER NOT NULL CHECK (observed_at_ns >= 0),
        time_quality TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="time_quality")}),
        causal_unit_id TEXT NOT NULL REFERENCES execution_units (causal_unit_id)
            ON DELETE CASCADE,
        parent_record_id TEXT CHECK (
            parent_record_id IS NULL OR ({_SHA256_CHECK.format(column="parent_record_id")})
        ),
        segment_id INTEGER NOT NULL REFERENCES execution_segments (segment_id),
        provenance_id TEXT NOT NULL REFERENCES execution_provenance (provenance_id),
        outcome TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="outcome")}),
        reason TEXT CHECK (reason IS NULL OR ({_IDENTITY_CHECK.format(column="reason")})),
        payload TEXT NOT NULL CHECK (json_valid(payload)),
        payload_bytes INTEGER NOT NULL CHECK (payload_bytes = length(CAST(payload AS BLOB))),
        committed_at_ns INTEGER NOT NULL CHECK (committed_at_ns >= 0)
    ) STRICT
    """,
    """
    CREATE INDEX execution_records_camera_time
        ON execution_records (camera_id, observed_at_ns, producer_sequence)
    """,
    """
    CREATE INDEX execution_records_unit ON execution_records (causal_unit_id)
    """,
    """
    CREATE INDEX execution_records_segment ON execution_records (segment_id)
    """,
    """
    CREATE INDEX execution_records_producer_sequence
        ON execution_records (camera_id, worker_boot_id, stream_epoch, producer, producer_sequence)
    """,
    """
    CREATE INDEX execution_units_prune_order
        ON execution_units (terminal, last_observed_ns)
    """,
    f"""
    CREATE TABLE execution_coverage (
        coverage_id INTEGER PRIMARY KEY,
        camera_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="camera_id")}),
        worker_boot_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="worker_boot_id")}),
        source_generation INTEGER NOT NULL CHECK (source_generation >= 0),
        stream_epoch INTEGER NOT NULL CHECK (stream_epoch >= 0),
        coverage_kind TEXT NOT NULL CHECK (coverage_kind IN ({_in_list(COVERAGE_KINDS)})),
        producer TEXT CHECK (producer IS NULL OR ({_IDENTITY_CHECK.format(column="producer")})),
        from_sequence INTEGER CHECK (from_sequence IS NULL OR from_sequence >= 0),
        to_sequence INTEGER CHECK (
            to_sequence IS NULL OR (from_sequence IS NOT NULL AND to_sequence >= from_sequence)
        ),
        from_ns INTEGER NOT NULL CHECK (from_ns >= 0),
        to_ns INTEGER NOT NULL CHECK (to_ns >= from_ns),
        record_count INTEGER NOT NULL CHECK (record_count >= 0),
        exact INTEGER NOT NULL CHECK (exact IN (0, 1)),
        cause TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="cause")}),
        recorded_at_ns INTEGER NOT NULL CHECK (recorded_at_ns >= 0)
    ) STRICT
    """,
    """
    CREATE INDEX execution_coverage_camera_time
        ON execution_coverage (camera_id, from_ns, to_ns)
    """,
    f"""
    CREATE TABLE execution_batches (
        batch_id TEXT PRIMARY KEY CHECK ({_SHA256_CHECK.format(column="batch_id")}),
        camera_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="camera_id")}),
        worker_boot_id TEXT NOT NULL CHECK ({_IDENTITY_CHECK.format(column="worker_boot_id")}),
        received_at_ns INTEGER NOT NULL CHECK (received_at_ns >= 0),
        accepted_records INTEGER NOT NULL CHECK (accepted_records >= 0),
        duplicate_records INTEGER NOT NULL CHECK (duplicate_records >= 0),
        rejected_records INTEGER NOT NULL CHECK (rejected_records >= 0),
        receipt TEXT NOT NULL CHECK (json_valid(receipt))
    ) STRICT
    """,
)

__all__ = [
    "COVERAGE_KINDS",
    "EXECUTION_RECORD_CREATE_STATEMENTS",
    "EXECUTION_RECORD_KINDS",
    "EXECUTION_RECORD_TABLES",
    "SEGMENT_STORAGE_STATES",
    "UNIT_CAUSAL_STATES",
]
