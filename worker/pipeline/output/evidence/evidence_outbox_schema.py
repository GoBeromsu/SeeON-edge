"""Forward-only SQLite schema for the worker evidence outbox."""

from typing import Final

SCHEMA_VERSION: Final = 4

SCHEMA_V1_STATEMENTS: Final = (
    """
    CREATE TABLE evidence_events (
        edge_event_id TEXT PRIMARY KEY,
        detected_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('STAGED', 'READY', 'IN_FLIGHT', 'ACKED')),
        queued_at REAL NOT NULL,
        next_attempt_at REAL NOT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
        lease_owner TEXT,
        lease_expires_at REAL,
        CHECK (
            (state = 'IN_FLIGHT' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
            OR
            (state != 'IN_FLIGHT' AND lease_owner IS NULL AND lease_expires_at IS NULL)
        )
    ) STRICT
    """,
    """
    CREATE TABLE evidence_clips (
        clip_id TEXT PRIMARY KEY
    ) STRICT
    """,
    """
    CREATE TABLE clip_events (
        clip_id TEXT NOT NULL REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
        edge_event_id TEXT NOT NULL UNIQUE
            REFERENCES evidence_events(edge_event_id) ON DELETE RESTRICT,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        PRIMARY KEY (clip_id, ordinal)
    ) STRICT
    """,
    """
    CREATE INDEX evidence_events_claim_idx
    ON evidence_events (state, next_attempt_at, lease_expires_at, queued_at, edge_event_id)
    """,
)

SCHEMA_V2_STATEMENTS: Final = (
    """
    ALTER TABLE evidence_clips ADD COLUMN local_state TEXT NOT NULL
        DEFAULT 'AWAITING_FINALIZE'
        CHECK (local_state IN ('AWAITING_FINALIZE', 'VERIFIED', 'UNAVAILABLE', 'CORRUPT'))
    """,
    "ALTER TABLE evidence_clips ADD COLUMN manifest_path TEXT",
    """
    ALTER TABLE evidence_clips ADD COLUMN state_version INTEGER NOT NULL DEFAULT 1
        CHECK (state_version >= 1)
    """,
    "ALTER TABLE evidence_clips ADD COLUMN media_relpath TEXT",
    "ALTER TABLE evidence_clips ADD COLUMN sha256 TEXT",
    """
    ALTER TABLE evidence_clips ADD COLUMN size_bytes INTEGER
        CHECK (size_bytes IS NULL OR size_bytes > 0)
    """,
    "ALTER TABLE evidence_clips ADD COLUMN mime_type TEXT",
    "ALTER TABLE evidence_clips ADD COLUMN codec TEXT",
    """
    ALTER TABLE evidence_clips ADD COLUMN duration_ms INTEGER
        CHECK (duration_ms IS NULL OR duration_ms BETWEEN 1 AND 120000)
    """,
    "ALTER TABLE evidence_clips ADD COLUMN clip_start_at TEXT",
    "ALTER TABLE evidence_clips ADD COLUMN clip_end_at TEXT",
    "ALTER TABLE evidence_clips ADD COLUMN finalized_at TEXT",
    """
    ALTER TABLE evidence_clips ADD COLUMN unavailable_reason TEXT
        CHECK (
            unavailable_reason IS NULL OR unavailable_reason IN (
                'ENCODER_FAILED', 'NO_FRAMES', 'FINALIZE_FAILED',
                'INTERRUPTED_FINALIZE', 'MISSING', 'CORRUPT'
            )
        )
    """,
    """
    CREATE INDEX evidence_clips_local_state_idx
    ON evidence_clips (local_state, clip_id)
    """,
)

SCHEMA_V3_STATEMENTS: Final = (
    """
    ALTER TABLE evidence_events ADD COLUMN delivery_state TEXT NOT NULL
        DEFAULT 'PENDING'
        CHECK (delivery_state IN ('PENDING', 'ACKED', 'PERMANENT', 'COMPATIBILITY'))
    """,
    "ALTER TABLE evidence_events ADD COLUMN backend_event_id TEXT",
    "ALTER TABLE evidence_events ADD COLUMN last_error_code TEXT",
    "UPDATE evidence_events SET delivery_state = 'ACKED' WHERE state = 'ACKED'",
    """
    ALTER TABLE evidence_clips ADD COLUMN publish_state TEXT NOT NULL
        DEFAULT 'WAITING'
        CHECK (publish_state IN ('WAITING', 'IN_FLIGHT', 'PUBLISHED', 'PERMANENT', 'COMPATIBILITY'))
    """,
    """
    ALTER TABLE evidence_clips ADD COLUMN publish_attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (publish_attempt_count >= 0)
    """,
    """
    ALTER TABLE evidence_clips ADD COLUMN publish_next_attempt_at REAL NOT NULL DEFAULT 0
    """,
    "ALTER TABLE evidence_clips ADD COLUMN publish_lease_owner TEXT",
    "ALTER TABLE evidence_clips ADD COLUMN publish_lease_expires_at REAL",
    "ALTER TABLE evidence_clips ADD COLUMN remote_state TEXT",
    "ALTER TABLE evidence_clips ADD COLUMN backend_ack_at REAL",
    "ALTER TABLE evidence_clips ADD COLUMN last_error_code TEXT",
    """
    CREATE INDEX evidence_events_delivery_idx
    ON evidence_events (delivery_state, state, next_attempt_at, lease_expires_at)
    """,
    """
    CREATE INDEX evidence_clips_publish_idx
    ON evidence_clips (
        publish_state, local_state, publish_next_attempt_at,
        publish_lease_expires_at, clip_id
    )
    """,
)

SCHEMA_V4_STATEMENTS: Final = ()
"""No-op version marker: the outbox DB was renamed from evidence-outbox.sqlite3
to worker-state.sqlite3 (constant-name change only, no on-disk migration)."""

MIGRATIONS: Final = (
    SCHEMA_V1_STATEMENTS,
    SCHEMA_V2_STATEMENTS,
    SCHEMA_V3_STATEMENTS,
    SCHEMA_V4_STATEMENTS,
)

__all__ = ["MIGRATIONS", "SCHEMA_VERSION"]
