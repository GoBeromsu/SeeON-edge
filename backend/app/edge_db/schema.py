"""Forward-only DDL ledger owned exclusively by the edge DB migrator."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from backend.app.edge_db.application_schema import APPLICATION_SCHEMA_STATEMENTS
from backend.app.edge_db.evidence_backfill import EVIDENCE_BACKFILL_STATEMENTS
from backend.app.edge_db.review_migration import LEGACY_LABEL_MIGRATION_STATEMENTS


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]
    writable_tables: frozenset[str] = frozenset()
    preflight: Callable[[sqlite3.Connection], None] | None = None

    @property
    def checksum(self) -> str:
        payload = "\x1e".join(statement.strip() for statement in self.statements)
        return hashlib.sha256(payload.encode()).hexdigest()


class SchemaV17MigrationError(RuntimeError):
    """The direct schema-17 ownership migration cannot safely proceed."""


_DRAIN_INCOMPLETE_SENTINEL: Final = "EDGE_DB_DRAIN_INCOMPLETE"


def _require_schema17_drain(connection: sqlite3.Connection) -> None:
    """Refuse the irreversible ownership cutover until worker-owned work drains."""
    blocked = connection.execute(
        """
        SELECT EXISTS(SELECT 1 FROM evidence_events WHERE state IN ('STAGED', 'READY', 'IN_FLIGHT'))
            -- Unfinalized local evidence blocks: the clip has no complete media
            -- yet, which is the condition the 1053 stalled live clips are in.
            OR EXISTS(SELECT 1 FROM evidence_clips WHERE local_state = 'AWAITING_FINALIZE')
            -- A publish already in flight blocks: interrupting it mid-delivery
            -- would strand the upload. `publish_state = 'WAITING'` deliberately
            -- does NOT block -- it is the resting state of every clip that was
            -- finalized locally but never published upstream, so gating on it
            -- would make migration impossible on any real database forever.
            OR EXISTS(SELECT 1 FROM evidence_clips WHERE publish_state = 'IN_FLIGHT')
            -- RUNNING blocks alongside PENDING: stopping the legacy worker can
            -- leave a job mid-work, and the backend-only runtime has no actor to
            -- finish it, so migrating would strand it in a state nothing resolves.
            OR EXISTS(SELECT 1 FROM derivative_jobs WHERE state IN ('PENDING', 'RUNNING'))
            OR EXISTS(SELECT 1 FROM derivative_evidence_slots WHERE state = 'PENDING')
            -- A PENDING retention row means the legacy worker recorded an intent
            -- to purge media and never completed it. After the ownership change
            -- nothing completes it either, so it must be resolved explicitly
            -- rather than migrated over.
            OR EXISTS(SELECT 1 FROM evidence_retention_states WHERE state = 'PENDING')
        """
    ).fetchone()
    if blocked == (1,):
        raise SchemaV17MigrationError(_DRAIN_INCOMPLETE_SENTINEL)


# Historical v1 statements are immutable checksum identities. Schema 17 performs
# the ownership reassignment for every deployed v1 database.
_FAMILY_INSERTS = (
    "INSERT INTO schema_table_families (prefix, writer, purpose) VALUES "
    "('control_', 'api', 'operator and deployment control state')",
    "INSERT INTO schema_table_families (prefix, writer, purpose) VALUES "
    "('qa_', 'api', 'internal replay and QA state')",
    "INSERT INTO schema_table_families (prefix, writer, purpose) VALUES "
    "('runtime_', 'worker', 'applied worker runtime state')",
    "INSERT INTO schema_table_families (prefix, writer, purpose) VALUES "
    "('evidence_', 'worker', 'event and evidence state')",
    "INSERT INTO schema_table_families (prefix, writer, purpose) VALUES "
    "('derivative_', 'worker', 'derived media state')",
    "INSERT INTO schema_table_families (prefix, writer, purpose) VALUES "
    "('schema_', 'migrator', 'schema ledger and ownership metadata')",
)

SCHEMA_V1 = Migration(
    version=1,
    name="edge_database_foundation",
    statements=(
        """
        CREATE TABLE schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) STRICT
        """,
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY CHECK (version > 0),
            name TEXT NOT NULL UNIQUE,
            checksum TEXT NOT NULL CHECK (length(checksum) = 64),
            applied_at TEXT NOT NULL
        ) STRICT
        """,
        """
        CREATE TABLE schema_table_families (
            prefix TEXT PRIMARY KEY CHECK (prefix GLOB '*_'),
            writer TEXT NOT NULL CHECK (writer IN ('api', 'worker', 'migrator')),
            purpose TEXT NOT NULL
        ) STRICT
        """,
        "INSERT INTO schema_metadata (key, value) VALUES ('format', 'seeon-edge-v1')",
        *_FAMILY_INSERTS,
    ),
)

SCHEMA_V2 = Migration(
    version=2,
    name="single_edge_application_schema",
    statements=APPLICATION_SCHEMA_STATEMENTS,
)

SCHEMA_V3 = Migration(
    version=3,
    name="initialize_clip_listing_generation",
    statements=(
        "INSERT INTO clip_listing_generation "
        "(id, active_generation, next_generation) VALUES (1, 0, 1)",
    ),
    writable_tables=frozenset({"clip_listing_generation"}),
)

SCHEMA_V4 = Migration(
    version=4,
    name="versioned_numeric_detection_policies",
    statements=(
        """
        CREATE TABLE control_detection_policy_revisions (
            revision_id INTEGER PRIMARY KEY,
            facility_id TEXT NOT NULL CHECK (length(facility_id) > 0),
            camera_id TEXT CHECK (camera_id IS NULL OR length(camera_id) > 0),
            module_id TEXT NOT NULL CHECK (length(module_id) > 0),
            module_version INTEGER NOT NULL CHECK (module_version > 0),
            schema_id TEXT NOT NULL CHECK (length(schema_id) > 0),
            schema_version INTEGER NOT NULL CHECK (schema_version > 0),
            values_json TEXT NOT NULL CHECK (json_valid(values_json)),
            content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
            created_at TEXT NOT NULL
        ) STRICT
        """,
        """
        CREATE TRIGGER control_detection_policy_revisions_immutable_update
        BEFORE UPDATE ON control_detection_policy_revisions
        BEGIN
            SELECT RAISE(ABORT, 'detection policy revisions are immutable');
        END
        """,
        """
        CREATE TRIGGER control_detection_policy_revisions_immutable_delete
        BEFORE DELETE ON control_detection_policy_revisions
        BEGIN
            SELECT RAISE(ABORT, 'detection policy revisions are immutable');
        END
        """,
        """
        CREATE TABLE control_detection_policy_activations (
            activation_id INTEGER PRIMARY KEY,
            facility_id TEXT NOT NULL CHECK (length(facility_id) > 0),
            camera_id TEXT CHECK (camera_id IS NULL OR length(camera_id) > 0),
            module_id TEXT NOT NULL CHECK (length(module_id) > 0),
            module_version INTEGER NOT NULL CHECK (module_version > 0),
            active_revision_id INTEGER REFERENCES control_detection_policy_revisions(revision_id),
            previous_revision_id INTEGER REFERENCES control_detection_policy_revisions(revision_id),
            activation_generation INTEGER NOT NULL CHECK (activation_generation > 0),
            status TEXT NOT NULL CHECK (status IN ('pending', 'applied', 'failed')),
            refusal_reason TEXT,
            activated_at TEXT NOT NULL,
            applied_at TEXT
        ) STRICT
        """,
        """
        CREATE UNIQUE INDEX control_policy_facility_activation_idx
        ON control_detection_policy_activations(facility_id, module_id, module_version)
        WHERE camera_id IS NULL
        """,
        """
        CREATE UNIQUE INDEX control_policy_camera_activation_idx
        ON control_detection_policy_activations(
            facility_id, camera_id, module_id, module_version
        ) WHERE camera_id IS NOT NULL
        """,
        """
        CREATE INDEX control_policy_revision_scope_idx
        ON control_detection_policy_revisions(
            facility_id, camera_id, module_id, module_version, revision_id
        )
        """,
        """
        CREATE TABLE control_detection_policy_state (
            facility_id TEXT PRIMARY KEY CHECK (length(facility_id) > 0),
            activation_generation INTEGER NOT NULL CHECK (activation_generation >= 0)
        ) STRICT
        """,
    ),
)

SCHEMA_V5 = Migration(
    version=5,
    name="applied_runtime_provenance_manifests",
    statements=(
        """
        CREATE TABLE runtime_manifest_contents (
            manifest_sha256 TEXT PRIMARY KEY CHECK (length(manifest_sha256) = 64),
            manifest_schema_version INTEGER NOT NULL CHECK (manifest_schema_version > 0),
            canonical_json TEXT NOT NULL CHECK (json_valid(canonical_json)),
            created_at TEXT NOT NULL
        ) STRICT
        """,
        """
        CREATE TRIGGER runtime_manifest_contents_immutable_update
        BEFORE UPDATE ON runtime_manifest_contents
        BEGIN
            SELECT RAISE(ABORT, 'runtime manifest contents are immutable');
        END
        """,
        """
        CREATE TRIGGER runtime_manifest_contents_immutable_delete
        BEFORE DELETE ON runtime_manifest_contents
        BEGIN
            SELECT RAISE(ABORT, 'runtime manifest contents are immutable');
        END
        """,
        """
        CREATE TABLE runtime_manifest_boots (
            boot_instance_id TEXT PRIMARY KEY CHECK (length(boot_instance_id) > 0),
            manifest_sha256 TEXT NOT NULL REFERENCES runtime_manifest_contents(manifest_sha256),
            applied_at TEXT NOT NULL
        ) STRICT
        """,
        """
        CREATE TABLE runtime_manifest_cameras (
            boot_instance_id TEXT NOT NULL REFERENCES runtime_manifest_boots(boot_instance_id),
            camera_id TEXT NOT NULL CHECK (length(camera_id) > 0),
            manifest_sha256 TEXT NOT NULL REFERENCES runtime_manifest_contents(manifest_sha256),
            applied_at TEXT NOT NULL,
            PRIMARY KEY (boot_instance_id, camera_id)
        ) STRICT
        """,
        """
        CREATE INDEX runtime_manifest_cameras_lookup_idx
        ON runtime_manifest_cameras(camera_id, applied_at DESC, boot_instance_id DESC)
        """,
    ),
)

SCHEMA_V6 = Migration(
    version=6,
    name="bounded_analysis_decision_traces",
    statements=(
        """
        CREATE TABLE runtime_analysis_traces (
            trace_id TEXT PRIMARY KEY CHECK (length(trace_id) = 64),
            trace_schema_version INTEGER NOT NULL CHECK (trace_schema_version = 1),
            worker_boot_id TEXT NOT NULL,
            camera_id TEXT NOT NULL CHECK (length(camera_id) > 0),
            stream_epoch INTEGER NOT NULL CHECK (stream_epoch >= 0),
            frame_seq INTEGER NOT NULL CHECK (frame_seq >= 0),
            pts REAL,
            pts_missing_reason TEXT CHECK (
                (pts IS NULL AND pts_missing_reason IS NOT NULL) OR
                (pts IS NOT NULL AND pts_missing_reason IS NULL)
            ),
            source_time_sec REAL,
            source_time_missing_reason TEXT CHECK (
                (source_time_sec IS NULL AND source_time_missing_reason IS NOT NULL) OR
                (source_time_sec IS NOT NULL AND source_time_missing_reason IS NULL)
            ),
            frame_width INTEGER NOT NULL CHECK (frame_width > 0),
            frame_height INTEGER NOT NULL CHECK (frame_height > 0),
            bed_region_provenance TEXT NOT NULL CHECK (
                bed_region_provenance IN ('fresh','cached','empty','expired','unknown')
            ),
            UNIQUE(worker_boot_id, camera_id, stream_epoch, frame_seq)
        ) STRICT
        """,
        """
        CREATE TABLE runtime_analysis_components (
            analysis_trace_id TEXT NOT NULL REFERENCES runtime_analysis_traces(trace_id)
                ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            component_qualified_id TEXT NOT NULL CHECK (length(component_qualified_id) > 0),
            observation_state TEXT NOT NULL CHECK (
                observation_state IN ('observed','not-scheduled','missing')
            ),
            PRIMARY KEY (analysis_trace_id, ordinal)
        ) STRICT
        """,
        """
        CREATE TABLE runtime_analysis_persons (
            analysis_trace_id TEXT NOT NULL REFERENCES runtime_analysis_traces(trace_id)
                ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            track_id INTEGER,
            track_missing_reason TEXT CHECK (
                (track_id IS NULL AND track_missing_reason IS NOT NULL) OR
                (track_id IS NOT NULL AND track_missing_reason IS NULL)
            ),
            x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL, y2 INTEGER NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            PRIMARY KEY (analysis_trace_id, ordinal)
        ) STRICT
        """,
        """
        CREATE TABLE runtime_analysis_beds (
            analysis_trace_id TEXT NOT NULL REFERENCES runtime_analysis_traces(trace_id)
                ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL, y2 INTEGER NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            provenance TEXT NOT NULL CHECK (
                provenance IN ('fresh','cached','empty','expired','unknown')
            ),
            PRIMARY KEY (analysis_trace_id, ordinal)
        ) STRICT
        """,
        """
        CREATE TABLE evidence_decision_traces (
            trace_id TEXT PRIMARY KEY CHECK (length(trace_id) = 64),
            trace_schema_version INTEGER NOT NULL CHECK (trace_schema_version = 1),
            analysis_trace_id TEXT REFERENCES runtime_analysis_traces(trace_id)
                ON DELETE SET NULL,
            module_qualified_id TEXT NOT NULL CHECK (length(module_qualified_id) > 0),
            policy_qualified_id TEXT NOT NULL CHECK (length(policy_qualified_id) > 0),
            effective_policy_id TEXT NOT NULL CHECK (length(effective_policy_id) = 64),
            runtime_manifest_sha256 TEXT NOT NULL
                REFERENCES runtime_manifest_contents(manifest_sha256),
            reason TEXT NOT NULL CHECK (length(reason) > 0),
            previous_state TEXT NOT NULL CHECK (length(previous_state) > 0),
            current_state TEXT NOT NULL CHECK (length(current_state) > 0),
            triggered INTEGER NOT NULL CHECK (triggered IN (0, 1)),
            track_id INTEGER,
            track_missing_reason TEXT CHECK (
                (track_id IS NULL AND track_missing_reason IS NOT NULL) OR
                (track_id IS NOT NULL AND track_missing_reason IS NULL)
            ),
            bed_id INTEGER,
            bed_missing_reason TEXT CHECK (
                (bed_id IS NULL AND bed_missing_reason IS NOT NULL) OR
                (bed_id IS NOT NULL AND bed_missing_reason IS NULL)
            )
        ) STRICT
        """,
        """
        CREATE TABLE evidence_decision_values (
            decision_trace_id TEXT NOT NULL REFERENCES evidence_decision_traces(trace_id)
                ON DELETE CASCADE,
            name TEXT NOT NULL CHECK (length(name) > 0),
            numeric_value REAL,
            missing_reason TEXT CHECK (
                (numeric_value IS NULL AND missing_reason IS NOT NULL) OR
                (numeric_value IS NOT NULL AND missing_reason IS NULL)
            ),
            PRIMARY KEY (decision_trace_id, name)
        ) STRICT
        """,
        """
        CREATE TABLE runtime_trace_cursors (
            camera_id TEXT PRIMARY KEY CHECK (length(camera_id) > 0),
            handoff_dropped_frames INTEGER NOT NULL DEFAULT 0 CHECK (handoff_dropped_frames >= 0),
            pruned_frames INTEGER NOT NULL DEFAULT 0 CHECK (pruned_frames >= 0),
            oldest_retained_seq INTEGER,
            newest_retained_seq INTEGER,
            updated_at_source_sec REAL
        ) STRICT
        """,
        """
        CREATE TABLE evidence_event_trace_refs (
            edge_event_id TEXT PRIMARY KEY REFERENCES evidence_events(edge_event_id)
                ON DELETE RESTRICT,
            decision_trace_id TEXT NOT NULL UNIQUE
                REFERENCES evidence_decision_traces(trace_id) ON DELETE RESTRICT
        ) STRICT
        """,
        "CREATE INDEX runtime_analysis_camera_ring_idx ON runtime_analysis_traces("
        "camera_id, source_time_sec, stream_epoch, frame_seq)",
        "CREATE INDEX evidence_decision_analysis_idx ON evidence_decision_traces("
        "analysis_trace_id, module_qualified_id, trace_id)",
    ),
)

SCHEMA_V7 = Migration(
    version=7,
    name="trace_persistence_integrity_and_bounds",
    statements=(
        "ALTER TABLE runtime_analysis_traces ADD COLUMN storage_bytes INTEGER "
        "NOT NULL DEFAULT 0 CHECK (storage_bytes >= 0)",
        "ALTER TABLE runtime_trace_cursors ADD COLUMN persistence_failed_frames "
        "INTEGER NOT NULL DEFAULT 0 CHECK (persistence_failed_frames >= 0)",
        "ALTER TABLE runtime_trace_cursors ADD COLUMN retention_blocked_frames "
        "INTEGER NOT NULL DEFAULT 0 CHECK (retention_blocked_frames >= 0)",
        "ALTER TABLE runtime_trace_cursors ADD COLUMN oldest_retained_boot_id TEXT",
        "ALTER TABLE runtime_trace_cursors ADD COLUMN oldest_retained_stream_epoch INTEGER",
        "ALTER TABLE runtime_trace_cursors ADD COLUMN oldest_retained_trace_id TEXT",
        "ALTER TABLE runtime_trace_cursors ADD COLUMN newest_retained_boot_id TEXT",
        "ALTER TABLE runtime_trace_cursors ADD COLUMN newest_retained_stream_epoch INTEGER",
        "ALTER TABLE runtime_trace_cursors ADD COLUMN newest_retained_trace_id TEXT",
        """
        CREATE TABLE evidence_clip_trace_refs (
            clip_id TEXT NOT NULL REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
            edge_event_id TEXT NOT NULL UNIQUE REFERENCES evidence_events(edge_event_id)
                ON DELETE RESTRICT,
            decision_trace_id TEXT NOT NULL
                REFERENCES evidence_decision_traces(trace_id) ON DELETE RESTRICT,
            PRIMARY KEY (clip_id, decision_trace_id)
        ) STRICT
        """,
        """
        INSERT INTO evidence_clip_trace_refs (clip_id, edge_event_id, decision_trace_id)
        SELECT relation.clip_id, relation.edge_event_id, event_ref.decision_trace_id
        FROM clip_events AS relation
        JOIN evidence_event_trace_refs AS event_ref USING (edge_event_id)
        """,
        """
        CREATE TABLE runtime_provenance_retention (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            pruned_boots INTEGER NOT NULL DEFAULT 0 CHECK (pruned_boots >= 0),
            pruned_camera_bindings INTEGER NOT NULL DEFAULT 0
                CHECK (pruned_camera_bindings >= 0),
            retention_blocked_boots INTEGER NOT NULL DEFAULT 0
                CHECK (retention_blocked_boots >= 0),
            retention_blocked_camera_bindings INTEGER NOT NULL DEFAULT 0
                CHECK (retention_blocked_camera_bindings >= 0)
        ) STRICT
        """,
        "INSERT INTO runtime_provenance_retention (id) VALUES (1)",
        """
        CREATE TRIGGER evidence_clip_trace_match_insert
        BEFORE INSERT ON evidence_clip_trace_refs
        WHEN NOT EXISTS (
            SELECT 1 FROM clip_events AS relation
            JOIN evidence_event_trace_refs AS event_ref USING (edge_event_id)
            WHERE relation.clip_id = NEW.clip_id
              AND relation.edge_event_id = NEW.edge_event_id
              AND event_ref.decision_trace_id = NEW.decision_trace_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'clip trace reference does not match its event relation');
        END
        """,
        """
        CREATE TRIGGER runtime_analysis_event_evidence_restrict_delete
        BEFORE DELETE ON runtime_analysis_traces
        WHEN EXISTS (
            SELECT 1 FROM evidence_decision_traces AS decision
            JOIN evidence_event_trace_refs AS event_ref
              ON event_ref.decision_trace_id = decision.trace_id
            WHERE decision.analysis_trace_id = OLD.trace_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'event-linked analysis traces are immutable evidence');
        END
        """,
        """
        CREATE TRIGGER evidence_decision_manifest_camera_match_insert
        BEFORE INSERT ON evidence_decision_traces
        WHEN NOT EXISTS (
            SELECT 1 FROM runtime_analysis_traces AS analysis
            JOIN runtime_manifest_cameras AS camera
              ON camera.boot_instance_id = analysis.worker_boot_id
             AND camera.camera_id = analysis.camera_id
             AND camera.manifest_sha256 = NEW.runtime_manifest_sha256
            WHERE analysis.trace_id = NEW.analysis_trace_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'trace camera is absent from the runtime manifest boot');
        END
        """,
        """
        CREATE TRIGGER runtime_manifest_camera_trace_restrict_delete
        BEFORE DELETE ON runtime_manifest_cameras
        WHEN EXISTS (
            SELECT 1 FROM runtime_analysis_traces AS analysis
            WHERE analysis.worker_boot_id = OLD.boot_instance_id
              AND analysis.camera_id = OLD.camera_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'runtime manifest camera is referenced by analysis traces');
        END
        """,
        """
        CREATE TRIGGER runtime_manifest_boot_trace_restrict_delete
        BEFORE DELETE ON runtime_manifest_boots
        WHEN EXISTS (
            SELECT 1 FROM runtime_analysis_traces AS analysis
            WHERE analysis.worker_boot_id = OLD.boot_instance_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'runtime manifest boot is referenced by analysis traces');
        END
        """,
        "CREATE INDEX runtime_analysis_total_retention_idx ON runtime_analysis_traces("
        "source_time_sec, worker_boot_id, camera_id, stream_epoch, frame_seq, trace_id)",
        "CREATE INDEX evidence_clip_trace_decision_idx ON evidence_clip_trace_refs("
        "decision_trace_id, clip_id)",
    ),
    writable_tables=frozenset({"evidence_clip_trace_refs", "runtime_provenance_retention"}),
)

SCHEMA_V8 = Migration(
    version=8,
    name="truthful_trace_component_states",
    statements=(
        "ALTER TABLE runtime_analysis_components RENAME TO runtime_analysis_components_v7",
        """
        CREATE TABLE runtime_analysis_components (
            analysis_trace_id TEXT NOT NULL REFERENCES runtime_analysis_traces(trace_id)
                ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            component_qualified_id TEXT NOT NULL CHECK (length(component_qualified_id) > 0),
            observation_state TEXT NOT NULL CHECK (
                observation_state IN (
                    'observed','executed','not-applicable','not-scheduled','missing'
                )
            ),
            PRIMARY KEY (analysis_trace_id, ordinal)
        ) STRICT
        """,
        """
        INSERT INTO runtime_analysis_components (
            analysis_trace_id, ordinal, component_qualified_id, observation_state
        )
        SELECT analysis_trace_id, ordinal, component_qualified_id, observation_state
        FROM runtime_analysis_components_v7
        """,
        "DROP TABLE runtime_analysis_components_v7",
    ),
    writable_tables=frozenset({"runtime_analysis_components", "runtime_analysis_components_v7"}),
)

SCHEMA_V9 = Migration(
    version=9,
    name="authoritative_central_evidence_records",
    statements=(
        """
        CREATE TABLE evidence_incidents (
            incident_id TEXT PRIMARY KEY CHECK (length(incident_id) > 0),
            record_schema_version INTEGER NOT NULL DEFAULT 1
                CHECK (record_schema_version = 1),
            edge_event_id TEXT NOT NULL UNIQUE REFERENCES evidence_events(edge_event_id)
                ON DELETE RESTRICT,
            camera_id TEXT NOT NULL CHECK (length(camera_id) > 0),
            event_type TEXT NOT NULL CHECK (length(event_type) > 0),
            detected_at TEXT NOT NULL CHECK (length(detected_at) > 0),
            runtime_manifest_sha256 TEXT REFERENCES runtime_manifest_contents(manifest_sha256),
            decision_trace_id TEXT UNIQUE REFERENCES evidence_decision_traces(trace_id),
            module_qualified_id TEXT,
            policy_qualified_id TEXT,
            effective_policy_id TEXT,
            provenance_state TEXT NOT NULL DEFAULT 'MISSING'
                CHECK (provenance_state IN ('QUALIFIED','MISSING')),
            provenance_missing_reason TEXT,
            primary_clip_id TEXT REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
            lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN (
                'STAGING','MEDIA_READY','PUBLISHED','DERIVATIVE_PENDING','COMPLETE','FAILED'
            )),
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
            failure_reason TEXT CHECK (failure_reason IS NULL OR failure_reason IN (
                'MISSING','UNAVAILABLE','CORRUPT','INTERRUPTED','PROVENANCE_MISSING',
                'PUBLICATION_FAILED','RETENTION_FAILED'
            )),
            created_at TEXT NOT NULL CHECK (length(created_at) > 0),
            updated_at TEXT NOT NULL CHECK (length(updated_at) > 0),
            CHECK (
                (provenance_state = 'QUALIFIED' AND provenance_missing_reason IS NULL
                 AND runtime_manifest_sha256 IS NOT NULL AND decision_trace_id IS NOT NULL
                 AND module_qualified_id IS NOT NULL AND policy_qualified_id IS NOT NULL
                 AND effective_policy_id IS NOT NULL)
                OR
                (provenance_state = 'MISSING' AND provenance_missing_reason IS NOT NULL)
            ),
            CHECK ((lifecycle_state = 'FAILED') = (failure_reason IS NOT NULL))
        ) STRICT
        """,
        """
        CREATE TABLE evidence_media_objects (
            media_id TEXT PRIMARY KEY CHECK (length(media_id) > 0),
            content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
            size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
            mime_type TEXT NOT NULL CHECK (length(mime_type) > 0),
            root_kind TEXT NOT NULL DEFAULT 'CLIP_STORE' CHECK (root_kind = 'CLIP_STORE'),
            contained_relpath TEXT NOT NULL UNIQUE CHECK (
                length(contained_relpath) > 0
                AND substr(contained_relpath, 1, 1) != '/'
                AND instr(contained_relpath, '\\') = 0
                AND instr('/' || contained_relpath || '/', '/../') = 0
            ),
            basename TEXT NOT NULL CHECK (
                length(basename) > 0 AND instr(basename, '/') = 0
                AND instr(basename, '\\') = 0 AND basename NOT IN ('.', '..')
            ),
            created_at TEXT NOT NULL,
            UNIQUE(content_sha256, size_bytes, mime_type, contained_relpath)
        ) STRICT
        """,
        """
        CREATE TABLE evidence_artifact_slots (
            incident_id TEXT NOT NULL REFERENCES evidence_incidents(incident_id)
                ON DELETE RESTRICT,
            slot_name TEXT NOT NULL CHECK (slot_name IN ('PRIMARY_CLIP','SNAPSHOT')),
            state TEXT NOT NULL CHECK (state IN ('PENDING','AVAILABLE','UNAVAILABLE','CORRUPT')),
            media_id TEXT REFERENCES evidence_media_objects(media_id) ON DELETE RESTRICT,
            reason TEXT,
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (incident_id, slot_name),
            CHECK (
                (state = 'PENDING' AND media_id IS NULL AND reason IS NULL)
                OR (state = 'AVAILABLE' AND media_id IS NOT NULL AND reason IS NULL)
                OR (state IN ('UNAVAILABLE','CORRUPT') AND reason IS NOT NULL)
            )
        ) STRICT
        """,
        """
        CREATE TABLE evidence_primary_clips (
            incident_id TEXT PRIMARY KEY REFERENCES evidence_incidents(incident_id)
                ON DELETE RESTRICT,
            clip_id TEXT NOT NULL REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
            manifest_relpath TEXT,
            manifest_sha256 TEXT CHECK (manifest_sha256 IS NULL OR length(manifest_sha256) = 64),
            manifest_size_bytes INTEGER CHECK (
                manifest_size_bytes IS NULL OR manifest_size_bytes > 0
            ),
            media_id TEXT REFERENCES evidence_media_objects(media_id) ON DELETE RESTRICT,
            codec TEXT,
            audio_codec TEXT,
            duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms BETWEEN 1 AND 120000),
            clip_start_at TEXT,
            clip_end_at TEXT,
            finalized_at TEXT,
            source_packet_preserved INTEGER NOT NULL CHECK (source_packet_preserved IN (0,1)),
            source_missing_reason TEXT,
            source_media_json TEXT CHECK (
                source_media_json IS NULL OR json_valid(source_media_json)
            ),
            time_origin_json TEXT CHECK (
                time_origin_json IS NULL OR json_valid(time_origin_json)
            ),
            truncation_json TEXT NOT NULL CHECK (json_valid(truncation_json)),
            unavailable_reason TEXT,
            created_at TEXT NOT NULL,
            CHECK (
                (media_id IS NOT NULL AND manifest_relpath IS NOT NULL
                 AND manifest_sha256 IS NOT NULL AND manifest_size_bytes IS NOT NULL
                 AND unavailable_reason IS NULL)
                OR (media_id IS NULL AND unavailable_reason IS NOT NULL)
            ),
            CHECK (
                (source_packet_preserved = 1 AND source_missing_reason IS NULL
                 AND source_media_json IS NOT NULL)
                OR (source_packet_preserved = 0 AND source_missing_reason IS NOT NULL)
            )
        ) STRICT
        """,
        """
        CREATE TABLE evidence_incident_snapshots (
            incident_id TEXT PRIMARY KEY REFERENCES evidence_incidents(incident_id)
                ON DELETE RESTRICT,
            snapshot_id TEXT NOT NULL UNIQUE CHECK (length(snapshot_id) > 0),
            media_id TEXT NOT NULL REFERENCES evidence_media_objects(media_id) ON DELETE RESTRICT,
            captured_at TEXT NOT NULL,
            camera_id TEXT NOT NULL CHECK (length(camera_id) > 0),
            created_at TEXT NOT NULL
        ) STRICT
        """,
        """
        CREATE TABLE derivative_evidence_slots (
            incident_id TEXT NOT NULL REFERENCES evidence_incidents(incident_id)
                ON DELETE RESTRICT,
            derivative_kind TEXT NOT NULL CHECK (derivative_kind IN ('ANNOTATED_CLIP')),
            state TEXT NOT NULL CHECK (state IN ('PENDING','AVAILABLE','UNAVAILABLE','CORRUPT')),
            media_id TEXT REFERENCES evidence_media_objects(media_id) ON DELETE RESTRICT,
            reason TEXT,
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (incident_id, derivative_kind),
            CHECK (
                (state = 'PENDING' AND media_id IS NULL AND reason IS NULL)
                OR (state = 'AVAILABLE' AND media_id IS NOT NULL AND reason IS NULL)
                OR (state IN ('UNAVAILABLE','CORRUPT') AND reason IS NOT NULL)
            )
        ) STRICT
        """,
        *EVIDENCE_BACKFILL_STATEMENTS,
        """
        CREATE TABLE evidence_retention_states (
            clip_id TEXT PRIMARY KEY REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
            state TEXT NOT NULL CHECK (state IN ('PENDING','PURGED','FAILED')),
            reason TEXT,
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
            requested_at TEXT NOT NULL CHECK (length(requested_at) > 0),
            updated_at TEXT NOT NULL CHECK (length(updated_at) > 0),
            CHECK ((state = 'FAILED') = (reason IS NOT NULL))
        ) STRICT
        """,
        """
        CREATE TRIGGER evidence_incidents_immutable_identity
        BEFORE UPDATE ON evidence_incidents
        WHEN NEW.incident_id IS NOT OLD.incident_id
          OR NEW.record_schema_version IS NOT OLD.record_schema_version
          OR NEW.edge_event_id IS NOT OLD.edge_event_id
          OR NEW.camera_id IS NOT OLD.camera_id
          OR NEW.event_type IS NOT OLD.event_type
          OR NEW.detected_at IS NOT OLD.detected_at
          OR NEW.runtime_manifest_sha256 IS NOT OLD.runtime_manifest_sha256
          OR NEW.decision_trace_id IS NOT OLD.decision_trace_id
          OR NEW.module_qualified_id IS NOT OLD.module_qualified_id
          OR NEW.policy_qualified_id IS NOT OLD.policy_qualified_id
          OR NEW.effective_policy_id IS NOT OLD.effective_policy_id
          OR NEW.provenance_state IS NOT OLD.provenance_state
          OR NEW.provenance_missing_reason IS NOT OLD.provenance_missing_reason
          OR NEW.created_at IS NOT OLD.created_at
          OR (OLD.primary_clip_id IS NOT NULL AND NEW.primary_clip_id IS NOT OLD.primary_clip_id)
        BEGIN
            SELECT RAISE(ABORT, 'central evidence identity and provenance are immutable');
        END
        """,
        """
        CREATE TRIGGER evidence_incidents_legal_lifecycle
        BEFORE UPDATE OF lifecycle_state ON evidence_incidents
        WHEN NEW.lifecycle_state IS NOT OLD.lifecycle_state AND NOT (
            (OLD.lifecycle_state = 'STAGING' AND NEW.lifecycle_state IN ('MEDIA_READY','FAILED'))
            OR (OLD.lifecycle_state = 'MEDIA_READY'
                AND NEW.lifecycle_state IN ('PUBLISHED','FAILED'))
            OR (OLD.lifecycle_state = 'PUBLISHED'
                AND NEW.lifecycle_state IN ('DERIVATIVE_PENDING','COMPLETE','FAILED'))
            OR (OLD.lifecycle_state = 'DERIVATIVE_PENDING'
                AND NEW.lifecycle_state IN ('COMPLETE','FAILED'))
            OR (OLD.lifecycle_state = 'COMPLETE' AND NEW.lifecycle_state = 'FAILED')
        )
        BEGIN
            SELECT RAISE(ABORT, 'illegal central evidence lifecycle transition');
        END
        """,
        """
        CREATE TRIGGER evidence_incidents_required_state_refs
        BEFORE UPDATE OF lifecycle_state ON evidence_incidents
        WHEN
            (NEW.lifecycle_state = 'MEDIA_READY' AND (
                NEW.primary_clip_id IS NULL
                OR NOT EXISTS (
                    SELECT 1 FROM evidence_primary_clips AS primary_record
                    JOIN evidence_artifact_slots AS slot
                      ON slot.incident_id = primary_record.incident_id
                     AND slot.slot_name = 'PRIMARY_CLIP'
                     AND slot.state = 'AVAILABLE'
                    WHERE primary_record.incident_id = NEW.incident_id
                      AND primary_record.clip_id = NEW.primary_clip_id
                      AND primary_record.media_id IS NOT NULL
                )
            ))
            OR (NEW.lifecycle_state IN ('PUBLISHED','COMPLETE') AND NOT EXISTS (
                SELECT 1 FROM evidence_events AS event
                JOIN evidence_clips AS clip ON clip.clip_id = NEW.primary_clip_id
                WHERE event.edge_event_id = NEW.edge_event_id
                  AND event.delivery_state = 'ACKED'
                  AND clip.publish_state = 'PUBLISHED'
            ))
            OR (NEW.lifecycle_state = 'DERIVATIVE_PENDING' AND NOT EXISTS (
                SELECT 1 FROM derivative_evidence_slots AS derivative
                WHERE derivative.incident_id = NEW.incident_id
                  AND derivative.state = 'PENDING'
            ))
            OR (NEW.lifecycle_state = 'COMPLETE' AND EXISTS (
                SELECT 1 FROM derivative_evidence_slots AS derivative
                WHERE derivative.incident_id = NEW.incident_id
                  AND derivative.state = 'PENDING'
            ))
        BEGIN
            SELECT RAISE(ABORT, 'central evidence lifecycle references are incomplete');
        END
        """,
        """
        CREATE TRIGGER evidence_incidents_revision_guard
        BEFORE UPDATE ON evidence_incidents
        WHEN NEW.revision != OLD.revision + 1
        BEGIN
            SELECT RAISE(ABORT, 'central evidence revision must advance exactly once');
        END
        """,
        """
        CREATE TRIGGER evidence_incidents_immutable_delete
        BEFORE DELETE ON evidence_incidents
        BEGIN
            SELECT RAISE(ABORT, 'central evidence records are retained by policy');
        END
        """,
        """
        CREATE TRIGGER evidence_media_objects_immutable_update
        BEFORE UPDATE ON evidence_media_objects
        BEGIN
            SELECT RAISE(ABORT, 'evidence media identities are immutable');
        END
        """,
        """
        CREATE TRIGGER evidence_media_objects_immutable_delete
        BEFORE DELETE ON evidence_media_objects
        BEGIN
            SELECT RAISE(ABORT, 'evidence media identities require retention transition');
        END
        """,
        """
        CREATE TRIGGER evidence_primary_clips_immutable_update
        BEFORE UPDATE ON evidence_primary_clips
        BEGIN
            SELECT RAISE(ABORT, 'primary evidence facts are immutable');
        END
        """,
        """
        CREATE TRIGGER evidence_incident_snapshots_immutable_update
        BEFORE UPDATE ON evidence_incident_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'snapshot evidence facts are immutable');
        END
        """,
        """
        CREATE TRIGGER evidence_retention_states_legal_transition
        BEFORE UPDATE OF state ON evidence_retention_states
        WHEN NEW.state IS NOT OLD.state AND NOT (
            (OLD.state = 'PENDING' AND NEW.state IN ('PURGED','FAILED'))
            OR (OLD.state = 'FAILED' AND NEW.state = 'PENDING')
        )
        BEGIN
            SELECT RAISE(ABORT, 'illegal central evidence retention transition');
        END
        """,
        """
        CREATE TRIGGER evidence_retention_states_revision_guard
        BEFORE UPDATE ON evidence_retention_states
        WHEN NEW.revision != OLD.revision + 1
        BEGIN
            SELECT RAISE(ABORT, 'central evidence retention revision must advance exactly once');
        END
        """,
        """
        CREATE TRIGGER evidence_artifact_slots_no_content_rewrite
        BEFORE UPDATE ON evidence_artifact_slots
        WHEN OLD.media_id IS NOT NULL AND NEW.media_id IS NOT OLD.media_id
        BEGIN
            SELECT RAISE(ABORT, 'evidence artifact content cannot be rewritten');
        END
        """,
        """
        CREATE TRIGGER evidence_artifact_slots_revision_guard
        BEFORE UPDATE ON evidence_artifact_slots
        WHEN NEW.revision != OLD.revision + 1
        BEGIN
            SELECT RAISE(ABORT, 'evidence artifact revision must advance exactly once');
        END
        """,
        """
        CREATE TRIGGER derivative_evidence_slots_no_content_rewrite
        BEFORE UPDATE ON derivative_evidence_slots
        WHEN OLD.media_id IS NOT NULL AND NEW.media_id IS NOT OLD.media_id
        BEGIN
            SELECT RAISE(ABORT, 'derivative evidence content cannot be rewritten');
        END
        """,
        "CREATE INDEX evidence_incident_lifecycle_idx ON evidence_incidents("
        "lifecycle_state, updated_at, incident_id)",
        "CREATE INDEX evidence_incident_camera_idx ON evidence_incidents("
        "camera_id, detected_at DESC, incident_id DESC)",
        "CREATE INDEX evidence_media_content_idx ON evidence_media_objects("
        "content_sha256, size_bytes)",
    ),
    writable_tables=frozenset(
        {
            "evidence_artifact_slots",
            "evidence_incidents",
            "evidence_primary_clips",
        }
    ),
)

SCHEMA_V10 = Migration(
    version=10,
    name="versioned_operator_evidence_reviews",
    statements=(
        """
        CREATE TABLE control_evidence_review_revisions (
            review_id TEXT PRIMARY KEY CHECK (length(review_id) BETWEEN 1 AND 80),
            incident_id TEXT NOT NULL REFERENCES evidence_incidents(incident_id)
                ON DELETE RESTRICT,
            clip_id TEXT NOT NULL REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
            review_version INTEGER NOT NULL CHECK (review_version > 0),
            actor_id TEXT NOT NULL CHECK (
                length(actor_id) BETWEEN 1 AND 128 AND instr(actor_id, char(0)) = 0
            ),
            reviewed_at TEXT NOT NULL CHECK (
                length(reviewed_at) BETWEEN 1 AND 64 AND instr(reviewed_at, char(0)) = 0
            ),
            disposition TEXT NOT NULL CHECK (
                disposition IN ('TRUE_POSITIVE','FALSE_POSITIVE')
            ),
            notes TEXT CHECK (
                notes IS NULL OR (
                    length(notes) BETWEEN 1 AND 1000 AND instr(notes, char(0)) = 0
                )
            ),
            UNIQUE (incident_id, clip_id, review_version)
        ) STRICT
        """,
        """
        CREATE TABLE control_evidence_review_state (
            incident_id TEXT PRIMARY KEY REFERENCES evidence_incidents(incident_id)
                ON DELETE RESTRICT,
            clip_id TEXT NOT NULL REFERENCES evidence_clips(clip_id) ON DELETE RESTRICT,
            current_version INTEGER NOT NULL CHECK (current_version > 0),
            FOREIGN KEY (incident_id, clip_id, current_version)
                REFERENCES control_evidence_review_revisions(
                    incident_id, clip_id, review_version
                ) ON DELETE RESTRICT
        ) STRICT
        """,
        """
        CREATE TABLE control_legacy_label_migrations (
            source_clip_id TEXT PRIMARY KEY CHECK (length(source_clip_id) > 0),
            classification TEXT NOT NULL CHECK (classification IN (
                'MIGRATED','ORPHAN_CLIP','ORPHAN_INCIDENT','AMBIGUOUS_INCIDENT',
                'UNSUPPORTED_DISPOSITION','UNSAFE_METADATA','REVIEW_EXISTS'
            )),
            incident_id TEXT REFERENCES evidence_incidents(incident_id) ON DELETE RESTRICT,
            review_id TEXT REFERENCES control_evidence_review_revisions(review_id)
                ON DELETE RESTRICT,
            CHECK (
                (classification = 'MIGRATED' AND incident_id IS NOT NULL
                 AND review_id IS NOT NULL)
                OR (classification != 'MIGRATED' AND incident_id IS NULL
                    AND review_id IS NULL)
            )
        ) STRICT
        """,
        """
        CREATE TRIGGER control_evidence_review_relation_insert
        BEFORE INSERT ON control_evidence_review_revisions
        WHEN NOT EXISTS (
            SELECT 1 FROM evidence_primary_clips AS primary_record
            WHERE primary_record.incident_id = NEW.incident_id
              AND primary_record.clip_id = NEW.clip_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'review must reference its central evidence relation');
        END
        """,
        """
        CREATE TRIGGER control_evidence_review_sequential_insert
        BEFORE INSERT ON control_evidence_review_revisions
        WHEN NEW.review_version != coalesce((
            SELECT current.current_version + 1
            FROM control_evidence_review_state AS current
            WHERE current.incident_id = NEW.incident_id
        ), 1)
        BEGIN
            SELECT RAISE(ABORT, 'review revision must advance exactly once');
        END
        """,
        """
        CREATE TRIGGER control_evidence_review_revisions_immutable_update
        BEFORE UPDATE ON control_evidence_review_revisions
        BEGIN
            SELECT RAISE(ABORT, 'evidence review revisions are immutable');
        END
        """,
        """
        CREATE TRIGGER control_evidence_review_revisions_immutable_delete
        BEFORE DELETE ON control_evidence_review_revisions
        BEGIN
            SELECT RAISE(ABORT, 'evidence review revisions are immutable');
        END
        """,
        """
        CREATE TRIGGER control_evidence_review_state_identity_guard
        BEFORE UPDATE ON control_evidence_review_state
        WHEN NEW.incident_id IS NOT OLD.incident_id OR NEW.clip_id IS NOT OLD.clip_id
        BEGIN
            SELECT RAISE(ABORT, 'evidence review relation is immutable');
        END
        """,
        """
        CREATE TRIGGER control_evidence_review_state_revision_guard
        BEFORE UPDATE ON control_evidence_review_state
        WHEN NEW.current_version != OLD.current_version + 1
        BEGIN
            SELECT RAISE(ABORT, 'evidence review revision must advance exactly once');
        END
        """,
        """
        CREATE TRIGGER control_evidence_review_state_immutable_delete
        BEFORE DELETE ON control_evidence_review_state
        BEGIN
            SELECT RAISE(ABORT, 'evidence review state is retained by policy');
        END
        """,
        *LEGACY_LABEL_MIGRATION_STATEMENTS,
        "CREATE INDEX control_evidence_review_clip_idx "
        "ON control_evidence_review_state(clip_id, current_version)",
    ),
    writable_tables=frozenset(
        {
            "control_evidence_review_revisions",
            "control_evidence_review_state",
            "control_legacy_label_migrations",
        }
    ),
)

SCHEMA_V11 = Migration(
    version=11,
    name="exhaustive_evidence_unavailable_reasons",
    statements=(
        """
        ALTER TABLE evidence_clips ADD COLUMN unavailable_reason_code TEXT CHECK (
            unavailable_reason_code IS NULL OR unavailable_reason_code IN (
                'ENCODER_FAILED','NO_FRAMES','FINALIZE_FAILED','STREAM_EPOCH_MISMATCH',
                'INTERRUPTED_FINALIZE','MISSING','CORRUPT'
            )
        )
        """,
        "UPDATE evidence_clips SET unavailable_reason_code = unavailable_reason "
        "WHERE unavailable_reason IS NOT NULL",
    ),
    writable_tables=frozenset({"evidence_clips"}),
)

SCHEMA_V12 = Migration(
    version=12,
    name="retire_legacy_system_test_operator_state",
    statements=(
        # Retention triggers intentionally block ordinary deletes. Drop them for
        # this one-time SYSTEM_TEST retirement, then restore identical bodies.
        "DROP TRIGGER IF EXISTS control_evidence_review_state_immutable_delete",
        "DROP TRIGGER IF EXISTS control_evidence_review_revisions_immutable_delete",
        "DROP TRIGGER IF EXISTS evidence_incidents_immutable_delete",
        """
        DELETE FROM control_evidence_review_state
        WHERE incident_id IN (
            SELECT incident.incident_id
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event
              ON event.edge_event_id = incident.edge_event_id
            WHERE event.operator_only = 1
        )
        """,
        """
        DELETE FROM control_evidence_review_revisions
        WHERE incident_id IN (
            SELECT incident.incident_id
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event
              ON event.edge_event_id = incident.edge_event_id
            WHERE event.operator_only = 1
        )
        """,
        """
        DELETE FROM control_legacy_label_migrations
        WHERE incident_id IN (
            SELECT incident.incident_id
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event
              ON event.edge_event_id = incident.edge_event_id
            WHERE event.operator_only = 1
        )
        """,
        """
        DELETE FROM derivative_evidence_slots
        WHERE incident_id IN (
            SELECT incident.incident_id
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event
              ON event.edge_event_id = incident.edge_event_id
            WHERE event.operator_only = 1
        )
        """,
        """
        DELETE FROM evidence_artifact_slots
        WHERE incident_id IN (
            SELECT incident.incident_id
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event
              ON event.edge_event_id = incident.edge_event_id
            WHERE event.operator_only = 1
        )
        """,
        """
        DELETE FROM evidence_incident_snapshots
        WHERE incident_id IN (
            SELECT incident.incident_id
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event
              ON event.edge_event_id = incident.edge_event_id
            WHERE event.operator_only = 1
        )
        """,
        """
        DELETE FROM evidence_primary_clips
        WHERE incident_id IN (
            SELECT incident.incident_id
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event
              ON event.edge_event_id = incident.edge_event_id
            WHERE event.operator_only = 1
        )
        """,
        """
        DELETE FROM evidence_incidents
        WHERE edge_event_id IN (
            SELECT edge_event_id FROM evidence_events WHERE operator_only = 1
        )
        """,
        """
        DELETE FROM evidence_clip_trace_refs
        WHERE edge_event_id IN (
            SELECT edge_event_id FROM evidence_events WHERE operator_only = 1
        )
        """,
        """
        DELETE FROM evidence_event_trace_refs
        WHERE edge_event_id IN (
            SELECT edge_event_id FROM evidence_events WHERE operator_only = 1
        )
        """,
        """
        DELETE FROM clip_events
        WHERE edge_event_id IN (
            SELECT edge_event_id FROM evidence_events WHERE operator_only = 1
        )
        """,
        "DROP TABLE IF EXISTS system_test_runs",
        "DELETE FROM evidence_events WHERE operator_only = 1",
        "DROP INDEX IF EXISTS evidence_events_operator_claim_idx",
        "ALTER TABLE evidence_events DROP COLUMN operator_only",
        """
        CREATE TRIGGER control_evidence_review_state_immutable_delete
        BEFORE DELETE ON control_evidence_review_state
        BEGIN
            SELECT RAISE(ABORT, 'evidence review state is retained by policy');
        END
        """,
        """
        CREATE TRIGGER control_evidence_review_revisions_immutable_delete
        BEFORE DELETE ON control_evidence_review_revisions
        BEGIN
            SELECT RAISE(ABORT, 'evidence review revisions are immutable');
        END
        """,
        """
        CREATE TRIGGER evidence_incidents_immutable_delete
        BEFORE DELETE ON evidence_incidents
        BEGIN
            SELECT RAISE(ABORT, 'central evidence records are retained by policy');
        END
        """,
        "INSERT INTO schema_metadata (key, value) VALUES ('system_test_state_policy', 'retired')",
    ),
    writable_tables=frozenset(
        {
            "clip_events",
            "control_evidence_review_revisions",
            "control_evidence_review_state",
            "control_legacy_label_migrations",
            "derivative_evidence_slots",
            "evidence_artifact_slots",
            "evidence_clip_trace_refs",
            "evidence_event_trace_refs",
            "evidence_events",
            "evidence_incident_snapshots",
            "evidence_incidents",
            "evidence_primary_clips",
            "schema_metadata",
            "system_test_runs",
        }
    ),
)
"""Retire temporary SYSTEM_TEST operator-only state to match released main outbox
schema 9. Ordinary evidence events, clips, config snapshots, faults, clip-deletion
reasons, and non-operator central projections are preserved. Dependent central
rows that referenced operator-only events are removed first so SQLite RESTRICT
foreign keys and DROP COLUMN succeed inside the migrator transaction."""

SCHEMA_V13 = Migration(
    version=13,
    name="canonical_overlay_scenes_and_derivatives",
    statements=(
        """
        CREATE TABLE runtime_analysis_keypoints (
            analysis_trace_id TEXT NOT NULL REFERENCES runtime_analysis_traces(trace_id)
                ON DELETE CASCADE,
            person_ordinal INTEGER NOT NULL CHECK (person_ordinal >= 0),
            keypoint_index INTEGER NOT NULL CHECK (keypoint_index >= 0),
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
            PRIMARY KEY (analysis_trace_id, person_ordinal, keypoint_index),
            FOREIGN KEY (analysis_trace_id, person_ordinal)
                REFERENCES runtime_analysis_persons(analysis_trace_id, ordinal)
                ON DELETE CASCADE
        ) STRICT
        """,
        """
        CREATE TABLE runtime_analysis_bed_points (
            analysis_trace_id TEXT NOT NULL REFERENCES runtime_analysis_traces(trace_id)
                ON DELETE CASCADE,
            bed_ordinal INTEGER NOT NULL CHECK (bed_ordinal >= 0),
            point_index INTEGER NOT NULL CHECK (point_index >= 0),
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            PRIMARY KEY (analysis_trace_id, bed_ordinal, point_index),
            FOREIGN KEY (analysis_trace_id, bed_ordinal)
                REFERENCES runtime_analysis_beds(analysis_trace_id, ordinal)
                ON DELETE CASCADE
        ) STRICT
        """,
        """
        CREATE TABLE derivative_render_records (
            incident_id TEXT NOT NULL,
            derivative_kind TEXT NOT NULL CHECK (derivative_kind = 'ANNOTATED_CLIP'),
            derivative_id TEXT NOT NULL UNIQUE CHECK (length(derivative_id) = 64),
            media_id TEXT NOT NULL REFERENCES evidence_media_objects(media_id)
                ON DELETE RESTRICT,
            primary_clip_id TEXT NOT NULL REFERENCES evidence_clips(clip_id)
                ON DELETE RESTRICT,
            primary_media_sha256 TEXT NOT NULL CHECK (length(primary_media_sha256) = 64),
            decision_trace_id TEXT NOT NULL CHECK (length(decision_trace_id) = 64),
            runtime_manifest_sha256 TEXT NOT NULL CHECK (length(runtime_manifest_sha256) = 64),
            scene_id TEXT NOT NULL CHECK (length(scene_id) = 64),
            scene_schema_version INTEGER NOT NULL CHECK (scene_schema_version = 1),
            render_backend TEXT NOT NULL CHECK (length(render_backend) > 0),
            render_device TEXT NOT NULL CHECK (render_device = 'cpu'),
            input_memory_kind TEXT NOT NULL CHECK (input_memory_kind = 'host'),
            render_version TEXT NOT NULL CHECK (length(render_version) > 0),
            width INTEGER NOT NULL CHECK (width > 0),
            height INTEGER NOT NULL CHECK (height > 0),
            start_time_ms INTEGER NOT NULL CHECK (start_time_ms >= 0),
            end_time_ms INTEGER NOT NULL CHECK (end_time_ms >= start_time_ms),
            created_at TEXT NOT NULL CHECK (length(created_at) > 0),
            PRIMARY KEY (incident_id, derivative_kind),
            FOREIGN KEY (incident_id, derivative_kind)
                REFERENCES derivative_evidence_slots(incident_id, derivative_kind)
                ON DELETE RESTRICT
        ) STRICT
        """,
        """
        CREATE TRIGGER derivative_render_records_immutable_update
        BEFORE UPDATE ON derivative_render_records
        BEGIN
            SELECT RAISE(ABORT, 'derivative render facts are immutable');
        END
        """,
        """
        CREATE TRIGGER derivative_render_records_immutable_delete
        BEFORE DELETE ON derivative_render_records
        BEGIN
            SELECT RAISE(ABORT, 'derivative render facts require retention transition');
        END
        """,
        """
        CREATE TRIGGER derivative_evidence_slots_legal_transition_v13
        BEFORE UPDATE OF state ON derivative_evidence_slots
        WHEN NEW.state IS NOT OLD.state AND NOT (
            (OLD.state = 'PENDING'
             AND NEW.state IN ('AVAILABLE','UNAVAILABLE','CORRUPT'))
            OR (OLD.state = 'AVAILABLE' AND NEW.state = 'CORRUPT')
        )
        BEGIN
            SELECT RAISE(ABORT, 'illegal derivative evidence transition');
        END
        """,
        """
        CREATE TRIGGER derivative_evidence_slots_revision_guard_v13
        BEFORE UPDATE ON derivative_evidence_slots
        WHEN NEW.revision != OLD.revision + 1
        BEGIN
            SELECT RAISE(ABORT, 'derivative evidence revision must advance exactly once');
        END
        """,
        "CREATE INDEX derivative_render_content_idx ON derivative_render_records(derivative_id)",
    ),
)

SCHEMA_V14 = Migration(
    version=14,
    name="still_video_derivative_lifecycle",
    statements=(
        """
        CREATE TABLE derivative_jobs (
            incident_id TEXT NOT NULL REFERENCES evidence_incidents(incident_id)
                ON DELETE RESTRICT,
            derivative_kind TEXT NOT NULL CHECK (derivative_kind IN ('STILL','VIDEO')),
            request_id TEXT NOT NULL UNIQUE CHECK (length(request_id) = 64),
            state TEXT NOT NULL CHECK (state IN (
                'PENDING','RUNNING','AVAILABLE','UNAVAILABLE','CORRUPT','CANCELLED'
            )),
            media_id TEXT REFERENCES evidence_media_objects(media_id) ON DELETE RESTRICT,
            reason TEXT CHECK (reason IS NULL OR length(reason) BETWEEN 1 AND 128),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0,1)),
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
            created_at TEXT NOT NULL CHECK (length(created_at) > 0),
            updated_at TEXT NOT NULL CHECK (length(updated_at) > 0),
            PRIMARY KEY (incident_id, derivative_kind),
            CHECK (
                (state IN ('PENDING','RUNNING') AND media_id IS NULL AND reason IS NULL)
                OR (state = 'AVAILABLE' AND media_id IS NOT NULL AND reason IS NULL)
                OR (state IN ('UNAVAILABLE','CANCELLED')
                    AND media_id IS NULL AND reason IS NOT NULL)
                OR (state = 'CORRUPT' AND reason IS NOT NULL)
            )
        ) STRICT
        """,
        """
        CREATE TABLE derivative_artifacts (
            incident_id TEXT NOT NULL,
            derivative_kind TEXT NOT NULL CHECK (derivative_kind IN ('STILL','VIDEO')),
            derivative_id TEXT NOT NULL UNIQUE CHECK (length(derivative_id) = 64),
            media_id TEXT NOT NULL REFERENCES evidence_media_objects(media_id)
                ON DELETE RESTRICT,
            primary_clip_id TEXT NOT NULL REFERENCES evidence_clips(clip_id)
                ON DELETE RESTRICT,
            primary_media_sha256 TEXT NOT NULL CHECK (length(primary_media_sha256) = 64),
            decision_trace_id TEXT NOT NULL CHECK (length(decision_trace_id) = 64),
            runtime_manifest_sha256 TEXT NOT NULL CHECK (length(runtime_manifest_sha256) = 64),
            scene_id TEXT NOT NULL CHECK (length(scene_id) = 64),
            scene_schema_version INTEGER NOT NULL CHECK (scene_schema_version = 1),
            render_backend TEXT NOT NULL CHECK (length(render_backend) > 0),
            render_device TEXT NOT NULL CHECK (length(render_device) > 0),
            input_memory_kind TEXT NOT NULL CHECK (length(input_memory_kind) > 0),
            render_version TEXT NOT NULL CHECK (length(render_version) > 0),
            width INTEGER NOT NULL CHECK (width > 0),
            height INTEGER NOT NULL CHECK (height > 0),
            start_time_ms INTEGER NOT NULL CHECK (start_time_ms >= 0),
            end_time_ms INTEGER NOT NULL CHECK (end_time_ms >= start_time_ms),
            created_at TEXT NOT NULL CHECK (length(created_at) > 0),
            PRIMARY KEY (incident_id, derivative_kind),
            FOREIGN KEY (incident_id, derivative_kind)
                REFERENCES derivative_jobs(incident_id, derivative_kind)
                ON DELETE RESTRICT
        ) STRICT
        """,
        """
        CREATE TRIGGER derivative_jobs_immutable_identity
        BEFORE UPDATE ON derivative_jobs
        WHEN NEW.incident_id IS NOT OLD.incident_id
          OR NEW.derivative_kind IS NOT OLD.derivative_kind
          OR NEW.request_id IS NOT OLD.request_id
          OR NEW.created_at IS NOT OLD.created_at
        BEGIN
            SELECT RAISE(ABORT, 'derivative request identity is immutable');
        END
        """,
        """
        CREATE TRIGGER derivative_jobs_legal_transition
        BEFORE UPDATE OF state ON derivative_jobs
        WHEN NEW.state IS NOT OLD.state AND NOT (
            (OLD.state = 'PENDING' AND NEW.state IN (
                'RUNNING','AVAILABLE','UNAVAILABLE','CANCELLED'
            ))
            OR (OLD.state = 'RUNNING' AND NEW.state IN (
                'PENDING','AVAILABLE','UNAVAILABLE','CANCELLED'
            ))
            OR (OLD.state = 'AVAILABLE' AND NEW.state = 'CORRUPT')
        )
        BEGIN
            SELECT RAISE(ABORT, 'illegal derivative job transition');
        END
        """,
        """
        CREATE TRIGGER derivative_jobs_revision_guard
        BEFORE UPDATE ON derivative_jobs
        WHEN NEW.revision != OLD.revision + 1
        BEGIN
            SELECT RAISE(ABORT, 'derivative job revision must advance exactly once');
        END
        """,
        """
        CREATE TRIGGER derivative_jobs_no_content_rewrite
        BEFORE UPDATE ON derivative_jobs
        WHEN OLD.media_id IS NOT NULL AND NEW.media_id IS NOT OLD.media_id
        BEGIN
            SELECT RAISE(ABORT, 'derivative job content cannot be rewritten');
        END
        """,
        """
        CREATE TRIGGER derivative_artifacts_immutable_update
        BEFORE UPDATE ON derivative_artifacts
        BEGIN
            SELECT RAISE(ABORT, 'derivative artifact facts are immutable');
        END
        """,
        """
        CREATE TRIGGER derivative_artifacts_immutable_delete
        BEFORE DELETE ON derivative_artifacts
        BEGIN
            SELECT RAISE(ABORT, 'derivative artifact facts require retention transition');
        END
        """,
        "CREATE INDEX derivative_jobs_state_idx ON derivative_jobs(state, updated_at, request_id)",
        "CREATE INDEX derivative_artifacts_content_idx ON derivative_artifacts(derivative_id)",
        "CREATE INDEX derivative_artifacts_primary_idx ON derivative_artifacts(primary_clip_id)",
    ),
)

SCHEMA_V15 = Migration(
    version=15,
    name="internal_replay_qa",
    statements=(
        """
        CREATE TABLE qa_replay_runs (
            run_id TEXT PRIMARY KEY CHECK (length(run_id) = 64),
            camera_id TEXT NOT NULL CHECK (length(camera_id) > 0),
            module_qualified_id TEXT NOT NULL CHECK (length(module_qualified_id) > 0),
            policy_qualified_id TEXT NOT NULL CHECK (length(policy_qualified_id) > 0),
            effective_policy_id TEXT NOT NULL CHECK (length(effective_policy_id) = 64),
            frame_count INTEGER NOT NULL CHECK (frame_count >= 0),
            event_count INTEGER NOT NULL CHECK (event_count >= 0),
            source_kind TEXT NOT NULL CHECK (source_kind IN ('captured','replay')),
            source_run_id TEXT REFERENCES qa_replay_runs(run_id) ON DELETE RESTRICT,
            requested_by TEXT NOT NULL CHECK (
                length(requested_by) BETWEEN 1 AND 128 AND instr(requested_by, char(0)) = 0
            ),
            requested_at TEXT NOT NULL CHECK (
                length(requested_at) BETWEEN 1 AND 64 AND instr(requested_at, char(0)) = 0
            ),
            result_sha256 TEXT NOT NULL CHECK (length(result_sha256) = 64),
            result_json TEXT NOT NULL CHECK (length(result_json) > 0),
            CHECK (
                (source_kind = 'captured' AND source_run_id IS NULL)
                OR (source_kind = 'replay' AND source_run_id IS NOT NULL)
            )
        ) STRICT
        """,
        """
        CREATE TRIGGER qa_replay_runs_immutable_update
        BEFORE UPDATE ON qa_replay_runs
        BEGIN
            SELECT RAISE(ABORT, 'replay run facts are immutable');
        END
        """,
        """
        CREATE TRIGGER qa_replay_runs_immutable_delete
        BEFORE DELETE ON qa_replay_runs
        BEGIN
            SELECT RAISE(ABORT, 'replay run facts are immutable');
        END
        """,
        """
        CREATE TABLE qa_replay_comparisons (
            comparison_id TEXT PRIMARY KEY CHECK (length(comparison_id) = 64),
            baseline_run_id TEXT NOT NULL REFERENCES qa_replay_runs(run_id)
                ON DELETE RESTRICT,
            candidate_run_id TEXT NOT NULL REFERENCES qa_replay_runs(run_id)
                ON DELETE RESTRICT,
            identical INTEGER NOT NULL CHECK (identical IN (0,1)),
            mismatch_count INTEGER NOT NULL CHECK (mismatch_count >= 0),
            created_at TEXT NOT NULL CHECK (
                length(created_at) BETWEEN 1 AND 64 AND instr(created_at, char(0)) = 0
            ),
            comparison_sha256 TEXT NOT NULL CHECK (length(comparison_sha256) = 64),
            comparison_json TEXT NOT NULL CHECK (length(comparison_json) > 0),
            CHECK (baseline_run_id != candidate_run_id),
            CHECK ((identical = 1 AND mismatch_count = 0) OR (identical = 0 AND mismatch_count > 0))
        ) STRICT
        """,
        """
        CREATE TRIGGER qa_replay_comparisons_immutable_update
        BEFORE UPDATE ON qa_replay_comparisons
        BEGIN
            SELECT RAISE(ABORT, 'replay comparison facts are immutable');
        END
        """,
        """
        CREATE TRIGGER qa_replay_comparisons_immutable_delete
        BEFORE DELETE ON qa_replay_comparisons
        BEGIN
            SELECT RAISE(ABORT, 'replay comparison facts are immutable');
        END
        """,
        """
        CREATE TABLE qa_label_revisions (
            label_id TEXT PRIMARY KEY CHECK (length(label_id) BETWEEN 1 AND 80),
            comparison_id TEXT NOT NULL REFERENCES qa_replay_comparisons(comparison_id)
                ON DELETE RESTRICT,
            label_version INTEGER NOT NULL CHECK (label_version > 0),
            actor_id TEXT NOT NULL CHECK (
                length(actor_id) BETWEEN 1 AND 128 AND instr(actor_id, char(0)) = 0
            ),
            labeled_at TEXT NOT NULL CHECK (
                length(labeled_at) BETWEEN 1 AND 64 AND instr(labeled_at, char(0)) = 0
            ),
            disposition TEXT NOT NULL CHECK (disposition IN ('TP','FP','FN','TN','INCONCLUSIVE')),
            notes TEXT CHECK (
                notes IS NULL OR (
                    length(notes) BETWEEN 1 AND 1000 AND instr(notes, char(0)) = 0
                )
            ),
            UNIQUE (comparison_id, label_version)
        ) STRICT
        """,
        """
        CREATE TABLE qa_label_state (
            comparison_id TEXT PRIMARY KEY REFERENCES qa_replay_comparisons(comparison_id)
                ON DELETE RESTRICT,
            current_version INTEGER NOT NULL CHECK (current_version > 0),
            FOREIGN KEY (comparison_id, current_version)
                REFERENCES qa_label_revisions(comparison_id, label_version)
                ON DELETE RESTRICT
        ) STRICT
        """,
        """
        CREATE TRIGGER qa_label_revisions_relation_insert
        BEFORE INSERT ON qa_label_revisions
        WHEN NOT EXISTS (
            SELECT 1 FROM qa_replay_comparisons
            WHERE comparison_id = NEW.comparison_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'label must reference its comparison');
        END
        """,
        """
        CREATE TRIGGER qa_label_revisions_sequential_insert
        BEFORE INSERT ON qa_label_revisions
        WHEN NEW.label_version != coalesce((
            SELECT current.current_version + 1
            FROM qa_label_state AS current
            WHERE current.comparison_id = NEW.comparison_id
        ), 1)
        BEGIN
            SELECT RAISE(ABORT, 'label revision must advance exactly once');
        END
        """,
        """
        CREATE TRIGGER qa_label_revisions_immutable_update
        BEFORE UPDATE ON qa_label_revisions
        BEGIN
            SELECT RAISE(ABORT, 'qa label revisions are immutable');
        END
        """,
        """
        CREATE TRIGGER qa_label_revisions_immutable_delete
        BEFORE DELETE ON qa_label_revisions
        BEGIN
            SELECT RAISE(ABORT, 'qa label revisions are immutable');
        END
        """,
        """
        CREATE TRIGGER qa_label_state_identity_guard
        BEFORE UPDATE ON qa_label_state
        WHEN NEW.comparison_id IS NOT OLD.comparison_id
        BEGIN
            SELECT RAISE(ABORT, 'qa label relation is immutable');
        END
        """,
        """
        CREATE TRIGGER qa_label_state_revision_guard
        BEFORE UPDATE ON qa_label_state
        WHEN NEW.current_version != OLD.current_version + 1
        BEGIN
            SELECT RAISE(ABORT, 'qa label revision must advance exactly once');
        END
        """,
        """
        CREATE TRIGGER qa_label_state_immutable_delete
        BEFORE DELETE ON qa_label_state
        BEGIN
            SELECT RAISE(ABORT, 'qa label state is retained by policy');
        END
        """,
        "CREATE INDEX qa_replay_runs_camera_idx ON qa_replay_runs(camera_id, requested_at)",
        "CREATE INDEX qa_replay_comparisons_baseline_idx ON qa_replay_comparisons(baseline_run_id)",
        "CREATE INDEX qa_replay_comparisons_candidate_idx "
        "ON qa_replay_comparisons(candidate_run_id)",
    ),
    writable_tables=frozenset(
        {
            "qa_replay_runs",
            "qa_replay_comparisons",
            "qa_label_revisions",
            "qa_label_state",
        }
    ),
)

SCHEMA_V16 = Migration(
    version=16,
    name="live_runtime_clip_export_settings",
    statements=(
        """
        CREATE TABLE runtime_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            clip_export_enabled INTEGER NOT NULL CHECK (clip_export_enabled IN (0, 1)),
            version INTEGER NOT NULL CHECK (version >= 0)
        ) STRICT
        """,
    ),
    writable_tables=frozenset({"runtime_settings"}),
)

SCHEMA_V17 = Migration(
    version=17,
    name="backend_only_application_ownership",
    statements=(
        """
        UPDATE schema_table_families
        SET writer = CASE WHEN prefix = 'schema_' THEN 'migrator' ELSE 'api' END
        WHERE writer != CASE WHEN prefix = 'schema_' THEN 'migrator' ELSE 'api' END
        """,
    ),
    writable_tables=frozenset({"schema_table_families"}),
    preflight=_require_schema17_drain,
)

MIGRATIONS: Final = (
    SCHEMA_V1,
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMA_V4,
    SCHEMA_V5,
    SCHEMA_V6,
    SCHEMA_V7,
    SCHEMA_V8,
    SCHEMA_V9,
    SCHEMA_V10,
    SCHEMA_V11,
    SCHEMA_V12,
    SCHEMA_V13,
    SCHEMA_V14,
    SCHEMA_V15,
    SCHEMA_V16,
    SCHEMA_V17,
)
SCHEMA_VERSION: Final = MIGRATIONS[-1].version

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "Migration",
    "SchemaV17MigrationError",
]
