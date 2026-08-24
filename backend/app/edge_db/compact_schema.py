"""Schema 18 compact application tables and rebuild statements."""

from __future__ import annotations

from typing import Final

from backend.app.edge_db.compact_schema_ddl import COMPACT_SCHEMA_CREATE_STATEMENTS

COMPACT_APPLICATION_TABLES: Final = frozenset(
    {
        "artifacts",
        "audit_events",
        "cameras",
        "clips",
        "credentials",
        "edge_site",
        "incidents",
        "locations",
        "policies",
        "schema_migrations",
    }
)
COMPACT_API_TABLES: Final = frozenset(
    table for table in COMPACT_APPLICATION_TABLES if table != "schema_migrations"
)

# FK-safe drop order for the 71 non-ledger tables present at schema 17.
SCHEMA17_RETIRED_TABLES: Final = (
    "audit",
    "camera_bed_zone",
    "camera_registry",
    "camera_topology_cameras",
    "cameras",
    "clip_events",
    "clip_listing_generation",
    "clip_listing_rows",
    "clip_listing_summary",
    "clip_listing_thumbnails",
    "clip_storage_location",
    "clips",
    "config_current",
    "config_history",
    "connection_settings",
    "connection_store_migrations",
    "control_detection_policy_activations",
    "control_detection_policy_state",
    "control_evidence_review_state",
    "control_heartbeats",
    "control_legacy_label_migrations",
    "credentials",
    "derivative_artifacts",
    "derivative_render_records",
    "detection_settings",
    "edge_topology_confirmation_preview",
    "edge_topology_sync_state",
    "events",
    "evidence_artifact_slots",
    "evidence_clip_trace_refs",
    "evidence_decision_values",
    "evidence_event_trace_refs",
    "evidence_incident_snapshots",
    "evidence_primary_clips",
    "evidence_retention_states",
    "faults",
    "labels",
    "qa_label_state",
    "runtime_analysis_bed_points",
    "runtime_analysis_components",
    "runtime_analysis_keypoints",
    "runtime_latency",
    "runtime_manifest_cameras",
    "runtime_provenance_retention",
    "runtime_settings",
    "runtime_trace_cursors",
    "schema_import_receipts",
    "schema_import_sources",
    "schema_metadata",
    "schema_table_families",
    "snapshots",
    "topology_dirty",
    "camera_topology_rooms",
    "control_detection_policy_revisions",
    "control_evidence_review_revisions",
    "derivative_evidence_slots",
    "derivative_jobs",
    "qa_label_revisions",
    "runtime_analysis_beds",
    "runtime_analysis_persons",
    "runtime_manifest_boots",
    "camera_topology_floors",
    "evidence_incidents",
    "evidence_media_objects",
    "qa_replay_comparisons",
    "evidence_clips",
    "evidence_decision_traces",
    "evidence_events",
    "runtime_analysis_traces",
    "runtime_manifest_contents",
    "qa_replay_runs",
)

SCHEMA_V18_STATEMENTS: Final = (
    """
    ALTER TABLE schema_migrations ADD COLUMN source_schema_version INTEGER
        CHECK (source_schema_version IS NULL OR source_schema_version > 0)
    """,
    """
    ALTER TABLE schema_migrations ADD COLUMN source_db_sha256 TEXT
        CHECK (
            source_db_sha256 IS NULL
            OR (
                length(source_db_sha256) = 64
                AND source_db_sha256 NOT GLOB '*[^0-9a-f]*'
            )
        )
    """,
    """
    ALTER TABLE schema_migrations ADD COLUMN reconciliation_sha256 TEXT
        CHECK (
            reconciliation_sha256 IS NULL
            OR (
                length(reconciliation_sha256) = 64
                AND reconciliation_sha256 NOT GLOB '*[^0-9a-f]*'
            )
        )
    """,
    *(f"DROP TABLE {_table}" for _table in SCHEMA17_RETIRED_TABLES if _table != "qa_replay_runs"),
    "DROP TRIGGER qa_replay_runs_immutable_update",
    "DROP TRIGGER qa_replay_runs_immutable_delete",
    "UPDATE qa_replay_runs SET source_kind = 'captured', source_run_id = NULL",
    "DELETE FROM qa_replay_runs",
    "DROP TABLE qa_replay_runs",
    *COMPACT_SCHEMA_CREATE_STATEMENTS,
)

__all__ = [
    "COMPACT_API_TABLES",
    "COMPACT_APPLICATION_TABLES",
    "SCHEMA17_RETIRED_TABLES",
    "SCHEMA_V18_STATEMENTS",
]
