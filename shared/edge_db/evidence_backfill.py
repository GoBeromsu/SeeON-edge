"""Deterministic promotion of legacy outbox facts into central evidence records."""

from __future__ import annotations

import sqlite3
from typing import Final

_CAMERA = """CASE
    WHEN json_valid(event.payload_json)
     AND json_type(event.payload_json, '$.camera_id') = 'text'
     AND length(json_extract(event.payload_json, '$.camera_id')) > 0
    THEN json_extract(event.payload_json, '$.camera_id')
    ELSE 'LEGACY_CAMERA_NOT_RECORDED'
END"""
_EVENT_TYPE = """CASE
    WHEN json_valid(event.payload_json)
     AND json_type(event.payload_json, '$.event_type') = 'text'
     AND length(json_extract(event.payload_json, '$.event_type')) > 0
    THEN json_extract(event.payload_json, '$.event_type')
    WHEN json_valid(event.payload_json)
     AND json_type(event.payload_json, '$.type') = 'text'
     AND length(json_extract(event.payload_json, '$.type')) > 0
    THEN json_extract(event.payload_json, '$.type')
    ELSE 'LEGACY_EVENT_TYPE_NOT_RECORDED'
END"""

EVIDENCE_BACKFILL_STATEMENTS: Final = (
    f"""
    INSERT INTO evidence_incidents (
        incident_id, edge_event_id, camera_id, event_type, detected_at,
        runtime_manifest_sha256, decision_trace_id, module_qualified_id,
        policy_qualified_id, effective_policy_id, provenance_state,
        provenance_missing_reason, primary_clip_id, lifecycle_state,
        failure_reason, created_at, updated_at
    )
    SELECT event.edge_event_id, event.edge_event_id, {_CAMERA}, {_EVENT_TYPE},
           event.detected_at,
           CASE WHEN decision.trace_id IS NOT NULL
                      AND analysis.camera_id = {_CAMERA}
                THEN decision.runtime_manifest_sha256 END,
           CASE WHEN decision.trace_id IS NOT NULL
                      AND analysis.camera_id = {_CAMERA}
                THEN decision.trace_id END,
           CASE WHEN decision.trace_id IS NOT NULL
                      AND analysis.camera_id = {_CAMERA}
                THEN decision.module_qualified_id END,
           CASE WHEN decision.trace_id IS NOT NULL
                      AND analysis.camera_id = {_CAMERA}
                THEN decision.policy_qualified_id END,
           CASE WHEN decision.trace_id IS NOT NULL
                      AND analysis.camera_id = {_CAMERA}
                THEN decision.effective_policy_id END,
           CASE WHEN decision.trace_id IS NOT NULL
                      AND analysis.camera_id = {_CAMERA}
                THEN 'QUALIFIED' ELSE 'MISSING' END,
           CASE WHEN decision.trace_id IS NOT NULL
                      AND analysis.camera_id = {_CAMERA}
                THEN NULL ELSE 'LEGACY_PROVENANCE_NOT_RECORDED' END,
           relation.clip_id,
           CASE WHEN clip.local_state = 'AWAITING_FINALIZE' THEN 'STAGING'
                ELSE 'FAILED' END,
           CASE WHEN clip.local_state = 'AWAITING_FINALIZE' THEN NULL
                WHEN relation.clip_id IS NULL THEN 'MISSING'
                WHEN clip.local_state = 'CORRUPT' THEN 'CORRUPT'
                WHEN clip.unavailable_reason = 'MISSING' THEN 'MISSING'
                WHEN clip.unavailable_reason = 'INTERRUPTED_FINALIZE' THEN 'INTERRUPTED'
                WHEN clip.local_state = 'VERIFIED' THEN 'PUBLICATION_FAILED'
                ELSE 'UNAVAILABLE' END,
           event.detected_at, event.detected_at
    FROM evidence_events AS event
    LEFT JOIN clip_events AS relation USING (edge_event_id)
    LEFT JOIN evidence_clips AS clip USING (clip_id)
    LEFT JOIN evidence_event_trace_refs AS event_trace USING (edge_event_id)
    LEFT JOIN evidence_decision_traces AS decision
      ON decision.trace_id = event_trace.decision_trace_id
    LEFT JOIN runtime_analysis_traces AS analysis
      ON analysis.trace_id = decision.analysis_trace_id
    WHERE NOT EXISTS (
        SELECT 1 FROM evidence_incidents AS existing
        WHERE existing.edge_event_id = event.edge_event_id
    )
    ORDER BY event.edge_event_id
    """,
    """
    INSERT INTO evidence_artifact_slots (
        incident_id, slot_name, state, reason, created_at, updated_at
    )
    SELECT incident.incident_id, 'PRIMARY_CLIP',
           CASE WHEN clip.local_state = 'AWAITING_FINALIZE' THEN 'PENDING'
                WHEN clip.local_state = 'CORRUPT' THEN 'CORRUPT'
                ELSE 'UNAVAILABLE' END,
           CASE WHEN clip.local_state = 'AWAITING_FINALIZE' THEN NULL
                WHEN relation.clip_id IS NULL THEN 'LEGACY_CLIP_RELATION_NOT_RECORDED'
                WHEN clip.local_state = 'VERIFIED'
                  THEN 'LEGACY_MANIFEST_FACTS_NOT_RECORDED'
                ELSE coalesce(clip.unavailable_reason, 'LEGACY_CLIP_STATE_NOT_RECORDED') END,
           incident.created_at, incident.updated_at
    FROM evidence_incidents AS incident
    LEFT JOIN clip_events AS relation USING (edge_event_id)
    LEFT JOIN evidence_clips AS clip USING (clip_id)
    WHERE NOT EXISTS (
        SELECT 1 FROM evidence_artifact_slots AS slot
        WHERE slot.incident_id = incident.incident_id
          AND slot.slot_name = 'PRIMARY_CLIP'
    )
    ORDER BY incident.incident_id
    """,
    """
    INSERT INTO evidence_artifact_slots (
        incident_id, slot_name, state, reason, created_at, updated_at
    )
    SELECT incident_id, 'SNAPSHOT', 'UNAVAILABLE',
           'LEGACY_SNAPSHOT_NOT_RECORDED', created_at, updated_at
    FROM evidence_incidents AS incident
    WHERE NOT EXISTS (
        SELECT 1 FROM evidence_artifact_slots AS slot
        WHERE slot.incident_id = incident.incident_id AND slot.slot_name = 'SNAPSHOT'
    )
    ORDER BY incident_id
    """,
    """
    INSERT INTO evidence_primary_clips (
        incident_id, clip_id, clip_start_at, clip_end_at, finalized_at,
        source_packet_preserved, source_missing_reason, truncation_json,
        unavailable_reason, created_at
    )
    SELECT incident.incident_id, clip.clip_id, clip.clip_start_at, clip.clip_end_at,
           clip.finalized_at, 0, 'LEGACY_SOURCE_FACTS_NOT_RECORDED', '[]',
           CASE WHEN clip.local_state = 'VERIFIED'
                  THEN 'LEGACY_MANIFEST_FACTS_NOT_RECORDED'
                ELSE coalesce(clip.unavailable_reason, 'LEGACY_CLIP_STATE_NOT_RECORDED') END,
           coalesce(clip.finalized_at, incident.created_at)
    FROM evidence_incidents AS incident
    JOIN clip_events AS relation USING (edge_event_id)
    JOIN evidence_clips AS clip USING (clip_id)
    WHERE clip.local_state != 'AWAITING_FINALIZE'
      AND NOT EXISTS (
          SELECT 1 FROM evidence_primary_clips AS existing
          WHERE existing.incident_id = incident.incident_id
      )
    ORDER BY incident.incident_id
    """,
)


def backfill_legacy_evidence(connection: sqlite3.Connection) -> None:
    """Idempotently materialize central records without changing legacy source rows."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in EVIDENCE_BACKFILL_STATEMENTS:
            connection.execute(statement)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


__all__ = ["EVIDENCE_BACKFILL_STATEMENTS", "backfill_legacy_evidence"]
