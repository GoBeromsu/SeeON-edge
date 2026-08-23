from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.edge_db.migrator import migrate_database
from backend.app.features.evidence.record_store import CentralEvidenceQuery


def test_central_evidence_schema_has_owned_strict_records_and_integrity_guards(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)

    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type = 'table' "
                "AND (name LIKE 'evidence_%' OR name LIKE 'derivative_%')"
            )
        }
        assert {
            "evidence_incidents",
            "evidence_media_objects",
            "evidence_artifact_slots",
            "evidence_primary_clips",
            "evidence_incident_snapshots",
            "evidence_retention_states",
            "derivative_evidence_slots",
        } <= tables.keys()
        assert all(
            "STRICT" in tables[name]
            for name in (
                "evidence_incidents",
                "evidence_media_objects",
                "evidence_artifact_slots",
                "evidence_primary_clips",
                "evidence_incident_snapshots",
                "evidence_retention_states",
                "derivative_evidence_slots",
            )
        )
        foreign_key_failures = connection.execute("PRAGMA foreign_key_check").fetchall()
        assert foreign_key_failures == []

    worker = open_runtime_database(database, actor=RuntimeActor.API)
    api = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="CHECK constraint failed"):
            api.execute(
                "INSERT INTO evidence_incidents "
                "(incident_id, edge_event_id, camera_id, event_type, detected_at, "
                "lifecycle_state, created_at, updated_at) "
                "VALUES ('i','e','c','fall','2026-08-13T00:00:00Z',"
                "'STAGING','2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')"
            )
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            worker.execute("ALTER TABLE evidence_incidents ADD COLUMN forbidden TEXT")
    finally:
        worker.close()
        api.close()


def test_backend_central_evidence_query_is_privacy_bounded(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_events "
            "(edge_event_id, detected_at, payload_json, state, queued_at, next_attempt_at) "
            "VALUES ('event:query','2026-08-13T00:00:00Z',?, 'STAGED',1,1)",
            ('{"facility_id":"private","snapshot_jpeg_base64":"private"}',),
        )
        connection.execute(
            """
            INSERT INTO evidence_incidents (
                incident_id, edge_event_id, camera_id, event_type, detected_at,
                provenance_missing_reason, lifecycle_state, created_at, updated_at
            ) VALUES ('incident:query','event:query','camera:opaque','fall',
                      '2026-08-13T00:00:00Z','NOT_RECORDED','STAGING',
                      '2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')
            """
        )
        connection.commit()

    summary = CentralEvidenceQuery(database).get("event:query")

    assert summary is not None
    assert summary.incident_id == "incident:query"
    assert summary.camera_id == "camera:opaque"
    assert summary.lifecycle_state == "STAGING"
    assert "private" not in repr(summary)
    assert not hasattr(summary, "payload_json")
    assert not hasattr(summary, "operator_only")



