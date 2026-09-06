from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.features.evidence.record_store import CentralEvidenceQuery

COMPACT_EVIDENCE_TABLES = ("incidents", "artifacts", "clips")
RETIRED_EVIDENCE_TABLES = (
    "evidence_events",
    "evidence_incidents",
    "evidence_media_objects",
    "evidence_artifact_slots",
    "evidence_primary_clips",
    "evidence_incident_snapshots",
    "evidence_retention_states",
    "derivative_evidence_slots",
)


def test_central_evidence_schema_has_owned_strict_records_and_integrity_guards(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)

    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert set(COMPACT_EVIDENCE_TABLES) <= tables.keys()
        assert set(RETIRED_EVIDENCE_TABLES).isdisjoint(tables)
        assert all("STRICT" in tables[name] for name in COMPACT_EVIDENCE_TABLES)
        foreign_key_failures = connection.execute("PRAGMA foreign_key_check").fetchall()
        assert foreign_key_failures == []

    api = open_runtime_database(database, actor=RuntimeActor.API)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="CHECK constraint failed"):
            api.execute(
                "INSERT INTO incidents "
                "(incident_id, edge_event_id, facility_id, camera_id, event_type, "
                "detected_at, lifecycle_state, provenance_state, "
                "provenance_missing_reason, review_version, revision, created_at, updated_at) "
                "VALUES ('i','e','f','c','fall','2026-08-13T00:00:00Z',"
                "'STAGING','MISSING','NOT_RECORDED',0,1,"
                "'2026-08-13T00:00:00Z','2026-08-13T00:00:00Z')"
            )
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            api.execute("ALTER TABLE incidents ADD COLUMN forbidden TEXT")
    finally:
        api.close()


def test_backend_central_evidence_query_is_privacy_bounded(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO incidents (
                incident_id, edge_event_id, facility_id, camera_id, event_type,
                probability, detected_at, lifecycle_state, provenance_state,
                provenance_missing_reason, review_version, revision, created_at, updated_at
            ) VALUES ('incident:query','event:query','private-facility','camera:opaque','fall',
                      0.8, '2026-08-13T00:00:00Z', 'OPEN', 'MISSING', 'NOT_RECORDED',
                      0, 1, '2026-08-13T00:00:00Z', '2026-08-13T00:00:00Z')
            """
        )
        connection.commit()

    summary = CentralEvidenceQuery(database).get("event:query")

    assert summary is not None
    assert summary.incident_id == "incident:query"
    assert summary.camera_id == "camera:opaque"
    assert summary.lifecycle_state == "OPEN"
    assert summary.schema_version == 18
    assert "private-facility" not in repr(summary)
    assert not hasattr(summary, "payload_json")
    assert not hasattr(summary, "operator_only")
    assert not hasattr(summary, "facility_id")
