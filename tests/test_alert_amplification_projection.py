"""WP4 measurement: real B -> I incident staging and authenticated projection.

Media-free by construction. This drives the actual product composition
(``DurableEvidenceStager`` -> ``EvidenceOutbox.stage`` -> central incident
staging -> ``CentralEvidenceQuery`` -> authenticated ``GET /api/v1/incidents``)
on a disposable migrated edge database. No RTSP, no frames, no clip bytes, no
human adjudication, and no model/policy attribution.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.main import create_app, no_lifespan
from shared.edge_db.migrator import migrate_database
from tests_support.alert_amplification_harness import (
    DiagnosticOutcome,
    IncidentProjection,
    classify_rows,
    rows_from_relations,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager

_EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000b1"
_BACKEND_EVENT_ID = "d39d274b-5ecb-53f4-b892-74937e902c65"
_DETECTED_AT = "2026-08-16T00:00:00.000Z"


def _stager(database: Path) -> DurableEvidenceStager:
    return DurableEvidenceStager(
        database_path=database,
        camera_id="room-camera",
        facility_id="facility-1",
        resident_id=None,
        config_version=1,
        clock=lambda: 1.0,
    )


def _event(edge_event_id: str = _EDGE_EVENT_ID) -> dict[str, object]:
    return {
        "edge_event_id": edge_event_id,
        "event_type": "fall",
        "probability": 0.91,
        "detected_at": _DETECTED_AT,
        "camera_id": "room-camera",
        "facility_id": "facility-1",
    }


def _migrated(tmp_path: Path) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    return database


def _incidents_via_api(database: Path) -> list[dict[str, object]]:
    app = create_app(lifespan=no_lifespan)
    app.state.central_evidence_query = CentralEvidenceQuery(database)
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
            ).status_code
            == 204
        )
        first = client.get("/api/v1/incidents")
        assert first.status_code == 200, first.text
        # Repeated polling must not mint a second incident identity.
        second = client.get("/api/v1/incidents")
        assert second.status_code == 200
        assert first.json() == second.json()
        return list(first.json()["incidents"])


def test_duplicate_staging_projects_one_incident_identity(tmp_path: Path) -> None:
    database = _migrated(tmp_path)
    stager = _stager(database)

    stager.stage(_event())
    stager.stage(_event())

    incidents = _incidents_via_api(database)
    assert len(incidents) == 1
    projected = incidents[0]
    assert projected["edge_event_id"] == _EDGE_EVENT_ID
    assert projected["review"] is None


def test_measured_b_to_i_chain_classifies_healthy_convergence(tmp_path: Path) -> None:
    database = _migrated(tmp_path)
    stager = _stager(database)
    stager.stage(_event())
    stager.stage(_event())

    incidents = _incidents_via_api(database)
    projections = [
        IncidentProjection(
            str(item["incident_id"]),
            str(item["edge_event_id"]),
            str(item["detected_at"]),
            str(item["lifecycle_state"]),
            None if item["event_delivery_state"] is None else str(item["event_delivery_state"]),
            None,
        )
        for item in incidents
    ]
    rows = rows_from_relations(
        transitions={_EDGE_EVENT_ID: "transition-1"},
        attempts={_EDGE_EVENT_ID: [1, 2]},
        backend_event_ids={_EDGE_EVENT_ID: [_BACKEND_EVENT_ID]},
        incidents=projections,
        terminal_states={_EDGE_EVENT_ID: "ACKED"},
        clock_order_valid=True,
    )

    assert len(rows) == 1
    assert rows[0].incident_ids == (projections[0].incident_id,)
    assert classify_rows(rows).outcome is DiagnosticOutcome.TRANSPORT_RETRY


def test_api_projection_lacks_a_projection_timestamp_field(tmp_path: Path) -> None:
    """Measured finite gap, recorded rather than fabricated."""

    database = _migrated(tmp_path)
    _stager(database).stage(_event())

    [projected] = _incidents_via_api(database)

    assert "projection_timestamp" not in projected
    assert "detected_at" in projected


def test_incident_multiplication_is_structurally_impossible(tmp_path: Path) -> None:
    """The falsifier cannot even be staged: the schema forbids two I for one E.

    This is stronger than detecting duplication after the fact — one
    ``edge_event_id`` can never own two incident identities, so API incident
    amplification is ruled out by construction rather than by observation.
    """

    database = _migrated(tmp_path)
    _stager(database).stage(_event())
    [projected] = _incidents_via_api(database)

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="evidence_incidents.edge_event_id"):
            connection.execute(
                """
                INSERT INTO evidence_incidents (
                    incident_id, edge_event_id, camera_id, event_type, detected_at,
                    provenance_missing_reason, lifecycle_state, created_at, updated_at
                ) VALUES (?, ?, 'room-camera', 'fall', ?, 'NOT_RECORDED', 'STAGING', ?, ?)
                """,
                (
                    f"{projected['incident_id']}-duplicate",
                    _EDGE_EVENT_ID,
                    _DETECTED_AT,
                    _DETECTED_AT,
                    _DETECTED_AT,
                ),
            )

    assert len(_incidents_via_api(database)) == 1
