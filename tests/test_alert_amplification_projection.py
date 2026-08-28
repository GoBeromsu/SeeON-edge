"""WP4 measurement: real B -> I incident staging and authenticated projection.

Media-free by construction. This drives the actual product composition
(``DurableEvidenceStager`` -> ``EvidenceOutbox.stage`` -> central incident
staging -> ``CentralEvidenceQuery`` -> authenticated ``GET /api/v1/incidents``)
on a disposable migrated edge database. No RTSP, no frames, no clip bytes, no
human adjudication, and no model/policy attribution.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.features.evidence.relay_projection import RelayEvidenceProjection
from backend.app.main import create_app, no_lifespan
from tests_support.alert_amplification_harness import (
    DiagnosticOutcome,
    IncidentProjection,
    classify_rows,
    rows_from_relations,
)
from tests_support.alert_amplification_runtime import RELAY_TOKEN, ServedFixture, relay_client
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager

_EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000b1"
_BACKEND_EVENT_ID = "d39d274b-5ecb-53f4-b892-74937e902c65"
_DETECTED_AT = "2026-08-16T00:00:00.000Z"


def _stager(queue_directory: Path) -> DurableEvidenceStager:
    return DurableEvidenceStager(
        queue_directory=queue_directory,
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
    bootstrap_database(database)
    return database


def _stage_and_deliver(database: Path, tmp_path: Path, edge_event_id: str = _EDGE_EVENT_ID) -> None:
    stager = _stager(tmp_path / "delivery-queue")
    stager.stage(_event(edge_event_id))
    entry = next(
        item for item in stager.queue.entries() if item["edge_event_id"] == edge_event_id
    )
    payload = json.loads(base64.b64decode(str(entry["values_b64"])))
    with ServedFixture() as served:
        relay = relay_client(served.origin, tmp_path, database=database)
        response = relay.post(
            "/api/v1/relay/alerts",
            json=payload,
            headers={"X-Edge-Relay-Token": RELAY_TOKEN},
        )
        assert response.status_code == 202, response.text


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


def test_idempotent_relay_redelivery_projects_one_incident_identity(tmp_path: Path) -> None:
    database = _migrated(tmp_path)
    _stage_and_deliver(database, tmp_path)
    _stage_and_deliver(database, tmp_path)

    incidents = _incidents_via_api(database)
    assert len(incidents) == 1
    projected = incidents[0]
    assert projected["edge_event_id"] == _EDGE_EVENT_ID
    assert projected["review"] is None


def test_measured_b_to_i_chain_classifies_healthy_convergence(tmp_path: Path) -> None:
    database = _migrated(tmp_path)
    _stage_and_deliver(database, tmp_path)
    _stage_and_deliver(database, tmp_path)

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
    _stage_and_deliver(database, tmp_path)

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
    _stage_and_deliver(database, tmp_path)
    [projected] = _incidents_via_api(database)

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="incidents.edge_event_id"):
            connection.execute(
                """
                INSERT INTO incidents (
                    incident_id, edge_event_id, facility_id, camera_id, event_type,
                    detected_at, lifecycle_state, provenance_state,
                    provenance_missing_reason, review_version, revision, created_at, updated_at
                ) VALUES (?, ?, 'facility-1', 'room-camera', 'fall', ?, 'OPEN',
                          'MISSING', 'NOT_RECORDED', 0, 1, ?, ?)
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


def test_snapshot_companions_bind_without_mutating_the_delivered_event(tmp_path: Path) -> None:
    database = _migrated(tmp_path)
    _stage_and_deliver(database, tmp_path)
    projection = RelayEvidenceProjection(database)

    projection.attach_snapshot(
        edge_event_id=_EDGE_EVENT_ID,
        snapshot_id="snapshot-1",
        sha256="a" * 64,
        media_reference="snapshots/snapshot-1.jpg",
        size_bytes=1,
        mime_type="image/jpeg",
    )
    projection.attach_snapshot(
        edge_event_id=_EDGE_EVENT_ID,
        snapshot_id="snapshot-1",
        sha256="a" * 64,
        media_reference="snapshots/snapshot-1.jpg",
        size_bytes=1,
        mime_type="image/jpeg",
    )

    second_event_id = "00000000-0000-4000-8000-0000000000b2"
    _stage_and_deliver(database, tmp_path, second_event_id)
    projection.record_snapshot_disposition(
        edge_event_id=second_event_id,
        snapshot_id="snapshot-2",
        disposition="UNAVAILABLE",
        reason="capture_failed",
    )

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT review_version, revision FROM incidents WHERE edge_event_id = ?",
            (_EDGE_EVENT_ID,),
        ).fetchone() == (0, 1)
        assert connection.execute(
            "SELECT state FROM artifacts WHERE incident_id = ? AND kind = 'SNAPSHOT'",
            (f"incident:{_EDGE_EVENT_ID}",),
        ).fetchone() == ("AVAILABLE",)
        assert connection.execute(
            "SELECT state, reason FROM artifacts "
            "WHERE incident_id = ? AND kind = 'SNAPSHOT'",
            (f"incident:{second_event_id}",),
        ).fetchone() == ("UNAVAILABLE", "UNAVAILABLE:capture_failed")
