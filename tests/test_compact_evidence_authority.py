from __future__ import annotations

import base64
import hashlib
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.configuration import open_configuration_database
from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras import store as camera_store_module
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.features.evidence.relay_projection import (
    RelayEvent,
    RelayEvidenceProjection,
    RelayEvidenceProjectionConflict,
    RelaySnapshot,
)
from backend.app.main import create_app, no_lifespan

EVENT_ID = "00000000-0000-4000-8000-000000000088"
TS = "2026-08-24T01:02:03Z"
SNAPSHOT_BYTES = b"jpeg-snapshot"


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "edge.sqlite3"
    migrate_database(path)
    return path


def _event(*, probability: float = 0.8) -> RelayEvent:
    return RelayEvent(
        edge_event_id=EVENT_ID,
        event_type="fall",
        probability=probability,
        detected_at=TS,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        evidence=None,
        audit=None,
    )


def _snapshot() -> RelaySnapshot:
    return RelaySnapshot(
        snapshot_id="snapshot-1",
        path="snapshots/camera-1/snapshot-1.jpg",
        sha256=hashlib.sha256(SNAPSHOT_BYTES).hexdigest(),
        size_bytes=len(SNAPSHOT_BYTES),
        mime_type="image/jpeg",
        captured_at=TS,
    )


def test_alert_and_inline_snapshot_commit_atomically_and_replay_idempotently(
    tmp_path: Path,
) -> None:
    # Given: a fresh compact database and one alert with inline snapshot metadata.
    database = _database(tmp_path)
    projection = RelayEvidenceProjection(database)

    # When: the exact alert is delivered twice.
    projection.project_event(_event(), _snapshot())
    projection.project_event(_event(), _snapshot())

    # Then: one incident and one available snapshot survive process restart.
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM incidents").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)
    summary = CentralEvidenceQuery(database).get(EVENT_ID)
    assert summary is not None
    assert summary.snapshot_artifact_state == "AVAILABLE"


def test_edge_event_id_replay_rejects_changed_identity(tmp_path: Path) -> None:
    # Given: an already committed edge_event_id.
    projection = RelayEvidenceProjection(_database(tmp_path))
    projection.project_event(_event())

    # When/Then: a changed immutable probability cannot reuse that key.
    with pytest.raises(RelayEvidenceProjectionConflict):
        projection.project_event(_event(probability=0.7))


def test_snapshot_database_failure_rolls_back_incident(tmp_path: Path) -> None:
    # Given: an artifact tuple whose MIME exceeds the locked schema boundary.
    database = _database(tmp_path)
    invalid = RelaySnapshot(
        snapshot_id="snapshot-1",
        path="snapshots/camera-1/snapshot-1.jpg",
        sha256=hashlib.sha256(SNAPSHOT_BYTES).hexdigest(),
        size_bytes=len(SNAPSHOT_BYTES),
        mime_type="x" * 129,
        captured_at=TS,
    )

    # When: the invalid artifact fails after incident insertion in the real transaction.
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        RelayEvidenceProjection(database).project_event(_event(), invalid)

    # Then: neither authority contains a partial fact.
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM incidents").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


def test_unmapped_relay_alert_is_locally_accepted_on_real_http_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a registered camera without a Hub mapping and a compact projection store.
    database = _database(tmp_path)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.relay_evidence_projection = RelayEvidenceProjection(database)
    ddl_attempts: list[int] = []

    def captured_open(path: Path) -> sqlite3.Connection:
        connection = open_configuration_database(path)

        def authorize(
            action: int,
            _argument_one: str | None,
            _argument_two: str | None,
            _database_name: str | None,
            _source: str | None,
        ) -> int:
            if action in {sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_ALTER_TABLE}:
                ddl_attempts.append(action)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(camera_store_module, "open_configuration_database", captured_open)
    registry = CameraRegistryStore(database)
    assert ddl_attempts == []
    registry.create(
        camera_id="camera-1",
        label="Camera 1",
        rtsp_url="rtsp://example/camera-1",
        space_id=None,
        status="online",
        backend_camera_id=None,
    )
    app.state.camera_registry = registry
    snapshot = _snapshot()
    payload = {
        "edge_event_id": EVENT_ID,
        "event_type": "fall",
        "probability": 0.8,
        "detected_at": TS,
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "snapshot_jpeg_base64": base64.b64encode(SNAPSHOT_BYTES).decode(),
        "snapshot": {
            "snapshot_id": snapshot.snapshot_id,
            "path": snapshot.path,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
            "mime_type": snapshot.mime_type,
            "captured_at": snapshot.captured_at,
            "camera_id": "camera-1",
            "edge_event_id": EVENT_ID,
        },
    }

    # When: the worker POSTs through the authenticated relay route.
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts",
            json=payload,
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    # Then: local atomic acceptance does not require an upstream camera id, and
    # it says so by name. The bare {"status": "accepted"} this used to pin was
    # unactionable for the sender: it needs a receipt echoing its edge_event_id,
    # an absent one is indistinguishable from a mangled response, so it retried
    # this event forever and every newer event queued behind it never left the
    # edge (#431). The backend is the only party that knows the push was
    # deliberately skipped, so the backend is the party that must state it.
    assert response.status_code == 202
    assert response.json() == {"status": "accepted_local", "edge_event_id": EVENT_ID}
    assert CentralEvidenceQuery(database).get(EVENT_ID) is not None
