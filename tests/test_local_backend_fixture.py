from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from contracts.edge_provisioning_enrollment import parse_enrollment_verification_result
from contracts.edge_provisioning_response import parse_topology_success_envelope
from tests_support.local_backend_fixture import LocalBackendFixture

_AUTH = {"Authorization": "Bearer fixture-token"}
_SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de801"
_UNKNOWN_SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de802"


def _event(edge_event_id: str = "edge-event-1") -> dict[str, object]:
    return {
        "edge_event_id": edge_event_id,
        "camera_id": "room-camera",
        "type": "fall",
        "detected_at": "2026-08-16T00:00:00Z",
        "confidence": 0.9,
    }


def _topology_body(fixture: LocalBackendFixture) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "edgeInstallationId": fixture.edge_installation_id,
        "enrollmentGeneration": 1,
        "clientRevision": 1,
        "expectedServerRevision": 0,
        "floors": [],
    }


def test_enrollment_and_config_match_current_client_contracts() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)

    enrollment = client.post(
        "/api/v1/edge/enrollments/verify",
        headers=_AUTH,
        json={
            "schemaVersion": 1,
            "facilityCode": "NH-7H2K9M4QXP",
            "clientInstallationRef": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
        },
    )
    assert enrollment.status_code == 200
    parsed = parse_enrollment_verification_result(enrollment.json())
    assert parsed.principal.edge_installation_id == fixture.edge_installation_id
    assert parsed.facility.facility_id == fixture.facility_id

    config = client.get(f"/api/v1/ml-config/{fixture.facility_id}", headers=_AUTH)
    assert config.status_code == 200
    assert config.json() == {"configVersion": 1, "detectionWindows": {}, "cameras": []}


def test_auth_is_required_without_retaining_token() -> None:
    fixture = LocalBackendFixture(bearer_token="do-not-retain-this-token")
    client = TestClient(fixture.app)

    response = client.get(
        f"/api/v1/ml-config/{fixture.facility_id}",
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401
    assert "do-not-retain-this-token" not in repr(fixture.route_ledger)


def test_capabilities_and_heartbeat_are_bounded() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)

    capabilities = client.get(
        "/api/v1/events/capabilities",
        headers=_AUTH,
        params={"camera_id": "room-camera"},
    )
    heartbeat = client.post(
        "/api/v1/events/heartbeat", headers=_AUTH, json={"camera_id": "room-camera"}
    )
    oversized = client.post(
        "/api/v1/events/heartbeat",
        headers={**_AUTH, "Content-Type": "application/json"},
        content=b'{"camera_id":"' + b"x" * (256 * 1024) + b'"}',
    )

    assert capabilities.json() == {"event_idempotency": 1, "clip_export": 0}
    assert heartbeat.status_code == 200
    assert heartbeat.json() == {"ok": True}
    assert oversized.status_code == 413


def test_event_replay_is_stable_and_conflicting_replay_is_rejected() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)

    first = client.post("/api/v1/events", headers=_AUTH, json=_event())
    replay = client.post("/api/v1/events", headers=_AUTH, json=_event())
    conflict_body = _event()
    conflict_body["type"] = "bed-exit"
    conflict = client.post("/api/v1/events", headers=_AUTH, json=conflict_body)

    assert first.status_code == replay.status_code == 201
    assert first.json() == replay.json()
    assert conflict.status_code == 409
    assert fixture.event_for_edge_id("edge-event-1") is not None


def test_event_contract_uses_frozen_wire_vocabulary() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)
    unknown = {**_event(), "unexpected": True}
    invalid_type = {**_event(), "edge_event_id": "edge-invalid", "type": "bed_exit"}
    bed_exit = {**_event(), "edge_event_id": "edge-bed", "type": "bed-exit"}
    detection_lost = {
        **_event(),
        "edge_event_id": "edge-lost",
        "type": "detection-lost",
    }

    assert client.post("/api/v1/events", headers=_AUTH, json=unknown).status_code == 422
    assert (
        client.post("/api/v1/events", headers=_AUTH, json=invalid_type).status_code
        == 422
    )
    assert client.post("/api/v1/events", headers=_AUTH, json=bed_exit).status_code == 201
    assert (
        client.post("/api/v1/events", headers=_AUTH, json=detection_lost).status_code
        == 201
    )


def test_fault_mode_exposes_hub_identity_multiplication() -> None:
    fixture = LocalBackendFixture(faulty_event_identity=True)
    client = TestClient(fixture.app)

    first = client.post("/api/v1/events", headers=_AUTH, json=_event()).json()
    replay = client.post("/api/v1/events", headers=_AUTH, json=_event()).json()

    assert first["edge_event_id"] == replay["edge_event_id"]
    assert first["event_id"] != replay["event_id"]


def test_snapshot_is_discarded_and_has_no_read_surface() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)
    receipt = client.post("/api/v1/events", headers=_AUTH, json=_event()).json()

    response = client.put(
        f"/api/v1/events/{receipt['event_id']}/snapshot",
        headers={**_AUTH, "Content-Type": "image/jpeg"},
        content=b"\xff\xd8fixture\xff\xd9",
    )
    read_back = client.get(f"/api/v1/events/{receipt['event_id']}/snapshot", headers=_AUTH)

    assert response.status_code == 201
    assert response.json()["snapshotKey"].endswith("/snapshot.jpg")
    assert fixture.snapshot_bytes_discarded == 11
    assert fixture.retained_media_bytes == 0
    assert read_back.status_code == 405


def test_snapshot_rejects_unknown_event_type_and_size() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)

    unknown = client.put(
        f"/api/v1/events/{uuid4()}/snapshot",
        headers={**_AUTH, "Content-Type": "image/jpeg"},
        content=b"jpeg",
    )
    receipt = client.post("/api/v1/events", headers=_AUTH, json=_event()).json()
    wrong_type = client.put(
        f"/api/v1/events/{receipt['event_id']}/snapshot",
        headers={**_AUTH, "Content-Type": "image/png"},
        content=b"png",
    )
    too_large = client.put(
        f"/api/v1/events/{receipt['event_id']}/snapshot",
        headers={**_AUTH, "Content-Type": "image/jpeg"},
        content=b"x" * (512 * 1024 + 1),
    )

    assert unknown.status_code == 404
    assert wrong_type.status_code == 415
    assert too_large.status_code == 413
    assert fixture.retained_media_bytes == 0


def test_topology_snapshot_and_confirmation_are_deterministic() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)
    snapshot_id = _SNAPSHOT_ID

    snapshot = client.put(
        f"/api/v1/edge/topology-snapshots/{snapshot_id}",
        headers=_AUTH,
        json=_topology_body(fixture),
    )
    parsed = parse_topology_success_envelope(snapshot.json())
    confirmation = client.post(
        f"/api/v1/edge/topology-snapshots/{snapshot_id}/confirm",
        headers=_AUTH,
        json={
            "schemaVersion": 1,
            "confirmationId": snapshot.json()["omissions"]["confirmationId"],
            "digest": snapshot.json()["omissions"]["digest"],
            "expectedServerRevision": parsed.server_revision,
        },
    )

    assert snapshot.status_code == confirmation.status_code == 200
    assert confirmation.json()["snapshotId"] == snapshot_id
    assert confirmation.json()["serverRevision"] == parsed.server_revision


def test_topology_exact_replay_is_stable_and_changed_replay_conflicts() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)
    path = f"/api/v1/edge/topology-snapshots/{_SNAPSHOT_ID}"
    body = _topology_body(fixture)

    first = client.put(path, headers=_AUTH, json=body)
    replay = client.put(path, headers=_AUTH, json=body)
    changed = client.put(
        path,
        headers=_AUTH,
        json={**body, "clientRevision": 2},
    )

    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert changed.status_code == 409


def test_topology_confirmation_rejects_wrong_digest() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)
    snapshot = client.put(
        f"/api/v1/edge/topology-snapshots/{_SNAPSHOT_ID}",
        headers=_AUTH,
        json=_topology_body(fixture),
    ).json()

    response = client.post(
        f"/api/v1/edge/topology-snapshots/{_SNAPSHOT_ID}/confirm",
        headers=_AUTH,
        json={
            "schemaVersion": 1,
            "confirmationId": snapshot["omissions"]["confirmationId"],
            "digest": "0" * 64,
            "expectedServerRevision": snapshot["serverRevision"],
        },
    )

    assert response.status_code == 409


def test_topology_rejects_malformed_or_unknown_confirmation() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)
    snapshot_id = _SNAPSHOT_ID

    malformed = client.put(
        f"/api/v1/edge/topology-snapshots/{snapshot_id}",
        headers=_AUTH,
        json={**_topology_body(fixture), "edgeInstallationId": str(uuid4())},
    )
    unknown = client.post(
        f"/api/v1/edge/topology-snapshots/{_UNKNOWN_SNAPSHOT_ID}/confirm",
        headers=_AUTH,
        json={
            "schemaVersion": 1,
            "confirmationId": _UNKNOWN_SNAPSHOT_ID,
            "digest": "0" * 64,
            "expectedServerRevision": 1,
        },
    )

    assert malformed.status_code == 422
    assert unknown.status_code == 404


def test_obsolete_and_unknown_routes_do_not_mask_drift() -> None:
    fixture = LocalBackendFixture()
    client = TestClient(fixture.app)

    assert client.post("/v1/events", headers=_AUTH, json=_event()).status_code == 404
    assert client.get(f"/v1/ml-config/{fixture.facility_id}", headers=_AUTH).status_code == 404
    assert client.get("/api/v1/clips/secret/video", headers=_AUTH).status_code == 404
    assert [record.path for record in fixture.route_ledger] == [
        "/v1/events",
        f"/v1/ml-config/{fixture.facility_id}",
        "/api/v1/clips/secret/video",
    ]


def test_route_ledger_retains_metadata_only() -> None:
    fixture = LocalBackendFixture(bearer_token="sensitive-token")
    client = TestClient(fixture.app)
    body = _event("privacy-event")
    body["camera_id"] = "corridor-camera"

    response = client.post(
        "/api/v1/events",
        headers={"Authorization": "Bearer sensitive-token"},
        json=body,
    )

    assert response.status_code == 201
    ledger_text = repr(fixture.route_ledger)
    assert "sensitive-token" not in ledger_text
    assert "corridor-camera" not in ledger_text
    assert "privacy-event" not in ledger_text
    assert fixture.route_ledger[-1].path == "/api/v1/events"
