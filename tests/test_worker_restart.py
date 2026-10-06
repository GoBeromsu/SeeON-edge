from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox
from worker.runtime.config.restart import RestartDirective, RestartDirectiveTracker

pytest_plugins = ("tests_support.postgres_sandbox",)

_FACILITY_ID = "facility/restart-acceptance"
_CAMERA_ID = "camera/restart-acceptance"
_RELAY_HEADERS = {"X-Edge-Relay-Token": "restart-relay-token"}


def _directive(config: dict[str, object]) -> RestartDirective:
    return RestartDirective(
        generation=int(config.get("restart_epoch", 0)),
        version=int(config.get("config_version", 0)),
    )


def _activation_status(client: TestClient) -> str:
    response = client.get("/api/v1/detection-policies")
    assert response.status_code == 200
    activations = response.json()["activations"]
    assert len(activations) == 1
    return str(activations[0]["status"])


def test_policy_restart_requires_matching_worker_ack_before_pending_becomes_applied(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.edge_relay_token = "restart-relay-token"
    app.state.connection_settings_store.save(
        {
            "facility_code": "NH-RESTART01",
            "client_installation_ref": "restart-installation",
            "facility_id": _FACILITY_ID,
            "facility_token": "facility-secret-not-returned",
            "edge_installation_id": "edge/restart-acceptance",
            "enrollment_generation": 1,
        }
    )
    app.state.camera_registry.create(
        camera_id="local-restart-camera",
        label="Restart acceptance camera",
        rtsp_url="rtsp://camera.invalid/restart",
        space_id=None,
        status="offline",
        backend_camera_id=_CAMERA_ID,
    )

    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/session",
                json={"username": "admin", "password": "admin"},
            ).status_code
            == 204
        )
        before_response = client.get("/api/v1/cameras/worker-config", headers=_RELAY_HEADERS)
        assert before_response.status_code == 200
        before = _directive(before_response.json())

        applied = client.post(
            "/api/v1/detection-policies/apply",
            json={
                "module_id": "fall",
                "module_version": 2,
                "schema_id": "fall.policy",
                "schema_version": 2,
                "camera_id": None,
                "expected_revision_id": 0,
                "values": {"transition_threshold": 0.73},
            },
        )
        assert applied.status_code == 202
        assert applied.json()["status"] == "pending"

        pending_config_response = client.get(
            "/api/v1/cameras/worker-config", headers=_RELAY_HEADERS
        )
        assert pending_config_response.status_code == 200
        pending_config = pending_config_response.json()
        pending = _directive(pending_config)
        tracker = RestartDirectiveTracker(before)
        assert tracker.observe(pending) is True
        assert tracker.current == pending
        assert _activation_status(client) == "pending"

        stale_ack = client.post(
            "/api/v1/relay/heartbeat",
            headers=_RELAY_HEADERS,
            json={
                "camera_id": _CAMERA_ID,
                "facility_id": _FACILITY_ID,
                "config_version": before.version,
            },
        )
        assert stale_ack.status_code == 202
        assert _activation_status(client) == "pending"

        matching_ack = client.post(
            "/api/v1/relay/heartbeat",
            headers=_RELAY_HEADERS,
            json={
                "camera_id": _CAMERA_ID,
                "facility_id": _FACILITY_ID,
                "config_version": pending.version,
            },
        )
        assert matching_ack.status_code == 202
        assert _activation_status(client) == "applied"

        unchanged = client.get("/api/v1/cameras/worker-config", headers=_RELAY_HEADERS).json()
        assert _directive(unchanged) == pending
        assert tracker.observe(_directive(unchanged)) is False
