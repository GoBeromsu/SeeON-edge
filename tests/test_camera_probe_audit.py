from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.cameras.camera_values import ProbeResult
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def _app_with_camera(sandbox: ProductSandbox, audit_runtime: PostgresAuditRuntime):
    app = postgres_api_app(sandbox, audit_runtime)
    app.state.camera_registry.create(
        camera_id="camera-a",
        label="A",
        rtsp_url="rtsp://camera.example/live",
        space_id=None,
        status="offline",
    )
    return app


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def _action_count(sandbox: ProductSandbox, action: AuditAction) -> int:
    return sandbox.admin.execute(
        "SELECT COUNT(*) FROM audit_events WHERE action=%s", (action.value,)
    ).fetchone()[0]


def test_camera_probe_persisted_outcomes_append_exactly_one_typed_audit(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = postgres_product_sandbox
    app = _app_with_camera(sandbox, postgres_audit_runtime)
    outcomes = iter((ProbeResult(True, width=640, height=480), ProbeResult(False, "timeout")))
    monkeypatch.setattr(
        "backend.app.features.cameras.router._probe_rtsp_url",
        lambda *_args: next(outcomes),
    )
    with TestClient(app) as client:
        _login(client)
        before = _action_count(sandbox, AuditAction.CAMERA_PROBE)
        online = client.post("/api/v1/cameras/camera-a/test")
        middle = _action_count(sandbox, AuditAction.CAMERA_PROBE)
        offline = client.post("/api/v1/cameras/camera-a/test")
        after = _action_count(sandbox, AuditAction.CAMERA_PROBE)

    assert online.status_code == offline.status_code == 200
    assert (middle - before, after - middle) == (1, 1)
    details = [
        json.loads(row[0])
        for row in sandbox.admin.execute(
            "SELECT detail_json FROM audit_events WHERE action=%s ORDER BY audit_id",
            (AuditAction.CAMERA_PROBE.value,),
        )
    ]
    assert details == [
        {"error_class": None, "ok": True, "version": 1},
        {"error_class": "timeout", "ok": False, "version": 1},
    ]


def test_camera_probe_error_without_persistence_appends_no_success(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = postgres_product_sandbox
    app = _app_with_camera(sandbox, postgres_audit_runtime)
    store = app.state.camera_registry
    calls = 0

    def unavailable(*_args):
        nonlocal calls
        calls += 1
        raise HTTPException(status_code=503, detail="probe unavailable")

    monkeypatch.setattr("backend.app.features.cameras.router._probe_rtsp_url", unavailable)
    with TestClient(app) as client:
        _login(client)
        before = store.get("camera-a")
        before_count = _action_count(sandbox, AuditAction.CAMERA_PROBE)
        failed = client.post("/api/v1/cameras/camera-a/test")
        unauthorized = TestClient(app).post("/api/v1/cameras/camera-a/test")

    assert failed.status_code == 503
    assert unauthorized.status_code == 401
    assert calls == 1
    assert store.get("camera-a") == before
    assert _action_count(sandbox, AuditAction.CAMERA_PROBE) == before_count


def test_camera_probe_audit_denial_rolls_back_persisted_outcome(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = postgres_product_sandbox
    app = _app_with_camera(sandbox, postgres_audit_runtime)
    store = app.state.camera_registry
    monkeypatch.setattr(
        "backend.app.features.cameras.router._probe_rtsp_url",
        lambda *_args: ProbeResult(True, width=640, height=480),
    )
    with TestClient(app) as client:
        _login(client)
        before = store.get("camera-a")
        before_count = _action_count(sandbox, AuditAction.CAMERA_PROBE)
        sandbox.admin.execute(
            "CREATE FUNCTION reject_audit_test() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'injected audit failure'; END $$"
        )
        sandbox.admin.execute(
            "CREATE TRIGGER reject_audit_test BEFORE INSERT ON audit_events "
            "FOR EACH ROW EXECUTE FUNCTION reject_audit_test()"
        )
        response = client.post("/api/v1/cameras/camera-a/test")

    assert (response.status_code, response.content) == (503, b"")
    assert "set-cookie" not in response.headers
    assert store.get("camera-a") == before
    assert _action_count(sandbox, AuditAction.CAMERA_PROBE) == before_count


def test_fail_open_heartbeat_does_not_invoke_audit_verification(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = PostgresAuditStore.verify

    def counted(self: PostgresAuditStore, checkpoint=None):
        nonlocal calls
        calls += 1
        return original(self, checkpoint)

    monkeypatch.setattr(PostgresAuditStore, "verify", counted)
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.edge_relay_token = "relay-token"
    with TestClient(app) as client:
        responses = tuple(
            client.post(
                "/api/v1/relay/heartbeat",
                json={"camera_id": "camera-a", "facility_id": "facility-a"},
                headers={"X-Edge-Relay-Token": "relay-token"},
            )
            for _ in range(20)
        )

    assert all(response.status_code in {202, 403} for response in responses)
    assert calls == 0


def _wired_actions(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "AuditAction"
    }


def test_camera_probe_production_wiring_is_covered_and_mutation_sensitive() -> None:
    from backend.app.features.cameras import router

    source = Path(router.__file__).read_text(encoding="utf-8")
    governed = {
        "AUTH_LOGIN",
        "AUTH_SESSION_READ",
        "AUTH_LOGOUT",
        "CREDENTIAL_ROTATE",
        "CAMERA_CREATE",
        "CAMERA_UPDATE",
        "CAMERA_DELETE",
        "CAMERA_PROBE",
        "LOCATION_CREATE",
        "LOCATION_UPDATE",
        "LOCATION_DELETE",
        "BED_ZONE_UPDATE",
        "CONNECTION_UPDATE",
        "CLIP_STORAGE_UPDATE",
        "DETECTION_SETTINGS_UPDATE",
        "RUNTIME_SETTINGS_UPDATE",
        "POLICY_APPLY",
        "POLICY_ROLLBACK",
        "INCIDENT_LIST",
        "INCIDENT_DETAIL",
        "INCIDENT_REVIEW",
        "CLIP_LIST",
        "CLIP_DETAIL",
        "CLIP_PLAY",
        "CLIP_THUMBNAIL",
        "CLIP_ARTIFACT",
        "AUDIT_LIST",
        "AUDIT_DETAIL",
        "RELAY_ALERT",
    }
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path(router.__file__).parents[1]).rglob("*.py")
        if path.name != "catalog.py"
    )
    assert governed <= _wired_actions(production)
    mutated = source.replace("AuditAction.CAMERA_PROBE", "AuditAction.CAMERA_UPDATE")
    assert "CAMERA_PROBE" not in _wired_actions(mutated)
