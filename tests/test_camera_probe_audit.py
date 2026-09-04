from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.store import AuditEvent, AuditRecord, AuditStore
from backend.app.features.cameras.camera_values import ProbeResult
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan


def _database_path() -> Path:
    from backend.app.features.audit import store

    return store.EDGE_DATABASE_PATH


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def test_camera_probe_persisted_outcomes_append_exactly_one_typed_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(lifespan=no_lifespan)
    store = app.state.camera_registry = CameraRegistryStore(_database_path())
    store.create(
        camera_id="camera-a",
        label="A",
        rtsp_url="rtsp://camera.example/live",
        space_id=None,
        status="offline",
    )
    outcomes = iter((ProbeResult(True, width=640, height=480), ProbeResult(False, "timeout")))
    monkeypatch.setattr(
        "backend.app.features.cameras.router._probe_rtsp_url",
        lambda *_args: next(outcomes),
    )
    with TestClient(app) as client:
        _login(client)
        before = _action_count(AuditAction.CAMERA_PROBE)
        online = client.post("/api/v1/cameras/camera-a/test")
        middle = _action_count(AuditAction.CAMERA_PROBE)
        offline = client.post("/api/v1/cameras/camera-a/test")
        after = _action_count(AuditAction.CAMERA_PROBE)

    assert online.status_code == offline.status_code == 200
    assert (middle - before, after - middle) == (1, 1)
    with sqlite3.connect(_database_path()) as connection:
        details = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT detail_json FROM audit_events WHERE action='camera.probe' ORDER BY audit_id"
            )
        ]
    assert details == [
        {"error_class": None, "ok": True, "version": 1},
        {"error_class": "timeout", "ok": False, "version": 1},
    ]


def test_camera_probe_error_without_persistence_appends_no_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(lifespan=no_lifespan)
    store = app.state.camera_registry = CameraRegistryStore(_database_path())
    store.create(
        camera_id="camera-a",
        label="A",
        rtsp_url="rtsp://camera.example/live",
        space_id=None,
        status="offline",
    )
    calls = 0

    def unavailable(*_args):
        nonlocal calls
        calls += 1
        raise HTTPException(status_code=503, detail="probe unavailable")

    monkeypatch.setattr("backend.app.features.cameras.router._probe_rtsp_url", unavailable)
    with TestClient(app) as client:
        _login(client)
        before = store.get("camera-a")
        before_count = _action_count(AuditAction.CAMERA_PROBE)
        failed = client.post("/api/v1/cameras/camera-a/test")
        unauthorized = TestClient(app).post("/api/v1/cameras/camera-a/test")

    assert failed.status_code == 503
    assert unauthorized.status_code == 401
    assert calls == 1
    assert store.get("camera-a") == before
    assert _action_count(AuditAction.CAMERA_PROBE) == before_count


class _DenyAuditInsertStore(AuditStore):
    def _append(self, connection: sqlite3.Connection, event: AuditEvent) -> AuditRecord:
        def authorize(
            action: int,
            arg1: str | None,
            _arg2: str | None,
            _database: str | None,
            _source: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_INSERT and arg1 == "audit_events":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return super()._append(connection, event)


def test_camera_probe_audit_denial_rolls_back_persisted_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(lifespan=no_lifespan)
    store = app.state.camera_registry = CameraRegistryStore(_database_path())
    store.create(
        camera_id="camera-a",
        label="A",
        rtsp_url="rtsp://camera.example/live",
        space_id=None,
        status="offline",
    )
    monkeypatch.setattr(
        "backend.app.features.cameras.router._probe_rtsp_url",
        lambda *_args: ProbeResult(True, width=640, height=480),
    )
    with TestClient(app) as client:
        _login(client)
        before = store.get("camera-a")
        before_count = _action_count(AuditAction.CAMERA_PROBE)
        app.state.audit_store = _DenyAuditInsertStore(_database_path())
        response = client.post("/api/v1/cameras/camera-a/test")

    assert (response.status_code, response.content) == (503, b"")
    assert "set-cookie" not in response.headers
    assert store.get("camera-a") == before
    assert _action_count(AuditAction.CAMERA_PROBE) == before_count


def _action_count(action: AuditAction) -> int:
    with sqlite3.connect(_database_path()) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action=?", (action.value,)
        ).fetchone()[0]


def test_fail_open_heartbeat_does_not_invoke_audit_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = AuditStore.verify

    def counted(self: AuditStore, checkpoint=None):
        nonlocal calls
        calls += 1
        return original(self, checkpoint)

    monkeypatch.setattr(AuditStore, "verify", counted)
    app = create_app(lifespan=no_lifespan)
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
