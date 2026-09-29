"""Real native mutation HTTP paths; explicit DI, not lifespan/login qualification."""

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import AuditAction
from backend.app.features.cameras.bed_zone_store import BedZoneRegion, BedZoneStore
from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.store import CameraRegistryStore, ProbeResult
from backend.app.features.clips.storage_location_store import ClipStorageLocationStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.connection.topology_retry_coordinator import TopologyRetryCoordinator
from backend.app.features.detection_settings.policy_store import DetectionPolicyStore
from backend.app.features.detection_settings.store import DetectionSettingsStore
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.main import create_app, no_lifespan
from backend.app.shared.dashboard_auth import (
    DASHBOARD_SESSION_COOKIE,
    DashboardSessionStore,
    PlaintextDashboardCredentials,
)
from shared.rtsp_url_policy import ALLOW_PRIVATE_RTSP_ENV

pytest_plugins = ("tests_support.postgres_sandbox",)


@pytest.fixture
def setup(postgres_product_sandbox, postgres_audit_runtime, tmp_path, monkeypatch):
    sandbox, runtime = postgres_product_sandbox, postgres_audit_runtime
    monkeypatch.setenv(ALLOW_PRIVATE_RTSP_ENV, "1")
    app = create_app(lifespan=no_lifespan)
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    bed = BedZoneStore(sandbox.database, sandbox.authority)
    app.state.camera_registry = registry
    app.state.bed_zone_store = bed
    app.state.runtime_settings_store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    app.state.clip_storage_location_store = ClipStorageLocationStore(
        sandbox.database, sandbox.authority
    )
    app.state.detection_settings_store = DetectionSettingsStore(sandbox.database, sandbox.authority)
    app.state.detection_policy_store = DetectionPolicyStore(sandbox.database, sandbox.authority)
    app.state.connection_settings_store = ConnectionSettingsStore(
        sandbox.database, sandbox.authority
    )
    app.state.audit_runtime = runtime
    app.state.topology_retry_coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(sandbox.database, sandbox.authority), lambda: None
    )
    sessions = DashboardSessionStore(PlaintextDashboardCredentials("operator-1", "test-password"))
    token = sessions.authenticate("operator-1", "test-password")
    assert token is not None
    app.state.dashboard_sessions = sessions
    root = tmp_path / "clips"
    (root / "room").mkdir(parents=True)
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    from backend.app.features.cameras import router as cameras

    monkeypatch.setattr(cameras, "_probe_rtsp_url", lambda request, url: ProbeResult(ok=True))

    def reject_sqlite(*args, **kwargs):
        pytest.fail("native mutation opened a SQLite owner")

    with TestClient(app) as client, monkeypatch.context() as guard:
        guard.setattr(sqlite3, "connect", reject_sqlite)
        client.cookies.set(DASHBOARD_SESSION_COOKIE, token)
        yield SimpleNamespace(
            sandbox=sandbox,
            runtime=runtime,
            app=app,
            client=client,
            registry=registry,
            bed=bed,
            root=root,
        )


def _actions(setup):
    return setup.sandbox.admin.execute(
        "SELECT action,target_id FROM audit_events WHERE action <> %s ORDER BY audit_id",
        (AuditAction.AUDIT_SESSION_START,),
    ).fetchall()


def _create(setup, camera_id="camera-1", *, backend_id=None):
    return setup.registry.create(
        camera_id=camera_id,
        label="Before",
        rtsp_url=f"rtsp://10.0.0.10/{camera_id}",
        space_id=None,
        status="unknown",
        backend_camera_id=backend_id,
    )


def _zone(setup, camera_id):
    return setup.bed.put(
        camera_id,
        regions=(BedZoneRegion(camera_id + "-bed", ((0, 0), (10, 0), (0, 10)), "manual"),),
        image_width=20,
        image_height=20,
        recognized_at="2026-09-27T00:00:00Z",
    )


@pytest.mark.parametrize("kind", ["floor", "room"])
def test_location_mutations_publish_only_their_declared_successes(setup, kind):
    if kind == "room":
        setup.registry.create_floor(edge_ref="parent", name="Parent", order_index=1)
    path = f"/api/v1/cameras/topology/{kind}s"
    payload = {"edge_ref": "target", "name": "Original"}
    payload.update({"order_index": 1} if kind == "floor" else {"floor_edge_ref": "parent"})
    assert setup.client.post(path, json=payload).status_code == 201
    update = {"name": "Changed"}
    if kind == "floor":
        update["order_index"] = 2
    assert setup.client.patch(path + "/target", json=update).status_code == 200
    assert setup.client.patch(path + "/absent", json=update).status_code == 404
    assert setup.client.delete(path + "/target").status_code == 204
    assert setup.client.delete(path + "/target").status_code == 404
    assert _actions(setup) == [
        (AuditAction.LOCATION_CREATE, "target"),
        (AuditAction.LOCATION_UPDATE, "target"),
        (AuditAction.LOCATION_DELETE, "target"),
    ]
    assert not setup.runtime._pending


def test_location_conflict_has_no_success_audit(setup):
    setup.registry.create_floor(edge_ref="floor", name="First", order_index=1)
    setup.registry.create_room(edge_ref="room", floor_edge_ref="floor", name="Room")
    response = setup.client.delete("/api/v1/cameras/topology/floors/floor")
    assert response.status_code == 409 and _actions(setup) == []
    assert setup.registry.topology_snapshot().floors[0].edge_ref == "floor"


def test_camera_create_update_duplicate_and_missing_paths(setup):
    path = "/api/v1/cameras"
    created = setup.client.post(path, json={"label": "First", "rtsp_url": "rtsp://10.0.0.10/live"})
    assert created.status_code == 201
    camera_id = created.json()["id"]
    assert setup.client.patch(f"{path}/{camera_id}", json={"label": "Changed"}).status_code == 200
    assert setup.client.patch(f"{path}/{camera_id}", json={}).status_code == 200
    assert (
        setup.client.post(
            path, json={"label": "Duplicate", "rtsp_url": "rtsp://10.0.0.10/live"}
        ).status_code
        == 409
    )
    assert setup.client.patch(path + "/absent", json={"label": "Missing"}).status_code == 404
    assert _actions(setup) == [
        (AuditAction.CAMERA_CREATE, camera_id),
        (AuditAction.CAMERA_UPDATE, camera_id),
    ]
    assert setup.registry.get(camera_id)["label"] == "Changed"


def test_camera_deletion_does_not_clear_another_local_rows_zone(setup):
    _create(setup, "local", backend_id="other-local")
    _create(setup, "other-local")
    local_zone = _zone(setup, "local")
    other_zone = _zone(setup, "other-local")
    snapshot = setup.client.get("/api/v1/cameras")
    assert snapshot.status_code == 200
    zones = {camera["id"]: camera["bed_zone"] for camera in snapshot.json()["cameras"]}
    assert zones == {"local": local_zone.as_dict(), "other-local": other_zone.as_dict()}
    assert setup.client.delete("/api/v1/cameras/local").status_code == 204
    assert setup.client.delete("/api/v1/cameras/local").status_code == 404
    assert setup.registry.get("local") is None and setup.bed.get("local") is None
    assert setup.bed.get("other-local") == other_zone
    assert _actions(setup) == [(AuditAction.CAMERA_DELETE, "local")]


def test_private_camera_requires_the_existing_explicit_admission_policy(setup, monkeypatch):
    from backend.app.features.cameras import router as cameras

    monkeypatch.delenv(ALLOW_PRIVATE_RTSP_ENV, raising=False)
    effects = []
    monkeypatch.setattr(cameras, "_probe_rtsp_url", lambda *args: effects.append(True))
    response = setup.client.post(
        "/api/v1/cameras", json={"label": "Denied", "rtsp_url": "rtsp://10.0.0.10/live"}
    )
    assert response.status_code == 400 and not effects and _actions(setup) == []


def test_bed_zone_save_clear_and_empty_clear_are_one_one_zero(setup):
    _create(setup)
    path = "/api/v1/cameras/camera-1/bed-zone"
    payload = {
        "regions": [{"id": "bed-1", "polygon": [[0, 0], [10, 0], [0, 10]], "origin": "manual"}],
        "image_width": 20,
        "image_height": 20,
    }
    assert setup.client.put(path, json=payload).status_code == 200
    assert setup.bed.get("camera-1") is not None
    payload["regions"] = []
    assert setup.client.put(path, json=payload).json() == {"bed_zone": None}
    assert setup.client.put(path, json=payload).json() == {"bed_zone": None}
    assert setup.bed.get("camera-1") is None
    assert _actions(setup) == [(AuditAction.BED_ZONE_UPDATE, "camera-1")] * 2


@pytest.mark.parametrize("race", [False, True])
def test_probe_persists_and_audits_only_an_existing_camera(setup, monkeypatch, race):
    from backend.app.features.cameras import router as cameras

    _create(setup)
    if race:

        def probe(request, url):
            assert setup.registry.delete("camera-1")
            return ProbeResult(ok=True)

        monkeypatch.setattr(cameras, "_probe_rtsp_url", probe)
    response = setup.client.post("/api/v1/cameras/camera-1/test")
    assert response.status_code == 200 and response.json()["ok"]
    assert _actions(setup) == ([] if race else [(AuditAction.CAMERA_PROBE, "camera-1")])


def test_runtime_setting_matching_write_audits_but_conflict_does_not(setup):
    path = "/api/v1/runtime-settings"
    first = setup.client.put(path, json={"clip_export_enabled": True, "expected_version": 0})
    assert first.status_code == 200 and first.json() == {"clip_export_enabled": True, "version": 1}
    second = setup.client.put(path, json={"clip_export_enabled": True, "expected_version": 1})
    assert second.status_code == 200 and second.json() == first.json()
    conflict = setup.client.put(path, json={"clip_export_enabled": False, "expected_version": 0})
    assert conflict.status_code == 409
    assert setup.client.get(path).json() == first.json()
    assert _actions(setup) == [(AuditAction.RUNTIME_SETTINGS_UPDATE, "runtime-settings")] * 2


def test_storage_publication_precedes_filesystem_response_work(setup, monkeypatch):
    from backend.app.features.clips import storage_router

    original = storage_router._storage_snapshot
    trace = []
    publish = setup.runtime.publish_committed

    def publication(token):
        trace.append("publication")
        return publish(token)

    def snapshot(app):
        assert trace == ["publication"]
        assert _actions(setup) == [(AuditAction.CLIP_STORAGE_UPDATE, "room")]
        assert app.state.clip_storage_location_store.get() == "room"
        trace.append("snapshot")
        return original(app)

    monkeypatch.setattr(setup.runtime, "publish_committed", publication)
    monkeypatch.setattr(storage_router, "_storage_snapshot", snapshot)
    response = setup.client.put("/api/v1/clips/storage/location", json={"path": "room"})
    assert response.status_code == 200 and trace == ["publication", "snapshot"]


def test_detection_settings_use_native_owner_and_one_audit(setup):
    domains = {
        "fall": {"on": True, "mode": "window", "start": "08:00", "end": "20:00"},
        "bed_exit": {"on": False, "mode": "always"},
    }
    response = setup.client.put("/api/v1/detection-settings", json={"domains": domains})
    assert response.status_code == 200
    assert setup.client.get("/api/v1/detection-settings").json() == response.json()
    assert _actions(setup) == [(AuditAction.DETECTION_SETTINGS_UPDATE, "detection-settings")]


def _enroll(setup):
    setup.app.state.connection_settings_store.save(
        {
            "facility_code": "NH-1234",
            "client_installation_ref": "install-1",
            "facility_id": "facility-1",
            "facility_token": "test-token",
            "edge_installation_id": "c72bd9a7-3e04-47ba-a8cd-a56e54f98152",
            "enrollment_generation": 1,
        }
    )


@pytest.mark.parametrize("missing", [True, False])
def test_policy_list_refuses_missing_or_wrong_registry_without_constructing_one(setup, missing):
    _enroll(setup)
    if missing:
        del setup.app.state.camera_registry
    else:
        setup.app.state.camera_registry = object()
    expected = RuntimeError if missing else TypeError
    message = "camera registry is not injected" if missing else "camera registry has invalid type"
    with pytest.raises(expected, match=message):
        setup.client.get("/api/v1/detection-policies")
    assert _actions(setup) == []
    setup.app.state.camera_registry = setup.registry
    assert setup.client.get("/api/v1/detection-policies").status_code == 200


def test_policy_apply_conflict_and_rollback_use_native_publication(setup):
    _enroll(setup)
    payload = {
        "module_id": "fall",
        "module_version": 2,
        "schema_id": "fall.policy",
        "schema_version": 2,
        "camera_id": None,
        "values": {"transition_threshold": 0.62},
        "expected_revision_id": 0,
    }
    path = "/api/v1/detection-policies"
    first = setup.client.post(path + "/apply", json=payload)
    assert first.status_code == 202
    revision = first.json()["active_revision_id"]
    assert setup.client.post(path + "/apply", json=payload).status_code == 409
    second = setup.client.post(
        path + "/apply",
        json={
            **payload,
            "expected_revision_id": revision,
            "values": {"transition_threshold": 0.72},
        },
    )
    assert second.status_code == 202
    reverted = setup.client.post(
        path + "/rollback",
        json={
            "module_id": "fall",
            "module_version": 2,
            "camera_id": None,
            "expected_revision_id": second.json()["active_revision_id"],
        },
    )
    assert reverted.status_code == 202
    assert _actions(setup) == [(AuditAction.POLICY_APPLY, "fall")] * 2 + [
        (AuditAction.POLICY_ROLLBACK, "fall")
    ]


def test_http_mutation_publication_follows_pool_return(setup, monkeypatch):
    trace = []
    acquire, publish = setup.sandbox.database._pool.connection, setup.runtime.publish_committed

    @contextmanager
    def complete_exit(*args, **kwargs):
        with acquire(*args, **kwargs) as connection:
            yield connection
        trace.append("pool-exit")

    def publication(token):
        assert trace == ["pool-exit"]
        assert setup.sandbox.admin.execute(
            "SELECT clip_export_enabled FROM edge_site WHERE id=1"
        ).fetchone() == (1,)
        trace.append("publication")
        return publish(token)

    monkeypatch.setattr(setup.sandbox.database._pool, "connection", complete_exit)
    monkeypatch.setattr(setup.runtime, "publish_committed", publication)
    response = setup.client.put(
        "/api/v1/runtime-settings", json={"clip_export_enabled": True, "expected_version": 0}
    )
    assert response.status_code == 200 and trace == ["pool-exit", "publication"]


@pytest.mark.parametrize("fault", ["deferred", "unknown"])
def test_http_commit_failure_is_not_a_success_or_a_replay(setup, monkeypatch, fault):
    database, admin = setup.sandbox.database, setup.sandbox.admin
    commits = []
    if fault == "deferred":
        admin.execute(
            "CREATE FUNCTION reject_setting() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'private detail' USING ERRCODE='23514'; END $$"
        )
        admin.execute(
            "CREATE CONSTRAINT TRIGGER reject_setting AFTER UPDATE ON edge_site "
            "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_setting()"
        )
    else:
        transact = database.transact

        def unknown(callback):
            transact(callback)
            commits.append(True)
            raise CommitOutcomeUnknown()

        monkeypatch.setattr(database, "transact", unknown)
    response = setup.client.put(
        "/api/v1/runtime-settings", json={"clip_export_enabled": True, "expected_version": 0}
    )
    assert (response.status_code, response.content) == (503, b"")
    assert admin.execute("SELECT clip_export_enabled FROM edge_site WHERE id=1").fetchone() == (
        (0,) if fault == "deferred" else (1,)
    )
    assert _actions(setup) == (
        [] if fault == "deferred" else [(AuditAction.RUNTIME_SETTINGS_UPDATE, "runtime-settings")]
    )
    assert commits == ([] if fault == "deferred" else [True])
    assert not setup.runtime._pending and not setup.runtime.snapshot().ready
