from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_runtime import (
    AuditMutation,
    AuditRuntimeUnavailable,
    PostgresAuditRuntime,
)
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.record_store import (
    CentralEvidenceReviewStore,
    ReviewDisposition,
)
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_STAMP = "2026-08-24T00:00:00.000Z"


def _write_clip(root: Path, clip_id: str) -> None:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"verified-video")
    (clip_dir / "thumbnail.jpg").write_bytes(b"jpeg")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "camera-a",
                "event_ref": "event-a",
                "event_type": "fall",
                "started_at": "2026-08-24T00:00:00Z",
                "duration_s": 1.0,
                "codec": "h264",
                "path": f"clips/{clip_id}",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def _reject_audit_inserts(sandbox: ProductSandbox) -> None:
    sandbox.admin.execute(
        "CREATE OR REPLACE FUNCTION reject_audit_test() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'injected audit failure'; END $$"
    )
    sandbox.admin.execute(
        "CREATE TRIGGER reject_audit_test BEFORE INSERT ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION reject_audit_test()"
    )


def _restore_audit_inserts(sandbox: ProductSandbox, runtime: PostgresAuditRuntime) -> None:
    # The extra trigger fails the exact trigger contract, so re-verify only after dropping it.
    sandbox.admin.execute("DROP TRIGGER reject_audit_test ON audit_events")
    assert runtime.verify_once()


def _action_count(sandbox: ProductSandbox, action: AuditAction) -> int:
    return sandbox.admin.execute(
        "SELECT COUNT(*) FROM audit_events WHERE action=%s", (action.value,)
    ).fetchone()[0]


def _event_count(sandbox: ProductSandbox) -> int:
    return sandbox.admin.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]


def _event(action: AuditAction, target_id: str) -> AuditEvent:
    return AuditEvent(
        occurred_at=_STAMP,
        actor_id="admin",
        action=action,
        target_id=target_id,
        detail=empty_detail(action),
    )


def test_stored_evidence_audit_failure_has_empty_503_and_live_probe_survives(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: authenticated stored evidence and a real PostgreSQL audit INSERT rejection.
    sandbox = postgres_product_sandbox
    root = tmp_path / "clips"
    _write_clip(root, "clip-a")
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = postgres_api_app(sandbox, postgres_audit_runtime)
    opened_handles: list[BinaryIO] = []
    original = ClipStore.open_located_video

    def capture_open(store: ClipStore, located):
        opened = original(store, located)
        opened_handles.append(opened.handle)
        return opened

    monkeypatch.setattr(ClipStore, "open_located_video", capture_open)
    with TestClient(app) as client:
        _login(client)
        _reject_audit_inserts(sandbox)

        # When: JSON and descriptor-backed reads reach the audit commit boundary.
        listed = client.get("/api/v1/clips")
        metadata = client.get("/api/v1/clips/clip-a/metadata")
        artifacts = client.get("/api/v1/clips/clip-a/artifacts")
        video = client.get("/api/v1/clips/clip-a/video")
        thumbnail = client.get("/api/v1/clips/clip-a/thumbnail")
        readiness = client.get("/health/ready")
        liveness = client.get("/health/live")

    # Then: neither response starts product content, the descriptor closes, and liveness stays open.
    assert (listed.status_code, listed.content) == (503, b"")
    assert (metadata.status_code, metadata.content) == (503, b"")
    assert (artifacts.status_code, artifacts.content) == (503, b"")
    assert (video.status_code, video.content) == (503, b"")
    assert (thumbnail.status_code, thumbnail.content) == (503, b"")
    for header in ("accept-ranges", "content-range", "content-disposition"):
        assert header not in video.headers
        assert header not in thumbnail.headers
    assert opened_handles and all(handle.closed for handle in opened_handles)
    assert readiness.status_code == 503
    assert readiness.json()["reason"] == "audit unavailable"
    assert liveness.status_code == 200


def test_valid_video_200_and_206_append_one_success_audit_each(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one authenticated descriptor-backed clip.
    sandbox = postgres_product_sandbox
    root = tmp_path / "clips"
    _write_clip(root, "clip-a")
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = postgres_api_app(sandbox, postgres_audit_runtime)
    with TestClient(app) as client:
        _login(client)
        before = _action_count(sandbox, AuditAction.CLIP_PLAY)

        # When: complete and satisfiable-range responses are prepared and served.
        complete = client.get("/api/v1/clips/clip-a/video")
        partial = client.get("/api/v1/clips/clip-a/video", headers={"Range": "bytes=0-7"})

        after = _action_count(sandbox, AuditAction.CLIP_PLAY)

    # Then: both valid response classes are audited exactly once.
    assert (complete.status_code, complete.content) == (200, b"verified-video")
    assert (partial.status_code, partial.content) == (206, b"verified")
    assert partial.headers["content-range"] == "bytes 0-7/14"
    assert after - before == 2


def test_credential_rotation_rolls_back_before_cookie_or_session_mutation(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    # Given: a valid dashboard session and an audit table that refuses the shared transaction.
    sandbox = postgres_product_sandbox
    app = postgres_api_app(sandbox, postgres_audit_runtime)

    def credentials():
        return sandbox.admin.execute(
            "SELECT username, salt, password_hash, updated_at FROM credentials ORDER BY id"
        ).fetchall()

    with TestClient(app) as client:
        _login(client)
        before = credentials()
        rotations = _action_count(sandbox, AuditAction.CREDENTIAL_ROTATE)
        _reject_audit_inserts(sandbox)

        # When: credential rotation attempts its credentials + audit transaction.
        response = client.put(
            "/api/v1/auth/credentials",
            json={"username": "rotated", "new_password": "new-password"},
        )

    # Then: no response cookie and no durable credential row escaped rollback.
    assert (response.status_code, response.content) == (503, b"")
    assert "set-cookie" not in response.headers
    assert credentials() == before
    assert _action_count(sandbox, AuditAction.CREDENTIAL_ROTATE) == rotations


def test_camera_and_topology_mutations_roll_back_with_audit_failure(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    # Given: one camera authority sharing the audit runtime's database and a rejecting audit table.
    sandbox = postgres_product_sandbox
    runtime = postgres_audit_runtime
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    events = _event_count(sandbox)
    _reject_audit_inserts(sandbox)

    # When: camera and location mutations reach their shared transaction callback.
    with pytest.raises(AuditRuntimeUnavailable):
        AuditMutation(runtime, lambda: _event(AuditAction.CAMERA_CREATE, "camera-a")).apply(
            store,
            lambda append: store.create(
                camera_id="camera-a",
                label="Camera A",
                rtsp_url="rtsp://example/camera-a",
                space_id=None,
                status="unknown",
                after_write=append,
            ),
        )
    _restore_audit_inserts(sandbox, runtime)
    _reject_audit_inserts(sandbox)
    with pytest.raises(AuditRuntimeUnavailable):
        AuditMutation(runtime, lambda: _event(AuditAction.LOCATION_CREATE, "floor-a")).apply(
            store,
            lambda append: store.create_floor(
                edge_ref="floor-a", name="Floor A", order_index=0, after_write=append
            ),
        )

    # Then: neither authority changed and no audit row escaped.
    assert store.snapshot()["cameras"] == []
    assert store.topology_snapshot().floors == ()
    assert _event_count(sandbox) == events


def test_review_cas_rolls_back_with_audit_failure(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    # Given: one reviewable incident and a rejecting audit table.
    sandbox = postgres_product_sandbox
    sandbox.admin.execute(
        "INSERT INTO incidents(incident_id,edge_event_id,facility_id,camera_id,event_type,"
        "probability,detected_at,lifecycle_state,provenance_state,"
        "provenance_missing_reason,review_version,revision,created_at,updated_at) "
        "VALUES('incident-a','event-a','facility-a','camera-a','fall',0.9,%s,'OPEN',"
        "'MISSING','NOT_RECORDED',0,1,%s,%s)",
        (_STAMP, _STAMP, _STAMP),
    )
    store = CentralEvidenceReviewStore(sandbox.database, sandbox.authority)
    _reject_audit_inserts(sandbox)

    # When: the review and audit share a transaction whose append fails.
    with pytest.raises(AuditRuntimeUnavailable):
        AuditMutation(
            postgres_audit_runtime, lambda: _event(AuditAction.INCIDENT_REVIEW, "incident-a")
        ).apply(
            store,
            lambda append: store.update(
                incident_id="incident-a",
                expected_version=0,
                actor_id="admin",
                reviewed_at=_STAMP,
                disposition=ReviewDisposition.TRUE_POSITIVE,
                notes=None,
                after_write=append,
            ),
        )

    # Then: CAS state remains unchanged.
    version = sandbox.admin.execute(
        "SELECT review_version FROM incidents WHERE incident_id='incident-a'"
    ).fetchone()[0]
    assert version == 0


def test_audit_router_uses_unique_descending_keyset_pages(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    # Given: three existing events plus the authenticated login event.
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    for action in ("clip.list", "clip.detail", "clip.thumbnail"):
        parsed = AuditAction(action)
        postgres_audit_runtime.append_owned(
            AuditEvent(
                occurred_at=_STAMP,
                actor_id="seed",
                action=parsed,
                target_id=action,
                detail=empty_detail(parsed),
            )
        )
    with TestClient(app) as client:
        _login(client)

        # When: the caller follows the bounded keyset cursor.
        first = client.get("/api/v1/audit", params={"limit": 2})
        cursor = first.json()["next_before_id"]
        second = client.get("/api/v1/audit", params={"limit": 2, "before_id": cursor})

    # Then: ordering is deterministic and pages do not overlap.
    first_ids = [event["audit_id"] for event in first.json()["events"]]
    second_ids = [event["audit_id"] for event in second.json()["events"]]
    assert first_ids == sorted(first_ids, reverse=True)
    assert second_ids == sorted(second_ids, reverse=True)
    assert set(first_ids).isdisjoint(second_ids)


def test_recovered_audit_interval_writes_one_fence(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    # Given: one failed governed read marks a bounded degraded interval.
    sandbox = postgres_product_sandbox
    app = postgres_api_app(sandbox, postgres_audit_runtime)
    with TestClient(app) as client:
        _login(client)
        _reject_audit_inserts(sandbox)
        assert client.get("/api/v1/audit").status_code == 503

        # When: audit recovers and two governed reads succeed.
        _restore_audit_inserts(sandbox, postgres_audit_runtime)
        first = client.get("/api/v1/audit")
        second = client.get("/api/v1/audit")
        readiness = client.get("/health/ready")

    # Then: recovery is summarized once, without a per-request backlog, and readiness heals.
    assert first.status_code == second.status_code == 200
    assert readiness.status_code == 200
    assert _action_count(sandbox, AuditAction.RECOVERY_FENCE) == 1
