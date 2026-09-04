from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.store import AuditEvent, AuditRecord, AuditStore, utc_now
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.artifacts import CentralClipArtifactQuery, CentralClipArtifacts
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.record_store import (
    CentralEvidenceReviewStore,
    ReviewDisposition,
)
from backend.app.main import create_app, no_lifespan


class EmptyArtifactQuery(CentralClipArtifactQuery):
    def get(self, clip_id: str) -> CentralClipArtifacts | None:
        del clip_id
        return None


class AuthorizerAuditDenyStore(AuditStore):
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


class FailingAuditStore(AuditStore):
    def _append(self, connection: sqlite3.Connection, event: AuditEvent) -> AuditRecord:
        del connection, event
        raise sqlite3.OperationalError("database or disk is full")


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


def test_stored_evidence_audit_failure_has_empty_503_and_live_probe_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: authenticated stored evidence and a real SQLite audit INSERT denial.
    root = tmp_path / "clips"
    _write_clip(root, "clip-a")
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = create_app(lifespan=no_lifespan)
    app.state.central_clip_artifact_query = EmptyArtifactQuery()
    opened_handles: list[BinaryIO] = []
    original = ClipStore.open_located_video

    def capture_open(store: ClipStore, located):
        opened = original(store, located)
        opened_handles.append(opened.handle)
        return opened

    monkeypatch.setattr(ClipStore, "open_located_video", capture_open)
    with TestClient(app) as client:
        _login(client)
        healthy_store = app.state.audit_store
        app.state.audit_store = AuthorizerAuditDenyStore(healthy_store.path)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: one authenticated descriptor-backed clip.
    root = tmp_path / "clips"
    _write_clip(root, "clip-a")
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        with sqlite3.connect(AuditStore().path) as connection:
            before = connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE action='clip.play'"
            ).fetchone()[0]

        # When: complete and satisfiable-range responses are prepared and served.
        complete = client.get("/api/v1/clips/clip-a/video")
        partial = client.get("/api/v1/clips/clip-a/video", headers={"Range": "bytes=0-7"})

        with sqlite3.connect(AuditStore().path) as connection:
            after = connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE action='clip.play'"
            ).fetchone()[0]

    # Then: both valid response classes are audited exactly once.
    assert (complete.status_code, complete.content) == (200, b"verified-video")
    assert (partial.status_code, partial.content) == (206, b"verified")
    assert partial.headers["content-range"] == "bytes 0-7/14"
    assert after - before == 2


def test_credential_rotation_rolls_back_before_cookie_or_session_mutation(tmp_path: Path) -> None:
    # Given: a valid dashboard session and an audit store that refuses the shared transaction.
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        healthy_store = app.state.audit_store
        app.state.audit_store = FailingAuditStore(healthy_store.path)

        # When: credential rotation attempts its credentials + audit transaction.
        response = client.put(
            "/api/v1/auth/credentials",
            json={"username": "rotated", "new_password": "new-password"},
        )

    # Then: no response cookie and no durable credential row escaped rollback.
    assert (response.status_code, response.content) == (503, b"")
    assert "set-cookie" not in response.headers
    with sqlite3.connect(healthy_store.path) as connection:
        assert connection.execute("SELECT username FROM credentials WHERE id=1").fetchone() is None


def test_camera_and_topology_mutations_roll_back_with_audit_failure() -> None:
    # Given: one schema-18 camera store and deterministic audit rejection callback.
    store = CameraRegistryStore(AuditStore().path)

    def reject(connection: sqlite3.Connection) -> None:
        del connection
        raise sqlite3.OperationalError("database or disk is full")

    # When: camera and location mutations reach their shared transaction callback.
    with pytest.raises(sqlite3.OperationalError, match="full"):
        store.create(
            camera_id="camera-a",
            label="Camera A",
            rtsp_url="rtsp://example/camera-a",
            space_id=None,
            status="unknown",
            after_write=reject,
        )
    with pytest.raises(sqlite3.OperationalError, match="full"):
        store.create_floor(edge_ref="floor-a", name="Floor A", order_index=0, after_write=reject)

    # Then: neither authority changed.
    assert store.snapshot()["cameras"] == []
    assert store.topology_snapshot().floors == ()


def test_review_cas_rolls_back_with_audit_failure() -> None:
    # Given: one reviewable incident.
    path = AuditStore().path
    stamp = "2026-08-24T00:00:00.000Z"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO incidents(incident_id,edge_event_id,facility_id,camera_id,event_type,"
            "probability,detected_at,lifecycle_state,provenance_state,"
            "provenance_missing_reason,review_version,revision,created_at,updated_at) "
            "VALUES('incident-a','event-a','facility-a','camera-a','fall',0.9,?,'OPEN',"
            "'MISSING','NOT_RECORDED',0,1,?,?)",
            (stamp, stamp, stamp),
        )

    def reject(connection: sqlite3.Connection) -> None:
        del connection
        raise sqlite3.OperationalError("database or disk is full")

    # When: the review and audit share a transaction whose append fails.
    with pytest.raises(sqlite3.OperationalError, match="full"):
        CentralEvidenceReviewStore(path).update(
            incident_id="incident-a",
            expected_version=0,
            actor_id="admin",
            reviewed_at=stamp,
            disposition=ReviewDisposition.TRUE_POSITIVE,
            notes=None,
            after_write=reject,
        )

    # Then: CAS state remains unchanged.
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT review_version FROM incidents WHERE incident_id='incident-a'"
        ).fetchone()[0]
    assert version == 0



def test_audit_router_uses_unique_descending_keyset_pages() -> None:
    # Given: three existing events plus the authenticated login event.
    app = create_app(lifespan=no_lifespan)
    store = AuditStore()
    for action in ("clip.list", "clip.detail", "clip.thumbnail"):
        parsed = AuditAction(action)
        store.append(
            AuditEvent(
                occurred_at=utc_now(),
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


def test_recovered_audit_interval_writes_one_fence() -> None:
    # Given: one failed governed read marks a bounded degraded interval.
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        healthy_store = app.state.audit_store
        app.state.audit_store = FailingAuditStore(healthy_store.path)
        assert client.get("/api/v1/audit").status_code == 503

        # When: audit recovers and two governed reads succeed.
        app.state.audit_store = healthy_store
        first = client.get("/api/v1/audit")
        second = client.get("/api/v1/audit")

    # Then: recovery is summarized once, without a per-request backlog.
    assert first.status_code == second.status_code == 200
    with sqlite3.connect(healthy_store.path) as connection:
        fence_count = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='audit.recovery-fence'"
        ).fetchone()[0]
    assert fence_count == 1
