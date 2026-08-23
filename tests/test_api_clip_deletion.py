"""Authenticated ``DELETE /api/v1/clips/{clip_id}``: operator clip deletion.

Backend contract only -- worker-owned deletion mechanics (hold checks,
containment verification, filesystem removal, DB tombstone, crash recovery)
are covered by ``tests/test_clip_deletion_lifecycle.py`` and
``tests/test_evidence_reconciliation.py``. This file proves the HTTP surface:
auth, exact confirmation, unknown clip, accepted/idempotent status
convergence, audit events, and the real backend-to-worker control seam.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.clips import artifacts as artifact_module
from backend.app.features.clips.audit_log import AuditLogStore
from backend.app.main import create_app, no_lifespan
from worker.pipeline.output.evidence.clip_maintenance import ClipMaintenance
from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
)
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig
from worker.runtime.clip_deletion_control import ClipDeletionControlService

NOW = "2026-08-13T00:00:00Z"
LATER = "2026-08-13T00:00:01Z"


def _write_finalized_clip(store_dir: Path, clip_id: str) -> Path:
    clip_dir = store_dir / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"clean-source")
    manifest = {
        "clip_id": clip_id,
        "camera_id": "camera-a",
        "event_ref": "event-a",
        "event_type": "fall",
        "started_at": NOW,
        "duration_s": 1.0,
        "codec": "h264",
        "path": f"clips/{clip_id}/clip.mp4",
        "video_available": True,
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return clip_dir


@pytest.fixture
def clip_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "labels"))
    _write_finalized_clip(root, "clip-a")
    return root


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def _worker_server(database: Path, store_dir: Path) -> MjpegServer:
    del database
    config = ClipRecorderConfig(store_dir=store_dir)
    retention: dict[str, str] = {}

    def begin(clip_id: str) -> bool:
        if retention.get(clip_id) == "PURGED":
            return False
        retention[clip_id] = "PENDING"
        return True

    def complete(clip_id: str) -> None:
        retention[clip_id] = "PURGED"

    def fail(clip_id: str, reason: str) -> None:
        del reason
        retention[clip_id] = "FAILED"

    def retention_state(clip_id: str) -> str | None:
        return retention.get(clip_id)

    import shutil

    maintenance = ClipMaintenance(
        config,
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=lambda _path: shutil.disk_usage(store_dir),
        begin_clip_purge=begin,
        complete_clip_purge=complete,
        fail_clip_purge=fail,
    )
    control = ClipDeletionControlService(
        delete_clip=maintenance.purge_clip,
        retention_state=retention_state,
    )
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        clip_deletion_control=control,
    )
    server.start()
    return server


def _seed_published_clip(database: Path, clip_id: str = "clip-a") -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_clips (clip_id,local_state,state_version,publish_state) "
            "VALUES (?,'VERIFIED',2,'PUBLISHED')",
            (clip_id,),
        )
        connection.commit()


def _wire_worker_origin(monkeypatch: pytest.MonkeyPatch, server: MjpegServer) -> None:
    monkeypatch.setattr(
        "backend.app.features.clips.deletion_control.get_settings",
        lambda: SimpleNamespace(
            worker_stream_origin=f"http://127.0.0.1:{server.port}",
            worker_stream_timeout_s=2.0,
        ),
    )


def test_delete_requires_authentication(clip_store: Path) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.request(
            "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
        )
    assert response.status_code == 401


def test_delete_rejects_confirmation_mismatch(clip_store: Path) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.request(
            "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "not-clip-a"}
        )
    assert response.status_code == 422


def test_delete_malformed_clip_id_is_404(clip_store: Path) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.request(
            "DELETE",
            "/api/v1/clips/not%20valid%2Fid",
            json={"confirm_clip_id": "not valid/id"},
        )
    assert response.status_code == 404


def test_delete_unknown_clip_reports_truthful_missing_status(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A syntactically valid clip_id the worker has never heard of is
    distinct from PURGED/HELD -- reported as ``MISSING``, not falsely as a
    successful deletion, and not a bare 404 that would collide with a
    duplicate request against an already-purged (and now file-absent) clip.
    """
    database = artifact_module.EDGE_DATABASE_PATH
    migrate_database(database)
    server = _worker_server(database, clip_store)
    _wire_worker_origin(monkeypatch, server)

    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    try:
        with TestClient(app) as client:
            _login(client)
            response = client.request(
                "DELETE",
                "/api/v1/clips/never-existed",
                json={"confirm_clip_id": "never-existed"},
            )
    finally:
        server.stop()

    assert response.status_code == 202
    assert response.json() == {"clip_id": "never-existed", "status": "MISSING"}


def test_delete_reaches_real_worker_purges_and_is_idempotent(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = artifact_module.EDGE_DATABASE_PATH
    migrate_database(database)
    _seed_published_clip(database)
    server = _worker_server(database, clip_store)
    _wire_worker_origin(monkeypatch, server)
    audit_path = tmp_path / "labels" / "audit.jsonl"

    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    try:
        with TestClient(app) as client:
            _login(client)
            first = client.request(
                "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
            )
            assert first.status_code == 202
            assert first.json() == {"clip_id": "clip-a", "status": "PURGED"}

            second = client.request(
                "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
            )
            assert second.status_code == 202
            assert second.json() == {"clip_id": "clip-a", "status": "PURGED"}
    finally:
        server.stop()

    assert not (clip_store / "clips" / "clip-a").exists()
    entries = AuditLogStore(audit_path).list_entries()
    delete_entries = [entry for entry in entries if entry["clip_id"] == "clip-a"]
    actions = [entry["action"] for entry in delete_entries]
    assert actions == ["clip-delete-completed"]


def test_delete_worker_control_failure_is_reported_and_audited(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "backend.app.features.clips.deletion_control.get_settings",
        lambda: SimpleNamespace(worker_stream_origin="", worker_stream_timeout_s=2.0),
    )
    audit_path = tmp_path / "labels" / "audit.jsonl"

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.request(
            "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
        )

    assert response.status_code == 503
    entries = AuditLogStore(audit_path).list_entries()
    actions = [entry["action"] for entry in entries if entry["clip_id"] == "clip-a"]
    assert actions == ["clip-delete-failed"]


def test_delete_response_is_strictly_typed(clip_store: Path) -> None:
    from pydantic import ValidationError

    from backend.app.features.clips.schemas import DeleteClipResponse

    response = DeleteClipResponse.model_validate({"clip_id": "clip-a", "status": "PURGED"})
    assert response.status == "PURGED"
    with pytest.raises(ValidationError):
        DeleteClipResponse.model_validate({"clip_id": "clip-a", "status": "DELETED"})
