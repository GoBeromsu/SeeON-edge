"""Authenticated ``DELETE /api/v1/clips/{clip_id}``: operator clip deletion.

Backend contract only -- worker-owned deletion mechanics (hold checks,
containment verification, filesystem removal, DB tombstone, crash recovery)
are covered by ``tests/test_clip_deletion_lifecycle.py`` and
``tests/test_evidence_reconciliation.py``. This file proves the HTTP surface:
auth, exact confirmation, unknown clip, accepted/idempotent status
convergence, audit events, and the real backend-to-worker control seam.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.store import AuditEvent, AuditRecord, AuditStore
from backend.app.features.clips import artifacts as artifact_module
from backend.app.features.clips.deletion_lifecycle import reconcile_pending_clip_deletions
from backend.app.main import create_app, no_lifespan
from worker.pipeline.output.evidence.clip_maintenance import ClipMaintenance
from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
)
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.mjpeg_server import MjpegServer, MjpegServerConfig
from worker.runtime.clip_deletion_control import ClipDeletionControlService

NOW = "2026-05-01T00:00:00Z"
LATER = "2026-05-01T00:00:01Z"


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
    import shutil

    maintenance = ClipMaintenance(
        ClipRecorderConfig(store_dir=store_dir),
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=lambda _path: shutil.disk_usage(store_dir),
    )
    control = ClipDeletionControlService(
        preflight_clip=maintenance.preflight_clip,
        delete_clip=maintenance.purge_clip,
    )
    server = MjpegServer(
        LatestFrameStore(),
        MjpegServerConfig(port=0, probe_token="relay-token"),
        clip_deletion_control=control,
    )
    server.start()
    return server


def _seed_published_clip(database: Path, store_dir: Path, clip_id: str = "clip-a") -> None:
    manifest = store_dir / "clips" / clip_id / "manifest.json"
    media = manifest.with_name("clip.mp4")
    manifest_bytes = manifest.read_bytes()
    media_bytes = media.read_bytes()
    incident_id = f"incident:{clip_id}"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO incidents(incident_id,edge_event_id,facility_id,camera_id,event_type,"
            "probability,detected_at,lifecycle_state,provenance_state,"
            "provenance_missing_reason,review_version,revision,created_at,updated_at) "
            "VALUES (?,?,?,?,?,1.0,?,'OPEN','MISSING','NOT_RECORDED',0,1,?,?)",
            (incident_id, "event-a", "facility-a", "camera-a", "fall", NOW, NOW, NOW),
        )
        connection.execute(
            "INSERT INTO clips(clip_id,camera_id,event_facet,started_at,duration_ms,codec,"
            "mime_type,manifest_relpath,media_relpath,manifest_sha256,media_sha256,"
            "manifest_size_bytes,media_size_bytes,local_state,publish_state,published_at,"
            "retention_state,revision,created_at,updated_at) "
            "VALUES (?,?,?,?,1000,'h264','video/mp4',?,?,?,?,?,?,'AVAILABLE','PUBLISHED',?,"
            "'RETAINED',1,?,?)",
            (
                clip_id,
                "camera-a",
                "fall",
                NOW,
                f"clips/{clip_id}/manifest.json",
                f"clips/{clip_id}/clip.mp4",
                hashlib.sha256(manifest_bytes).hexdigest(),
                hashlib.sha256(media_bytes).hexdigest(),
                len(manifest_bytes),
                len(media_bytes),
                NOW,
                NOW,
                NOW,
            ),
        )
        connection.execute(
            "INSERT INTO artifacts(incident_id,kind,artifact_id,clip_id,state,contained_relpath,"
            "content_sha256,size_bytes,mime_type,codec,revision,created_at,updated_at) "
            "VALUES (?,'PRIMARY_CLIP',?,?, 'AVAILABLE',?,?,?,?, 'h264',1,?,?)",
            (
                incident_id,
                f"primary:{clip_id}",
                clip_id,
                f"clips/{clip_id}/clip.mp4",
                hashlib.sha256(media_bytes).hexdigest(),
                len(media_bytes),
                "video/mp4",
                NOW,
                NOW,
            ),
        )


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


def test_backend_commits_pending_and_request_audit_before_worker_filesystem_delete(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = artifact_module.EDGE_DATABASE_PATH
    migrate_database(database)
    _seed_published_clip(database, clip_store)
    observed: list[tuple[str, list[tuple[str]]]] = []

    def worker_delete(_request: object, clip_id: str) -> dict[str, object]:
        with sqlite3.connect(database) as connection:
            state = connection.execute(
                "SELECT retention_state FROM clips WHERE clip_id = ?", (clip_id,)
            ).fetchone()
            actions = connection.execute(
                "SELECT action FROM audit_events WHERE target_id = ? ORDER BY audit_id",
                (clip_id,),
            ).fetchall()
        assert state is not None
        observed.append((str(state[0]), [(str(action),) for (action,) in actions]))
        return {"clip_id": clip_id, "status": "PURGED"}

    monkeypatch.setattr(
        "backend.app.features.clips.router.preflight_clip_deletion",
        lambda _request, clip_id: {"clip_id": clip_id, "status": "READY"},
    )
    monkeypatch.setattr(
        "backend.app.features.clips.router.control_clip_deletion", worker_delete
    )
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        response = client.request(
            "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
        )

    assert response.status_code == 202
    assert observed == [("PENDING", [("clip.delete.request",)])]


def test_delete_reaches_real_worker_purges_and_is_idempotent(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = artifact_module.EDGE_DATABASE_PATH
    migrate_database(database)
    _seed_published_clip(database, clip_store)
    server = _worker_server(database, clip_store)
    _wire_worker_origin(monkeypatch, server)

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
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT retention_state FROM clips WHERE clip_id='clip-a'"
        ).fetchone() == ("PURGED",)
        assert connection.execute(
            "SELECT state FROM artifacts WHERE clip_id='clip-a' AND kind='PRIMARY_CLIP'"
        ).fetchone() == ("PURGED",)
        actions = connection.execute(
            "SELECT action FROM audit_events WHERE target_id='clip-a' ORDER BY audit_id"
        ).fetchall()
    assert actions == [("clip.delete.request",), ("clip.delete.complete",)]


def test_delete_worker_control_failure_is_reported_and_audited(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = artifact_module.EDGE_DATABASE_PATH
    migrate_database(database)
    _seed_published_clip(database, clip_store)
    monkeypatch.setattr(
        "backend.app.features.clips.router.preflight_clip_deletion",
        lambda _request, clip_id: {"clip_id": clip_id, "status": "READY"},
    )
    monkeypatch.setattr(
        "backend.app.features.clips.deletion_control.get_settings",
        lambda: SimpleNamespace(worker_stream_origin="", worker_stream_timeout_s=2.0),
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.request(
            "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
        )

    assert response.status_code == 503
    with sqlite3.connect(artifact_module.EDGE_DATABASE_PATH) as connection:
        actions = connection.execute(
            "SELECT action FROM audit_events WHERE target_id='clip-a'"
        ).fetchall()
    assert actions == [("clip.delete.request",)]
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT retention_state FROM clips WHERE clip_id='clip-a'"
        ).fetchone() == ("PENDING",)
    assert (clip_store / "clips" / "clip-a").exists()


def test_hold_preflight_leaves_state_files_and_audit_untouched(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = artifact_module.EDGE_DATABASE_PATH
    migrate_database(database)
    _seed_published_clip(database, clip_store)
    monkeypatch.setattr(
        "backend.app.features.clips.router.preflight_clip_deletion",
        lambda _request, clip_id: {"clip_id": clip_id, "status": "HELD"},
    )
    monkeypatch.setattr(
        "backend.app.features.clips.router.control_clip_deletion",
        lambda *_args: pytest.fail("held preflight must not issue destructive command"),
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.request(
            "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
        )

    assert response.status_code == 202
    assert response.json() == {"clip_id": "clip-a", "status": "HELD"}
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT retention_state FROM clips WHERE clip_id='clip-a'"
        ).fetchone() == ("RETAINED",)
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE target_id='clip-a'"
        ).fetchone() == (0,)
    assert (clip_store / "clips" / "clip-a" / "clip.mp4").is_file()


def test_request_audit_full_rolls_back_pending_and_never_commands_worker(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = artifact_module.EDGE_DATABASE_PATH
    migrate_database(database)
    _seed_published_clip(database, clip_store)

    class FullAuditStore(AuditStore):
        def append(
            self,
            event: AuditEvent,
            *,
            connection: sqlite3.Connection | None = None,
        ) -> AuditRecord:
            del event, connection
            raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(
        "backend.app.features.clips.router.preflight_clip_deletion",
        lambda _request, clip_id: {"clip_id": clip_id, "status": "READY"},
    )
    monkeypatch.setattr(
        "backend.app.features.clips.router.control_clip_deletion",
        lambda *_args: pytest.fail("uncommitted intent must never command deletion"),
    )
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        app.state.audit_store = FullAuditStore(database)
        response = client.request(
            "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
        )

    assert response.status_code == 503
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT retention_state FROM clips WHERE clip_id='clip-a'"
        ).fetchone() == ("RETAINED",)
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE target_id='clip-a'"
        ).fetchone() == (0,)
    assert (clip_store / "clips" / "clip-a").is_dir()


def test_post_delete_completion_failure_reconciles_once_on_startup(
    clip_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = artifact_module.EDGE_DATABASE_PATH
    migrate_database(database)
    _seed_published_clip(database, clip_store)
    server = _worker_server(database, clip_store)
    _wire_worker_origin(monkeypatch, server)

    class CompletionFullAuditStore(AuditStore):
        fail_completion = True

        def append(
            self,
            event: AuditEvent,
            *,
            connection: sqlite3.Connection | None = None,
        ) -> AuditRecord:
            if self.fail_completion and event.action is AuditAction.CLIP_DELETE_COMPLETE:
                raise sqlite3.OperationalError("database or disk is full")
            return super().append(event, connection=connection)

    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = CompletionFullAuditStore(database)
    app.state.audit_store = store
    try:
        with TestClient(app) as client:
            _login(client)
            response = client.request(
                "DELETE", "/api/v1/clips/clip-a", json={"confirm_clip_id": "clip-a"}
            )
    finally:
        server.stop()

    assert response.status_code == 503
    assert not (clip_store / "clips" / "clip-a").exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT retention_state FROM clips WHERE clip_id='clip-a'"
        ).fetchone() == ("PENDING",)
        assert connection.execute(
            "SELECT action FROM audit_events WHERE target_id='clip-a' ORDER BY audit_id"
        ).fetchall() == [("clip.delete.request",)]

    store.fail_completion = False
    monkeypatch.setattr(
        "backend.app.features.clips.deletion_lifecycle.preflight_clip_deletion",
        lambda _app, clip_id: {"clip_id": clip_id, "status": "MISSING"},
    )
    first = reconcile_pending_clip_deletions(app, database)
    second = reconcile_pending_clip_deletions(app, database)

    assert first.completed == ("clip-a",)
    assert second.completed == ()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT retention_state FROM clips WHERE clip_id='clip-a'"
        ).fetchone() == ("PURGED",)
        assert connection.execute(
            "SELECT action FROM audit_events WHERE target_id='clip-a' ORDER BY audit_id"
        ).fetchall() == [
            ("clip.delete.request",),
            ("clip.delete.complete",),
        ]


def test_delete_response_is_strictly_typed(clip_store: Path) -> None:
    from pydantic import ValidationError

    from backend.app.features.clips.schemas import DeleteClipResponse

    response = DeleteClipResponse.model_validate({"clip_id": "clip-a", "status": "PURGED"})
    assert response.status == "PURGED"
    with pytest.raises(ValidationError):
        DeleteClipResponse.model_validate({"clip_id": "clip-a", "status": "DELETED"})
