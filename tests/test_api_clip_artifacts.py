from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from receipt_helpers import add_accepted_media_receipts

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


@pytest.fixture
def make_app(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> Callable[[], FastAPI]:
    """An app on the sandbox, with receipts for the clips written so far."""

    def make() -> FastAPI:
        app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
        add_accepted_media_receipts(app)
        return app

    return make


NOW = "2026-08-13T00:00:00Z"
PRIMARY = b"clean-source-packet-media"


@pytest.fixture
def clip_artifact_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    clip_dir = root / "clips" / "clip-a"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(PRIMARY)
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-a",
                "camera_id": "camera-a",
                "event_ref": "event-a",
                "event_type": "fall",
                "started_at": NOW,
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/clip-a/clip.mp4",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    return root


HASH = "ab" * 32


def _seed_snapshot(sandbox: ProductSandbox) -> None:
    with sandbox.admin.transaction():
        sandbox.admin.execute(
            "INSERT INTO incidents ("
            "incident_id, edge_event_id, facility_id, camera_id, event_type, "
            "probability, detected_at, lifecycle_state, provenance_state, "
            "provenance_missing_reason, review_version, revision, created_at, updated_at"
            ") VALUES ('incident-a','event-a','facility-1','camera-a','fall',0.8,%s,'OPEN',"
            "'MISSING','NOT_RECORDED',0,1,%s,%s)",
            (NOW, NOW, NOW),
        )
        sandbox.admin.execute(
            "INSERT INTO clips ("
            "clip_id, camera_id, event_facet, started_at, manifest_relpath, "
            "media_relpath, manifest_sha256, media_sha256, manifest_size_bytes, "
            "media_size_bytes, local_state, publish_state, retention_state, "
            "revision, created_at, updated_at"
            ") VALUES ('clip-a','camera-a','fall',%s,'clips/clip-a/manifest.json',"
            "'clips/clip-a/clip.mp4',%s,%s,%s,%s,'AVAILABLE','WAITING','RETAINED',1,%s,%s)",
            (NOW, HASH, HASH, len(PRIMARY), len(PRIMARY), NOW, NOW),
        )
        sandbox.admin.execute(
            "INSERT INTO artifacts ("
            "incident_id, kind, artifact_id, clip_id, state, contained_relpath, "
            "content_sha256, size_bytes, mime_type, codec, revision, created_at, updated_at"
            ") VALUES ('incident-a','PRIMARY_CLIP','primary-a','clip-a','AVAILABLE',"
            "'clips/clip-a/clip.mp4',%s,%s,'video/mp4','h264',1,%s,%s)",
            (HASH, len(PRIMARY), NOW, NOW),
        )
        sandbox.admin.execute(
            "INSERT INTO artifacts ("
            "incident_id, kind, artifact_id, state, contained_relpath, content_sha256, "
            "size_bytes, mime_type, captured_at, revision, created_at, updated_at"
            ") VALUES ('incident-a','SNAPSHOT','snapshot-a','AVAILABLE',"
            "'snapshots/camera-a/snapshot-a.jpg',%s,%s,'image/jpeg',%s,1,%s,%s)",
            (HASH, 12, NOW, NOW, NOW),
        )


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def test_authenticated_artifact_views_are_clean_and_optional_snapshot(
    clip_artifact_store: Path,
    postgres_product_sandbox: ProductSandbox,
    make_app: Callable[[], FastAPI],
) -> None:
    del clip_artifact_store
    _seed_snapshot(postgres_product_sandbox)
    with TestClient(make_app()) as client:
        assert client.get("/api/v1/clips/clip-a/artifacts").status_code == 401
        _login(client)
        views = client.get("/api/v1/clips/clip-a/artifacts")
        video = client.get("/api/v1/clips/clip-a/video", headers={"Range": "bytes=2-8"})

    assert views.status_code == 200
    assert views.json() == {
        "clip_id": "clip-a",
        "clean": "AVAILABLE",
        "snapshot": "AVAILABLE",
    }
    assert set(views.json()) == {"clip_id", "clean", "snapshot"}
    assert video.status_code == 206
    assert video.content == PRIMARY[2:9]


def test_missing_snapshot_is_omitted_from_artifact_views(
    clip_artifact_store: Path,
    make_app: Callable[[], FastAPI],
) -> None:
    del clip_artifact_store
    with TestClient(make_app()) as client:
        _login(client)
        views = client.get("/api/v1/clips/clip-a/artifacts")
        video = client.get("/api/v1/clips/clip-a/video")

    assert views.status_code == 200
    assert views.json() == {
        "clip_id": "clip-a",
        "clean": "AVAILABLE",
        "snapshot": None,
    }
    assert video.status_code == 200
    assert video.content == PRIMARY
