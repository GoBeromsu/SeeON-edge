"""Diagnostic (path-less) clip manifest handling and media-type mapping."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.app.features.clips.media_response import media_type as _media_type
from backend.app.main import create_app, no_lifespan

# Dashboard auth now always resolves to a session store (persisted file > env
# > the built-in admin/admin default, see backend/app/shared/dashboard_auth.py),
# so a bare worker relay/bearer token is never sufficient on its own -- these
# tests log in as the zero-config default and rely on the TestClient's cookie
# jar to carry the session across subsequent calls.


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


def _write_playable_manifest(clip_store, clip_id: str) -> None:
    clip_dir = clip_store / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip.mp4").write_bytes(f"video:{clip_id}".encode())
    payload = {
        "clip_id": clip_id,
        "camera_id": "camera-1",
        "event_ref": f"event-{clip_id}",
        "event_type": "fall",
        "started_at": "2026-07-06T00:00:00Z",
        "duration_s": 30.0,
        "codec": "h264",
        "path": f"clips/{clip_id}",
        "video_available": True,
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_diagnostic_manifest(clip_store, clip_id: str, *, video_error: str) -> None:
    # A video-failed clip: no media file, path omitted, video_available false.
    clip_dir = clip_store / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": clip_id,
        "camera_id": "camera-1",
        "event_ref": f"event-{clip_id}",
        "event_type": "bed-exit",
        "started_at": "2026-07-06T00:05:00Z",
        "duration_s": 0.0,
        "codec": None,
        "path": None,
        "video_available": False,
        "video_error": video_error,
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(autouse=True)
def clip_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "label-store"))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.delenv("API_AUDIT_LOG", raising=False)
    monkeypatch.delenv("API_BACKEND_CLIP_EVENTS_URL", raising=False)
    monkeypatch.delenv("API_BACKEND_FACILITY_TOKEN", raising=False)
    monkeypatch.delenv("API_FACILITY_TOKEN", raising=False)
    return tmp_path


def test_path_less_manifest_is_listed_with_video_unavailable(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_diagnostic_manifest(clip_store, "clip-broken", video_error="no working codec")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")

    assert listed.status_code == 200
    clips = listed.json()["clips"]
    assert [clip["clip_id"] for clip in clips] == ["clip-broken"]
    row = clips[0]
    assert row["video_available"] is False
    assert row["path"] is None
    assert row["video_error"] == "no working codec"


def test_reason_code_is_surfaced_as_the_clips_failure_reason(clip_env) -> None:
    """#165: the worker writes the failure reason as `reason_code`
    (worker/pipeline/output/evidence/manifest_models.py:93), never as
    `video_error` -- so `GET /clips` must read `reason_code`, not the key
    the worker never populates.
    """
    clip_store = clip_env / "clip-store"
    clip_dir = clip_store / "clips" / "clip-no-frames"
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": "clip-no-frames",
        "camera_id": "camera-1",
        "event_ref": "event-clip-no-frames",
        "event_type": "bed-exit",
        "started_at": "2026-07-06T00:05:00Z",
        "duration_s": 0.0,
        "codec": None,
        "path": None,
        "video_available": False,
        "reason_code": "NO_FRAMES",
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")

    row = listed.json()["clips"][0]
    assert row["video_available"] is False
    assert row["video_error"] == "NO_FRAMES"


def test_reason_code_takes_priority_over_a_legacy_video_error_field(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    clip_dir = clip_store / "clips" / "clip-both-fields"
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": "clip-both-fields",
        "camera_id": "camera-1",
        "event_ref": "event-clip-both-fields",
        "event_type": "bed-exit",
        "started_at": "2026-07-06T00:05:00Z",
        "duration_s": 0.0,
        "codec": None,
        "path": None,
        "video_available": False,
        "video_error": "stale legacy reason",
        "reason_code": "ENCODER_FAILED",
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")

    row = listed.json()["clips"][0]
    assert row["video_error"] == "ENCODER_FAILED"


def test_legacy_manifest_without_field_defaults_video_available_true(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    clip_dir = clip_store / "clips" / "clip-legacy"
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip.mp4").write_bytes(b"video:clip-legacy")
    payload = {
        "clip_id": "clip-legacy",
        "camera_id": "camera-1",
        "event_ref": "event-legacy",
        "event_type": "fall",
        "started_at": "2026-07-06T00:00:00Z",
        "duration_s": 30.0,
        "codec": "h264",
        "path": "clips/clip-legacy",
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")

    row = listed.json()["clips"][0]
    assert row["video_available"] is True
    assert row["video_error"] is None


def test_video_endpoint_404_but_manifest_stays_listed(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_diagnostic_manifest(clip_store, "clip-broken", video_error="decode failed")
    _write_playable_manifest(clip_store, "clip-ok")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        broken_video = client.get("/api/v1/clips/clip-broken/video")
        ok_video = client.get("/api/v1/clips/clip-ok/video")
        listed = client.get("/api/v1/clips")

    assert broken_video.status_code == 404
    assert ok_video.status_code == 200
    assert ok_video.content == b"video:clip-ok"
    listed_ids = {clip["clip_id"] for clip in listed.json()["clips"]}
    assert listed_ids == {"clip-broken", "clip-ok"}


def test_media_type_maps_real_suffixes() -> None:
    assert _media_type("clip.mp4") == "video/mp4"
    assert _media_type("clip.webm") == "video/webm"
    assert _media_type("clip.mkv") == "video/x-matroska"
    assert _media_type("clip.avi") == "video/x-msvideo"
    assert _media_type("clip.mov") == "video/quicktime"
    assert _media_type("clip.m4v") == "video/x-m4v"
    assert _media_type("clip.MP4") == "video/mp4"
    assert _media_type("clip.txt") == "application/octet-stream"
    assert _media_type("clip") == "application/octet-stream"
