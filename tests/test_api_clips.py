from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan

# Dashboard auth now always resolves to a session store (persisted file > env
# > the built-in admin/admin default, see backend/app/shared/dashboard_auth.py),
# so a bare worker relay/bearer token is never sufficient on its own -- these
# tests log in as the zero-config default and rely on the TestClient's cookie
# jar to carry the session across subsequent calls.
DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


class FakeHTTPResponse:
    status = 202

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def _write_manifest(
    clip_store,
    clip_id: str,
    *,
    camera_id: str = "camera-1",
    event_ref: str | None = None,
    event_type: str | None = "fall",
    started_at: str = "2026-07-06T00:00:00Z",
    path: str | None = None,
    finalized: bool = True,
) -> None:
    clip_dir = clip_store / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip.mp4").write_bytes(f"video:{clip_id}".encode())
    payload = {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "event_ref": event_ref or f"event-{clip_id}",
        "started_at": started_at,
        "duration_s": 30.0,
        "codec": "h264",
        "path": path or f"clips/{clip_id}",
        "finalized": finalized,
    }
    if event_type is not None:
        payload["event_type"] = event_type
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


def test_list_clips_returns_only_finalized_latest_first_and_filters_camera(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(
        clip_store,
        "clip-old",
        camera_id="camera-1",
        started_at="2026-07-06T00:00:00Z",
    )
    _write_manifest(
        clip_store,
        "clip-new",
        camera_id="camera-2",
        started_at="2026-07-06T00:01:00Z",
    )
    _write_manifest(
        clip_store,
        "clip-open",
        camera_id="camera-1",
        started_at="2026-07-06T00:02:00Z",
        finalized=False,
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        filtered = client.get("/api/v1/clips", params={"camera_id": "camera-1"})

    assert listed.status_code == 200
    assert [clip["clip_id"] for clip in listed.json()["clips"]] == ["clip-new", "clip-old"]
    assert listed.json()["clips"][0]["event_type"] == "fall"
    assert filtered.status_code == 200
    assert [clip["clip_id"] for clip in filtered.json()["clips"]] == ["clip-old"]


def test_list_clips_preserves_event_type_when_event_ref_is_identity(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(
        clip_store,
        "clip-bed-exit",
        event_ref="0:0",
        event_type="bed-exit",
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert response.json()["clips"][0]["event_ref"] == "0:0"
    assert response.json()["clips"][0]["event_type"] == "bed-exit"


def test_streams_manifest_video_and_appends_audit(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-1")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        video = client.get("/api/v1/clips/clip-1/video")
        query_video = client.get("/api/v1/clips/clip-1/video", params={"token": "relay-token"})
        audit = client.get("/api/v1/audit")

    assert video.status_code == 200
    assert video.content == b"video:clip-1"
    assert video.headers["content-type"].startswith("video/mp4")
    assert query_video.status_code == 200
    assert query_video.content == b"video:clip-1"
    assert audit.status_code == 200
    # Both requests carry the same dashboard session cookie, so both are
    # attributed to the real session actor regardless of the (now-vestigial)
    # query token also present on the second call.
    assert audit.json()["entries"] == [
        {
            "ts": audit.json()["entries"][0]["ts"],
            "actor": "admin",
            "action": "play",
            "clip_id": "clip-1",
        },
        {
            "ts": audit.json()["entries"][1]["ts"],
            "actor": "admin",
            "action": "play",
            "clip_id": "clip-1",
        },
    ]


def test_label_clip_saves_sidecar_and_audit(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    label_store = clip_env / "label-store"
    _write_manifest(clip_store, "clip-1")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.put(
            "/api/v1/clips/clip-1/label",
            json={"label": "TRUE_POSITIVE", "reviewer": "reviewer-1"},
        )
        clear_response = client.put(
            "/api/v1/clips/clip-1/label",
            json={"label": None, "reviewer": "reviewer-2"},
        )
        default_reviewer = client.put(
            "/api/v1/clips/clip-1/label",
            json={"label": "FALSE_POSITIVE"},
        )
        audit = client.get("/api/v1/audit")

    assert response.status_code == 200
    assert response.json()["label"] == "TRUE_POSITIVE"
    assert clear_response.status_code == 200
    assert clear_response.json()["label"] is None
    assert default_reviewer.status_code == 200
    # No reviewer supplied -> defaults to the authenticated actor (the
    # dashboard session username, not a legacy bearer/operator placeholder).
    assert default_reviewer.json()["reviewer"] == "admin"
    saved = json.loads((label_store / "labels" / "clip-1.json").read_text(encoding="utf-8"))
    assert saved["label"] == "FALSE_POSITIVE"
    assert saved["reviewer"] == "admin"
    audit_rows = [
        (entry["actor"], entry["action"], entry["clip_id"]) for entry in audit.json()["entries"]
    ]
    assert audit_rows == [
        ("reviewer-1", "label", "clip-1"),
        ("reviewer-2", "label", "clip-1"),
        ("admin", "label", "clip-1"),
    ]


def test_label_clip_signals_degradation_when_label_store_is_unwritable(
    clip_env, monkeypatch
) -> None:
    """A degraded ``LabelStore.save`` (e.g. an unwritable labels dir) must not
    be reported to the caller as a successful 200 label write -- see #50."""
    clip_store = clip_env / "clip-store"
    label_store_root = clip_env / "label-store"
    _write_manifest(clip_store, "clip-1")
    label_store_root.mkdir(parents=True)
    labels_dir = label_store_root / "labels"
    original_mkdir = Path.mkdir

    def guarded_mkdir(self: Path, *, parents: bool = False, exist_ok: bool = False) -> None:
        if self == labels_dir:
            raise PermissionError("labels mount unavailable")
        original_mkdir(self, parents=parents, exist_ok=exist_ok)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
        response = client.put(
            "/api/v1/clips/clip-1/label",
            json={"label": "TRUE_POSITIVE", "reviewer": "reviewer-1"},
        )

    assert response.status_code == 503
    assert not (labels_dir / "clip-1.json").exists()


def test_clip_routes_require_a_dashboard_session(clip_env) -> None:
    """A bare worker relay token or forged bearer token is never a substitute
    for a real dashboard session -- the legacy bypass is unreachable now that
    dashboard auth always resolves (persisted file > env > built-in default)."""
    _write_manifest(clip_env / "clip-store", "clip-1")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        unauthenticated = client.get("/api/v1/clips")
        wrong_bearer = client.get("/api/v1/clips", headers={"Authorization": "Bearer wrong"})
        worker_relay_token = client.get(
            "/api/v1/clips", headers={"Authorization": "Bearer relay-token"}
        )

    assert unauthenticated.status_code == 401
    assert wrong_bearer.status_code == 401
    assert worker_relay_token.status_code == 401


def test_video_rejects_manifest_path_escape(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    (clip_env / "secret.mp4").write_bytes(b"secret")
    _write_manifest(clip_store, "clip-escape", path="../secret.mp4")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/clip-escape/video")
        invalid_id = client.get("/api/v1/clips/%2E%2E/video")

    assert response.status_code == 400
    assert invalid_id.status_code == 400


def test_list_clips_includes_size_bytes_stat_from_the_resolved_video_file(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-1")
    video_path = clip_store / "clips" / "clip-1" / "clip.mp4"

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert response.json()["clips"][0]["size_bytes"] == video_path.stat().st_size


def test_list_clips_size_bytes_is_null_when_video_is_unavailable(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    clip_dir = clip_store / "clips" / "clip-no-video"
    clip_dir.mkdir(parents=True)
    payload = {
        "clip_id": "clip-no-video",
        "camera_id": "camera-1",
        "event_ref": "event-clip-no-video",
        "started_at": "2026-07-06T00:00:00Z",
        "duration_s": 10.0,
        "codec": "h264",
        "path": None,
        "video_available": False,
        "video_error": "encode failed",
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert response.json()["clips"][0]["size_bytes"] is None


def test_list_clips_defaults_missing_duration_s_to_zero_instead_of_dropping_the_manifest(
    clip_env,
) -> None:
    clip_store = clip_env / "clip-store"
    clip_dir = clip_store / "clips" / "clip-no-duration"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"video")
    payload = {
        "clip_id": "clip-no-duration",
        "camera_id": "camera-1",
        "event_ref": "event-clip-no-duration",
        "started_at": "2026-07-06T00:00:00Z",
        "codec": "h264",
        "path": "clips/clip-no-duration",
        "finalized": True,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    clips = response.json()["clips"]
    assert [clip["clip_id"] for clip in clips] == ["clip-no-duration"]
    assert clips[0]["duration_s"] == 0.0


def test_list_clips_finds_manifests_under_the_root_and_subdirectory_layouts(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    # Root layout: root/clips/<id>/manifest.json.
    _write_manifest(clip_store, "clip-root", started_at="2026-07-06T00:00:00Z")
    # First-level subdir layout: root/<sub>/clips/<id>/manifest.json.
    _write_manifest(
        clip_store / "backup-drive",
        "clip-first-level",
        started_at="2026-07-06T00:01:00Z",
        path="backup-drive/clips/clip-first-level",
    )
    # Second-level subdir layout: root/<sub2>/<sub1>/clips/<id>/manifest.json.
    _write_manifest(
        clip_store / "external" / "drive-1",
        "clip-second-level",
        started_at="2026-07-06T00:02:00Z",
        path="external/drive-1/clips/clip-second-level",
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    clip_ids = {clip["clip_id"] for clip in response.json()["clips"]}
    assert clip_ids == {"clip-root", "clip-first-level", "clip-second-level"}


def test_label_and_audit_backend_backup_is_best_effort(
    clip_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_manifest(clip_env / "clip-store", "clip-1")
    monkeypatch.setenv("API_BACKEND_CLIP_EVENTS_URL", "http://backend/api/v1/clip-events")
    monkeypatch.setenv("API_BACKEND_FACILITY_TOKEN", "facility-token")
    calls: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "authorization": request.headers.get("Authorization"),
                "timeout": timeout,
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeHTTPResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.put(
            "/api/v1/clips/clip-1/label",
            json={"label": "FALSE_POSITIVE", "reviewer": "reviewer-1"},
        )

    assert response.status_code == 200
    assert [call["body"]["type"] for call in calls] == ["clip_label", "clip_audit"]
    assert {call["authorization"] for call in calls} == {"Bearer facility-token"}
    assert {call["url"] for call in calls} == {"http://backend/api/v1/clip-events"}
