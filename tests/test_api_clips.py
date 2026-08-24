from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Self, TypedDict

import pytest
from fastapi.testclient import TestClient
from receipt_helpers import add_accepted_media_receipts

from backend.app.features.connection.store import (
    API_CONNECTION_SETTINGS_PATH_ENV,
    ConnectionSettingsStore,
)
from backend.app.lifespan import apply_connection_settings
from backend.app.main import create_app as _create_app
from backend.app.main import no_lifespan
from tests_support.compact_authority_db import prepare_compact_database


def create_app(*, lifespan):
    app = _create_app(lifespan=lifespan)
    add_accepted_media_receipts(app)
    return app


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


class BackupCall(TypedDict):
    url: str
    method: str
    authorization: str | None
    timeout: float
    body: dict[str, object]


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


def test_clip_keyset_pages_equal_timestamps_without_skip_or_duplicate(clip_env) -> None:
    # Given: three verified manifests with the same start timestamp.
    clip_store = clip_env / "clip-store"
    for clip_id in ("clip-a", "clip-b", "clip-c"):
        _write_manifest(clip_store, clip_id, started_at="2026-07-06T00:00:00Z")

    # When: a dashboard traverses one-row keyset pages and probes a malformed cursor.
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        seen: list[str] = []
        cursor: str | None = None
        while True:
            params = {"limit": 1}
            if cursor is not None:
                params["cursor"] = cursor
            response = client.get("/api/v1/clips", params=params)
            assert response.status_code == 200
            body = response.json()
            seen.extend(clip["clip_id"] for clip in body["clips"])
            cursor = body["pagination"]["next_cursor"]
            if cursor is None:
                break
        malformed = client.get("/api/v1/clips", params={"limit": 1, "cursor": "%%%"})

    # Then: (started_at, clip_id) is unique and malformed cursors fail closed.
    assert seen == ["clip-c", "clip-b", "clip-a"]
    assert malformed.status_code == 400


def test_manifest_rebuild_rolls_back_when_one_tuple_is_invalid(clip_env) -> None:
    # Given: one valid manifest followed by a duration outside schema 18's clip bound.
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-a", started_at="2026-07-06T00:00:00Z")
    _write_manifest(clip_store, "clip-b", started_at="2026-07-06T00:00:01Z")
    invalid_path = clip_store / "clips" / "clip-b" / "manifest.json"
    invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
    invalid["duration_s"] = 121.0
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")

    # When: the request rebuilds both facts in one real SQLite transaction.
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips", params={"limit": 10})

    # Then: the request is not misleadingly successful and no partial clip row commits.
    assert response.status_code == 503
    with sqlite3.connect(clip_env / ".central-fixture" / "edge.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM clips").fetchone() == (0,)


def test_compact_rebuild_removes_stale_manifest_from_page_total_and_facets(clip_env) -> None:
    # Given: one manifest has been reconciled into compact clips.
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "stale")
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        first = client.get("/api/v1/clips", params={"limit": 10})
        assert first.status_code == 200
        assert first.json()["pagination"]["total"] == 1

        # When: filesystem truth removes the complete manifest/media directory.
        shutil.rmtree(clip_store / "clips" / "stale")
        rebuilt = client.get("/api/v1/clips", params={"limit": 10})

    # Then: page, total, and facets come from the same reconciled visible set.
    assert rebuilt.status_code == 200
    assert rebuilt.json()["clips"] == []
    assert rebuilt.json()["pagination"] == {
        "limit": 10,
        "offset": 0,
        "total": 0,
        "has_more": False,
        "next_cursor": None,
    }
    assert rebuilt.json()["event_type_counts"] == {}


def test_stale_referenced_clip_is_retained_unavailable_but_hidden(clip_env) -> None:
    # Given: a reconciled clip is retained by PRIMARY_CLIP history.
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "history")
    database = clip_env / ".central-fixture" / "edge.sqlite3"
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        assert client.get("/api/v1/clips", params={"limit": 10}).status_code == 200
        with sqlite3.connect(database) as connection:
            clip = connection.execute(
                "SELECT media_sha256,media_size_bytes,media_relpath FROM clips "
                "WHERE clip_id='history'"
            ).fetchone()
            connection.execute(
                """
                INSERT INTO incidents (
                    incident_id,edge_event_id,facility_id,camera_id,event_type,detected_at,
                    lifecycle_state,provenance_state,provenance_missing_reason,
                    review_version,revision,created_at,updated_at
                ) VALUES ('incident-history','event-history','facility-1','camera-1','fall',
                          '2026-07-06T00:00:00Z','OPEN','MISSING','NOT_RECORDED',0,1,
                          '2026-07-06T00:00:00Z','2026-07-06T00:00:00Z')
                """
            )
            connection.execute(
                """
                INSERT INTO artifacts (
                    incident_id,kind,artifact_id,clip_id,state,contained_relpath,
                    content_sha256,size_bytes,mime_type,codec,revision,created_at,updated_at
                ) VALUES ('incident-history','PRIMARY_CLIP','artifact-history','history',
                          'AVAILABLE',?,?,?,'video/mp4','h264',1,
                          '2026-07-06T00:00:00Z','2026-07-06T00:00:00Z')
                """,
                (clip[2], clip[0], clip[1]),
            )
        shutil.rmtree(clip_store / "clips" / "history")

        # When: compact reconciliation observes the missing filesystem fact.
        rebuilt = client.get("/api/v1/clips", params={"limit": 10})

    # Then: history remains referentially intact but is absent from every listing projection.
    assert rebuilt.status_code == 200
    assert rebuilt.json()["pagination"]["total"] == 0
    assert rebuilt.json()["event_type_counts"] == {}
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT local_state,local_reason,manifest_relpath,media_relpath FROM clips "
            "WHERE clip_id='history'"
        ).fetchone()
        relation = connection.execute(
            "SELECT clip_id FROM artifacts WHERE artifact_id='artifact-history'"
        ).fetchone()
    assert row == ("UNAVAILABLE", "MANIFEST_MISSING", None, None)
    assert relation == ("history",)


def test_compact_rebuild_rejects_changed_identity_without_mutating_row(clip_env) -> None:
    # Given: one immutable manifest/media identity has been reconciled.
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "stable")
    database = clip_env / ".central-fixture" / "edge.sqlite3"
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        first = client.get("/api/v1/clips", params={"limit": 10})
        assert first.status_code == 200
        with sqlite3.connect(database) as connection:
            before = connection.execute(
                "SELECT media_sha256, media_size_bytes, publish_state FROM clips "
                "WHERE clip_id='stable'"
            ).fetchone()

        # When: bytes change under the same immutable clip identity.
        (clip_store / "clips" / "stable" / "clip.mp4").write_bytes(b"changed-media")
        conflict = client.get("/api/v1/clips", params={"limit": 10})

    # Then: conflict is deterministic and the prior compact tuple is unchanged.
    assert conflict.status_code == 503
    with sqlite3.connect(database) as connection:
        after = connection.execute(
            "SELECT media_sha256, media_size_bytes, publish_state FROM clips "
            "WHERE clip_id='stable'"
        ).fetchone()
    assert after == before


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
    video_events = [
        event for event in audit.json()["events"] if event["action"] == "clip.play"
    ]
    assert [(event["actor_id"], event["target_id"]) for event in video_events] == [
        ("admin", "clip-1"),
        ("admin", "clip-1"),
    ]


def test_list_clips_and_audit_view_are_recorded_in_the_audit_log(clip_env) -> None:
    """List and audit-history access each append one closed-catalog event."""
    _write_manifest(clip_env / "clip-store", "clip-1")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        first_audit = client.get("/api/v1/audit")
        second_audit = client.get("/api/v1/audit")

    assert listed.status_code == 200
    assert first_audit.status_code == 200
    first_actions = [event["action"] for event in first_audit.json()["events"]]
    second_actions = [event["action"] for event in second_audit.json()["events"]]
    assert first_actions[:2] == ["clip.list", "auth.login"]
    assert second_actions[:3] == ["audit.list", "clip.list", "auth.login"]


def test_list_clips_returns_200_without_api_label_store_env_set(
    clip_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clip listing no longer creates legacy JSONL when label storage is unset."""
    monkeypatch.delenv("API_LABEL_STORE", raising=False)
    _write_manifest(clip_env / "clip-store", "clip-1")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert [clip["clip_id"] for clip in response.json()["clips"]] == ["clip-1"]
    audit_path = clip_env / ".local" / "state" / "ml-api" / "labels" / "audit.jsonl"
    assert not audit_path.exists()


def test_list_clips_succeeds_even_when_the_audit_log_is_unwritable(
    clip_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #152: ``list_clips`` (``router.py``) appends a "list" audit entry
    as a side effect of an otherwise pure read. Before this fix, an
    unwritable audit log (e.g. the real ``/var/lib/ml-api-labels`` default
    outside a container) crashed that append with an unhandled
    ``PermissionError``, turning ``GET /clips`` into a 500 -- the operations
    page couldn't load event history at all. The audit append is now
    best-effort, so listing must succeed regardless."""
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-1")
    audit_path = clip_env / "label-store" / "audit.jsonl"
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        if self == audit_path:
            raise PermissionError("audit log mount unavailable")
        return original_open(self, *args, **kwargs)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        monkeypatch.setattr(Path, "open", guarded_open)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert [clip["clip_id"] for clip in response.json()["clips"]] == ["clip-1"]
    # The append attempted to write and failed -- the file was never created.
    assert not audit_path.exists()


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
    assert all(event["action"] != "label" for event in audit.json()["events"])


def test_label_clip_signals_degradation_when_label_store_is_unwritable(
    clip_env, monkeypatch
) -> None:
    """A degraded ``LabelStore.save`` (e.g. an unwritable labels dir) must not
    be reported to the caller as a successful 200 label write -- see #50.

    It also must not report the label as saved *elsewhere*: a failed local
    save must short-circuit before the best-effort backend backup POST and
    before the audit log records a "label" action. Without an enrollment
    bundle, the trailing ``GET /audit`` remains local and makes no cloud call."""
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

    monkeypatch.setenv("API_BACKEND_CLIP_EVENTS_URL", "http://backend/api/v1/clip-events")
    monkeypatch.setenv("API_BACKEND_BASE_URL", "http://backend/api")
    monkeypatch.setenv("API_BACKEND_FACILITY_TOKEN", "facility-token")
    backup_calls: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        backup_calls.append(
            {
                "url": request.full_url,
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeHTTPResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
        response = client.put(
            "/api/v1/clips/clip-1/label",
            json={"label": "TRUE_POSITIVE", "reviewer": "reviewer-1"},
        )
        audit = client.get("/api/v1/audit")

    assert response.status_code == 503
    assert not (labels_dir / "clip-1.json").exists()
    assert all(event["action"] != "label" for event in audit.json()["events"])
    assert backup_calls == []


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
    monkeypatch.setenv("API_BACKEND_BASE_URL", "http://backend/api")
    monkeypatch.setenv("API_BACKEND_FACILITY_TOKEN", "facility-token")
    calls: list[BackupCall] = []

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
    monkeypatch.setenv(
        API_CONNECTION_SETTINGS_PATH_ENV,
        str(clip_env / "connection-settings.sqlite3"),
    )
    prepare_compact_database(clip_env / "connection-settings.sqlite3")
    _ = ConnectionSettingsStore.from_env().save(
        {
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
            "facility_token": "facility-token",
            "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollment_generation": 3,
        }
    )
    app = create_app(lifespan=no_lifespan)
    apply_connection_settings(app)

    with TestClient(app) as client:
        _login(client)
        response = client.put(
            "/api/v1/clips/clip-1/label",
            json={"label": "FALSE_POSITIVE", "reviewer": "reviewer-1"},
        )

    assert response.status_code == 200
    assert [call["body"]["type"] for call in calls] == ["clip_label"]
    assert {call["authorization"] for call in calls} == {"Bearer facility-token"}
    assert {call["url"] for call in calls} == {"http://backend/api/v1/clip-events"}
