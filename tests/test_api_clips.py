from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from hashlib import sha256

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from receipt_helpers import add_accepted_media_receipts

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.clips.store import (
    PLAYBACK_H264_MANIFEST_FILENAME,
    ClipStore,
)
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_clip_app import index_clips
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def _write_playback_bundle(clip_dir, *, pts_identical: bool) -> tuple[object, str]:
    rendition_bytes = b"browser-safe"
    digest = sha256(rendition_bytes).hexdigest()
    rendition = clip_dir / f"clip.playback-h264.{digest[:16]}.mp4"
    rendition.write_bytes(rendition_bytes)
    original_digest = sha256((clip_dir / "clip.mp4").read_bytes()).hexdigest()
    (clip_dir / PLAYBACK_H264_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "rendition": rendition.name,
                "rendition_sha256": digest,
                "source_sha256": original_digest,
                "pts_identical": pts_identical,
                "time_base": "1/1000",
                "frames": 1,
                "source_frames": 1,
            }
        ),
        encoding="ascii",
    )
    return rendition, digest


# Dashboard auth always resolves to a session store (bootstrapped from the
# API_DASHBOARD_* pair the suite sets to admin/admin), so a bare worker
# relay/bearer token is never sufficient on its own -- these tests log in and
# rely on the TestClient's cookie jar to carry the session across calls.
DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


def _write_manifest(
    clip_store,
    clip_id: str,
    *,
    camera_id: str = "camera-1",
    event_ref: str | None = None,
    event_type: str | None = "fall",
    started_at: str = "2026-07-06T00:00:00Z",
    detected_at: str | None = None,
    truncation_reasons: list[str] | None = None,
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
    if detected_at is not None:
        payload["detected_at"] = detected_at
    if truncation_reasons is not None:
        payload["truncation_reasons"] = truncation_reasons
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


@pytest.fixture
def make_app(
    clip_env,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> Callable[[], FastAPI]:
    """Each call is a new app on the one sandbox, with receipts for the clips written so far."""

    def make() -> FastAPI:
        app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
        add_accepted_media_receipts(app)
        return app

    return make


def test_list_clips_returns_only_finalized_latest_first_and_filters_camera(
    clip_env, make_app
) -> None:
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

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        filtered = client.get("/api/v1/clips", params={"camera_id": "camera-1"})

    assert listed.status_code == 200
    assert [clip["clip_id"] for clip in listed.json()["clips"]] == ["clip-new", "clip-old"]
    assert listed.json()["clips"][0]["event_type"] == "fall"
    assert filtered.status_code == 200
    assert [clip["clip_id"] for clip in filtered.json()["clips"]] == ["clip-old"]


def test_clip_keyset_pages_equal_timestamps_without_skip_or_duplicate(clip_env, make_app) -> None:
    # Given: three verified manifests with the same start timestamp.
    clip_store = clip_env / "clip-store"
    for clip_id in ("clip-a", "clip-b", "clip-c"):
        _write_manifest(clip_store, clip_id, started_at="2026-07-06T00:00:00Z")

    # When: a dashboard traverses one-row keyset pages and probes a malformed cursor.
    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
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


def test_manifest_rebuild_isolates_one_invalid_tuple(
    clip_env, make_app, postgres_product_sandbox: ProductSandbox
) -> None:
    # Given: one valid manifest followed by a duration outside the catalogue's clip bound.
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-a", started_at="2026-07-06T00:00:00Z")
    _write_manifest(clip_store, "clip-b", started_at="2026-07-06T00:00:01Z")
    invalid_path = clip_store / "clips" / "clip-b" / "manifest.json"
    invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
    invalid["duration_s"] = 121.0
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")

    # When: the catalogue indexes both manifests and the dashboard lists clips.
    app = make_app()
    outcomes = index_clips(app)
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips", params={"limit": 10})

    # Then: the invalid tuple is isolated and never commits; the valid clip still lists.
    assert sum(outcome.isolated for outcome in outcomes) == 1
    assert response.status_code == 200
    assert [clip["clip_id"] for clip in response.json()["clips"]] == ["clip-a"]
    rows = postgres_product_sandbox.admin.execute(
        "SELECT clip_id FROM clips ORDER BY clip_id"
    ).fetchall()
    assert rows == [("clip-a",)]


def test_compact_rebuild_removes_stale_manifest_from_page_total_and_facets(
    clip_env, make_app
) -> None:
    # Given: one manifest has been reconciled into the catalogue.
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "stale")
    app = make_app()
    index_clips(app)
    with TestClient(app) as client:
        _login(client)
        first = client.get("/api/v1/clips", params={"limit": 10})
        assert first.status_code == 200
        assert first.json()["pagination"]["total"] == 1

        # When: filesystem truth removes the complete manifest/media directory.
        shutil.rmtree(clip_store / "clips" / "stale")
        index_clips(app)
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


def test_stale_referenced_clip_is_retained_unavailable_but_hidden(
    clip_env, make_app, postgres_product_sandbox: ProductSandbox
) -> None:
    # Given: an indexed clip is retained by PRIMARY_CLIP history.
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "history")
    admin = postgres_product_sandbox.admin
    app = make_app()
    index_clips(app)
    clip = admin.execute(
        "SELECT media_sha256, media_size_bytes, media_relpath FROM clips WHERE clip_id = %s",
        ("history",),
    ).fetchone()
    assert clip is not None
    recorded_at = "2026-07-06T00:00:00Z"
    with admin.transaction():
        admin.execute(
            """
            INSERT INTO incidents (
                incident_id, edge_event_id, facility_id, camera_id, event_type, detected_at,
                lifecycle_state, provenance_state, provenance_missing_reason,
                review_version, revision, created_at, updated_at
            ) VALUES ('incident-history', 'event-history', 'facility-1', 'camera-1', 'fall',
                      %s, 'OPEN', 'MISSING', 'NOT_RECORDED', 0, 1, %s, %s)
            """,
            (recorded_at, recorded_at, recorded_at),
        )
        admin.execute(
            """
            INSERT INTO artifacts (
                incident_id, kind, artifact_id, clip_id, state, contained_relpath,
                content_sha256, size_bytes, mime_type, codec, revision, created_at, updated_at
            ) VALUES ('incident-history', 'PRIMARY_CLIP', 'artifact-history', 'history',
                      'AVAILABLE', %s, %s, %s, 'video/mp4', 'h264', 1, %s, %s)
            """,
            (clip[2], clip[0], clip[1], recorded_at, recorded_at),
        )
    shutil.rmtree(clip_store / "clips" / "history")

    # When: the catalogue re-indexes the missing filesystem fact and the dashboard lists.
    index_clips(app)
    with TestClient(app) as client:
        _login(client)
        rebuilt = client.get("/api/v1/clips", params={"limit": 10})

    # Then: history remains referentially intact but is absent from every listing projection.
    assert rebuilt.status_code == 200
    assert rebuilt.json()["pagination"]["total"] == 0
    assert rebuilt.json()["event_type_counts"] == {}
    row = admin.execute(
        "SELECT local_state, local_reason, manifest_relpath, media_relpath FROM clips "
        "WHERE clip_id = %s",
        ("history",),
    ).fetchone()
    relation = admin.execute(
        "SELECT clip_id FROM artifacts WHERE artifact_id = %s", ("artifact-history",)
    ).fetchone()
    assert row == ("UNAVAILABLE", "MANIFEST_MISSING", None, None)
    assert relation == ("history",)


def test_compact_rebuild_rejects_changed_identity_without_mutating_row(
    clip_env, make_app, postgres_product_sandbox: ProductSandbox
) -> None:
    # Given: one immutable manifest/media identity has been indexed.
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "stable")
    admin = postgres_product_sandbox.admin
    identity_sql = (
        "SELECT media_sha256, media_size_bytes, publish_state FROM clips WHERE clip_id = %s"
    )
    app = make_app()
    index_clips(app)
    before = admin.execute(identity_sql, ("stable",)).fetchone()

    # When: bytes change under the same immutable clip identity and the catalogue re-indexes.
    (clip_store / "clips" / "stable" / "clip.mp4").write_bytes(b"changed-media")
    index_clips(app)
    with TestClient(app) as client:
        _login(client)
        conflict = client.get("/api/v1/clips", params={"limit": 10})

    # Then: the conflict is recorded as corruption and the prior identity is unchanged.
    assert conflict.status_code == 200
    assert admin.execute(identity_sql, ("stable",)).fetchone() == before
    state = admin.execute(
        "SELECT local_state, local_reason FROM clips WHERE clip_id = %s", ("stable",)
    ).fetchone()
    assert state == ("CORRUPT", "IDENTITY_CONFLICT")


def test_list_clips_preserves_event_type_when_event_ref_is_identity(clip_env, make_app) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(
        clip_store,
        "clip-bed-exit",
        event_ref="0:0",
        event_type="bed-exit",
    )

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert response.json()["clips"][0]["event_ref"] == "0:0"
    assert response.json()["clips"][0]["event_type"] == "bed-exit"


def test_clip_responses_tolerate_prechange_and_detected_at_manifests(clip_env, make_app) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "older")
    _write_manifest(
        clip_store,
        "current",
        detected_at="2026-07-06T00:00:12Z",
        truncation_reasons=["POSTROLL_LIMIT"],
    )

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        older = client.get("/api/v1/clips/older/metadata")
        current = client.get("/api/v1/clips/current/metadata")

    assert listed.status_code == older.status_code == current.status_code == 200
    assert older.json()["detected_at"] is None
    assert older.json()["truncation_reasons"] == []
    assert current.json()["detected_at"] == "2026-07-06T00:00:12Z"
    assert current.json()["truncation_reasons"] == ["POSTROLL_LIMIT"]


def test_removed_clip_scene_route_returns_not_found(clip_env, make_app) -> None:
    with TestClient(make_app()) as client:
        _login(client)
        get = client.get("/api/v1/clips/clip-1/scene")
        head = client.head("/api/v1/clips/clip-1/scene")

    assert get.status_code == head.status_code == 404


def test_streams_manifest_video_and_appends_audit(clip_env, make_app) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-1")

    with TestClient(make_app()) as client:
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
    video_events = [event for event in audit.json()["events"] if event["action"] == "clip.play"]
    assert [(event["actor_id"], event["target_id"]) for event in video_events] == [
        ("admin", "clip-1"),
        ("admin", "clip-1"),
    ]


def test_playback_identity_reports_manifest_timing_status(clip_env) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-identity")
    clip_dir = clip_store / "clips" / "clip-identity"
    original = clip_dir / "clip.mp4"
    manifest = clip_dir / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sha256"] = sha256(original.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    rendition, _ = _write_playback_bundle(clip_dir, pts_identical=False)
    store = ClipStore(clip_store)
    located = store.locate_manifest("clip-identity")
    assert located is not None

    identity = store.open_located_playback_identity(located)
    try:
        assert identity.opened.path == rendition
        assert identity.original_sha256 == sha256(original.read_bytes()).hexdigest()
        assert identity.served_kind == "rendition"
        assert identity.served_pts_identical is False
    finally:
        identity.opened.handle.close()

    _write_playback_bundle(clip_dir, pts_identical=True)
    identity = store.open_located_playback_identity(located)
    try:
        assert identity.served_kind == "rendition"
        assert identity.served_pts_identical is True
    finally:
        identity.opened.handle.close()


@pytest.mark.parametrize("tamper", ["source_sha256", "rendition_sha256"])
def test_playback_manifest_with_unbound_media_falls_back_to_original(clip_env, tamper: str) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-unbound")
    clip_dir = clip_store / "clips" / "clip-unbound"
    original = clip_dir / "clip.mp4"
    manifest_path = clip_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_digest = sha256(original.read_bytes()).hexdigest()
    manifest["sha256"] = original_digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, rendition_digest = _write_playback_bundle(clip_dir, pts_identical=True)
    bundle_path = clip_dir / PLAYBACK_H264_MANIFEST_FILENAME
    bundle = json.loads(bundle_path.read_text(encoding="ascii"))
    bundle[tamper] = "a" * 64
    bundle_path.write_text(json.dumps(bundle), encoding="ascii")
    store = ClipStore(clip_store)
    located = store.locate_manifest("clip-unbound")
    assert located is not None

    identity = store.open_located_playback_identity(located)
    try:
        assert identity.opened.path == original
        assert identity.served_media_sha256 == original_digest
        assert identity.served_media_sha256 != rendition_digest
        assert identity.served_kind == "original"
        assert identity.served_pts_identical is True
    finally:
        identity.opened.handle.close()


@pytest.mark.parametrize("with_rendition", [False, True])
def test_video_media_parameter_binds_range_request_to_served_bytes(
    clip_env, make_app, with_rendition: bool
) -> None:
    clip_store = clip_env / "clip-store"
    clip_id = "clip-rendition" if with_rendition else "clip-original"
    _write_manifest(clip_store, clip_id)
    clip_dir = clip_store / "clips" / clip_id
    original = clip_dir / "clip.mp4"
    manifest_path = clip_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = sha256(original.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    served_digest = manifest["sha256"]
    if with_rendition:
        _, served_digest = _write_playback_bundle(clip_dir, pts_identical=True)

    with TestClient(make_app()) as client:
        _login(client)
        matched = client.get(
            f"/api/v1/clips/{clip_id}/video",
            params={"media": served_digest},
            headers={"Range": "bytes=0-1"},
        )
        mismatched = client.get(
            f"/api/v1/clips/{clip_id}/video",
            params={"media": "e" * 64},
        )
        head_mismatched = client.head(
            f"/api/v1/clips/{clip_id}/video",
            params={"media": "e" * 64},
        )

    assert matched.status_code == 206
    assert mismatched.status_code == head_mismatched.status_code == 409
    assert mismatched.json() == {"detail": "media_mismatch"}


def test_list_clips_and_audit_view_are_recorded_in_the_audit_log(clip_env, make_app) -> None:
    """List and audit-history access each append one closed-catalog event."""
    _write_manifest(clip_env / "clip-store", "clip-1")

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
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
    clip_env, make_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clip listing no longer creates legacy JSONL when label storage is unset."""
    monkeypatch.delenv("API_LABEL_STORE", raising=False)
    _write_manifest(clip_env / "clip-store", "clip-1")

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert [clip["clip_id"] for clip in response.json()["clips"]] == ["clip-1"]
    audit_path = clip_env / ".local" / "state" / "ml-api" / "labels" / "audit.jsonl"
    assert not audit_path.exists()


def test_list_clips_does_not_create_jsonl_audit_side_channel(
    clip_env, make_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-1")
    audit_path = clip_env / "label-store" / "audit.jsonl"

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips")
        audit = client.get("/api/v1/audit")

    assert response.status_code == 200
    assert [clip["clip_id"] for clip in response.json()["clips"]] == ["clip-1"]
    assert audit.status_code == 200
    assert "clip.list" in [event["action"] for event in audit.json()["events"]]
    assert not audit_path.exists()


def test_legacy_label_route_is_absent(clip_env, make_app) -> None:
    _write_manifest(clip_env / "clip-store", "clip-1")
    with TestClient(make_app()) as client:
        _login(client)
        response = client.put(
            "/api/v1/clips/clip-1/label",
            json={"label": "TRUE_POSITIVE", "reviewer": "reviewer-1"},
        )
        audit = client.get("/api/v1/audit")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert not (clip_env / "label-store" / "labels" / "clip-1.json").exists()
    assert all(event["action"] != "label" for event in audit.json()["events"])


def test_clip_routes_require_a_dashboard_session(clip_env, make_app) -> None:
    """A bare worker relay token or forged bearer token is never a substitute
    for a real dashboard session -- the legacy bypass is unreachable now that
    dashboard auth always resolves to a session store."""
    _write_manifest(clip_env / "clip-store", "clip-1")

    with TestClient(make_app()) as client:
        unauthenticated = client.get("/api/v1/clips")
        wrong_bearer = client.get("/api/v1/clips", headers={"Authorization": "Bearer wrong"})
        worker_relay_token = client.get(
            "/api/v1/clips", headers={"Authorization": "Bearer relay-token"}
        )

    assert unauthenticated.status_code == 401
    assert wrong_bearer.status_code == 401
    assert worker_relay_token.status_code == 401


def test_video_rejects_manifest_path_escape(clip_env, make_app) -> None:
    clip_store = clip_env / "clip-store"
    (clip_env / "secret.mp4").write_bytes(b"secret")
    _write_manifest(clip_store, "clip-escape", path="../secret.mp4")

    with TestClient(make_app()) as client:
        _login(client)
        response = client.get("/api/v1/clips/clip-escape/video")
        invalid_id = client.get("/api/v1/clips/%2E%2E/video")

    assert response.status_code == 400
    assert invalid_id.status_code == 400


def test_list_clips_includes_size_bytes_stat_from_the_resolved_video_file(
    clip_env, make_app
) -> None:
    clip_store = clip_env / "clip-store"
    _write_manifest(clip_store, "clip-1")
    video_path = clip_store / "clips" / "clip-1" / "clip.mp4"

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert response.json()["clips"][0]["size_bytes"] == video_path.stat().st_size


def test_list_clips_size_bytes_is_null_when_video_is_unavailable(clip_env, make_app) -> None:
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

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    assert response.json()["clips"][0]["size_bytes"] is None


def test_list_clips_defaults_missing_duration_s_to_zero_instead_of_dropping_the_manifest(
    clip_env,
    make_app,
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

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    clips = response.json()["clips"]
    assert [clip["clip_id"] for clip in clips] == ["clip-no-duration"]
    assert clips[0]["duration_s"] == 0.0


def test_list_clips_finds_manifests_under_the_root_and_subdirectory_layouts(
    clip_env, make_app
) -> None:
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

    app = make_app()
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips")

    assert response.status_code == 200
    clip_ids = {clip["clip_id"] for clip in response.json()["clips"]}
    assert clip_ids == {"clip-root", "clip-first-level", "clip-second-level"}
