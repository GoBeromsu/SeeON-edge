from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.clips import store as store_module
from backend.app.features.clips.store import ClipStore
from backend.app.main import create_app, no_lifespan

DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}
JPEG = b"\xff\xd8thumbnail\xff\xd9"
THUMBNAIL_LIMIT_BYTES = 2 * 1024 * 1024


def _write_clip(root: Path, clip_id: str, *, thumbnail: bool) -> Path:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"video")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "camera-1",
                "event_ref": f"event-{clip_id}",
                "event_type": "fall",
                "started_at": "2026-08-09T00:00:00Z",
                "duration_s": 10.0,
                "codec": "h264",
                "path": str(clip_dir / "clip.mp4"),
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    if thumbnail:
        (clip_dir / "thumbnail.jpg").write_bytes(JPEG)
    return clip_dir


@pytest.fixture(autouse=True)
def clip_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "label-store"))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    return root


def _login(client: TestClient) -> None:
    assert client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN).status_code == 204


def test_list_and_metadata_compute_thumbnail_availability_for_returned_items(
    clip_env: Path,
) -> None:
    _write_clip(clip_env, "clip-with", thumbnail=True)
    _write_clip(clip_env, "clip-without", thumbnail=False)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        metadata = client.get("/api/v1/clips/clip-with/metadata")

    availability = {
        clip["clip_id"]: clip["thumbnail_available"]
        for clip in listed.json()["clips"]
    }
    assert listed.status_code == 200
    assert availability == {"clip-with": True, "clip-without": False}
    assert metadata.status_code == 200
    assert metadata.json()["thumbnail_available"] is True


def test_compact_listing_rebuilds_thumbnail_identity(clip_env: Path) -> None:
    # Given: a finalized clip with a regular thumbnail.
    _write_clip(clip_env, "clip-with", thumbnail=True)

    # When: keyset listing rebuilds schema-18 clips.
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips", params={"limit": 10})

    # Then: the response and compact authority both retain thumbnail identity.
    assert response.status_code == 200
    assert response.json()["clips"][0]["thumbnail_available"] is True
    database = clip_env.parent / ".central-fixture" / "edge.sqlite3"
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT thumbnail_relpath, length(thumbnail_sha256), thumbnail_size_bytes "
            "FROM clips WHERE clip_id='clip-with'"
        ).fetchone()
    assert row == ("clips/clip-with/thumbnail.jpg", 64, len(JPEG))


def test_first_page_rebuilds_thumbnail_availability(
    clip_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for item_index in range(60):
        _write_clip(clip_env, f"clip-{item_index:03d}", thumbnail=item_index % 2 == 0)
    app = create_app(lifespan=no_lifespan)
    app.state.clip_store = ClipStore(clip_env)
    root_walks = 0
    original = store_module.bounded_clip_roots

    def instrumented(root: Path) -> tuple[Path, ...]:
        nonlocal root_walks
        root_walks += 1
        return original(root)

    monkeypatch.setattr(store_module, "bounded_clip_roots", instrumented)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips", params={"limit": 48, "offset": 0})

    assert response.status_code == 200
    clips = response.json()["clips"]
    assert len(clips) == 48
    assert root_walks > 0
    assert all(
        clip["thumbnail_available"] == (int(clip["clip_id"].removeprefix("clip-")) % 2 == 0)
        for clip in clips
    )


@pytest.mark.parametrize(
    "layout",
    (Path(), Path("archive"), Path("external") / "drive"),
)
def test_authenticated_thumbnail_endpoint_serves_all_bounded_layouts_with_cache_headers(
    clip_env: Path,
    layout: Path,
) -> None:
    _write_clip(clip_env / layout, "clip-layout", thumbnail=True)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        unauthorized = client.get("/api/v1/clips/clip-layout/thumbnail")
        _login(client)
        response = client.get("/api/v1/clips/clip-layout/thumbnail")

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.content == JPEG
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "private, no-store"


def test_thumbnail_payload_limit_is_enforced_for_availability_and_reads(
    clip_env: Path,
) -> None:
    accepted_dir = _write_clip(clip_env, "clip-accepted", thumbnail=False)
    rejected_dir = _write_clip(clip_env, "clip-rejected", thumbnail=False)
    (accepted_dir / "thumbnail.jpg").write_bytes(b"a" * THUMBNAIL_LIMIT_BYTES)
    (rejected_dir / "thumbnail.jpg").write_bytes(b"b" * (THUMBNAIL_LIMIT_BYTES + 1))

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        accepted = client.get("/api/v1/clips/clip-accepted/thumbnail")
        rejected = client.get("/api/v1/clips/clip-rejected/thumbnail")

    availability = {
        clip["clip_id"]: clip["thumbnail_available"]
        for clip in listed.json()["clips"]
    }
    assert availability == {"clip-accepted": True, "clip-rejected": False}
    assert accepted.status_code == 200
    assert len(accepted.content) == THUMBNAIL_LIMIT_BYTES
    assert rejected.status_code == 404


def test_thumbnail_endpoint_rejects_missing_symlink_and_duplicate_clip_ids(
    clip_env: Path,
    tmp_path: Path,
) -> None:
    missing_dir = _write_clip(clip_env, "clip-missing", thumbnail=False)
    symlink_dir = _write_clip(clip_env, "clip-symlink", thumbnail=False)
    external = tmp_path / "external.jpg"
    external.write_bytes(JPEG)
    os.symlink(external, symlink_dir / "thumbnail.jpg")
    _write_clip(clip_env, "clip-duplicate", thumbnail=True)
    _write_clip(clip_env / "archive", "clip-duplicate", thumbnail=True)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        missing = client.get("/api/v1/clips/clip-missing/thumbnail")
        symlink = client.get("/api/v1/clips/clip-symlink/thumbnail")
        duplicate = client.get("/api/v1/clips/clip-duplicate/thumbnail")

    assert missing_dir.is_dir()
    assert missing.status_code == 404
    assert symlink.status_code == 404
    assert duplicate.status_code == 409
