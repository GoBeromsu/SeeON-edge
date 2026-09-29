from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_clip_app import index_clips
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

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


@pytest.fixture
def app(
    clip_env: Path,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> FastAPI:
    return postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)


def _login(client: TestClient) -> None:
    assert client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN).status_code == 204


def test_list_and_metadata_compute_thumbnail_availability_for_returned_items(
    clip_env: Path,
    app: FastAPI,
) -> None:
    _write_clip(clip_env, "clip-with", thumbnail=True)
    _write_clip(clip_env, "clip-without", thumbnail=False)
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        metadata = client.get("/api/v1/clips/clip-with/metadata")

    availability = {clip["clip_id"]: clip["thumbnail_available"] for clip in listed.json()["clips"]}
    assert listed.status_code == 200
    assert availability == {"clip-with": True, "clip-without": False}
    assert metadata.status_code == 200
    assert metadata.json()["thumbnail_available"] is True


def test_compact_listing_rebuilds_thumbnail_identity(
    clip_env: Path,
    app: FastAPI,
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: a finalized clip with a regular thumbnail.
    _write_clip(clip_env, "clip-with", thumbnail=True)

    # When: the catalogue indexes the store and the keyset listing reads it.
    index_clips(app)
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips", params={"limit": 10})

    # Then: the response and the PostgreSQL catalogue both retain thumbnail identity.
    assert response.status_code == 200
    assert response.json()["clips"][0]["thumbnail_available"] is True
    row = postgres_product_sandbox.admin.execute(
        "SELECT thumbnail_relpath, thumbnail_sha256, thumbnail_size_bytes "
        "FROM clips WHERE clip_id = %s",
        ("clip-with",),
    ).fetchone()
    assert row == (
        "clips/clip-with/thumbnail.jpg",
        hashlib.sha256(JPEG).hexdigest(),
        len(JPEG),
    )


def test_first_page_rebuilds_thumbnail_availability(
    clip_env: Path,
    app: FastAPI,
) -> None:
    for item_index in range(60):
        _write_clip(clip_env, f"clip-{item_index:03d}", thumbnail=item_index % 2 == 0)
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips", params={"limit": 48, "offset": 0})

    assert response.status_code == 200
    clips = response.json()["clips"]
    assert len(clips) == 48
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
    app: FastAPI,
) -> None:
    _write_clip(clip_env / layout, "clip-layout", thumbnail=True)

    with TestClient(app) as client:
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
    app: FastAPI,
) -> None:
    accepted_dir = _write_clip(clip_env, "clip-accepted", thumbnail=False)
    rejected_dir = _write_clip(clip_env, "clip-rejected", thumbnail=False)
    (accepted_dir / "thumbnail.jpg").write_bytes(b"a" * THUMBNAIL_LIMIT_BYTES)
    (rejected_dir / "thumbnail.jpg").write_bytes(b"b" * (THUMBNAIL_LIMIT_BYTES + 1))
    index_clips(app)

    with TestClient(app) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        accepted = client.get("/api/v1/clips/clip-accepted/thumbnail")
        rejected = client.get("/api/v1/clips/clip-rejected/thumbnail")

    availability = {clip["clip_id"]: clip["thumbnail_available"] for clip in listed.json()["clips"]}
    assert availability == {"clip-accepted": True, "clip-rejected": False}
    assert accepted.status_code == 200
    assert len(accepted.content) == THUMBNAIL_LIMIT_BYTES
    assert rejected.status_code == 404


def test_thumbnail_endpoint_rejects_missing_symlink_and_duplicate_clip_ids(
    clip_env: Path,
    tmp_path: Path,
    app: FastAPI,
) -> None:
    missing_dir = _write_clip(clip_env, "clip-missing", thumbnail=False)
    symlink_dir = _write_clip(clip_env, "clip-symlink", thumbnail=False)
    external = tmp_path / "external.jpg"
    external.write_bytes(JPEG)
    os.symlink(external, symlink_dir / "thumbnail.jpg")
    _write_clip(clip_env, "clip-duplicate", thumbnail=True)
    _write_clip(clip_env / "archive", "clip-duplicate", thumbnail=True)

    with TestClient(app) as client:
        _login(client)
        missing = client.get("/api/v1/clips/clip-missing/thumbnail")
        symlink = client.get("/api/v1/clips/clip-symlink/thumbnail")
        duplicate = client.get("/api/v1/clips/clip-duplicate/thumbnail")

    assert missing_dir.is_dir()
    assert missing.status_code == 404
    assert symlink.status_code == 404
    assert duplicate.status_code == 409


def test_head_thumbnail_answers_with_the_get_header_section_and_no_body(
    clip_env: Path,
    app: FastAPI,
) -> None:
    """Same #452 gap as the video route: FastAPI does not synthesise HEAD."""
    _write_clip(clip_env, "clip-head", thumbnail=True)

    with TestClient(app) as client:
        unauthorized = client.head("/api/v1/clips/clip-head/thumbnail")
        _login(client)
        head = client.head("/api/v1/clips/clip-head/thumbnail")
        get = client.get("/api/v1/clips/clip-head/thumbnail")
        missing = client.head("/api/v1/clips/clip-absent/thumbnail")

    assert unauthorized.status_code == 401
    assert head.status_code == get.status_code == 200
    assert head.content == b""
    assert get.content == JPEG
    for header in ("content-type", "content-length", "cache-control"):
        assert head.headers[header] == get.headers[header]
    assert head.headers["content-type"] == "image/jpeg"
    assert head.headers["content-length"] == str(len(JPEG))
    assert missing.status_code == 404
