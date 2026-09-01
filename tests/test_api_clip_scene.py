from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.clips.manifest import read_manifest_file
from backend.app.main import create_app, no_lifespan

DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}
SCENE_LIMIT_BYTES = 8 * 1024 * 1024


def _write_clip(
    root: Path,
    clip_id: str,
    *,
    scene: bytes | None = None,
    claim: bool = True,
    claim_size: int | None = None,
) -> Path:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"video")
    manifest: dict[str, object] = {
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
    if scene is not None:
        (clip_dir / "scene-index.json").write_bytes(scene)
        if claim:
            manifest["scene_index"] = {
                "path": "scene-index.json",
                "sha256": hashlib.sha256(scene).hexdigest(),
                "size_bytes": len(scene) if claim_size is None else claim_size,
                "schema": 1,
                "count": 7,
            }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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


def test_scene_endpoint_serves_claimed_bytes_and_head_is_body_free(clip_env: Path) -> None:
    scene = b'{"frames":[]}'
    _write_clip(clip_env, "clip-scene", scene=scene)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        assert client.get("/api/v1/clips/clip-scene/scene").status_code == 401
        _login(client)
        get = client.get("/api/v1/clips/clip-scene/scene")
        head = client.head("/api/v1/clips/clip-scene/scene")

    assert get.status_code == head.status_code == 200
    assert get.content == scene
    assert head.content == b""
    for header in ("content-type", "content-length", "cache-control"):
        assert head.headers[header] == get.headers[header]
    assert get.headers["content-type"] == "application/json"
    assert str(clip_env) not in get.text
    assert str(clip_env) not in "\n".join(get.headers.values())


def test_scene_listing_and_metadata_require_a_valid_claim(clip_env: Path) -> None:
    _write_clip(clip_env, "clip-unclaimed", scene=b"{}", claim=False)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        metadata = client.get("/api/v1/clips/clip-unclaimed/metadata")
        get = client.get("/api/v1/clips/clip-unclaimed/scene")
        head = client.head("/api/v1/clips/clip-unclaimed/scene")

    assert listed.status_code == metadata.status_code == 200
    assert listed.json()["clips"][0]["scene_available"] is False
    assert metadata.json()["scene_available"] is False
    assert metadata.json()["scene_frame_count"] is None
    assert get.status_code == head.status_code == 404


def test_scene_claim_size_mismatch_is_unavailable(clip_env: Path) -> None:
    _write_clip(clip_env, "clip-size", scene=b"{}", claim_size=3)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        response = client.get("/api/v1/clips/clip-size/scene")

    assert listed.json()["clips"][0]["scene_available"] is False
    assert response.status_code == 404


def test_scene_get_rejects_same_size_tampering_without_hiding_listing(clip_env: Path) -> None:
    clip_dir = _write_clip(clip_env, "clip-tampered", scene=b'{"frames":[]}')

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        (clip_dir / "scene-index.json").write_bytes(b'{"frames":{}}')
        response = client.get("/api/v1/clips/clip-tampered/scene")

    assert listed.json()["clips"][0]["scene_available"] is True
    assert response.status_code == 404


def test_scene_claim_larger_than_backend_cap_degrades_to_unavailable(clip_env: Path) -> None:
    scene = b"a" * (SCENE_LIMIT_BYTES + 1)
    _write_clip(clip_env, "clip-large", scene=scene)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        listed = client.get("/api/v1/clips")
        response = client.get("/api/v1/clips/clip-large/scene")

    assert listed.json()["clips"][0]["scene_available"] is False
    assert response.status_code == 404


def test_manifest_scene_projection_defaults_and_preserves_valid_claim(clip_env: Path) -> None:
    absent = _write_clip(clip_env, "clip-absent", scene=None)
    claimed = _write_clip(clip_env, "clip-claimed", scene=b"{}")

    absent_manifest = read_manifest_file(absent / "manifest.json")
    claimed_manifest = read_manifest_file(claimed / "manifest.json")

    assert absent_manifest is not None
    assert absent_manifest.as_response()["scene_available"] is False
    assert absent_manifest.as_response()["scene_frame_count"] is None
    assert claimed_manifest is not None
    assert claimed_manifest.scene_index is not None
    assert claimed_manifest.as_response()["scene_frame_count"] == 7

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        missing = client.get("/api/v1/clips/clip-absent/scene")

    assert missing.status_code == 404
