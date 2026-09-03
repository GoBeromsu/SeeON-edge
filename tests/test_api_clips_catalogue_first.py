"""``GET /clips`` serves pages from the catalogue; the store is walked once per request."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.clips import compact_listing
from backend.app.features.clips import store as store_module
from backend.app.features.clips.store import ClipStore
from backend.app.main import create_app, no_lifespan


def _write_clip(
    root: Path,
    index: int,
    *,
    camera_id: str = "camera-a",
    media: bytes | None = None,
) -> str:
    clip_id = f"clip-{index:05d}"
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    _ = (clip_dir / "clip.mp4").write_bytes(
        bytes([index % 256]) * (512 + index % 97) if media is None else media
    )
    payload = {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "event_ref": f"event-{index}",
        "event_type": "fall" if index % 2 == 0 else "bed-exit",
        "started_at": (
            f"2026-08-{1 + index // 86400:02d}T{index // 3600 % 24:02d}:"
            f"{index // 60 % 60:02d}:{index % 60:02d}Z"
        ),
        "duration_s": 1.0,
        "codec": "h264",
        "path": f"clips/{clip_id}/clip.mp4",
        "video_available": True,
        "finalized": True,
    }
    _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return clip_id


def _client(app) -> TestClient:
    client = TestClient(app)
    login = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert login.status_code == 204
    return client


def _app(root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = create_app(lifespan=no_lifespan)
    app.state.clip_store = ClipStore(root)
    return app


def _catalogue_everything(client: TestClient, count: int) -> int:
    """Drive reconciliation until every clip is catalogued; return the call count."""
    calls = 0
    while True:
        response = client.get("/api/v1/clips", params={"limit": 1})
        assert response.status_code == 200
        calls += 1
        if response.json()["pagination"]["total"] == count:
            return calls
        assert calls <= count, "catalogue never converged"


def test_listing_walks_the_store_once_and_never_relocates_per_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "clip-store"
    count = 500
    for index in range(count):
        _write_clip(root, index)
    app = _app(root, monkeypatch)
    root_walks = 0
    locates = 0
    real_roots = store_module.bounded_clip_roots
    real_locate = ClipStore.locate_manifest

    def counting_roots(store_root: Path):
        nonlocal root_walks
        root_walks += 1
        return real_roots(store_root)

    def counting_locate(self, clip_id: str):
        nonlocal locates
        locates += 1
        return real_locate(self, clip_id)

    monkeypatch.setattr(store_module, "bounded_clip_roots", counting_roots)
    monkeypatch.setattr(ClipStore, "locate_manifest", counting_locate)
    with _client(app) as client:
        _catalogue_everything(client, count)
        root_walks = 0
        locates = 0
        response = client.get("/api/v1/clips", params={"limit": 20})
        assert response.status_code == 200
        assert len(response.json()["clips"]) == 20
        assert response.json()["pagination"]["total"] == count
        assert root_walks <= 1, "one store walk per request, not one per clip"
        assert locates == 0, "no per-clip locate_manifest on the listing path"

        root_walks = 0
        cursor = response.json()["pagination"]["next_cursor"]
        second = client.get("/api/v1/clips", params={"limit": 20, "cursor": cursor})
        assert second.status_code == 200
        assert root_walks <= 1
        assert locates == 0


def test_listing_two_thousand_clips_stays_within_two_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "clip-store"
    count = 2000
    for index in range(count):
        _write_clip(root, index)
    app = _app(root, monkeypatch)
    with _client(app) as client:
        durations: list[float] = []
        total = 0
        while total < count:
            started = time.perf_counter()
            response = client.get("/api/v1/clips", params={"limit": 20})
            durations.append(time.perf_counter() - started)
            assert response.status_code == 200
            total = response.json()["pagination"]["total"]
            assert len(durations) <= count // compact_listing.EXAMINE_BUDGET + 1
        started = time.perf_counter()
        steady = client.get("/api/v1/clips", params={"limit": 20})
        steady_duration = time.perf_counter() - started
    assert steady.status_code == 200
    assert steady.json()["pagination"]["total"] == count
    assert steady_duration < 2.0, f"steady-state listing took {steady_duration:.2f}s"
    assert max(durations) < 2.0, f"slowest reconciling call took {max(durations):.2f}s"


def test_new_clip_is_examined_and_visible_on_the_next_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "clip-store"
    for index in range(5):
        _write_clip(root, index)
    app = _app(root, monkeypatch)
    hashed: list[str] = []
    real_hash = compact_listing._hash_regular

    def counting_hash(store_root: Path, path: Path):
        if path.name == "clip.mp4":
            hashed.append(path.parent.name)
        return real_hash(store_root, path)

    monkeypatch.setattr(compact_listing, "_hash_regular", counting_hash)
    with _client(app) as client:
        first = client.get("/api/v1/clips", params={"limit": 10})
        assert first.status_code == 200
        assert first.json()["pagination"]["total"] == 5
        assert sorted(hashed) == [f"clip-{index:05d}" for index in range(5)]

        hashed.clear()
        new_id = _write_clip(root, 900)
        second = client.get("/api/v1/clips", params={"limit": 10})
        assert second.status_code == 200
        assert second.json()["pagination"]["total"] == 6
        assert second.json()["clips"][0]["clip_id"] == new_id
        assert second.json()["clips"][0]["video_available"] is True
        assert hashed == [new_id], "only the new clip's media is hashed"

        hashed.clear()
        third = client.get("/api/v1/clips", params={"limit": 10})
        assert third.status_code == 200
        assert hashed == []


def test_changed_media_still_conflicts_and_deleted_clip_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "clip-store"
    for index in range(4):
        _write_clip(root, index)
    app = _app(root, monkeypatch)
    with _client(app) as client:
        first = client.get("/api/v1/clips", params={"limit": 10})
        assert first.status_code == 200
        assert first.json()["pagination"]["total"] == 4

        (root / "clips" / "clip-00002" / "clip.mp4").write_bytes(b"different bytes now")
        conflict = client.get("/api/v1/clips", params={"limit": 10})
        assert conflict.status_code == 503

        shutil.rmtree(root / "clips" / "clip-00002")
        after_delete = client.get("/api/v1/clips", params={"limit": 10})
        assert after_delete.status_code == 200
        assert after_delete.json()["pagination"]["total"] == 3
        assert "clip-00002" not in {clip["clip_id"] for clip in after_delete.json()["clips"]}
    from backend.app.features.clips import router as router_module

    with sqlite3.connect(router_module.EDGE_DATABASE_PATH) as connection:
        assert connection.execute(
            "SELECT count(*) FROM clips WHERE clip_id='clip-00002'"
        ).fetchone() == (0,)


def test_examination_is_bounded_per_call_and_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "clip-store"
    monkeypatch.setattr(compact_listing, "EXAMINE_BUDGET", 7)
    count = 20
    for index in range(count):
        _write_clip(root, index)
    app = _app(root, monkeypatch)
    with _client(app) as client:
        first = client.get("/api/v1/clips", params={"limit": 5})
        assert first.status_code == 200
        assert first.json()["pagination"]["total"] == 7
        assert first.json()["clips"][0]["clip_id"] == "clip-00019"
        calls = _catalogue_everything(client, count)
    assert calls == 2  # ceil(20 / 7) - 1 further calls after the first
