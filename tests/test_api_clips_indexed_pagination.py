"""Catalogue clip listing walks keyset pages and never re-reads catalogued media."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.clips import catalog_indexer
from backend.app.features.clips.store import ClipStore
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_clip_app import index_clips
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_CLIP_COUNT = 60
_PAGE_SIZE = 20


def _write_fixture(root: Path) -> None:
    for index in range(_CLIP_COUNT):
        clip_id = f"clip-{index:05d}"
        clip_dir = root / "clips" / clip_id
        clip_dir.mkdir(parents=True)
        payload: dict[str, str | float | bool] = {
            "clip_id": clip_id,
            "camera_id": "camera-a",
            "event_ref": f"event-{index}",
            "started_at": (
                f"2026-08-09T{index // 3600:02d}:{index // 60 % 60:02d}:{index % 60:02d}Z"
            ),
            "duration_s": 0.0,
            "codec": "",
            "video_available": False,
            "finalized": True,
        }
        facet_case = index % 4
        if facet_case == 0:
            payload["event_type"] = "fall"
        elif facet_case == 1:
            payload["event_type"] = "bed-exit"
        elif facet_case == 2:
            payload["event_type"] = f"unknown-{index}"
        _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_compact_listing_cursor_walk_visits_each_clip_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    root = tmp_path / "clip-store"
    _write_fixture(root)
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.clip_store = ClipStore(root)
    index_clips(app)
    traversed_ids: list[str] = []
    first_facets: dict[str, int] = {}
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "admin", "password": "admin"},
        )
        assert login.status_code == 204
        cursor: str | None = None
        page_number = 0
        while True:
            params: dict[str, str | int] = {"limit": _PAGE_SIZE}
            if cursor is not None:
                params["cursor"] = cursor
            response = client.get("/api/v1/clips", params=params)
            assert response.status_code == 200
            body = response.json()
            if page_number == 0:
                first_facets = body["event_type_counts"]
                assert len(body["clips"]) == _PAGE_SIZE
                assert body["pagination"]["total"] == _CLIP_COUNT
                assert isinstance(body["pagination"]["next_cursor"], str)
            traversed_ids.extend(clip["clip_id"] for clip in body["clips"])
            cursor = body["pagination"]["next_cursor"]
            page_number += 1
            if cursor is None:
                break
    assert set(first_facets) == {"bed-exit", "fall", "other"}
    assert len(traversed_ids) == _CLIP_COUNT
    assert len(set(traversed_ids)) == _CLIP_COUNT


def _write_media_fixture(root: Path, count: int) -> None:
    for index in range(count):
        clip_id = f"clip-{index:05d}"
        clip_dir = root / "clips" / clip_id
        clip_dir.mkdir(parents=True)
        _ = (clip_dir / "clip.mp4").write_bytes(bytes([index % 256]) * (4096 + index))
        payload = {
            "clip_id": clip_id,
            "camera_id": "camera-a",
            "event_ref": f"event-{index}",
            "event_type": "fall",
            "started_at": f"2026-08-09T00:{index // 60 % 60:02d}:{index % 60:02d}Z",
            "duration_s": 1.0,
            "codec": "h264",
            "path": f"clips/{clip_id}/clip.mp4",
            "video_available": True,
            "finalized": True,
        }
        _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_compact_listing_does_not_rehash_catalogued_media_on_every_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    """A populated store must not re-read every clip's media per ``GET /clips``.

    On the live edge (7.8k clips, 20 GB) the per-request full re-hash took
    tens of seconds per request and concurrent dashboard polls never finished,
    so the events page and the room event history showed nothing at all.
    """
    root = tmp_path / "clip-store"
    count = 12
    _write_media_fixture(root, count)
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.clip_store = ClipStore(root)
    hashed_media: list[Path] = []
    real_hash = catalog_indexer._hash_regular

    def counting_hash(store_root: Path, path: Path) -> tuple[str, int]:
        if path.name == "clip.mp4":
            hashed_media.append(path)
        return real_hash(store_root, path)

    monkeypatch.setattr(catalog_indexer, "_hash_regular", counting_hash)
    index_clips(app)
    assert len(hashed_media) == count, "first catalogue pass must verify every media file"

    hashed_media.clear()
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "admin", "password": "admin"},
        )
        assert login.status_code == 204
        first = client.get("/api/v1/clips", params={"limit": 5})
        second = client.get("/api/v1/clips", params={"limit": 5})
    assert first.status_code == second.status_code == 200
    assert all(clip["video_available"] for clip in first.json()["clips"])
    assert second.json()["clips"] == first.json()["clips"]
    assert second.json()["pagination"]["total"] == count
    assert hashed_media == [], "listing pages must not read media"

    index_clips(app)
    assert hashed_media == [], "already-catalogued media must not be re-read per pass"

    # A media file that changes size is not trusted from the catalogue: it is
    # hashed again and the immutable-content conflict is recorded, not served.
    (root / "clips" / "clip-00003" / "clip.mp4").write_bytes(b"replaced-with-other-size")
    index_clips(app)
    assert [path.parent.name for path in hashed_media] == ["clip-00003"]
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "admin", "password": "admin"},
        )
        assert login.status_code == 204
        third = client.get("/api/v1/clips", params={"limit": 5})
    assert third.status_code == 200
    state = postgres_product_sandbox.admin.execute(
        "SELECT local_state, local_reason FROM clips WHERE clip_id = %s", ("clip-00003",)
    ).fetchone()
    assert state == ("CORRUPT", "IDENTITY_CONFLICT")
