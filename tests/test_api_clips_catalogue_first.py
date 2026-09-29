"""``GET /clips`` serves pages from the PostgreSQL catalogue; only the indexer walks the store."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.clips import catalog_indexer
from backend.app.features.clips.catalog_indexer import ReconcileOutcome
from backend.app.features.clips.store import ClipStore
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_clip_app import index_clips
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_STATE_SQL = "SELECT local_state, local_reason FROM clips WHERE clip_id = %s"


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


def _client(app: FastAPI) -> TestClient:
    client = TestClient(app)
    login = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert login.status_code == 204
    return client


@pytest.fixture
def make_app(
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> Callable[[Path], FastAPI]:
    def make(root: Path) -> FastAPI:
        monkeypatch.setenv("CLIP_STORE_DIR", str(root))
        app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
        app.state.clip_store = ClipStore(root)
        return app

    return make


def _listed(client: TestClient) -> tuple[int, list[str]]:
    response = client.get("/api/v1/clips", params={"limit": 10})
    assert response.status_code == 200
    body = response.json()
    return body["pagination"]["total"], [clip["clip_id"] for clip in body["clips"]]


def test_listing_walks_the_store_once_and_never_relocates_per_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_app: Callable[[Path], FastAPI]
) -> None:
    """Pages come from the catalogue: a GET neither walks the store nor locates a clip."""
    root = tmp_path / "clip-store"
    count = 500
    for index in range(count):
        _write_clip(root, index)
    app = make_app(root)
    index_clips(app)
    root_walks = 0
    locates = 0
    real_scan = ClipStore.scan_manifest_partition
    real_locate = ClipStore.locate_manifest

    def counting_scan(self: ClipStore):
        nonlocal root_walks
        root_walks += 1
        return real_scan(self)

    def counting_locate(self: ClipStore, clip_id: str):
        nonlocal locates
        locates += 1
        return real_locate(self, clip_id)

    monkeypatch.setattr(ClipStore, "scan_manifest_partition", counting_scan)
    monkeypatch.setattr(ClipStore, "locate_manifest", counting_locate)
    with _client(app) as client:
        response = client.get("/api/v1/clips", params={"limit": 20})
        assert response.status_code == 200
        assert len(response.json()["clips"]) == 20
        assert response.json()["pagination"]["total"] == count
        assert root_walks == 0, "a listing request does not walk the store"
        assert locates == 0, "no per-clip locate_manifest on the listing path"

        cursor = response.json()["pagination"]["next_cursor"]
        second = client.get("/api/v1/clips", params={"limit": 20, "cursor": cursor})
        assert second.status_code == 200
        assert len(second.json()["clips"]) == 20
        assert root_walks == 0
        assert locates == 0


def test_new_clip_is_examined_and_visible_on_the_next_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_app: Callable[[Path], FastAPI]
) -> None:
    root = tmp_path / "clip-store"
    for index in range(5):
        _write_clip(root, index)
    app = make_app(root)
    hashed: list[str] = []
    real_hash = catalog_indexer._hash_regular

    def counting_hash(store_root: Path, path: Path) -> tuple[str, int]:
        identity = real_hash(store_root, path)
        hashed.append(f"{path.parent.name}/{path.name}")
        return identity

    monkeypatch.setattr(catalog_indexer, "_hash_regular", counting_hash)
    index_clips(app)
    assert sorted(hashed) == [
        f"clip-{index:05d}/{name}" for index in range(5) for name in ("clip.mp4", "manifest.json")
    ]
    with _client(app) as client:
        assert _listed(client)[0] == 5

        hashed.clear()
        new_id = _write_clip(root, 900)
        index_clips(app)
        second = client.get("/api/v1/clips", params={"limit": 10})
        assert second.status_code == 200
        assert second.json()["pagination"]["total"] == 6
        assert second.json()["clips"][0]["clip_id"] == new_id
        assert second.json()["clips"][0]["video_available"] is True
        assert sorted(hashed) == [f"{new_id}/clip.mp4", f"{new_id}/manifest.json"], (
            "only the new clip is examined; catalogued clips are not re-read"
        )

        hashed.clear()
        index_clips(app)
        assert _listed(client)[0] == 6
        assert hashed == []


def test_changed_media_still_conflicts_and_deleted_clip_disappears(
    tmp_path: Path,
    make_app: Callable[[Path], FastAPI],
    postgres_product_sandbox: ProductSandbox,
) -> None:
    root = tmp_path / "clip-store"
    for index in range(4):
        _write_clip(root, index)
    app = make_app(root)
    admin = postgres_product_sandbox.admin
    identity_sql = "SELECT media_sha256, media_size_bytes FROM clips WHERE clip_id = %s"
    index_clips(app)
    before = admin.execute(identity_sql, ("clip-00002",)).fetchone()
    with _client(app) as client:
        assert _listed(client)[0] == 4

        (root / "clips" / "clip-00002" / "clip.mp4").write_bytes(b"different bytes now")
        index_clips(app)
        assert _listed(client)[0] == 4
        assert admin.execute(identity_sql, ("clip-00002",)).fetchone() == before
        assert admin.execute(_STATE_SQL, ("clip-00002",)).fetchone() == (
            "CORRUPT",
            "IDENTITY_CONFLICT",
        )

        shutil.rmtree(root / "clips" / "clip-00002")
        index_clips(app)
        total, clip_ids = _listed(client)
        assert total == 3
        assert "clip-00002" not in clip_ids
    assert admin.execute(
        "SELECT count(*) FROM clips WHERE clip_id = %s", ("clip-00002",)
    ).fetchone() == (0,)


def test_examination_is_bounded_per_call_and_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_app: Callable[[Path], FastAPI]
) -> None:
    root = tmp_path / "clip-store"
    monkeypatch.setattr(catalog_indexer, "EXAMINE_BUDGET", 7)
    count = 20
    for index in range(count):
        _write_clip(root, index)
    app = make_app(root)
    first_pass = app.state.clip_catalog_indexer.reconcile(ClipStore(root))
    assert first_pass == ReconcileOutcome(examined=7, remaining=13, isolated=0)
    with _client(app) as client:
        first = client.get("/api/v1/clips", params={"limit": 5})
        assert first.status_code == 200
        assert first.json()["pagination"]["total"] == 7
        assert first.json()["clips"][0]["clip_id"] == "clip-00019"

        assert index_clips(app) == (
            ReconcileOutcome(examined=7, remaining=6, isolated=0),
            ReconcileOutcome(examined=6, remaining=0, isolated=0),
        )
        assert _listed(client)[0] == count


def test_parked_row_whose_manifest_reappears_is_restored_not_refused(
    tmp_path: Path,
    make_app: Callable[[Path], FastAPI],
    postgres_product_sandbox: ProductSandbox,
) -> None:
    """A row parked UNAVAILABLE/MANIFEST_MISSING carries no identity. When the
    manifest is back on disk the catalogue must restore it, not report a content
    change: one such row made every ``GET /clips`` answer 503 in production."""
    root = tmp_path / "clip-store"
    for index in range(3):
        _write_clip(root, index)
    app = make_app(root)
    admin = postgres_product_sandbox.admin
    index_clips(app)
    with _client(app) as client:
        assert _listed(client)[0] == 3

        # The state the indexer leaves when a referenced clip's manifest vanishes.
        _ = admin.execute(
            """
            UPDATE clips SET
                manifest_relpath=NULL, media_relpath=NULL, thumbnail_relpath=NULL,
                manifest_sha256=NULL, media_sha256=NULL, thumbnail_sha256=NULL,
                manifest_size_bytes=NULL, media_size_bytes=NULL, thumbnail_size_bytes=NULL,
                local_state='UNAVAILABLE', local_reason='MANIFEST_MISSING',
                revision=revision+1
            WHERE clip_id=%s
            """,
            ("clip-00001",),
        )
        parked_total, parked_ids = _listed(client)
        assert parked_total == 2
        assert "clip-00001" not in parked_ids

        index_clips(app)
        restored_total, restored_ids = _listed(client)
        assert restored_total == 3
        assert "clip-00001" in restored_ids

    row = admin.execute(
        "SELECT local_state, local_reason, manifest_relpath, media_sha256, revision "
        "FROM clips WHERE clip_id = %s",
        ("clip-00001",),
    ).fetchone()
    assert row is not None
    assert row[0] == "AVAILABLE"
    assert row[1] is None
    assert row[2] == "clips/clip-00001/manifest.json"
    assert row[3] is not None
    assert row[4] == 3, "insert, park, restore"


def test_corrupt_row_whose_media_reappears_is_restored(
    tmp_path: Path,
    make_app: Callable[[Path], FastAPI],
    postgres_product_sandbox: ProductSandbox,
) -> None:
    root = tmp_path / "clip-store"
    media = b"stable media bytes"
    _write_clip(root, 0, media=media)
    app = make_app(root)
    admin = postgres_product_sandbox.admin
    media_path = root / "clips" / "clip-00000" / "clip.mp4"
    media_path.unlink()
    index_clips(app)
    with _client(app) as client:
        assert _listed(client)[1] == ["clip-00000"]
        assert admin.execute(_STATE_SQL, ("clip-00000",)).fetchone() == (
            "CORRUPT",
            "MEDIA_MISSING",
        )

        _ = media_path.write_bytes(media)
        index_clips(app)
        assert _listed(client)[1] == ["clip-00000"]
        assert admin.execute(_STATE_SQL, ("clip-00000",)).fetchone() == ("AVAILABLE", None)

        _ = media_path.write_bytes(b"different bytes now")
        index_clips(app)
        assert _listed(client)[1] == ["clip-00000"]
        assert admin.execute(_STATE_SQL, ("clip-00000",)).fetchone() == (
            "CORRUPT",
            "IDENTITY_CONFLICT",
        )
