from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.clips import listing_generation as listing_generation_module
from backend.app.features.clips import listing_runtime
from backend.app.features.clips.listing_index import (
    ClipListingIndex,
    ClipListingReconcileError,
    ReconcileStats,
)
from backend.app.features.clips.schemas import ClipListQuery
from backend.app.features.clips.store import ClipStore
from backend.app.main import create_app, no_lifespan


def _write_manifest(
    root: Path,
    clip_id: str,
    *,
    event_type: str | None = "fall",
    layout: str = "",
    started_at: str = "2026-08-09T00:00:00Z",
) -> Path:
    clip_dir = root / layout / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    video_path = clip_dir / "clip.mp4"
    _ = video_path.write_bytes(clip_id.encode())
    relative_path = "/".join(part for part in (layout, "clips", clip_id) if part)
    payload: dict[str, str | float | bool] = {
        "clip_id": clip_id,
        "camera_id": "camera-a",
        "event_ref": f"event-{clip_id}",
        "started_at": started_at,
        "duration_s": 5.0,
        "codec": "h264",
        "path": relative_path,
        "video_available": True,
        "finalized": True,
    }
    if event_type is not None:
        payload["event_type"] = event_type
    manifest_path = clip_dir / "manifest.json"
    _ = manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_initial_reconcile_indexes_every_historical_layout(tmp_path: Path) -> None:
    # Given: finalized manifests across every filesystem layout the legacy API serves.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-root", started_at="2026-08-09T00:00:01Z")
    _write_manifest(root, "clip-one", layout="archive", event_type="bed-exit")
    _write_manifest(root, "clip-two", layout="external/drive", event_type=None)
    index = ClipListingIndex.open(tmp_path / "catalog.sqlite3")

    try:
        # When: the derived index performs its first reconciliation.
        stats = index.reconcile(ClipStore(root))
        page = index.page(ClipListQuery(limit=48))
    finally:
        index.close()

    # Then: every manifest is read once and all bounded facets are represented.
    assert stats == ReconcileStats(3, 3, 3, 0, 0)
    assert [manifest.clip_id for manifest in page.manifests] == [
        "clip-root",
        "clip-two",
        "clip-one",
    ]
    assert page.total == 3
    assert page.event_type_counts == {"bed-exit": 1, "fall": 1, "other": 1}


def test_reconcile_reads_only_new_or_changed_manifests(tmp_path: Path) -> None:
    # Given: two manifests already represented by the index.
    root = tmp_path / "clip-store"
    unchanged_path = _write_manifest(root, "clip-unchanged")
    changed_path = _write_manifest(root, "clip-changed")
    index = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    _ = index.reconcile(ClipStore(root))

    # When: one pass is unchanged, then one manifest changes and one is published.
    unchanged = index.reconcile(ClipStore(root))
    payload = json.loads(changed_path.read_text(encoding="utf-8"))
    payload["event_type"] = "bed-exit"
    _ = changed_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_manifest(root, "clip-new", event_type="mystery")
    incremental = index.reconcile(ClipStore(root))
    index.close()

    # Then: unchanged fingerprints avoid manifest reads and only two rows are reparsed.
    assert unchanged == ReconcileStats(2, 0, 0, 0, 2)
    assert incremental == ReconcileStats(3, 2, 2, 0, 1)
    assert unchanged_path.is_file()


def test_reconcile_backfills_sidecar_creation_and_removal_without_manifest_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "clip-store"
    manifest_path = _write_manifest(root, "clip-sidecar")
    original_manifest = manifest_path.read_bytes()
    original_stat = manifest_path.stat()
    thumbnail_path = manifest_path.parent / "thumbnail.jpg"
    index = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    _ = index.reconcile(ClipStore(root))

    thumbnail_path.write_bytes(b"thumbnail")
    created = index.reconcile(ClipStore(root))
    created_page = index.page(ClipListQuery(limit=48))
    thumbnail_path.unlink()
    removed = index.reconcile(ClipStore(root))
    removed_page = index.page(ClipListQuery(limit=48))
    index.close()

    assert manifest_path.read_bytes() == original_manifest
    assert manifest_path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert created == ReconcileStats(1, 1, 1, 0, 0)
    assert created_page.manifests[0].thumbnail_available is True
    assert removed == ReconcileStats(1, 1, 1, 0, 0)
    assert removed_page.manifests[0].thumbnail_available is False


def test_reconcile_removes_rows_after_retention_deletes_media(tmp_path: Path) -> None:
    # Given: two indexed clips.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-kept")
    removed_path = _write_manifest(root, "clip-removed")
    index = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    _ = index.reconcile(ClipStore(root))

    # When: retention deletes one clip directory and reconciliation runs again.
    shutil.rmtree(removed_path.parent)
    stats = index.reconcile(ClipStore(root))
    page = index.page(ClipListQuery(limit=48))
    index.close()

    # Then: the stale projection row is removed atomically.
    assert stats == ReconcileStats(1, 0, 0, 1, 1)
    assert [manifest.clip_id for manifest in page.manifests] == ["clip-kept"]
    assert page.total == 1


def test_failed_reconcile_disables_pages_until_a_successful_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a successfully synchronized index.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-ready")
    index = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    _ = index.reconcile(ClipStore(root))
    original_discover = listing_generation_module.discover_manifest_paths

    def fail_discovery(_root: Path) -> list[Path]:
        raise OSError("filesystem unavailable")

    # When: the next filesystem synchronization fails.
    monkeypatch.setattr(listing_generation_module, "discover_manifest_paths", fail_discovery)
    with pytest.raises(ClipListingReconcileError):
        index.reconcile(ClipStore(root))

    # Then: stale rows are not served, and a later successful sync restores service.
    with pytest.raises(ClipListingReconcileError):
        index.page(ClipListQuery(limit=48))
    app = create_app(lifespan=no_lifespan)
    app.state.clip_store = ClipStore(root)
    app.state.clip_listing_index = index
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session",
            json={"username": "admin", "password": "admin"},
        )
        assert login.status_code == 204
        assert client.get("/api/v1/clips", params={"limit": 48}).status_code == 503
        assert client.get("/api/v1/clips").status_code == 200
    monkeypatch.setattr(listing_generation_module, "discover_manifest_paths", original_discover)
    _ = index.reconcile(ClipStore(root))
    assert index.page(ClipListQuery(limit=48)).total == 1
    index.close()


def test_lifespan_reconciles_listing_index_before_serving(
    tmp_path: Path,
) -> None:
    # Given: a clip published before ml-api starts.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-before-start")

    # When: the production application lifespan starts.
    app = create_app()
    app.state.clip_store = ClipStore(root)
    with TestClient(app) as client:
        _ = client.get("/health/ready")
        index = app.state.clip_listing_index
        page = index.page(ClipListQuery(limit=48))

    # Then: bounded listing state was synchronized before requests were accepted.
    assert [manifest.clip_id for manifest in page.manifests] == ["clip-before-start"]


def test_lifespan_background_reconcile_publishes_new_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a running app whose next periodic reconciliation is held at the boundary.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-initial")
    monkeypatch.setattr(listing_runtime, "RECONCILE_INTERVAL_SEC", 0.01)
    allow_background = threading.Event()
    background_complete = threading.Event()
    original_reconcile = ClipListingIndex.reconcile
    call_count = 0

    def observed_reconcile(self: ClipListingIndex, store: ClipStore) -> ReconcileStats:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            assert allow_background.wait(timeout=1)
        result = original_reconcile(self, store)
        if call_count > 1:
            background_complete.set()
        return result

    monkeypatch.setattr(ClipListingIndex, "reconcile", observed_reconcile)

    # When: a new manifest is atomically visible before the held pass continues.
    app = create_app()
    app.state.clip_store = ClipStore(root)
    with TestClient(app):
        _write_manifest(root, "clip-new")
        allow_background.set()
        assert background_complete.wait(timeout=1)
        index = app.state.clip_listing_index
        page = index.page(ClipListQuery(limit=48))

    # Then: the background pass makes the new publication queryable without restart.
    assert {manifest.clip_id for manifest in page.manifests} == {"clip-initial", "clip-new"}
