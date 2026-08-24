from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from time import perf_counter

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.clips.listing_index import (
    ClipListingIndex,
    ClipListingReconcileError,
)
from backend.app.features.clips.schemas import ClipListQuery
from backend.app.features.clips.store import ClipStore
from backend.app.main import create_app, no_lifespan


def _write_manifest(
    root: Path,
    clip_id: str,
    *,
    camera_id: str = "camera-a",
    event_type: str | None = "fall",
    event_ref: str = "event-ref",
) -> None:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, str | float | bool] = {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "event_ref": event_ref,
        "started_at": f"2026-08-09T00:00:{clip_id[-2:]}Z",
        "duration_s": 1.0,
        "codec": "h264",
        "video_available": False,
        "finalized": True,
    }
    if event_type is not None:
        payload["event_type"] = event_type
    _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


@pytest.mark.parametrize(
    ("query", "expected_index"),
    [
        (ClipListQuery(limit=48), "clip_listing_global_order_idx"),
        (
            ClipListQuery(limit=48, event_type="fall"),
            "clip_listing_global_facet_order_idx",
        ),
        (
            ClipListQuery(limit=48, camera_id="camera-a"),
            "clip_listing_camera_order_idx",
        ),
        (
            ClipListQuery(limit=48, camera_id="camera-a", event_type="fall"),
            "clip_listing_camera_facet_order_idx",
        ),
    ],
)
def test_page_query_plan_is_ordered_and_summary_is_precomputed(
    tmp_path: Path,
    query: ClipListQuery,
    expected_index: str,
) -> None:
    # Given: enough indexed rows for SQLite to exercise each filter/order plan.
    root = tmp_path / "clip-store"
    for index in range(120):
        _write_manifest(
            root,
            f"clip-{index:03d}",
            camera_id="camera-a" if index % 2 == 0 else "camera-b",
            event_type="fall" if index % 3 == 0 else "bed-exit",
        )
    listing = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    _ = listing.reconcile(ClipStore(root))

    # When: the exact request statements are explained.
    plans = listing.explain(query)
    listing.close()

    # Then: page order is index-backed and metadata comes from bounded summary rows.
    page_plan = " ".join(plans.page)
    summary_plan = " ".join(plans.summary)
    assert expected_index in page_plan
    assert "USE TEMP B-TREE" not in page_plan
    assert "SCAN clip_listing_rows" not in page_plan
    assert "clip_listing_summary" in summary_plan
    assert "COUNT(" not in plans.summary_sql
    assert "GROUP BY" not in plans.summary_sql


def test_reconcile_precomputes_global_and_camera_summaries_atomically(tmp_path: Path) -> None:
    # Given: two cameras and all canonical facets.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01", event_type="fall")
    _write_manifest(root, "clip-02", event_type="bed-exit")
    _write_manifest(root, "clip-03", event_type=None, event_ref="unknown")
    _write_manifest(root, "clip-04", camera_id="camera-b", event_type="fall")
    path = tmp_path / "catalog.sqlite3"
    listing = ClipListingIndex.open(path)

    # When: reconciliation publishes a new active generation.
    _ = listing.reconcile(ClipStore(root))

    # Then: global/per-camera totals and facets are complete in that same generation.
    with sqlite3.connect(path) as connection:
        generation = connection.execute(
            "SELECT active_generation FROM clip_listing_generation WHERE id = 1"
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT camera_id, event_facet, count FROM clip_listing_summary "
            "WHERE generation = ? ORDER BY camera_id, event_facet",
            (generation,),
        ).fetchall()
    listing.close()
    assert rows == [
        ("", "", 4),
        ("", "bed-exit", 1),
        ("", "fall", 2),
        ("", "other", 1),
        ("camera-a", "", 3),
        ("camera-a", "bed-exit", 1),
        ("camera-a", "fall", 1),
        ("camera-a", "other", 1),
        ("camera-b", "", 1),
        ("camera-b", "fall", 1),
    ]


def test_page_does_not_wait_for_filesystem_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a ready generation and a later reconciliation paused in manifest stat.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01")
    listing = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    _ = listing.reconcile(ClipStore(root))
    discovery_entered = threading.Event()
    allow_discovery = threading.Event()
    page_finished = threading.Event()
    original_stat = Path.stat

    def blocking_stat(path: Path, *, follow_symlinks: bool = True):
        if path.name == "manifest.json" and not discovery_entered.is_set():
            discovery_entered.set()
            assert allow_discovery.wait(timeout=2)
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", blocking_stat)
    reconcile_thread = threading.Thread(target=listing.reconcile, args=(ClipStore(root),))
    reconcile_thread.start()
    assert discovery_entered.wait(timeout=1)

    # When: a page is requested while O(N) filesystem work remains paused.
    started_at = perf_counter()
    page_thread = threading.Thread(
        target=lambda: (listing.page(ClipListQuery(limit=48)), page_finished.set())
    )
    page_thread.start()
    try:
        completed_without_discovery = page_finished.wait(timeout=0.25)
        elapsed = perf_counter() - started_at
    finally:
        allow_discovery.set()
        reconcile_thread.join(timeout=2)
        page_thread.join(timeout=2)
        listing.close()

    # Then: the previous generation remains readable without taking the writer lock.
    assert completed_without_discovery
    assert elapsed < 0.25


def test_facets_fall_back_to_event_ref_when_event_type_is_absent(tmp_path: Path) -> None:
    # Given: historical manifests whose domain exists only in event_ref.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01", event_type=None, event_ref="fall")
    _write_manifest(root, "clip-02", event_type=None, event_ref="bed-exit")
    _write_manifest(root, "clip-03", event_type=None, event_ref="legacy-event")

    # When: both indexed and legacy listings derive facets.
    listing = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    _ = listing.reconcile(ClipStore(root))
    indexed = listing.page(ClipListQuery(limit=48))
    app = create_app(lifespan=no_lifespan)
    app.state.clip_store = ClipStore(root)
    with TestClient(app) as client:
        _login(client)
        legacy = client.get("/api/v1/clips")
    listing.close()

    # Then: both paths expose the same bounded canonical categories.
    expected = {"bed-exit": 1, "fall": 1, "other": 1}
    assert indexed.event_type_counts == expected
    assert legacy.json()["event_type_counts"] == expected


@pytest.mark.parametrize(
    "params",
    [{"event_type": "legacy"}, {"event_type": "mystery", "limit": 48}],
)
def test_event_type_filter_rejects_noncanonical_values(
    tmp_path: Path,
    params: dict[str, str | int],
) -> None:
    # Given: an authenticated app for either legacy or paged listing.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01")
    app = create_app(lifespan=no_lifespan)
    app.state.clip_store = ClipStore(root)
    listing = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    _ = listing.reconcile(ClipStore(root))
    app.state.clip_listing_index = listing

    # When: a caller supplies a noncanonical event filter.
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips", params=params)
    listing.close()

    # Then: query-boundary validation rejects it consistently before either path runs.
    assert response.status_code == 422


def test_runtime_open_on_migrated_edge_database_executes_no_ddl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the one-shot migrator has provisioned the complete central database.
    path = tmp_path / "edge.sqlite3"
    migrate_database(path)
    from backend.app.edge_db.connection import RuntimeActor, open_runtime_database

    connection = open_runtime_database(path, actor=RuntimeActor.API)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(
                "CREATE TABLE clip_listing_generation (id INTEGER PRIMARY KEY, "
                "active_generation INTEGER NOT NULL, next_generation INTEGER NOT NULL) STRICT"
            )
    finally:
        connection.close()
    assert "clip_listing_generation" not in tables


def test_listing_lifespan_can_enter_same_app_twice(
    tmp_path: Path,
) -> None:
    # Given: one FastAPI application and one persisted clip.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01")
    app = create_app()
    app.state.clip_store = ClipStore(root)

    # When: the same application's production lifespan is entered twice.
    pages: list[int] = []
    for _ in range(2):
        with TestClient(app) as client:
            _login(client)
            response = client.get("/api/v1/clips", params={"limit": 48})
            assert response.status_code == 200
            pages.append(len(response.json()["clips"]))
            assert not hasattr(app.state, "clip_listing_index")
        assert not hasattr(app.state, "clip_listing_index")

    # Then: compact listing serves both entries without a listing-runtime index.
    assert pages == [1, 1]


def test_reconcile_after_close_returns_typed_unavailable_error(tmp_path: Path) -> None:
    # Given: an index whose lifecycle owner already closed it.
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01")
    listing = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    listing.close()

    # When: stale application state attempts another reconciliation.
    with pytest.raises(ClipListingReconcileError):
        listing.reconcile(ClipStore(root))

    # Then: no raw sqlite closed-connection error escapes the index boundary.
