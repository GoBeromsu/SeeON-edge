from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from backend.app.features.clips.listing_generation import prepare_generation
from backend.app.features.clips.listing_index import ClipListingIndex, ReconcileStats
from backend.app.features.clips.listing_repository import ListingRepository
from backend.app.features.clips.schemas import ClipListQuery
from backend.app.features.clips.store import ClipStore

_RELEASED_POSITIONAL_INSERT = (
    "INSERT INTO clip_listing_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)
_RELEASED_COLUMNS = (
    "generation",
    "clip_id",
    "manifest_path",
    "manifest_mtime_ns",
    "manifest_size_bytes",
    "camera_id",
    "event_ref",
    "event_type",
    "event_facet",
    "started_at",
    "duration_s",
    "codec",
    "media_path",
    "video_available",
    "video_error",
    "finalized",
    "size_bytes",
)
_THUMBNAIL_COLUMNS = (
    "generation",
    "clip_id",
    "thumbnail_mtime_ns",
    "thumbnail_size_bytes",
    "thumbnail_available",
)


def _write_manifest(root: Path, clip_id: str) -> Path:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip.mp4").write_bytes(clip_id.encode())
    manifest_path = clip_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "camera-a",
                "event_ref": f"event-{clip_id}",
                "event_type": "fall",
                "started_at": "2026-08-09T00:00:00Z",
                "duration_s": 5.0,
                "codec": "h264",
                "path": f"clips/{clip_id}",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _released_row(
    generation: int,
    clip_id: str,
    manifest_path: str,
    manifest_mtime_ns: int = 101,
    manifest_size_bytes: int = 202,
) -> tuple[str | int | float | None, ...]:
    return (
        generation,
        clip_id,
        manifest_path,
        manifest_mtime_ns,
        manifest_size_bytes,
        "camera-a",
        f"event-{clip_id}",
        "fall",
        "fall",
        "2026-08-09T00:00:00Z",
        5.0,
        "h264",
        f"clips/{clip_id}",
        1,
        None,
        1,
        len(clip_id),
    )


def test_released_positional_insert_remains_valid_after_current_reconciliation(
    tmp_path: Path,
) -> None:
    index = ClipListingIndex.open(tmp_path / "catalog.sqlite3")
    assert index.reconcile(ClipStore(tmp_path / "clip-store")) == ReconcileStats(0, 0, 0, 0, 0)
    index.close()

    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        connection.execute(
            _RELEASED_POSITIONAL_INSERT,
            _released_row(99, "rollback-clip", "rollback/manifest.json"),
        )
        stored = connection.execute(
            "SELECT clip_id, size_bytes FROM clip_listing_rows WHERE generation = 99"
        ).fetchone()

    assert stored == ("rollback-clip", len("rollback-clip"))


def test_open_rolls_forward_released_generation_without_thumbnail_row(tmp_path: Path) -> None:
    root = tmp_path / "clip-store"
    manifest_path = _write_manifest(root, "clip-legacy")
    manifest_stat = manifest_path.stat()
    path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE clip_listing_rows (
                generation INTEGER NOT NULL, clip_id TEXT NOT NULL,
                manifest_path TEXT NOT NULL, manifest_mtime_ns INTEGER NOT NULL,
                manifest_size_bytes INTEGER NOT NULL, camera_id TEXT NOT NULL,
                event_ref TEXT NOT NULL, event_type TEXT,
                event_facet TEXT NOT NULL CHECK (event_facet IN ('fall','bed-exit','other')),
                started_at TEXT NOT NULL, duration_s REAL NOT NULL, codec TEXT NOT NULL,
                media_path TEXT, video_available INTEGER NOT NULL, video_error TEXT,
                finalized INTEGER NOT NULL, size_bytes INTEGER,
                PRIMARY KEY (generation, clip_id), UNIQUE (generation, manifest_path)
            ) STRICT"""
        )
        connection.execute(
            "CREATE TABLE clip_listing_generation ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), active_generation INTEGER NOT NULL, "
            "next_generation INTEGER NOT NULL) STRICT"
        )
        connection.execute("INSERT INTO clip_listing_generation VALUES (1, 7, 8)")
        connection.execute(
            _RELEASED_POSITIONAL_INSERT,
            _released_row(
                7,
                "clip-legacy",
                str(manifest_path),
                manifest_stat.st_mtime_ns,
                manifest_stat.st_size,
            ),
        )

    index = ClipListingIndex.open(path)
    stats = index.reconcile(ClipStore(root))
    page = index.page(ClipListQuery(limit=48))
    index.close()
    with sqlite3.connect(path) as connection:
        row_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(clip_listing_rows)")
        )
        thumbnail_columns = tuple(
            row[1]
            for row in connection.execute("PRAGMA table_info(clip_listing_thumbnails)")
        )

    assert stats == ReconcileStats(1, 0, 0, 0, 1)
    assert page.manifests[0].thumbnail_available is False
    assert row_columns == _RELEASED_COLUMNS
    assert thumbnail_columns == _THUMBNAIL_COLUMNS


def test_publish_removes_stale_thumbnail_generations(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_manifest(first_root, "clip-01")
    _write_manifest(second_root, "clip-02")
    (first_root / "clips" / "clip-01" / "thumbnail.jpg").write_bytes(b"first")
    path = tmp_path / "catalog.sqlite3"
    repository = ListingRepository.open(path)
    repository.publish(prepare_generation(ClipStore(first_root), {}))
    repository.publish(prepare_generation(ClipStore(second_root), {}))
    repository.close()

    with sqlite3.connect(path) as connection:
        generations = connection.execute(
            "SELECT generation, clip_id FROM clip_listing_thumbnails ORDER BY generation"
        ).fetchall()

    assert generations == [(2, "clip-02")]
