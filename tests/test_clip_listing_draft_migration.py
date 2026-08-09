from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.features.clips.listing_repository import ListingRepository

_RELEASED_COLUMNS = (
    "generation", "clip_id", "manifest_path", "manifest_mtime_ns",
    "manifest_size_bytes", "camera_id", "event_ref", "event_type", "event_facet",
    "started_at", "duration_s", "codec", "media_path", "video_available",
    "video_error", "finalized", "size_bytes",
)
_THUMBNAIL_COLUMNS = (
    "generation", "clip_id", "thumbnail_mtime_ns", "thumbnail_size_bytes",
    "thumbnail_available",
)
_RELEASED_ROWS_DDL = """CREATE TABLE clip_listing_rows (
    generation INTEGER NOT NULL, clip_id TEXT NOT NULL, manifest_path TEXT NOT NULL,
    manifest_mtime_ns INTEGER NOT NULL, manifest_size_bytes INTEGER NOT NULL,
    camera_id TEXT NOT NULL, event_ref TEXT NOT NULL, event_type TEXT,
    event_facet TEXT NOT NULL CHECK (event_facet IN ('fall','bed-exit','other')),
    started_at TEXT NOT NULL, duration_s REAL NOT NULL, codec TEXT NOT NULL,
    media_path TEXT, video_available INTEGER NOT NULL, video_error TEXT,
    finalized INTEGER NOT NULL, size_bytes INTEGER,
    PRIMARY KEY (generation, clip_id), UNIQUE (generation, manifest_path)
) STRICT"""
_DRAFT_ROWS_DDL = _RELEASED_ROWS_DDL.replace(
    "    finalized INTEGER NOT NULL, size_bytes INTEGER,\n",
    "    finalized INTEGER NOT NULL, size_bytes INTEGER,\n"
    "    thumbnail_mtime_ns INTEGER, thumbnail_size_bytes INTEGER,\n"
    "    thumbnail_available INTEGER NOT NULL DEFAULT 0,\n",
)
_DRAFT_POSITIONAL_INSERT = (
    "INSERT INTO clip_listing_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


def _draft_values() -> tuple[str | int | float | None, ...]:
    return (
        7, "draft-clip", "draft/manifest.json", 101, 202, "camera-a", "event-ref",
        "fall", "fall", "2026-08-09T00:00:00Z", 3.5, "h264", "clips/clip-a", 1,
        None, 1, 303, 404, 505, 1,
    )


def _create_draft_catalog(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(_DRAFT_ROWS_DDL)
        connection.execute(
            "CREATE TABLE clip_listing_generation ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), active_generation INTEGER NOT NULL, "
            "next_generation INTEGER NOT NULL) STRICT"
        )
        connection.execute("INSERT INTO clip_listing_generation VALUES (1, 7, 8)")
        connection.execute("CREATE TABLE unrelated_catalog_state (value TEXT NOT NULL) STRICT")
        connection.execute("INSERT INTO unrelated_catalog_state VALUES ('preserved')")
        connection.execute(_DRAFT_POSITIONAL_INSERT, _draft_values())
        for statement in (
            "CREATE INDEX clip_listing_global_order_idx ON clip_listing_rows("
            "generation, started_at DESC, clip_id DESC)",
            "CREATE INDEX clip_listing_global_facet_order_idx ON clip_listing_rows("
            "generation, event_facet, started_at DESC, clip_id DESC)",
            "CREATE INDEX clip_listing_camera_order_idx ON clip_listing_rows("
            "generation, camera_id, started_at DESC, clip_id DESC)",
            "CREATE INDEX clip_listing_camera_facet_order_idx ON clip_listing_rows("
            "generation, camera_id, event_facet, started_at DESC, clip_id DESC)",
        ):
            connection.execute(statement)


def test_open_migrates_exact_thumbnail_draft_without_losing_catalog_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.sqlite3"
    _create_draft_catalog(path)

    repository = ListingRepository.open(path)
    active = repository.active_clips()["draft/manifest.json"]
    repository.close()

    with sqlite3.connect(path) as connection:
        base_row = connection.execute(
            "SELECT generation, clip_id, manifest_path, manifest_mtime_ns, "
            "manifest_size_bytes, camera_id, event_ref, event_type, event_facet, "
            "started_at, duration_s, codec, media_path, video_available, video_error, "
            "finalized, size_bytes FROM clip_listing_rows"
        ).fetchone()
        thumbnail_row = connection.execute(
            "SELECT generation, clip_id, thumbnail_mtime_ns, thumbnail_size_bytes, "
            "thumbnail_available FROM clip_listing_thumbnails"
        ).fetchone()
        table_rows = {
            row[1]: row for row in connection.execute("PRAGMA table_list")
            if row[1] in {"clip_listing_rows", "clip_listing_thumbnails"}
        }
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list('clip_listing_rows')")
        }
        row_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(clip_listing_rows)")
        )
        thumbnail_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(clip_listing_thumbnails)")
        )
        side_objects = connection.execute(
            "SELECT type, name FROM sqlite_schema WHERE "
            "tbl_name = 'clip_listing_thumbnails' AND sql IS NOT NULL "
            "AND type IN ('index', 'trigger')"
        ).fetchall()
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list('clip_listing_thumbnails')"
        ).fetchall()
        unrelated = connection.execute("SELECT value FROM unrelated_catalog_state").fetchone()
        backup = connection.execute(
            "SELECT name FROM sqlite_schema WHERE name = 'clip_listing_rows_draft_268'"
        ).fetchall()

    assert base_row == _draft_values()[:17]
    assert thumbnail_row == (7, "draft-clip", 404, 505, 1)
    assert unrelated == ("preserved",)
    assert active.thumbnail_mtime_ns == 404
    assert active.thumbnail_size_bytes == 505
    assert active.thumbnail_available is True
    assert table_rows["clip_listing_rows"][5] == 1
    assert table_rows["clip_listing_thumbnails"][5] == 1
    assert row_columns == _RELEASED_COLUMNS
    assert thumbnail_columns == _THUMBNAIL_COLUMNS
    assert side_objects == []
    assert foreign_keys == []
    assert backup == []
    assert {
        "clip_listing_global_order_idx", "clip_listing_global_facet_order_idx",
        "clip_listing_camera_order_idx", "clip_listing_camera_facet_order_idx",
    } <= indexes


def test_open_rejects_unknown_near_match_without_mutating_catalog(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    near_match = _DRAFT_ROWS_DDL.replace(
        "thumbnail_available INTEGER NOT NULL DEFAULT 0",
        "thumbnail_available INTEGER NOT NULL DEFAULT 1",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(near_match)
        connection.execute("CREATE TABLE unrelated_catalog_state (value TEXT NOT NULL) STRICT")
        connection.execute("INSERT INTO unrelated_catalog_state VALUES ('untouched')")
        before_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'clip_listing_rows'"
        ).fetchone()

    with pytest.raises(sqlite3.DatabaseError, match="unsupported clip_listing_rows schema"):
        ListingRepository.open(path)

    with sqlite3.connect(path) as connection:
        after_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'clip_listing_rows'"
        ).fetchone()
        unrelated = connection.execute("SELECT value FROM unrelated_catalog_state").fetchone()
        listing_objects = connection.execute(
            "SELECT name FROM sqlite_schema WHERE name IN "
            "('clip_listing_generation', 'clip_listing_summary', 'clip_listing_thumbnails')"
        ).fetchall()

    assert after_sql == before_sql
    assert unrelated == ("untouched",)
    assert listing_objects == []
