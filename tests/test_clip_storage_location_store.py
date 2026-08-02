"""Unit tests for ClipStorageLocationStore -- the single-row persisted
clip-storage subdirectory selection backing
``PUT /api/v1/clips/storage/location`` (see clips/storage_router.py)."""

from __future__ import annotations

from pathlib import Path

from backend.app.features.clips.storage_location_store import ClipStorageLocationStore


def test_get_defaults_to_the_empty_string_mount_root_when_nothing_selected(
    tmp_path: Path,
) -> None:
    store = ClipStorageLocationStore(tmp_path / "catalog.sqlite3")
    assert store.get() == ""


def test_put_then_get_round_trips_a_selected_subdirectory(tmp_path: Path) -> None:
    store = ClipStorageLocationStore(tmp_path / "catalog.sqlite3")

    returned = store.put("backup-drive/clips")

    assert returned == "backup-drive/clips"
    assert store.get() == "backup-drive/clips"


def test_put_overwrites_a_previous_selection(tmp_path: Path) -> None:
    store = ClipStorageLocationStore(tmp_path / "catalog.sqlite3")
    store.put("first-choice")

    store.put("second-choice")

    assert store.get() == "second-choice"


def test_put_empty_string_resets_the_selection_to_the_mount_root(tmp_path: Path) -> None:
    store = ClipStorageLocationStore(tmp_path / "catalog.sqlite3")
    store.put("some/subdir")

    store.put("")

    assert store.get() == ""


def test_store_persists_across_reopening_the_same_database_file(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    ClipStorageLocationStore(path).put("external-drive")

    reopened = ClipStorageLocationStore(path)

    assert reopened.get() == "external-drive"
