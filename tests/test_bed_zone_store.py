"""Unit tests for BedZoneStore -- the camera_id-PK'd persisted bed-zone table
(distinct from CameraRegistryStore's single-row JSON-blob shape) backing the
on-demand bed-zone recognition endpoint (see bed_zone_router.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.features.cameras.bed_zone_store import BedZone, BedZoneStore
from tests_support.compact_authority_db import prepare_compact_database, seed_camera


@pytest.fixture(autouse=True)
def _compact_camera_database(tmp_path: Path) -> None:
    path = prepare_compact_database(tmp_path / "catalog.sqlite3")
    seed_camera(path, "camera-1")
    seed_camera(path, "camera-2")


def test_get_returns_none_for_an_unknown_camera(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    assert store.get("camera-1") is None


def test_put_then_get_round_trips_the_polygon_and_dimensions(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")

    saved = store.put(
        "camera-1",
        polygon=[[1, 2], [9, 2], [9, 8], [1, 8]],
        image_width=640,
        image_height=480,
        recognized_at="2026-08-03T00:00:00.000Z",
    )

    assert saved == BedZone(
        polygon=((1, 2), (9, 2), (9, 8), (1, 8)),
        image_width=640,
        image_height=480,
        recognized_at="2026-08-03T00:00:00.000Z",
    )
    fetched = store.get("camera-1")
    assert fetched == saved
    assert fetched.as_dict() == {
        "polygon": [[1, 2], [9, 2], [9, 8], [1, 8]],
        "image_width": 640,
        "image_height": 480,
        "recognized_at": "2026-08-03T00:00:00.000Z",
    }


def test_put_overwrites_a_previous_recognition_for_the_same_camera(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    store.put(
        "camera-1",
        polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
        image_width=100,
        image_height=100,
        recognized_at="2026-08-01T00:00:00.000Z",
    )

    store.put(
        "camera-1",
        polygon=[[2, 2], [9, 2], [9, 9], [2, 9]],
        image_width=640,
        image_height=480,
        recognized_at="2026-08-02T00:00:00.000Z",
    )

    fetched = store.get("camera-1")
    assert fetched.polygon == ((2, 2), (9, 2), (9, 9), (2, 9))
    assert fetched.recognized_at == "2026-08-02T00:00:00.000Z"


def test_delete_removes_the_row_and_is_idempotent(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    store.put(
        "camera-1",
        polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
        image_width=100,
        image_height=100,
        recognized_at="2026-08-01T00:00:00.000Z",
    )

    assert store.delete("camera-1") is True
    assert store.get("camera-1") is None
    # Deleting an already-absent row is a no-op, not an error.
    assert store.delete("camera-1") is False


def test_get_all_returns_every_persisted_camera_keyed_by_camera_id(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    store.put(
        "camera-1",
        polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
        image_width=100,
        image_height=100,
        recognized_at="2026-08-01T00:00:00.000Z",
    )
    store.put(
        "camera-2",
        polygon=[[2, 2], [9, 2], [9, 9], [2, 9]],
        image_width=640,
        image_height=480,
        recognized_at="2026-08-02T00:00:00.000Z",
    )

    all_zones = store.get_all()

    assert set(all_zones) == {"camera-1", "camera-2"}
    assert all_zones["camera-2"].polygon == ((2, 2), (9, 2), (9, 9), (2, 9))


def test_store_persists_across_reopening_the_same_database_file(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    BedZoneStore(path).put(
        "camera-1",
        polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
        image_width=100,
        image_height=100,
        recognized_at="2026-08-01T00:00:00.000Z",
    )

    reopened = BedZoneStore(path)
    fetched = reopened.get("camera-1")

    assert fetched is not None
    assert fetched.polygon == ((0, 0), (1, 0), (1, 1), (0, 1))
