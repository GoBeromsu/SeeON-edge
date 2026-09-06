"""Persistence invariants for canonical multi-region bed zones."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from backend.app.features.cameras.bed_zone_store import (
    BedZone,
    BedZoneRegion,
    BedZoneStore,
)
from tests_support.compact_authority_db import prepare_compact_database, seed_camera


@pytest.fixture(autouse=True)
def _compact_camera_database(tmp_path: Path) -> None:
    path = prepare_compact_database(tmp_path / "catalog.sqlite3")
    seed_camera(path, "camera-1")
    seed_camera(path, "camera-2")


def _region(identifier: str = "bed-a") -> BedZoneRegion:
    return BedZoneRegion(
        id=identifier,
        polygon=((1, 2), (9, 2), (9, 8), (1, 8)),
        origin="manual",
    )


def test_put_then_get_round_trips_regions_and_dimensions(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    saved = store.put(
        "camera-1",
        regions=(_region(),),
        image_width=640,
        image_height=480,
        recognized_at="2026-08-03T00:00:00.000Z",
    )
    assert saved == BedZone(
        regions=(_region(),),
        image_width=640,
        image_height=480,
        recognized_at="2026-08-03T00:00:00.000Z",
    )
    assert store.get("camera-1") == saved
    assert saved.as_dict() == {
        "regions": [
            {
                "id": "bed-a",
                "polygon": [[1, 2], [9, 2], [9, 8], [1, 8]],
                "origin": "manual",
            }
        ],
        "image_width": 640,
        "image_height": 480,
        "recognized_at": "2026-08-03T00:00:00.000Z",
    }


def test_put_validates_before_attempting_the_sqlite_write(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    writes: list[sqlite3.Connection] = []
    with pytest.raises(ValueError, match="distinct"):
        store.put(
            "camera-1",
            regions=(_region("duplicate"), _region("duplicate")),
            image_width=640,
            image_height=480,
            recognized_at="2026-08-03T00:00:00.000Z",
            after_write=writes.append,
        )
    assert writes == []
    assert store.get("camera-1") is None


def test_put_reports_an_invalid_region_type_as_a_type_error(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    invalid_region: Any = {"id": "not-a-region"}
    with pytest.raises(TypeError, match="invalid shape"):
        store.put(
            "camera-1",
            regions=(invalid_region,),
            image_width=640,
            image_height=480,
            recognized_at="2026-08-03T00:00:00.000Z",
        )
    assert store.get("camera-1") is None


def test_delete_removes_all_bed_zone_columns_and_calls_hook_in_transaction(
    tmp_path: Path,
) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    store.put(
        "camera-1",
        regions=(_region(),),
        image_width=640,
        image_height=480,
        recognized_at="2026-08-03T00:00:00.000Z",
    )
    hook_transactions: list[bool] = []
    assert store.delete(
        "camera-1",
        after_write=lambda connection: hook_transactions.append(connection.in_transaction),
    )
    assert hook_transactions == [True]
    assert store.get("camera-1") is None
    assert store.delete("camera-1") is False


def test_get_all_returns_only_canonical_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    store = BedZoneStore(path)
    store.put(
        "camera-1",
        regions=(_region(),),
        image_width=640,
        image_height=480,
        recognized_at="2026-08-03T00:00:00.000Z",
    )

    # The retired singleton wire must not be interpreted as a model region.
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE cameras SET bed_polygon_json=?,bed_image_width=?,bed_image_height=?,"
        "bed_recognized_at=? WHERE camera_id='camera-2'",
        ("[[1,2],[9,2],[9,8],[1,8]]", 640, 480, "2026-08-03T00:00:00.000Z"),
    )
    connection.commit()
    connection.close()

    assert store.get("camera-2") is None
    assert store.get_all() == {"camera-1": store.get("camera-1")}


def test_missing_camera_write_raises_without_creating_data(tmp_path: Path) -> None:
    store = BedZoneStore(tmp_path / "catalog.sqlite3")
    with pytest.raises(sqlite3.IntegrityError, match="does not exist"):
        store.put(
            "missing",
            regions=(_region(),),
            image_width=640,
            image_height=480,
            recognized_at="2026-08-03T00:00:00.000Z",
        )
    assert store.get("missing") is None
