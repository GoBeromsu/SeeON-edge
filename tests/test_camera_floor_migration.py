"""Unit tests for the issue #155 floor migration (free-text -> integer).

Covers ``parse_legacy_floor`` (the pure parsing/coercion function, reused by
both the read-time self-heal in ``public_camera`` and the one-time rewrite
in ``CameraRegistryStore.migrate_legacy_string_floors``) and the store
method itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.features.cameras.store import (
    DEFAULT_FLOOR,
    CameraRegistryStore,
    floor_label,
    is_valid_floor,
    parse_legacy_floor,
)


def test_parse_legacy_floor_passes_none_through_untouched() -> None:
    assert parse_legacy_floor(None) is None


def test_parse_legacy_floor_passes_a_valid_int_through_untouched() -> None:
    assert parse_legacy_floor(2) == 2
    assert parse_legacy_floor(-1) == -1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2층", 2),
        ("10층", 10),
        ("2 층", 2),
        ("2", 2),
        ("-1", -1),
        ("B1", -1),
        ("b1", -1),
        ("B 1", -1),
    ],
)
def test_parse_legacy_floor_recognizes_known_legacy_shapes(raw: str, expected: int) -> None:
    assert parse_legacy_floor(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "지하 1층",  # unsupported shape (not B<n> or <n>층)
        "이층",
        "B2",  # parses but outside the fixed B1..10층 catalog
        "11층",
        "0층",
        "",
        "   ",
    ],
)
def test_parse_legacy_floor_defaults_and_logs_on_unparseable_or_out_of_range_values(
    raw: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        result = parse_legacy_floor(raw, camera_id="camera-1")

    assert result == DEFAULT_FLOOR
    assert "camera-1" in caplog.text
    assert repr(raw) in caplog.text


def test_parse_legacy_floor_defaults_and_logs_on_garbage_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        assert parse_legacy_floor(3.5, camera_id="camera-2") == DEFAULT_FLOOR
        assert parse_legacy_floor(["not", "a", "floor"], camera_id="camera-2") == DEFAULT_FLOOR
        assert parse_legacy_floor(True, camera_id="camera-2") == DEFAULT_FLOOR

    assert "camera-2" in caplog.text


def test_parse_legacy_floor_defaults_an_out_of_range_int(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert parse_legacy_floor(99, camera_id="camera-3") == DEFAULT_FLOOR

    assert "camera-3" in caplog.text


def test_floor_label_round_trips_the_fixed_catalog() -> None:
    assert floor_label(-1) == "B1"
    assert floor_label(1) == "1층"
    assert floor_label(10) == "10층"


def test_is_valid_floor_rejects_zero_and_out_of_range_values() -> None:
    assert is_valid_floor(0) is False
    assert is_valid_floor(-2) is False
    assert is_valid_floor(11) is False
    assert is_valid_floor(1) is True
    assert is_valid_floor(-1) is True
    assert is_valid_floor(10) is True


def test_migrate_legacy_string_floors_rewrites_stored_strings_in_place(tmp_path: Path) -> None:
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1", label="205호", rtsp_url="rtsp://cam/1",
        space_id=None, status="online", floor="2층",
    )
    store.create(
        camera_id="camera-2", label="B1호", rtsp_url="rtsp://cam/2",
        space_id=None, status="online", floor="B1",
    )
    store.create(
        camera_id="camera-3", label="already-int", rtsp_url="rtsp://cam/3",
        space_id=None, status="online", floor=5,
    )
    store.create(
        camera_id="camera-4", label="unset", rtsp_url="rtsp://cam/4",
        space_id=None, status="online",
    )

    version_before = store.snapshot()["registry_version"]
    changes = store.migrate_legacy_string_floors()

    assert {change["camera_id"]: change["new"] for change in changes} == {
        "camera-1": 2,
        "camera-2": -1,
    }
    assert store.get("camera-1")["floor"] == 2
    assert store.get("camera-2")["floor"] == -1
    # Already-int and unset records are untouched.
    assert store.get("camera-3")["floor"] == 5
    assert store.get("camera-4").get("floor") is None
    assert store.snapshot()["registry_version"] == version_before + 1


def test_migrate_legacy_string_floors_is_idempotent(tmp_path: Path) -> None:
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1", label="205호", rtsp_url="rtsp://cam/1",
        space_id=None, status="online", floor="2층",
    )

    first_pass = store.migrate_legacy_string_floors()
    version_after_first = store.snapshot()["registry_version"]
    second_pass = store.migrate_legacy_string_floors()

    assert len(first_pass) == 1
    assert second_pass == []
    assert store.snapshot()["registry_version"] == version_after_first


def test_migrate_legacy_string_floors_defaults_an_unparseable_value_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1", label="garbled", rtsp_url="rtsp://cam/1",
        space_id=None, status="online", floor="지하 1층",
    )

    with caplog.at_level("WARNING"):
        changes = store.migrate_legacy_string_floors()

    assert changes == [{"camera_id": "camera-1", "old": "지하 1층", "new": DEFAULT_FLOOR}]
    assert store.get("camera-1")["floor"] == DEFAULT_FLOOR
    assert "camera-1" in caplog.text
