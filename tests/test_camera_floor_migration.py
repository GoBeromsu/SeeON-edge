"""Unit tests for the issue #155 floor migration (free-text -> integer).

Covers ``parse_legacy_floor`` (the pure parsing/coercion function, reused by
both the read-time self-heal in ``public_camera`` and the one-time rewrite
in ``CameraRegistryStore.migrate_legacy_string_floors``) and the store
method itself.
"""

from __future__ import annotations

import pytest

from backend.app.features.cameras.store import (
    DEFAULT_FLOOR,
    CameraRegistryStore,
    floor_label,
    is_valid_floor,
    parse_legacy_floor,
)
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


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


def test_migrate_legacy_string_floors_defaults_an_unparseable_value(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    store.create(
        camera_id="camera-1",
        label="garbled",
        rtsp_url="rtsp://cam/1",
        space_id=None,
        status="online",
        floor=5,
    )
    # A legacy free-text value the typed API can no longer write.
    sandbox.admin.execute(
        "UPDATE cameras SET floor_override=%s WHERE camera_id=%s", ("지하 1층", "camera-1")
    )

    changes = store.migrate_legacy_string_floors()

    assert changes == [{"camera_id": "camera-1", "old": "지하 1층", "new": DEFAULT_FLOOR}]
    assert store.get("camera-1")["floor"] == DEFAULT_FLOOR
