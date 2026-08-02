"""Unit tests for DetectionSettingsStore -- the domain-PK'd persisted
per-domain detection on/off + window table backing
``GET``/``PUT /api/v1/detection-settings`` (see detection_settings/router.py)."""

from __future__ import annotations

from pathlib import Path

from backend.app.features.detection_settings.store import (
    DetectionSettingsStore,
    DomainDetectionSetting,
)


def test_get_all_is_empty_for_a_fresh_store(tmp_path: Path) -> None:
    store = DetectionSettingsStore(tmp_path / "catalog.sqlite3")
    assert store.get_all() == {}


def test_replace_all_then_get_all_round_trips_an_always_on_domain(tmp_path: Path) -> None:
    store = DetectionSettingsStore(tmp_path / "catalog.sqlite3")

    store.replace_all(
        {"fall": DomainDetectionSetting(on=True, mode="always", start=None, end=None)}
    )

    fetched = store.get_all()
    assert fetched == {
        "fall": DomainDetectionSetting(on=True, mode="always", start=None, end=None)
    }
    assert fetched["fall"].as_dict() == {
        "on": True,
        "mode": "always",
        "start": None,
        "end": None,
    }


def test_replace_all_then_get_all_round_trips_a_window_mode_domain(tmp_path: Path) -> None:
    store = DetectionSettingsStore(tmp_path / "catalog.sqlite3")

    store.replace_all(
        {
            "bed_exit": DomainDetectionSetting(
                on=True, mode="window", start="22:00", end="06:00"
            )
        }
    )

    fetched = store.get_all()["bed_exit"]
    assert fetched.on is True
    assert fetched.mode == "window"
    assert fetched.start == "22:00"
    assert fetched.end == "06:00"


def test_replace_all_writes_both_domains_in_a_single_call(tmp_path: Path) -> None:
    store = DetectionSettingsStore(tmp_path / "catalog.sqlite3")

    store.replace_all(
        {
            "fall": DomainDetectionSetting(on=True, mode="always", start=None, end=None),
            "bed_exit": DomainDetectionSetting(
                on=False, mode="window", start="21:00", end="05:00"
            ),
        }
    )

    fetched = store.get_all()
    assert set(fetched) == {"fall", "bed_exit"}
    assert fetched["fall"].on is True
    assert fetched["bed_exit"].on is False


def test_replace_all_upserts_overwriting_a_previous_setting_for_the_same_domain(
    tmp_path: Path,
) -> None:
    store = DetectionSettingsStore(tmp_path / "catalog.sqlite3")
    store.replace_all(
        {"fall": DomainDetectionSetting(on=True, mode="always", start=None, end=None)}
    )

    store.replace_all(
        {"fall": DomainDetectionSetting(on=False, mode="window", start="20:00", end="07:00")}
    )

    fetched = store.get_all()["fall"]
    assert fetched.on is False
    assert fetched.mode == "window"
    assert fetched.start == "20:00"
    assert fetched.end == "07:00"


def test_replace_all_leaves_domains_not_included_in_the_call_untouched(tmp_path: Path) -> None:
    store = DetectionSettingsStore(tmp_path / "catalog.sqlite3")
    store.replace_all(
        {
            "fall": DomainDetectionSetting(on=True, mode="always", start=None, end=None),
            "bed_exit": DomainDetectionSetting(
                on=True, mode="window", start="22:00", end="06:00"
            ),
        }
    )

    store.replace_all(
        {"fall": DomainDetectionSetting(on=False, mode="always", start=None, end=None)}
    )

    fetched = store.get_all()
    assert fetched["fall"].on is False
    assert fetched["bed_exit"].mode == "window"
    assert fetched["bed_exit"].start == "22:00"


def test_store_persists_across_reopening_the_same_database_file(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    DetectionSettingsStore(path).replace_all(
        {"fall": DomainDetectionSetting(on=False, mode="always", start=None, end=None)}
    )

    reopened = DetectionSettingsStore(path)
    fetched = reopened.get_all()

    assert fetched == {
        "fall": DomainDetectionSetting(on=False, mode="always", start=None, end=None)
    }
