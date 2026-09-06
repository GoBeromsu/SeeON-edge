"""Round-trip characterization of ``contracts.worker_config.PulledWorkerConfig``
after generalizing the bed_exit-only ``night_window`` into per-domain
``detection_windows`` (issue #24).

Covers ``from_dict``/``as_dict`` precedence: an explicit ``detection_windows``
map wins over the deprecated single ``night_window`` field; a legacy
``night_window``-only payload still populates ``detection_windows["bed_exit"]``;
and ``night_window`` on a constructed instance always mirrors
``detection_windows.get("bed_exit")`` for from_dict-built configs.
"""

from __future__ import annotations

import pytest

from contracts.worker_config import PulledNightWindow, PulledWorkerConfig


def _camera_payload() -> dict[str, object]:
    return {
        "camera_id": "camera-1",
        "space_id": "space-1",
        "label": "Room 1",
        "rtsp_url": "rtsp://camera-1.local",
        "online": True,
    }


def test_from_dict_with_only_detection_windows_populates_night_window_alias() -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "detection_windows": {
            "bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"},
            "fall": {"start": "22:00", "end": "05:00", "tz": "Asia/Seoul"},
        },
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {
        "bed_exit": PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
        "fall": PulledNightWindow(start="22:00", end="05:00", tz="Asia/Seoul"),
    }
    assert config.night_window == PulledNightWindow(start="21:00", end="06:00", tz="UTC")


def test_from_dict_with_only_legacy_night_window_falls_back_to_bed_exit_entry() -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "night_window": {"start": "21:00", "end": "06:00", "tz": "UTC"},
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {
        "bed_exit": PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
    }
    assert config.night_window == PulledNightWindow(start="21:00", end="06:00", tz="UTC")


def test_from_dict_prefers_detection_windows_over_legacy_night_window_when_both_present() -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        # Deliberately different from detection_windows["bed_exit"] below, to
        # prove detection_windows wins outright rather than merging.
        "night_window": {"start": "01:00", "end": "02:00", "tz": "UTC"},
        "detection_windows": {
            "bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"},
        },
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {
        "bed_exit": PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
    }
    assert config.night_window == PulledNightWindow(start="21:00", end="06:00", tz="UTC")


def test_from_dict_with_neither_field_has_no_windows() -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {}
    assert config.night_window is None


def test_as_dict_round_trips_detection_windows_and_legacy_night_window() -> None:
    original = PulledWorkerConfig.from_dict(
        {
            "config_version": 1,
            "restart_epoch": 0,
            "cameras": [_camera_payload()],
            "detection_windows": {
                "bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"},
                "fall": {"start": "22:00", "end": "05:00", "tz": "Asia/Seoul"},
            },
        }
    )

    dumped = original.as_dict()
    assert dumped["night_window"] == {"start": "21:00", "end": "06:00", "tz": "UTC"}
    assert dumped["detection_windows"] == {
        "bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"},
        "fall": {"start": "22:00", "end": "05:00", "tz": "Asia/Seoul"},
    }

    restored = PulledWorkerConfig.from_dict(dumped)
    assert restored == original


def test_as_dict_round_trips_empty_detection_windows() -> None:
    original = PulledWorkerConfig.from_dict(
        {"config_version": 1, "restart_epoch": 0, "cameras": [_camera_payload()]}
    )

    dumped = original.as_dict()
    assert dumped["detection_windows"] == {}
    assert dumped["night_window"] is None

    restored = PulledWorkerConfig.from_dict(dumped)
    assert restored == original


def test_from_dict_drops_explicit_null_entry_in_detection_windows_map() -> None:
    """An explicit ``null`` entry means ALWAYS/24-7 for that domain: dropped
    from the map (not an error, so nothing is logged) rather than raised."""
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "detection_windows": {
            "bed_exit": None,
            "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
        },
    }

    config = PulledWorkerConfig.from_dict(data)

    assert "bed_exit" not in config.detection_windows
    assert config.detection_windows["fall"] == PulledNightWindow(
        start="22:00", end="05:00", tz="UTC"
    )
    assert config.night_window is None


def test_from_dict_drops_start_equal_end_window_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """start == end would make DetectionWindow.contains match nothing --
    fail open to ALWAYS (dropped from the map) with a loud stderr log."""
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "detection_windows": {"bed_exit": {"start": "09:00", "end": "09:00", "tz": "UTC"}},
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {}
    assert config.night_window is None
    err = capsys.readouterr().err
    assert "bed_exit" in err
    assert "matches nothing" in err


def test_from_dict_drops_malformed_hhmm_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "detection_windows": {"bed_exit": {"start": "25:00", "end": "17:00", "tz": "UTC"}},
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {}
    err = capsys.readouterr().err
    assert "bed_exit" in err
    assert "HH:MM" in err


def test_from_dict_drops_invalid_timezone_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "detection_windows": {"bed_exit": {"start": "21:00", "end": "06:00", "tz": "Not/A_Zone"}},
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {}
    err = capsys.readouterr().err
    assert "bed_exit" in err
    assert "timezone" in err


def test_from_dict_drops_non_dict_window_value_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "detection_windows": {"bed_exit": "not-a-window"},
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {}
    err = capsys.readouterr().err
    assert "bed_exit" in err


def test_from_dict_one_invalid_domain_does_not_affect_other_valid_domains() -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "detection_windows": {
            "bed_exit": {"start": "09:00", "end": "09:00", "tz": "UTC"},
            "fall": {"start": "22:00", "end": "05:00", "tz": "Asia/Seoul"},
        },
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {
        "fall": PulledNightWindow(start="22:00", end="05:00", tz="Asia/Seoul"),
    }


def test_from_dict_legacy_night_window_with_start_equal_end_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [_camera_payload()],
        "night_window": {"start": "09:00", "end": "09:00", "tz": "UTC"},
    }

    config = PulledWorkerConfig.from_dict(data)

    assert config.detection_windows == {}
    assert config.night_window is None
    err = capsys.readouterr().err
    assert "bed_exit" in err
