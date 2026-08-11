"""``worker.runtime.config.pull_models.BackendWorkerConfigPayload`` must degrade
per-entry on a malformed field, matching the two sibling parse boundaries that
already do (``contracts/worker_config.py``'s ``_pulled_detection_windows`` and
``backend/app/lifespan.py``'s ``_pulled_night_window``) -- see issue #28.

Before this fix, only an explicit ``null`` detection-window entry was
tolerated (fixed in #24); a wrong-typed window value, a window missing
start/end/tz, or a malformed camera entry each raised a pydantic
``ValidationError`` that rejected the *entire* payload. Whole-payload
rejection is still correct for top-level shape violations (e.g. a missing
version or a missing ``cameras`` key); this module only covers per-entry
degradation within an otherwise well-shaped payload.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.worker_config import PulledNightWindow
from worker.runtime.config.pull_models import BackendWorkerConfigPayload


def _camera_payload(
    camera_id: str = "camera-1",
    facility_id: str | None = "facility-1",
    rtsp_url: str = "rtsp://camera/stream",
) -> dict[str, object]:
    return {"camera_id": camera_id, "facility_id": facility_id, "rtsp_url": rtsp_url}


def test_null_domain_window_drops_only_that_domain() -> None:
    """Regression guard for #24: an explicit ``null`` window is ALWAYS for
    that domain, not an error, and must not affect the rest of the payload."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [_camera_payload()],
            "detection_windows": {
                "bed_exit": None,
                "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
            },
        }
    )

    pulled = payload.to_pulled_config()

    assert "bed_exit" not in pulled.detection_windows
    assert pulled.detection_windows["fall"] == PulledNightWindow(
        start="22:00", end="05:00", tz="UTC"
    )


def test_string_window_value_drops_only_that_domain_and_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [_camera_payload()],
            "detection_windows": {
                "bed_exit": "not-a-window",
                "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
            },
        }
    )

    pulled = payload.to_pulled_config()

    assert "bed_exit" not in pulled.detection_windows
    assert pulled.detection_windows["fall"] == PulledNightWindow(
        start="22:00", end="05:00", tz="UTC"
    )
    err = capsys.readouterr().err
    assert "bed_exit" in err


def test_window_missing_required_fields_drops_only_that_domain_and_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [_camera_payload()],
            "detection_windows": {
                # Missing "end" and "tz".
                "bed_exit": {"start": "21:00"},
                "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
            },
        }
    )

    pulled = payload.to_pulled_config()

    assert "bed_exit" not in pulled.detection_windows
    assert pulled.detection_windows["fall"] == PulledNightWindow(
        start="22:00", end="05:00", tz="UTC"
    )
    err = capsys.readouterr().err
    assert "bed_exit" in err


def test_camera_with_bad_fps_type_drops_only_that_camera_and_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                _camera_payload(camera_id="camera-1"),
                {**_camera_payload(camera_id="camera-2"), "fps": "fast"},
            ],
        }
    )

    pulled = payload.to_pulled_config()

    assert [camera.camera_id for camera in pulled.cameras] == ["camera-1"]
    err = capsys.readouterr().err
    assert "camera-2" in err


def test_camera_without_facility_or_space_is_accepted_as_local() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                _camera_payload(camera_id="camera-1"),
                # camera_id + rtsp is enough; facility defaults to "local".
                {"camera_id": "camera-2", "rtsp_url": "rtsp://camera-2/stream"},
            ],
        }
    )

    config = payload.to_worker_config("http://ml-api:8000", "relay-token")
    by_id = {camera.camera_id: camera for camera in config.cameras}
    assert set(by_id) == {"camera-1", "camera-2"}
    assert by_id["camera-2"].facility_id == "local"


def test_missing_cameras_key_still_rejects_whole_payload() -> None:
    """Whole-payload rejection remains for top-level shape violations -- the
    per-entry degradation above only applies within an otherwise well-shaped
    payload, not to a payload missing a required top-level key."""
    with pytest.raises(ValidationError):
        BackendWorkerConfigPayload.model_validate({"config_version": 5})


def test_missing_version_still_rejects_whole_payload() -> None:
    with pytest.raises(ValidationError):
        BackendWorkerConfigPayload.model_validate({"cameras": [_camera_payload()]})
