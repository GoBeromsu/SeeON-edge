"""Direct camera-model residue plus the supported static-YAML boundary.

Camera roster validation still belongs to ``CameraRuntimeConfig`` and
``WorkerConfig`` because pulled, versioned configuration constructs those
models. ``load_worker_config`` no longer accepts a non-empty static roster; its
remaining YAML contract is limited to non-mutable local settings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worker.runtime.config import (
    CameraRuntimeConfig,
    WorkerConfig,
    WorkerConfigError,
    load_worker_config,
)


def _relay_payload() -> dict[str, object]:
    return {
        "relay": {
            "url": "http://127.0.0.1:8000",
            "token": "relay-token-1",
        },
        "cameras": [],
    }


def _camera(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "rtsp_url": "rtsp://camera.local/trackID=2",
    }
    payload.update(overrides)
    return payload


def test_worker_config_loads_supported_yaml_and_redacts_relay_token(tmp_path: Path) -> None:
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(_relay_payload()), encoding="utf-8")

    config = load_worker_config(config_path)

    assert config.cameras == ()
    assert config.relay.url == "http://127.0.0.1:8000"
    assert config.relay_alert_url == "http://127.0.0.1:8000/api/v1/relay/alerts"
    assert config.relay_heartbeat_url == "http://127.0.0.1:8000/api/v1/relay/heartbeat"
    assert "relay-token-1" not in repr(config)


def test_worker_config_rejects_static_camera_roster_before_model_validation(
    tmp_path: Path,
) -> None:
    payload = _relay_payload()
    payload["cameras"] = [_camera(), _camera(camera_id="camera-1")]
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="static camera roster is retired"):
        load_worker_config(config_path)


def test_worker_config_model_rejects_duplicate_camera_ids() -> None:
    with pytest.raises(WorkerConfigError, match="duplicate camera_id"):
        WorkerConfig.model_validate(
            {
                **_relay_payload(),
                "cameras": [_camera(), _camera(camera_id="camera-1")],
            }
        )


def test_camera_runtime_config_requires_rtsp_url() -> None:
    with pytest.raises(ValueError, match="rtsp_url or streams.sub"):
        CameraRuntimeConfig(camera_id="camera-1", facility_id="facility-1")


def test_camera_runtime_config_normalizes_blank_resident_id() -> None:
    config = CameraRuntimeConfig(**_camera(resident_id="  "))

    assert config.resident_id is None


def test_camera_runtime_config_defaults_to_opencv_decode_backend() -> None:
    config = CameraRuntimeConfig(**_camera())

    assert config.decode_backend is None
    assert config.fps == 5.0
    assert config.inference_rtsp_url == "rtsp://camera.local/trackID=2"
    assert config.main_rtsp_url is None


def test_camera_runtime_config_accepts_opencv_decode_backend() -> None:
    config = CameraRuntimeConfig(**_camera(decode_backend="OpenCV"))

    assert config.decode_backend == "opencv"


def test_camera_runtime_config_accepts_nvdec_and_auto_decode_backend() -> None:
    for value, expected in (("NVDEC", "nvdec"), ("Auto", "auto"), ("cpu", "cpu")):
        config = CameraRuntimeConfig(**_camera(decode_backend=value))
        assert config.decode_backend == expected


def test_camera_runtime_config_accepts_dual_streams_and_fps() -> None:
    config = CameraRuntimeConfig(
        camera_id="camera-1",
        facility_id="facility-1",
        streams={
            "sub": " rtsp://camera.local/sub ",
            "main": "rtsp://camera.local/main",
        },
        fps=7.5,
    )

    assert config.rtsp_url is None
    assert config.inference_rtsp_url == "rtsp://camera.local/sub"
    assert config.main_rtsp_url == "rtsp://camera.local/main"
    assert config.fps == 7.5


def test_camera_runtime_config_prefers_sub_stream_for_inference() -> None:
    config = CameraRuntimeConfig(
        **_camera(
            rtsp_url="rtsp://camera.local/legacy",
            streams={"sub": "rtsp://camera.local/sub"},
        )
    )

    assert config.inference_rtsp_url == "rtsp://camera.local/sub"


def test_camera_runtime_config_rejects_missing_inference_url() -> None:
    with pytest.raises(ValueError, match="rtsp_url or streams.sub"):
        CameraRuntimeConfig(camera_id="camera-1", facility_id="facility-1")


def test_camera_runtime_config_rejects_invalid_stream_url() -> None:
    with pytest.raises(ValueError, match="streams must start with rtsp://"):
        CameraRuntimeConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            streams={"sub": "http://camera.local/sub"},
        )


def test_camera_runtime_config_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError):
        CameraRuntimeConfig(**_camera(fps=0))


def test_camera_runtime_config_rejects_unknown_decode_backend() -> None:
    with pytest.raises(ValueError, match="decode_backend must be one of"):
        CameraRuntimeConfig(**_camera(decode_backend="gstreamer"))


def test_worker_yaml_rejects_relative_relay_url(tmp_path: Path) -> None:
    payload = _relay_payload()
    relay = payload["relay"]
    assert isinstance(relay, dict)
    relay["url"] = "/relay"
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match=r"relay\.url"):
        load_worker_config(config_path)


def test_worker_config_error_does_not_include_relay_token_value(tmp_path: Path) -> None:
    payload = _relay_payload()
    relay = payload["relay"]
    assert isinstance(relay, dict)
    relay["token"] = "super-secret-value"
    relay["url"] = "not-http"
    config_path = tmp_path / "ml-worker.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError) as exc_info:
        load_worker_config(config_path)

    assert "super-secret-value" not in str(exc_info.value)
