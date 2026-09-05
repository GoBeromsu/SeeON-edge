from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.loader import load_worker_config


def _supported_yaml(path: Path) -> Path:
    payload = {
        "version": 1,
        "relay": {
            "url": "http://127.0.0.1:8000",
            "token": "relay-token-1",
        },
        "runtime": {
            "max_failures": 30,
            "open_timeout_ms": 5000,
            "read_timeout_ms": 5000,
        },
        "dev_mjpeg": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8091,
        },
        "cameras": [],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_ml_worker_yaml_loads_supported_non_mutable_contract(tmp_path: Path) -> None:
    config = load_worker_config(_supported_yaml(tmp_path / "ml-worker.yaml"))

    assert config.version == 1
    assert config.relay.url == "http://127.0.0.1:8000"
    assert config.relay_alert_url == "http://127.0.0.1:8000/api/v1/relay/alerts"
    assert config.relay_heartbeat_url == "http://127.0.0.1:8000/api/v1/relay/heartbeat"
    assert config.runtime.max_failures == 30
    assert config.runtime.open_timeout_ms == 5000
    assert config.runtime.read_timeout_ms == 5000
    assert config.dev_mjpeg.enabled is True
    assert config.dev_mjpeg.port == 8091
    assert config.cameras == ()
    # The YAML hatch declares no fall model; the runtime's local overlay settles
    # one, and a composed runtime refuses if it never does. A parsed YAML with
    # no models block therefore has none - not an empty placeholder.
    assert config.models is None
    assert config.domains.resolved_overrides() == {}
    assert config.clip.enabled is True
    assert "relay-token-1" not in repr(config)


def test_ml_worker_yaml_allows_omitted_empty_camera_roster(tmp_path: Path) -> None:
    path = _supported_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["cameras"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(path)

    assert config.cameras == ()


def test_ml_worker_yaml_rejects_static_camera_roster(tmp_path: Path) -> None:
    path = _supported_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["cameras"] = [
        {
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "rtsp_url": "rtsp://camera-1.local/trackID=2",
        }
    ]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="static camera roster is retired"):
        load_worker_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("models", {}),
        ("domains", {}),
        ("clip", {"enabled": True}),
    ],
)
def test_ml_worker_yaml_rejects_retired_mutable_authority(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = _supported_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match=rf"static {field} policy is retired"):
        load_worker_config(path)


def test_ml_worker_rejects_json_config(tmp_path: Path) -> None:
    path = tmp_path / "edge-cameras.json"
    path.write_text('{"cameras":[]}', encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="YAML"):
        load_worker_config(path)


def test_ml_worker_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "ml-worker.yaml"
    path.write_text("cameras: [", encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="not valid YAML"):
        load_worker_config(path)


def test_ml_worker_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "ml-worker.yaml"
    path.write_text("- relay\n", encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="must contain a YAML mapping"):
        load_worker_config(path)
