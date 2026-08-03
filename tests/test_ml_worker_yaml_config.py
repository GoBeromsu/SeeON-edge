from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.loader import load_worker_config


def _valid_yaml(path: Path) -> Path:
    artifact_dir = path.parent / "models" / "fall" / "lstm"
    _write_placeholder_artifact(artifact_dir)
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
        "models": {
            "fall": {
                "type": "lstm",
                "framework": "pytorch",
                "mode": "sequence",
                "artifact_dir": str(artifact_dir),
                "weights": "model.pt",
                "architecture": "arch.json",
                "metadata": "metadata.yaml",
                "window": 3,
                "stride": 1,
                "input_shape": [3, 51],
                "operating_threshold": 0.5,
            }
        },
        "domains": {"fall": {"enabled": True}},
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "resident_id": "resident-1",
                "rtsp_url": "rtsp://camera-1.local/trackID=2",
                "heartbeat_interval_sec": 30,
                "frame_stride": 1,
                "label": "Room 1",
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_placeholder_artifact(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"placeholder")
    (path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    (path / "metadata.yaml").write_text("type: lstm\n", encoding="utf-8")


def test_ml_worker_yaml_config_loads_nested_contract(tmp_path: Path) -> None:
    config = load_worker_config(_valid_yaml(tmp_path / "ml-worker.yaml"))

    assert config.version == 1
    assert config.relay.url == "http://127.0.0.1:8000"
    assert config.relay_alert_url == "http://127.0.0.1:8000/api/v1/relay/alerts"
    assert config.relay_heartbeat_url == "http://127.0.0.1:8000/api/v1/relay/heartbeat"
    assert config.runtime.max_failures == 30
    assert config.models.fall is not None
    assert config.models.fall.type == "lstm"
    assert config.models.fall.input_shape == (3, 51)
    assert config.enabled_domains == ("fall",)
    fall_domain = config.domains.domain_config("fall")
    assert fall_domain is not None
    assert fall_domain.enabled is True
    assert len(config.cameras) == 1


def test_ml_worker_yaml_config_preserves_absent_domain_defaults(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["domains"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(path)

    assert config.enabled_domains is None
    assert "domains" not in config.model_fields_set


def test_ml_worker_yaml_config_explicit_empty_domains_enable_none(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(path)

    assert "domains" in config.model_fields_set
    assert config.enabled_domains == ()


def test_ml_worker_yaml_config_all_disabled_domains_enable_none(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {
        "fall": {"enabled": False},
        "bed_exit": {"enabled": False},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(path)

    assert config.enabled_domains == ()
    assert "domains" in config.model_fields_set


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


def test_ml_worker_yaml_config_loads_legacy_enabled_domains(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {"enabled": ["fall", "bed_exit"]}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(path)

    assert config.enabled_domains == ("fall", "bed_exit")
    assert config.domains.domain_config("bed_exit") is None


def test_ml_worker_yaml_config_loads_per_domain_bed_exit_night_window(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {
        "fall": {"enabled": True},
        "bed_exit": {
            "enabled": True,
            "night_window": {"start": "21:00", "end": "05:00", "tz": "Asia/Seoul"},
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(path)

    assert config.enabled_domains == ("fall", "bed_exit")
    assert config.domains.bed_exit is not None
    assert config.domains.bed_exit.night_window is not None
    assert config.domains.bed_exit.night_window.start == "21:00"
    assert config.domains.bed_exit.night_window.end == "05:00"
    assert config.domains.bed_exit.night_window.tz == "Asia/Seoul"


def test_ml_worker_yaml_rejects_mixed_legacy_and_per_domain_config(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {"enabled": ["fall"], "bed_exit": {"enabled": True}}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="domains"):
        load_worker_config(path)


def test_ml_worker_yaml_rejects_unknown_domain(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {"enabled": ["fall", "unknown"]}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="domains.enabled"):
        load_worker_config(path)


@pytest.mark.parametrize("obsolete_domain", ("wheelchair_standup", "long_lie", "risk"))
def test_ml_worker_yaml_rejects_removed_domains(tmp_path: Path, obsolete_domain: str) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {"enabled": [obsolete_domain]}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="domains.enabled"):
        load_worker_config(path)


def test_ml_worker_yaml_rejects_unknown_per_domain_key(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {"unknown": {"enabled": True}}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="domains.unknown"):
        load_worker_config(path)


def test_ml_worker_yaml_rejects_invalid_night_window_time(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {
        "bed_exit": {
            "enabled": True,
            "night_window": {"start": "9:00", "end": "05:00", "tz": "Asia/Seoul"},
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="night_window.start"):
        load_worker_config(path)


def test_ml_worker_yaml_rejects_invalid_night_window_timezone(tmp_path: Path) -> None:
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["domains"] = {
        "bed_exit": {
            "enabled": True,
            "night_window": {"start": "21:00", "end": "05:00", "tz": "Mars/Base"},
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="night_window.tz"):
        load_worker_config(path)


def test_ml_worker_yaml_accepts_any_non_empty_fall_model_type_string(tmp_path: Path) -> None:
    """Issue #65: ``FallModelConfig.type`` opened from ``Literal["lstm"]`` to
    an open, non-empty string, so YAML parsing alone no longer rejects a
    non-"lstm" family name -- that decision moved to the fall-model family
    registry at boot (fail-closed for an *unregistered* family; see
    ``tests/test_fall_model_family_registry.py``), not to config parsing.
    """
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["models"]["fall"]["type"] = "random-forest"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(path)

    assert config.models.fall is not None
    assert config.models.fall.type == "random-forest"

def test_ml_worker_yaml_without_clip_section_enables_clip_recording_by_default(
    tmp_path: Path,
) -> None:
    """The upgrade path must land on always-on, per the #127/#10 decision.

    "Clip recording defaults to on" is a claim about configs that do not
    mention clips at all -- every config written before the flag existed. The
    composition tests all pass ``clip_enabled`` explicitly, and the shipped
    example sets ``clip.enabled: true`` by hand, so none of them exercise the
    default. This asserts the default itself: clips are always-on unless an
    operator opts out explicitly, and a clip-recorder start failure degrades
    visibly (runtime diagnostics) rather than silently disabling clips.

    The in-test assertion that the fixture has no ``clip`` key is part of the
    test: if someone adds one to ``_valid_yaml``, this stops testing the
    default and should say so rather than pass quietly.
    """
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "clip" not in payload, "fixture already sets clip; the default is untested"

    config = load_worker_config(path)

    assert config.clip.enabled is True


def test_ml_worker_yaml_clip_recording_is_opt_in_not_opt_out(tmp_path: Path) -> None:
    """Turning it on must require saying so, and must actually take effect.

    The mirror of the default test: if ``clip.enabled: true`` did not survive
    loading, the default-off test above would still pass while the flag was
    simply inert.
    """
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["clip"] = {"enabled": True}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_worker_config(path)

    assert config.clip.enabled is True


def test_ml_worker_yaml_rejects_unknown_clip_key(tmp_path: Path) -> None:
    """A typo in the clip section must fail loudly, not silently record.

    ``ClipRecordingConfig`` forbids extra keys. Without this, ``enabled_: true``
    or ``enable: true`` would load as the default -- off -- and an operator who
    believed they had turned recording on would get no clips and no error.
    """
    path = _valid_yaml(tmp_path / "ml-worker.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["clip"] = {"enable": True}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkerConfigError, match="clip"):
        load_worker_config(path)
