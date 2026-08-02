"""Issue #65: fall-model family registry -- config/metadata-driven dispatch.

Owner acceptance criteria under test (from the issue's comments):
  (a) the fall-model family in use is selected by config (``FallModelConfig
      .type``), not by which class is imported and called in
      ``worker/runtime/worker.py``.
  (b) onboarding a new family means implementing ``FallModelProtocol`` and
      registering a factory -- no edits to ``_create_fall_model``'s call
      sites and no widening of the ``type`` pin beyond the one-time
      Literal["lstm"] -> str change that landed this dispatch mechanism.
  (c) an unrecognized ``type`` value refuses to boot with a loud, diagnostic
      error naming every registered family -- never a silent fallback to
      "lstm" or any other default.
  (d) coverage includes a second, fake/test-only model family that loads and
      runs purely through config, with no hardcoded reference to it in any
      production module -- proving the registry actually decouples model
      family from ``worker.py`` rather than moving the hardcoding one layer
      down.

The #43 none-config boot refusal (``config.models.fall is None``) is
untouched by this issue and stays covered in
``tests/test_worker_fall_model_selection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import pytest

import worker.runtime.worker as worker_module
from worker.adapters.model.fall_family_registry import (
    FallModelFamilyRegistry,
    UnknownFallModelTypeError,
    default_fall_model_family_registry,
)
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime


@final
class _ForbiddenServingClient:
    """Fall model dispatch must never reach the serving client (mirrors the
    #43 test double in ``test_worker_fall_model_selection.py``)."""

    def create(self, task: str, **_options: object) -> object:
        raise AssertionError(f"fall model dispatch must not call serving.create({task!r})")


@dataclass(frozen=True, slots=True)
class _FakeMetadata:
    window: int = 3
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class _FakeFamilyFallModel:
    """A test-only fall-model family.

    No production module (``worker.py``, the default registry, or any
    adapter) references this class. It exists purely to prove requirement
    (d): the registry can load a brand-new family purely through
    ``config.models.fall.type``, without any hardcoded branch for it.
    """

    def __init__(self) -> None:
        self.metadata = _FakeMetadata()
        self.operating_threshold = 0.9
        self.predict_calls = 0

    def predict(self, _features: object) -> float:
        self.predict_calls += 1
        return 0.42


def _write_fall_artifact(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"placeholder")
    (path / "arch.json").write_text('{"hidden":4,"layers":1,"dropout":0.0}', encoding="utf-8")
    (path / "metadata.yaml").write_text("type: lstm\n", encoding="utf-8")
    return path


def _config(fall_type: str, artifact_dir: Path) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://example.test/camera-a",
                    "heartbeat_interval_sec": 30.0,
                }
            ],
            "models": {
                "fall": {
                    "type": fall_type,
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
        }
    )


def _runtime(config: WorkerConfig, state_dir: Path) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=_ForbiddenServingClient(),
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
    )


def test_registry_register_and_create_round_trip(tmp_path: Path) -> None:
    artifact_dir = _write_fall_artifact(tmp_path / "fake")
    config = _config("fake-family", artifact_dir)
    registry = FallModelFamilyRegistry()
    sentinel = _FakeFamilyFallModel()
    registry.register("fake-family", lambda _config, _device: sentinel)

    assert registry.types() == ("fake-family",)
    assert registry.create("fake-family", config.models.fall, "cpu") is sentinel


def test_registry_get_factory_raises_for_unregistered_type_naming_known_families() -> None:
    registry = FallModelFamilyRegistry()
    registry.register("lstm", lambda _config, _device: object())
    registry.register("fake-family", lambda _config, _device: object())

    with pytest.raises(UnknownFallModelTypeError) as excinfo:
        registry.get_factory("gru")

    message = str(excinfo.value)
    assert "gru" in message
    assert "lstm" in message
    assert "fake-family" in message


def test_default_registry_only_registers_lstm() -> None:
    registry = default_fall_model_family_registry()
    assert registry.types() == ("lstm",)


def test_create_fall_model_loads_a_fake_family_registered_purely_via_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Requirement (d): a fake second family, registered on a throwaway
    registry, loads through ``WorkerRuntime._create_fall_model`` purely
    because ``config.models.fall.type`` names it -- there is no "fake-family"
    branch anywhere in ``worker.py``."""
    artifact_dir = _write_fall_artifact(tmp_path / "models" / "fall" / "fake")
    config = _config("fake-family", artifact_dir)
    fake_model = _FakeFamilyFallModel()

    registry = FallModelFamilyRegistry()
    registry.register("fake-family", lambda _config, _device: fake_model)
    monkeypatch.setattr(worker_module, "DEFAULT_FALL_MODEL_FAMILY_REGISTRY", registry)

    runtime = _runtime(config, tmp_path)
    model = runtime._create_fall_model("cpu")  # noqa: SLF001

    assert model is fake_model


def test_create_fall_model_refuses_to_boot_for_an_unknown_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Requirement (c): an unrecognized ``type`` refuses to boot with a loud
    error naming the registered families -- never a silent fallback to
    "lstm" or any other default."""
    artifact_dir = _write_fall_artifact(tmp_path / "models" / "fall" / "unknown")
    config = _config("gru", artifact_dir)

    registry = FallModelFamilyRegistry()
    registry.register("lstm", lambda _config, _device: object())
    monkeypatch.setattr(worker_module, "DEFAULT_FALL_MODEL_FAMILY_REGISTRY", registry)

    runtime = _runtime(config, tmp_path)

    with pytest.raises(RuntimeError, match="unknown fall model type 'gru'"):
        runtime._create_fall_model("cpu")  # noqa: SLF001
