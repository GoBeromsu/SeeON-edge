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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import pytest
import torch
import yaml

import worker.runtime.worker as worker_module
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    FrameObservation,
)
from worker.adapters.model.fall_family_registry import (
    FallModelFamilyRegistry,
    UnknownFallModelTypeError,
    default_fall_model_family_registry,
)
from worker.adapters.model.torch_lstm_fall import build_lstm_module
from worker.domains.fall import FallWindowClassifier
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime
from worker.types import DecisionInput


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


def _config(
    fall_type: str, artifact_dir: Path, *, operating_threshold: float = 0.5
) -> WorkerConfig:
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
                    "operating_threshold": operating_threshold,
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


def _write_real_lstm_artifact(path: Path, *, manifest_operating_threshold: float) -> Path:
    """A genuinely loadable LSTM artifact (unlike ``_write_fall_artifact``'s
    placeholder ``model.pt``), whose manifest packages
    ``manifest_operating_threshold`` -- deliberately different from whatever
    ``FallModelConfig.operating_threshold`` the test configures, so the
    regression test below can tell "the manifest's own value" apart from
    "the configured override" when it inspects what actually got wired up.
    """
    path.mkdir(parents=True)
    module = build_lstm_module(hidden=4, layers=1, dropout=0.0)
    torch.save(module.state_dict(), path / "model.pt")
    (path / "arch.json").write_text(
        json.dumps({"hidden": 4, "layers": 1, "dropout": 0.0}), encoding="utf-8"
    )
    (path / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "lstm",
                "framework": "pytorch",
                "mode": "sequence",
                "artifact_dir": str(path),
                "schema_version": 2,
                "preprocessing_identity": "coco17-xyc-frame-normalized-zero-fill-v1",
                "weights": "model.pt",
                "architecture": "arch.json",
                "metadata": "metadata.yaml",
                "window": 3,
                "stride": 1,
                "input_shape": [3, 51],
                "operating_threshold": manifest_operating_threshold,
            }
        ),
        encoding="utf-8",
    )
    return path


def _classify_input(*, frame_index: int, track_id: int) -> DecisionInput:
    pose = tuple((25 + index, 100 + index, 0.9) for index in range(17))
    return DecisionInput(
        observation=FrameObservation(poses=(pose,), track_ids=(track_id,)),
        frame_width=100,
        frame_height=200,
        live_track_ids=(track_id,),
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_lstm_operating_threshold_override_reaches_the_classifiers_actual_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #217 regression.

    #204 correctly resolved ``FallModelConfig.operating_threshold`` from env
    and logged it, but the value never reached the object that actually
    decides: ``_create_lstm_fall_model`` dropped it on the floor and
    ``LstmFallRunner.from_artifact_dir`` unconditionally took the packaged
    manifest's own ``operating_threshold`` instead. A test that only checks
    ``FallModelConfig.operating_threshold`` (what #204 shipped) would have
    passed against that broken code -- this drives the real
    registry -> real ``LstmFallRunner.from_artifact_dir`` -> real
    ``FallWindowClassifier`` path (nothing about threshold delivery is
    mocked) and asserts on the classifier's actual fall/no-fall decision,
    the same read of ``self.model.operating_threshold`` that
    ``worker/domains/fall/classifier.py`` performs in production.
    """
    manifest_threshold = 0.9
    configured_threshold = 0.1
    artifact_dir = _write_real_lstm_artifact(
        tmp_path / "lstm", manifest_operating_threshold=manifest_threshold
    )
    config = _config("lstm", artifact_dir, operating_threshold=configured_threshold)
    fall_config = config.models.fall
    assert fall_config is not None

    registry = default_fall_model_family_registry()
    model = registry.create("lstm", fall_config, "cpu")

    # The packaged manifest's own value must not leak through the override.
    assert model.operating_threshold == pytest.approx(configured_threshold)
    assert model.operating_threshold != pytest.approx(manifest_threshold)

    classifier = FallWindowClassifier(model)
    # predict()'s own numerical correctness is covered elsewhere
    # (tests/test_runners_torch_lstm_fall.py); this isolates threshold
    # wiring by fixing the probability squarely between the two candidate
    # thresholds, so the fall/no-fall decision can only be explained by
    # which threshold the classifier actually compared against.
    monkeypatch.setattr(model, "predict", lambda _features: 0.5)

    for frame_index in range(2):
        result = classifier.classify(_classify_input(frame_index=frame_index, track_id=1))
    result = classifier.classify(_classify_input(frame_index=2, track_id=1))

    assert result.labels[0].confidence == 0.5
    # 0.5 >= 0.1 (configured) is True; 0.5 >= 0.9 (manifest) would be False --
    # against the pre-fix code this assertion fails.
    assert result.labels[0].is_fall


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
