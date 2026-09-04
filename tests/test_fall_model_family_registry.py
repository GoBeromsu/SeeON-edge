"""Issue #65: fall-model family registry -- config/metadata-driven dispatch.

Owner acceptance criteria under test (from the issue's comments):
  (a) the fall-model family in use is selected by config (``FallModelConfig
      .type``), not by which class is imported and called in
      ``worker/runtime/worker.py``.
  (b) onboarding a new family means implementing ``FallV2ModelProtocol`` and
      registering a factory -- no edits to ``_create_fall_model``'s call
      sites and no widening of the ``type`` pin beyond the one-time
      ``Literal`` -> ``str`` change that landed this dispatch mechanism.
  (c) an unrecognized ``type`` value refuses to boot with a loud, diagnostic
      error naming every registered family -- never a silent fallback to
      the packaged family or any other default.
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

from pathlib import Path
from typing import final

import pytest

import worker.runtime.worker as worker_module
from shared.detection_policies import FallPolicyV2, PolicySource, make_effective_policy
from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model.fall_family_registry import (
    FallModelFamilyRegistry,
    UnknownFallModelTypeError,
    default_fall_model_family_registry,
)
from worker.domains import DETECTION_MODULE_REGISTRY, CameraModuleContext
from worker.interfaces.fall_model import FallV2Probabilities
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


@final
class _FakeFamilyFallModel:
    """A test-only fall-model family.

    No production module (``worker.py``, the default registry, or any
    adapter) references this class. It exists purely to prove requirement
    (d): the registry can load a brand-new family purely through
    ``config.models.fall.type``, without any hardcoded branch for it.
    """

    device = "cpu"

    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, _features: object) -> FallV2Probabilities:
        self.predict_calls += 1
        return FallV2Probabilities(background=0.58, fall_transition=0.42, fallen=0.0)

    def warmup(self) -> None:
        self.predict(None)


def _write_fall_artifact(path: Path) -> Path:
    return write_pose_bbox56_bundle(path)


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
                    "window": 30,
                    "stride": 5,
                    "input_shape": [30, 56],
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
    registry.register("pose-bbox56-proxy-v0", lambda _config, _device: object())
    registry.register("fake-family", lambda _config, _device: object())

    with pytest.raises(UnknownFallModelTypeError) as excinfo:
        registry.get_factory("gru")

    message = str(excinfo.value)
    assert "gru" in message
    assert "pose-bbox56-proxy-v0" in message
    assert "fake-family" in message


def test_default_registry_only_registers_the_packaged_pose_bbox56_family() -> None:
    registry = default_fall_model_family_registry()
    assert registry.types() == ("pose-bbox56-proxy-v0",)
    assert registry.runtime_formats() == ("pose-bbox56-onnx-v0", "pose-bbox56-proxy-v0")


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


def _fall_module(
    model: object,
    transition_threshold: float = 0.5,
    *,
    source: PolicySource = "image-default",
):
    policy = make_effective_policy(
        module_id="fall",
        module_version=2,
        values=FallPolicyV2(transition_threshold=transition_threshold),
        source=source,
        facility_revision_id=1 if source == "facility-default" else None,
        camera_revision_id=1 if source == "camera-override" else None,
    )
    definition = DETECTION_MODULE_REGISTRY.get("fall", 2)
    context = CameraModuleContext(
        camera_id="camera-a",
        facility_id="facility-a",
        shared_components={"fall-classifier": model},
        camera_components={"episode-identity": ("boot", "1", 0)},
        detection_window=None,
        clock=lambda: pytest.fail("fall clock should not be called"),
        diagnostics=None,
        policy=policy,
    )
    return definition, definition.create_camera_module(context), context


def test_research_bundle_keeps_the_policy_default_threshold_and_audits_the_receipt(
    tmp_path: Path,
) -> None:
    """Threshold precedence (P1a-AC7): the packaged bundle's receipt is
    research-only (``promotion_eligible`` false), so the owner-fixed 0.5
    policy default governs the decider and the audit envelope names the
    source; the receipt value is still recorded for audit."""
    artifact_dir = write_pose_bbox56_bundle(
        tmp_path / "research", receipt_threshold=0.05, promotion_eligible=False
    )
    fall_config = _config("pose-bbox56-proxy-v0", artifact_dir).models.fall
    assert fall_config is not None
    model = default_fall_model_family_registry().create("pose-bbox56-proxy-v0", fall_config, "cpu")
    assert model.device == "cpu"

    definition, module, context = _fall_module(model)
    assert definition.audit_adapter is not None
    audit = definition.audit_adapter(context)

    assert module.decider.policy.policy.transition_threshold == 0.5
    assert audit.operating_threshold == 0.5
    assert audit.threshold_source == "default"
    assert audit.receipt_threshold == pytest.approx(0.05)


def test_promotion_eligible_receipt_overrides_the_policy_default_threshold(
    tmp_path: Path,
) -> None:
    artifact_dir = write_pose_bbox56_bundle(
        tmp_path / "promoted", receipt_threshold=0.3, promotion_eligible=True
    )
    fall_config = _config("pose-bbox56-proxy-v0", artifact_dir).models.fall
    assert fall_config is not None
    model = default_fall_model_family_registry().create("pose-bbox56-proxy-v0", fall_config, "cpu")

    definition, module, context = _fall_module(model)
    assert definition.audit_adapter is not None
    audit = definition.audit_adapter(context)

    assert module.decider.policy.policy.transition_threshold == pytest.approx(0.3)
    assert audit.operating_threshold == pytest.approx(0.3)
    assert audit.threshold_source == "receipt"


def test_operator_policy_override_is_audited_but_not_applied(tmp_path: Path) -> None:
    artifact_dir = write_pose_bbox56_bundle(
        tmp_path / "operator-policy", receipt_threshold=0.05, promotion_eligible=False
    )
    fall_config = _config("pose-bbox56-proxy-v0", artifact_dir).models.fall
    assert fall_config is not None
    model = default_fall_model_family_registry().create("pose-bbox56-proxy-v0", fall_config, "cpu")

    definition, module, context = _fall_module(model, 0.7, source="camera-override")
    assert definition.audit_adapter is not None
    audit = definition.audit_adapter(context)

    assert module.decider.policy.policy.transition_threshold == pytest.approx(0.5)
    assert audit.operating_threshold == pytest.approx(0.5)
    assert audit.threshold_source == "default"
    assert audit.unapplied_policy_threshold == pytest.approx(0.7)


def test_create_fall_model_refuses_to_boot_for_an_unknown_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Requirement (c): an unrecognized ``type`` refuses to boot with a loud
    error naming the registered families -- never a silent fallback to
    the packaged family or any other default."""
    artifact_dir = _write_fall_artifact(tmp_path / "models" / "fall" / "unknown")
    config = _config("gru", artifact_dir)

    registry = FallModelFamilyRegistry()
    registry.register("pose-bbox56-proxy-v0", lambda _config, _device: object())
    monkeypatch.setattr(worker_module, "DEFAULT_FALL_MODEL_FAMILY_REGISTRY", registry)

    runtime = _runtime(config, tmp_path)

    with pytest.raises(RuntimeError, match="unknown fall model type 'gru'"):
        runtime._create_fall_model("cpu")  # noqa: SLF001
