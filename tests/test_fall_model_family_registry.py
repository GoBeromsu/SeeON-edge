"""Packaged fall-model family registry behavior."""

from __future__ import annotations

from pathlib import Path
from typing import final

import pytest

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
    assert audit.operating_threshold == 0.5
    assert audit.threshold_source == "default"
    assert audit.unapplied_policy_threshold == pytest.approx(0.7)
