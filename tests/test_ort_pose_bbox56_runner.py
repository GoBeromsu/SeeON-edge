from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner


def test_ort_runner_matches_torch_proxy_bundle(tmp_path: Path) -> None:
    root = write_pose_bbox56_bundle(tmp_path)
    torch_runner = PoseBbox56BundleRunner.from_artifact_dir(root)
    ort_runner = OrtPoseBbox56Runner.from_artifact_dir(root)
    window = np.linspace(-1.0, 1.0, 30 * 56, dtype=np.float32).reshape(30, 56)

    assert ort_runner.device == "cpu"
    assert ort_runner.artifact_digest != torch_runner.artifact_digest
    assert ort_runner.preprocessing_identity == torch_runner.preprocessing_identity
    ort_result = ort_runner.predict(window)
    torch_result = torch_runner.predict(window)
    assert np.allclose(
        (
            ort_result.background,
            ort_result.fall_transition,
            ort_result.fallen,
        ),
        (
            torch_result.background,
            torch_result.fall_transition,
            torch_result.fallen,
        ),
        atol=1e-5,
    )
    assert ort_result.model_evidence is not None
    assert torch_result.model_evidence is not None
    assert ort_result.model_evidence.class_origins == (
        "derived_complement",
        "temperature_sigmoid",
        "constant_zero",
    )
    assert torch_result.model_evidence.class_origins == (
        "derived_complement",
        "temperature_sigmoid",
        "constant_zero",
    )
    assert ort_result.model_evidence.applied_temperature == 1.0
    assert torch_result.model_evidence.applied_temperature == 1.0
    assert ort_result.model_evidence.raw_logit == pytest.approx(
        torch_result.model_evidence.raw_logit,
        abs=1e-5,
    )
    for result in (ort_result, torch_result):
        evidence = result.model_evidence
        assert evidence is not None
        expected = 1.0 / (1.0 + np.exp(-evidence.raw_logit / evidence.applied_temperature))
        assert result.fall_transition == pytest.approx(expected)


def test_ort_runner_refuses_non_cpu_provider_before_session_creation(tmp_path: Path) -> None:
    calls: list[object] = []

    with pytest.raises(ModelLoadError, match="CPUExecutionProvider only"):
        OrtPoseBbox56Runner.from_artifact_dir(
            write_pose_bbox56_bundle(tmp_path),
            providers=["CUDAExecutionProvider"],
            session_factory=lambda *_args: calls.append("called"),
        )

    assert calls == []


def test_ort_runner_refuses_tampered_onnx_before_session_creation(tmp_path: Path) -> None:
    root = write_pose_bbox56_bundle(tmp_path)
    (root / "model.onnx").write_bytes(b"tampered")
    calls: list[object] = []

    with pytest.raises(ModelLoadError, match="identity mismatch: model.onnx"):
        OrtPoseBbox56Runner.from_artifact_dir(
            root, session_factory=lambda *_args: calls.append("called")
        )

    assert calls == []
