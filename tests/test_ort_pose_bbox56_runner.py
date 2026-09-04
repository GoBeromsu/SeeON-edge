from __future__ import annotations

import json
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.fall_family_registry import default_fall_model_family_registry
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner
from worker.runtime.config.worker_models import FallModelConfig


def test_ort_runner_matches_torch_proxy_bundle(tmp_path: Path) -> None:
    root = write_pose_bbox56_bundle(tmp_path)
    torch_runner = PoseBbox56BundleRunner.from_artifact_dir(root)
    ort_runner = OrtPoseBbox56Runner.from_artifact_dir(root)
    window = np.linspace(-1.0, 1.0, 30 * 56, dtype=np.float32).reshape(30, 56)

    assert ort_runner.device == "cpu"
    assert ort_runner.artifact_digest != torch_runner.artifact_digest
    assert ort_runner.preprocessing_identity == torch_runner.preprocessing_identity
    assert np.allclose(
        astuple(ort_runner.predict(window)),
        astuple(torch_runner.predict(window)),
        atol=1e-5,
    )


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


def test_registry_selects_onnx_format_and_keeps_proxy_format_torch(tmp_path: Path) -> None:
    root = write_pose_bbox56_bundle(tmp_path)
    registry = default_fall_model_family_registry()

    assert isinstance(
        registry.create_bundle("pose-bbox56-onnx-v0", root, "cpu"), OrtPoseBbox56Runner
    )

    manifest_path = root / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        item for item in manifest["files"] if item["relative_path"] != "model.onnx"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "model.onnx").unlink()
    assert isinstance(
        registry.create_bundle("pose-bbox56-proxy-v0", root, "cpu"), PoseBbox56BundleRunner
    )
    common = {
        "type": "pose-bbox56-proxy-v0",
        "mode": "sequence",
        "artifact_dir": root,
        "window": 30,
        "stride": 5,
        "input_shape": (30, 56),
        "operating_threshold": 0.5,
    }
    assert FallModelConfig(framework="pytorch", **common).framework == "pytorch"
    with pytest.raises(ValidationError, match="missing model.onnx"):
        FallModelConfig(framework="onnxruntime", **common)
