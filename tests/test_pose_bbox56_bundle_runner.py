from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner
from worker.interfaces.fall_model import FallV2ModelProtocol

_PACKAGED = Path(__file__).parents[1] / "models/fall/pose-bbox56-gru"


def test_bundle_runner_is_cpu_only_and_warms_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: calls.append("cuda"))
    runner = PoseBbox56BundleRunner.from_artifact_dir(write_pose_bbox56_bundle(tmp_path))
    assert runner.device == "cpu"
    assert isinstance(runner, FallV2ModelProtocol)
    runner.warmup()
    probabilities = runner.predict(np.zeros((30, 56), dtype=np.float32))
    assert probabilities.fallen == 0.0
    assert abs(probabilities.background + probabilities.fall_transition - 1.0) < 1e-6
    assert calls == []


def test_bundle_runner_refuses_non_cpu_device(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="pinned to cpu"):
        PoseBbox56BundleRunner.from_artifact_dir(write_pose_bbox56_bundle(tmp_path), device="cuda")


def test_bundle_runner_exposes_receipt_threshold_and_promotion_eligibility(tmp_path: Path) -> None:
    research = PoseBbox56BundleRunner.from_artifact_dir(
        write_pose_bbox56_bundle(tmp_path / "research", receipt_threshold=0.05)
    )
    assert (research.receipt_threshold, research.promotion_eligible) == (0.05, False)
    promoted = PoseBbox56BundleRunner.from_artifact_dir(
        write_pose_bbox56_bundle(
            tmp_path / "promoted", receipt_threshold=0.3, promotion_eligible=True
        )
    )
    assert (promoted.receipt_threshold, promoted.promotion_eligible) == (0.3, True)


def test_bundle_runner_verifies_every_member_before_loading_weights(tmp_path: Path) -> None:
    root = write_pose_bbox56_bundle(tmp_path)
    (root / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ModelLoadError, match="identity mismatch: model.pt"):
        PoseBbox56BundleRunner.from_artifact_dir(root)


def test_bundle_runner_rejects_wrong_input_shape(tmp_path: Path) -> None:
    runner = PoseBbox56BundleRunner.from_artifact_dir(write_pose_bbox56_bundle(tmp_path))
    with pytest.raises(ModelLoadError, match="shape \\(30, 56\\)"):
        runner.predict(np.zeros((30, 51), dtype=np.float32))


@pytest.mark.skipif(
    not (_PACKAGED / "bundle-manifest.json").exists(),
    reason="published pose+bbox56 bundle is not provisioned locally",
)
def test_published_bundle_loads_on_cpu() -> None:
    runner = PoseBbox56BundleRunner.from_artifact_dir(_PACKAGED)
    assert runner.device == "cpu"
    assert runner.promotion_eligible is False
    assert runner.receipt_threshold == 0.05
