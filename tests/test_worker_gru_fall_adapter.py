from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.torch_gru_fall import GruFallRunner, build_gru_module
from worker.domains.fall.pose_bbox56 import (
    POSE_BBOX56_DIM,
    PoseBbox56Track,
    native_pose_bbox56_row,
    pose_bbox56_row,
    pose_bbox56_tracks,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path) -> Path:
    root.mkdir()
    arch = root / "arch.json"
    arch.write_text(json.dumps({"hidden": 4, "layers": 1, "dropout": 0.0}), encoding="utf-8")
    weights = root / "model.pt"
    torch.save(build_gru_module(hidden=4, layers=1, dropout=0.0).state_dict(), weights)
    files = {
        "input_contract": "input.json",
        "policy": "policy.json",
        "calibration": "calibration.json",
        "conformance": "conformance.json",
    }
    for name in files.values():
        (root / name).write_text("{}", encoding="utf-8")
    metadata = {
        "type": "gru",
        "framework": "pytorch",
        "mode": "sequence",
        "artifact_dir": ".",
        "weights": "model.pt",
        "architecture": "arch.json",
        "metadata": "metadata.yaml",
        **files,
        "input_shape": [30, 56],
        "class_order": ["background", "fall_transition", "fallen"],
        "schema_version": 1,
        "preprocessing_identity": "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        "artifact_digest": _digest(weights),
        "architecture_digest": _digest(arch),
        "input_digest": _digest(root / files["input_contract"]),
        "policy_digest": _digest(root / files["policy"]),
        "calibration_digest": _digest(root / files["calibration"]),
        "conformance_digest": _digest(root / files["conformance"]),
    }
    metadata_path = root / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    return root


def test_pose_bbox56_fixture_rules_and_native_parity() -> None:
    keypoints = tuple(
        (index * 10 + 20, index * 5 + 10, 0.5 if index == 0 else 0.9) for index in range(17)
    )
    row = pose_bbox56_row(keypoints, (20, 10, 180, 90), 200, 100)
    assert isinstance(row, tuple)
    assert len(row) == POSE_BBOX56_DIM
    assert all(value == np.float32(value).item() for value in row)
    np.testing.assert_allclose(
        np.asarray(row, dtype=np.float32)[:3], (0.1, 0.1, 0.5), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(row, dtype=np.float32)[-5:],
        (0.1, 0.1, 0.9, 0.9, 1.0),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        row, native_pose_bbox56_row(keypoints, (20, 10, 180, 90), 200, 100, box_source="pose")
    )
    assert all(
        value == 0.0
        for value in native_pose_bbox56_row(
            keypoints, (20, 10, 180, 90), 200, 100, box_source="person"
        )
    )
    assert all(value == 0.0 for value in pose_bbox56_row(keypoints, (20, 10, 20, 90), 200, 100))
    assert all(
        value == 0.0
        for value in pose_bbox56_row(
            (*keypoints[:-1], (float("nan"), 1, 0.9)), (20, 10, 180, 90), 200, 100
        )
    )


def test_pose_bbox56_tracks_are_sorted() -> None:
    pose = tuple((1, 1, 0.9) for _ in range(17))
    rows = pose_bbox56_tracks(
        (
            PoseBbox56Track("bravo", pose, (1, 1, 2, 2)),
            PoseBbox56Track("alpha", pose, (1, 1, 2, 2)),
        ),
        10,
        10,
    )
    assert [track_id for track_id, _row in rows] == ["alpha", "bravo"]


def test_gru_runner_verifies_bundle_warms_up_and_returns_softmax(tmp_path: Path) -> None:
    runner = GruFallRunner.from_artifact_dir(_bundle(tmp_path / "gru"))
    probabilities = runner.predict(np.zeros((30, 56), dtype=np.float32))
    assert len(probabilities) == 3
    assert all(np.isfinite(probabilities))
    assert sum(probabilities) == pytest.approx(1.0)
    with pytest.raises(ModelLoadError, match="input shape"):
        runner.predict(np.zeros((29, 56), dtype=np.float32))
    with pytest.raises(ModelLoadError, match="non-finite"):
        runner.predict(np.full((30, 56), np.nan, dtype=np.float32))


def test_gru_runner_rejects_manifest_extra_fields_and_digest_mismatch(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "gru")
    metadata_path = root / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["unexpected"] = "field"
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="unknown fields"):
        GruFallRunner.from_artifact_dir(root)
    del metadata["unexpected"]
    metadata["artifact_digest"] = "0" * 64
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="digest"):
        GruFallRunner.from_artifact_dir(root)
