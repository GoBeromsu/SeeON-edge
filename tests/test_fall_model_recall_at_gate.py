from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import polars as pl
import pytest


@pytest.fixture(scope="module")
def recall_script() -> ModuleType:
    path = Path("scripts/qa/fall_model_recall_at_gate.py")
    spec = importlib.util.spec_from_file_location("fall_model_recall_at_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path, script: ModuleType) -> tuple[Path, str]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    positive_pose = [[[0.1, 0.1, 0.8]] * 17] * 300
    negative_pose = [[[0.1, 0.1, 0.2]] * 17] * 300
    boxes = [[0.0, 0.0, 1.0, 1.0, 1.0]] * 300
    clips = pl.DataFrame(
        {
            "width": [100, 100, 100],
            "height": [100, 100, 100],
            "split_membership": [
                {"split_role": "selection_validation"},
                {"split_role": "selection_validation"},
                {"split_role": "train"},
            ],
            "labels": [
                {"source_proxy_interval_15fps": {"start": 0, "end": 30}},
                {"source_proxy_interval_15fps": None},
                {"source_proxy_interval_15fps": {"start": 0, "end": 30}},
            ],
            "pose": [positive_pose, negative_pose, positive_pose],
            "pose_head_bbox": [boxes, boxes, boxes],
        }
    )
    clips.write_parquet(dataset / "clips.parquet")
    manifest = {
        "files": [
            {
                "relative_path": "clips.parquet",
                "size": (dataset / "clips.parquet").stat().st_size,
                "sha256": _sha256(dataset / "clips.parquet"),
            }
        ]
    }
    (dataset / "payload-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (dataset / "checksums.sha256").write_text(
        f"{manifest['files'][0]['sha256']}  clips.parquet\n", encoding="utf-8"
    )
    return dataset, script._canonical_json_digest(manifest)


def _bundle(tmp_path: Path, digest: str) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("model.onnx", "bundle-manifest.json"):
        (bundle / name).write_bytes(b"model")
    (bundle / "calibration.json").write_text('{"temperature": 1.0}', encoding="utf-8")
    receipt = {
        "dataset_publication": {
            "hf_repo": "example/dataset",
            "payload_revision": "revision",
            "dataset_payload_digest": digest,
        },
        "champion_seed": 1,
        "per_seed": {
            "gru": [
                {
                    "seed": 1,
                    "calibration": {
                        "selection_metrics": {
                            "confusion": {"true_positive_windows": 4, "false_negative_windows": 0},
                            "recall": 1.0,
                            "precision": 0.5,
                        }
                    },
                }
            ]
        },
    }
    (bundle / "evaluation-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return bundle


class _FakeRunner:
    receipt_threshold = 0.1

    def predict(self, features: object) -> SimpleNamespace:
        return SimpleNamespace(fall_transition=float(features[0, 2]))


def test_digest_verification_refuses_tampered_manifest(
    tmp_path: Path, recall_script: ModuleType
) -> None:
    dataset, digest = _snapshot(tmp_path, recall_script)
    (dataset / "payload-manifest.json").write_text('{"files": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest digest mismatch"):
        recall_script.verify_dataset_payload(dataset, digest)


def test_scores_selection_validation_and_writes_receipt(
    tmp_path: Path, recall_script: ModuleType
) -> None:
    dataset, digest = _snapshot(tmp_path, recall_script)
    bundle = _bundle(tmp_path, digest)

    receipt = recall_script.score_bundle(
        bundle, dataset, 0.5, runner_factory=lambda _: _FakeRunner()
    )
    out = tmp_path / "receipt.json"
    recall_script.write_receipt(out, receipt)

    assert receipt["dataset"]["split"]["clip_count"] == 2
    assert receipt["metrics"]["total_windows_scored"] == 110
    assert receipt["metrics"]["positive_window_count"] == 4
    assert receipt["metrics"]["threshold_0_5"]["recall"] == 1.0
    assert receipt["metrics"]["threshold_0_5"]["precision"] == pytest.approx(4 / 55)
    assert receipt["metrics"]["threshold_0_1"]["recall"] == 1.0
    assert receipt["metrics"]["threshold_0_1"]["precision"] == pytest.approx(4 / 55)
    assert receipt["metrics"]["maximum_fall_transition_score_on_positive_window"] == pytest.approx(
        0.8
    )
    persisted = json.loads(out.read_text(encoding="utf-8"))
    assert persisted["status"] == "measured"
    assert persisted["metrics"] == receipt["metrics"]
    # A model whose positive windows clear the gate must be told to proceed,
    # not to replace itself - the script once said "replace the model first"
    # unconditionally, which would have sent the owner in a circle.
    assert "proceed to a staged fall" in receipt["owner_instruction"]
    assert "replace the model" not in receipt["owner_instruction"]


def test_family_comes_from_the_receipt_and_a_failing_model_is_told_to_be_replaced(
    tmp_path: Path, recall_script: ModuleType
) -> None:
    """A replacement model's receipt carries its own family key; the script must
    not assume 'gru'. And a model whose positive windows never clear the gate is
    told to be replaced - the instruction follows the number in both directions.
    """
    script = recall_script
    dataset, digest = _snapshot(tmp_path, script)
    bundle = _bundle(tmp_path, digest)
    receipt_path = bundle / "evaluation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # A replacement's receipt names its own champion family, alongside the
    # comparators it was measured against - exactly the shipped receipt's shape.
    champion = receipt["per_seed"].pop("gru")
    receipt["per_seed"] = {"transformer": champion, "gru": champion, "rf45": champion}
    receipt["comparators"] = ["gru", "rf45"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    class _WeakRunner:
        receipt_threshold = 0.1

        def predict(self, features: object) -> SimpleNamespace:
            return SimpleNamespace(fall_transition=0.2)

    result = script.score_bundle(bundle, dataset, 0.5, runner_factory=lambda _: _WeakRunner())

    assert result["metrics"]["threshold_0_5"]["recall"] == 0.0
    assert "replace the model first" in result["owner_instruction"]
    assert "proceed" not in result["owner_instruction"]
