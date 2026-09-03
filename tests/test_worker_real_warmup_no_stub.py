"""Acceptance criterion 1: real_warmup passes with no stub injection.

The real-stack e2e (``tests/test_e2e_night_bed_exit_relay.py``) deliberately
injects ``ScriptedServingClient`` so it can drive deterministic detections, and
that client's ``warmup()`` is a no-op. That makes the e2e authoritative for
decode, relay, and liveness -- but *not* for warmup.

This module closes that gap without faking it. It provisions models through the
production path -- ``InProcessServingClient`` over ``default_registry()`` -- and
runs each adapter's real ``warmup()``, which performs one genuine forward on a
synthetic frame of the configured shape. Nothing here is mocked: no injected
``model=``, no patched loader, no fake registry.

The tests skip only when a required weight file is genuinely absent, so a
machine without artifacts reports "skipped", never a false pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner
from worker.runtime.config import WorkerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

# Task -> the weight file its production default points at.
_TASK_ARTIFACTS: dict[str, Path] = {
    "pose": REPO_ROOT / "models" / "pose" / "yolo26n-pose.pt",
    "person": REPO_ROOT / "models" / "person" / "yolo26n.pt",
    "bed": REPO_ROOT / "models" / "bed" / "yolo26m-seg.pt",
}


def _require(task: str) -> Path:
    artifact = _TASK_ARTIFACTS[task]
    if not artifact.is_file():
        pytest.skip(f"{task} weights are not present at {artifact}")
    return artifact


_REAL_WARMUP_TIMEOUT_SECONDS = 60.0
_REAL_WARMUP_COMPLETED = "REAL_WARMUP_COMPLETED"
_REAL_WARMUP_SCRIPT = f"""
import sys

from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import default_registry

TASK = sys.argv[1]
serving = InProcessServingClient(registry=default_registry())
adapter = serving.create(TASK, device="cpu")
adapter.warmup()
print("{_REAL_WARMUP_COMPLETED}:" + TASK, flush=True)
"""


@pytest.mark.heavy
@pytest.mark.parametrize("task", sorted(_TASK_ARTIFACTS))
def test_real_warmup_runs_a_genuine_forward_through_the_production_serving_path(
    task: str,
) -> None:
    """Provision through the real serving client and run the real warmup.

    A stubbed warmup cannot fail this: a fresh interpreter loads the actual
    artifact through ``default_registry()`` and executes a real CPU forward.
    Process completion plus the sentinel is the deterministic completion
    signal. The 60-second bound gives the measured 13-25 second cold pose
    warmup finite headroom without making this local-artifact test part of the
    default hardware-free CI suite.
    """
    _ = _require(task)

    completed = subprocess.run(
        [sys.executable, "-c", _REAL_WARMUP_SCRIPT, task],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=_REAL_WARMUP_TIMEOUT_SECONDS,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"{_REAL_WARMUP_COMPLETED}:{task}" in completed.stdout


def _example_fall_config() -> dict[str, object]:
    example = yaml.safe_load(
        (REPO_ROOT / "worker" / "ml-worker.example.yaml").read_text(encoding="utf-8")
    )
    return dict(example["models"]["fall"])


def test_example_config_fall_contract_matches_the_packaged_bundle() -> None:
    """The example config documents the packaged pose+bbox56 bundle contract:
    56-wide 30x5 windows, schema 2, the pose+bbox56 preprocessing identity, and
    the owner-fixed 0.5 transition threshold. It must validate as a
    ``FallModelConfig`` and, when the bundle is provisioned locally, boot the
    real CPU runner through the production loader."""
    fall_cfg = _example_fall_config()
    assert fall_cfg["type"] == "pose-bbox56-proxy-v0"
    assert fall_cfg["input_shape"] == [30, 56]
    assert (fall_cfg["window"], fall_cfg["stride"]) == (30, 5)
    assert fall_cfg["operating_threshold"] == 0.5
    assert fall_cfg["schema_version"] == 2
    assert fall_cfg["preprocessing_identity"] == "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"

    artifact_dir = REPO_ROOT / "models" / "fall" / "pose-bbox56-gru"
    if not (artifact_dir / "bundle-manifest.json").is_file():
        pytest.skip(f"packaged fall bundle is not present at {artifact_dir}")
    runner = PoseBbox56BundleRunner.from_artifact_dir(artifact_dir, device="cpu")
    assert runner.device == "cpu"


def test_example_config_fall_contract_boots_against_a_synthesized_bundle(
    tmp_path: Path,
) -> None:
    """CI-runnable companion: ``models/`` is gitignored, so synthesize a
    verifiable bundle in-process, point the example config's fall block at it,
    validate the full ``WorkerConfig``, and boot the real runner with a real
    warmup forward -- no mocks, no patched loader."""
    fall_cfg = _example_fall_config()
    fall_cfg["artifact_dir"] = str(write_pose_bbox56_bundle(tmp_path / "pose-bbox56-gru"))
    config = WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [],
            "models": {"fall": fall_cfg},
        }
    )
    assert config.models.fall is not None
    assert config.models.fall.input_shape == (30, 56)

    runner = PoseBbox56BundleRunner.from_artifact_dir(config.models.fall.artifact_dir)
    runner.warmup()
    assert runner.device == "cpu"
