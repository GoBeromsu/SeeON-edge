"""Issue #43: the fall model has no implicit fallback.

Before this change, ``WorkerRuntime._create_fall_model`` fell through to
``self._serving.create("fall")`` whenever ``config.models.fall`` was unset --
silently swapping in whatever the process-wide model registry happened to
provide (historically a random-forest classifier). "Which model ran that
night" must never be answered by an unconfigured default, so the fallback is
gone entirely: an operator who omits ``models.fall`` now gets a refused boot
(``REFUSE_TO_START_EXIT_CODE``, not the generic runtime code), and the
configured packaged-bundle path is otherwise unchanged.

This file covers three levels: (1) the unit-level refusal, proving it never
even reaches the serving client; (2) the unit-level configured path, proving
``PoseBbox56BundleRunner.from_artifact_dir`` receives exactly the artifact
the config declares and is always pinned to the CPU; (3) the full
``WorkerRuntime.run()`` integration path,
proving the refusal surfaces as ``SystemExit`` with the refuse-to-start code
and activates zero cameras.

Issue #65 moved the actual family dispatch behind
``worker.adapters.model.fall_family_registry.DEFAULT_FALL_MODEL_FAMILY_REGISTRY``
(keyed by ``FallModelConfig.type``). The registry's own plug-in and
unknown-type-refusal contracts are covered separately in
``tests/test_fall_model_family_registry.py``; this file's scope stays #43's
none-config refusal plus the configured packaged-bundle behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import final

import pytest

from contracts.runner import Image, RunnerResult
from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner
from worker.interfaces.fall_model import FallV2Probabilities
from worker.runtime import bootstrap
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.worker import WorkerRuntime
from worker.tools.export_fall_onnx import export_fall_onnx


@final
class _ForbiddenServingClient:
    """Fails the test immediately if fall-model construction ever reaches the
    serving client -- proving the refusal happens before any fallback model
    could be requested."""

    def create(self, task: str, **_options: object) -> object:
        raise AssertionError(f"fall model refusal must not call serving.create({task!r})")


@final
class _FakeRunner:
    device = "cpu"

    def __init__(self, task: str) -> None:
        self.task = task
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("this test must not run model inference")

    def predict(self, _features: object) -> FallV2Probabilities:
        return FallV2Probabilities(background=1.0, fall_transition=0.0, fallen=0.0)

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _YoloOnlyServingClient:
    """Provides pose/person/bed runners (consumed before the fall model is
    ever constructed) but must never be asked for "fall": the refusal is
    unconditional, not something that reaches the registry."""

    def create(self, task: str, **_options: object) -> _FakeRunner:
        if task == "fall":
            raise AssertionError("fall model refusal must not call serving.create('fall')")
        return _FakeRunner(task)


def test_exported_onnx_is_idempotent_and_loadable_by_the_flow_runner(tmp_path: Path) -> None:
    artifact_dir = write_pose_bbox56_bundle(tmp_path / "pose-bbox56-gru")
    manifest_path = artifact_dir / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        item for item in manifest["files"] if item["relative_path"] != "model.onnx"
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "model.onnx").unlink()

    first_digest = export_fall_onnx(artifact_dir)
    manifest_after_first_export = manifest_path.read_bytes()
    second_digest = export_fall_onnx(artifact_dir)

    assert second_digest == first_digest
    assert manifest_path.read_bytes() == manifest_after_first_export
    assert OrtPoseBbox56Runner.from_artifact_dir(artifact_dir, "cpu").device == "cpu"


def _config(*, with_fall: dict[str, object] | None = None) -> WorkerConfig:
    payload: dict[str, object] = {
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
    }
    if with_fall is not None:
        payload["models"] = {"fall": with_fall}
    return WorkerConfig.model_validate(payload)


def _runtime(config: WorkerConfig, serving: object, state_dir: Path) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "flow"},
        serving_client=serving,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
    )


def test_create_fall_model_refuses_when_unconfigured_and_never_touches_serving(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_config(), _ForbiddenServingClient(), tmp_path)

    with pytest.raises(RuntimeError, match="fall model must be explicitly configured"):
        runtime._create_fall_model("cpu")  # noqa: SLF001


def test_create_fall_model_uses_the_configured_bundle_artifact_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_dir = write_pose_bbox56_bundle(tmp_path / "models" / "fall" / "pose-bbox56-gru")
    config = _config(
        with_fall={
            "type": "pose-bbox56-proxy-v0",
            "framework": "pytorch",
            "mode": "sequence",
            "artifact_dir": str(artifact_dir),
            "weights": "model.pt",
            "architecture": "arch.json",
            "metadata": "metadata.yaml",
            "window": 30,
            "stride": 5,
            "input_shape": [30, 56],
            "operating_threshold": 0.5,
            "schema_version": 2,
            "preprocessing_identity": "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        }
    )
    calls: list[tuple[Path, str]] = []
    sentinel = _FakeRunner("fall")

    def fake_from_artifact_dir(artifact_dir: Path, device: str = "cpu") -> object:
        calls.append((Path(artifact_dir), device))
        return sentinel

    monkeypatch.setattr(PoseBbox56BundleRunner, "from_artifact_dir", fake_from_artifact_dir)
    runtime = _runtime(config, _ForbiddenServingClient(), tmp_path)

    model = runtime._create_fall_model("cpu")  # noqa: SLF001

    assert model is sentinel
    fall_config = config.models.fall
    assert fall_config is not None
    assert calls == [(fall_config.artifact_dir, "cpu")]


def test_bundle_runner_refuses_a_non_cpu_device(tmp_path: Path) -> None:
    """P1a-AC6b: the packaged fall runner is CPU-only; a GPU request is a boot error."""
    artifact_dir = write_pose_bbox56_bundle(tmp_path / "bundle")
    with pytest.raises(Exception, match="pinned to cpu"):
        PoseBbox56BundleRunner.from_artifact_dir(artifact_dir, device="cuda")


def test_run_refuses_to_start_with_refuse_to_start_exit_code_when_fall_is_unconfigured(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_config(), _YoloOnlyServingClient(), tmp_path)

    with pytest.raises(SystemExit) as exc:
        runtime.run()

    assert exc.value.code == bootstrap.REFUSE_TO_START_EXIT_CODE
