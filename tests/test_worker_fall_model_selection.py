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
import shutil
from pathlib import Path
from typing import final

import pytest

from contracts.model_selection import DatasetPublication, ModelPublication, ModelSelection
from contracts.runner import Image, RunnerResult
from shared.detection_policies import default_policy_bundle
from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model import ort_pose_bbox56
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner
from worker.interfaces.fall_model import FallV2Probabilities
from worker.runtime import bootstrap
from worker.runtime.config import WorkerConfig
from worker.runtime.config.worker_models import SelectedFallBundleConfig, WorkerModelsConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.provenance.manifest import (
    RuntimeEnvironmentFacts,
    build_applied_camera_state,
    build_applied_runtime_manifest,
)
from worker.runtime.provenance.model_bundle import DesiredModelBundle
from worker.runtime.worker import WorkerRuntime
from worker.tools.export_fall_onnx import export_fall_onnx
from worker.tools.fetch_models.fetcher import VerificationError, _require_loadable_fall_bundle


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


def _flow_boot() -> BootContext:
    profile = PROFILE_REGISTRY["flow"]
    return BootContext(
        profile=profile,
        device=profile.device,
        decode=profile.decode,
        encode=profile.encode,
        requested_profile="flow",
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


def test_boot_and_fetch_share_packaged_bundle_load_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_dir = tmp_path / "models" / "fall" / "pose-bbox56-gru"
    write_pose_bbox56_bundle(artifact_dir)
    config = _config(
        with_fall={
            "type": "pose-bbox56-proxy-v0",
            "framework": "onnxruntime",
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

    def raise_sentinel(_artifact_dir: Path) -> object:
        raise ModelLoadError("shared bundle sentinel")

    monkeypatch.setattr(ort_pose_bbox56, "load_packaged_fall_bundle", raise_sentinel)
    runtime = _runtime(config, _ForbiddenServingClient(), tmp_path)

    with pytest.raises(ModelLoadError, match="shared bundle sentinel"):
        runtime._create_fall_model("cpu", require_onnxruntime=True)  # noqa: SLF001

    fetch_bundle = tmp_path / "fetch" / "fall" / "pose-bbox56-gru"
    fetch_bundle.mkdir(parents=True)
    (fetch_bundle / "model.onnx").write_bytes(b"present")
    with pytest.raises(VerificationError, match="shared bundle sentinel"):
        _require_loadable_fall_bundle(tmp_path / "fetch")


def test_flow_composition_uses_the_loaded_bundle_published_weights_digest(tmp_path: Path) -> None:
    artifact_dir = write_pose_bbox56_bundle(tmp_path / "models" / "fall" / "pose-bbox56-gru")
    config = _config(
        with_fall={
            "type": "pose-bbox56-proxy-v0",
            "framework": "onnxruntime",
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
    runtime = _runtime(config, _ForbiddenServingClient(), tmp_path)

    graph = runtime._initialize_flow_policy_graph(_flow_boot())  # noqa: SLF001
    fall = next(
        identity for identity in graph.identities if identity.component_id == "fall-classifier"
    )

    assert fall.artifact_digest == runtime._packaged_fall_member_digest()  # noqa: SLF001
    # The swap proof: this synthetic bundle's digest is NOT the shipped model's,
    # and the manifest names it anyway, with no identity refusal and no code
    # change. Changing the model alone changes what the receipts name.
    shipped = "7bb75a2932e1a1250dc900013b2c80b220de5e23f3ea568e05f1db21d0a757e3"
    assert fall.artifact_digest != shipped
    assert len(fall.artifact_digest) == 64


def test_selected_bundle_composes_a_runtime_manifest_with_runner_preprocessing_identity(
    tmp_path: Path,
) -> None:
    bundle_sha256 = "a" * 64
    artifact_dir = write_pose_bbox56_bundle(tmp_path / "source-bundle")
    models_root = tmp_path / "models"
    selected_dir = models_root / "bundles" / bundle_sha256
    selected_dir.parent.mkdir(parents=True)
    shutil.copytree(artifact_dir, selected_dir)
    selection = ModelSelection(
        model_publication=ModelPublication("facility/fall", "1" * 40, bundle_sha256),
        bundle_members_digest="2" * 64,
        dataset_publication=DatasetPublication("facility/dataset", "3" * 40, "4" * 64),
        evaluation_receipt_digest="5" * 64,
        field_evaluation_receipt_digest="6" * 64,
        calibration_digest="7" * 64,
        conformance_digest="8" * 64,
        input_observation_schema="pose-bbox56.v1",
        output_class_count=2,
        output_class_semantics_digest="9" * 64,
        policy_digest="b" * 64,
        runtime_format="onnxruntime",
        bundle_format="bundle-manifest/proxy-v0",
        preprocessing_identity="coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        transition_threshold=0.5,
        threshold_source="default",
    )
    desired = DesiredModelBundle(
        bundle_sha256,
        {
            "dataset": selection.dataset_publication.payload_digest,
            "evaluation": selection.evaluation_receipt_digest,
            "field": selection.field_evaluation_receipt_digest,
            "calibration": selection.calibration_digest,
            "conformance": selection.conformance_digest,
            "class": selection.output_class_semantics_digest,
            "input": selection.input_observation_schema,
            "policy": selection.policy_digest,
            "members": selection.bundle_members_digest,
        },
        selection,
    )
    config = _config().model_copy(
        update={
            "models": WorkerModelsConfig(
                selected=SelectedFallBundleConfig(models_root=models_root, desired=desired)
            )
        }
    )
    runtime = _runtime(config, _ForbiddenServingClient(), tmp_path)
    boot = _flow_boot()
    graph = runtime._initialize_flow_policy_graph(boot)  # noqa: SLF001
    [fall] = [
        identity for identity in graph.identities if identity.component_id == "fall-classifier"
    ]

    assert fall.preprocessing_identity == selection.preprocessing_identity
    assert fall.preprocessing_identity != selection.input_observation_schema

    policy_bundle = default_policy_bundle(("camera-a",))
    camera = build_applied_camera_state(
        camera_id="camera-a",
        effective_decode_backend=boot.decode,
        ingest_target_fps=5.0,
        module_qualified_ids=("bed_exit.v1", "fall.v2"),
        schedule={"pose": 2, "bed": 30},
        detection_windows={"fall": None, "bed_exit": None},
        policies={
            "fall": policy_bundle.resolve("camera-a", "fall", 2),
            "bed_exit": policy_bundle.resolve("camera-a", "bed_exit", 1),
        },
        bed_zone_polygon=None,
        bed_zone_image_width=None,
        bed_zone_image_height=None,
    )
    manifest = build_applied_runtime_manifest(
        boot=boot,
        module_registry=runtime._module_registry,  # noqa: SLF001
        module_versions=runtime._module_versions,  # noqa: SLF001
        component_identities=graph.identities,
        cameras=(camera,),
        config_version=config.version,
        restart_generation=0,
        detector_version="worker-domain-detectors-v1",
        environment=RuntimeEnvironmentFacts(
            worker_build_revision="c" * 40,
            os_name="Linux",
            architecture="x86_64",
            python_version="3.12.11",
            model_runtime="onnxruntime",
            model_runtime_version="1.20.0",
            accelerator_runtime="CUDA 13.0",
            driver_version="580.65",
            device_name="NVIDIA RTX",
        ),
        edge_database_schema_version=5,
    )

    components = json.loads(manifest.canonical_json)["components"]
    [manifest_fall] = [
        component for component in components if component["component_id"] == "fall-classifier"
    ]
    assert manifest_fall["preprocessing_identity"] == selection.preprocessing_identity


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
