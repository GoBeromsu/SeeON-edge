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

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import final

import pytest

from contracts.model_selection import (
    DatasetPublication,
    ModelPublication,
    ModelSelection,
    canonical_digest,
)
from contracts.runner import Image, RunnerResult
from shared.detection_policies import default_policy_bundle
from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model import ort_pose_bbox56
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.adapters.model.pose_bbox56_bundle import PoseBbox56BundleRunner
from worker.domains import DETECTION_MODULE_REGISTRY, CameraModuleContext
from worker.domains.registry import _audit_snapshot, _effective_transition_threshold
from worker.interfaces.fall_model import FallV2Probabilities
from worker.runtime import bootstrap
from worker.runtime.config import WorkerConfig, local_env
from worker.runtime.config.worker_models import SelectedFallBundleConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.provenance.manifest import (
    RuntimeEnvironmentFacts,
    build_applied_camera_state,
    build_applied_runtime_manifest,
)
from worker.runtime.provenance.model_bundle import DesiredModelBundle, admit_model_bundle
from worker.runtime.worker import WorkerRuntime, _validate_fall_bundle_conformance
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
class _ZeroLogitSession:
    def run(self, _output_names: object, _input_feed: object) -> list[object]:
        return [[[0.0]]]


def _rewrite_packaged_json_member(
    root: Path, relative_path: str, mutation: Callable[[dict[str, object]], None]
) -> None:
    path = root / relative_path
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    manifest_path = root / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["relative_path"] == relative_path)
    payload = path.read_bytes()
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    entry["size"] = len(payload)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


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


def _selected_onnx_bundle(
    tmp_path: Path,
    *,
    transition_threshold: float = 0.5,
    threshold_source: str = "default",
    calibration_grants: tuple[bool, float] | None = None,
    class_order: list[str] | None = None,
    temporal_rule: object = None,
) -> tuple[Path, DesiredModelBundle]:
    source = write_pose_bbox56_bundle(tmp_path / "source")
    # A selection that declares its threshold comes from the receipt must be
    # backed by the calibration: a real promoted publication states
    # promotion_eligible and the granted threshold there. Write that in, so
    # the fixture is what such a publication looks like.
    grants_receipt = threshold_source == "receipt"
    if calibration_grants is None and grants_receipt:
        calibration_grants = (True, transition_threshold)
    if class_order is not None or temporal_rule is not None or calibration_grants is not None:
        calibration_path = source / "calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_grants is not None:
            calibration["promotion_eligible"], calibration["threshold"] = calibration_grants
        if class_order is not None:
            calibration["class_order"] = class_order
        if temporal_rule is not None:
            if temporal_rule == "missing":
                del calibration["temporal_rule"]
            else:
                calibration["temporal_rule"] = temporal_rule
        calibration_path.write_text(
            json.dumps(calibration, sort_keys=True),
            encoding="utf-8",
        )
    members = {
        path: (source / path).read_bytes()
        for path in (
            "model.onnx",
            "calibration.json",
            "conformance/pose-bbox56-v1.json",
            "bundle-manifest.json",
        )
    }
    calibration_document = json.loads(members["calibration.json"])
    identities = {
        "dataset": "1" * 64,
        "calibration": hashlib.sha256(members["calibration.json"]).hexdigest(),
        "conformance": hashlib.sha256(members["conformance/pose-bbox56-v1.json"]).hexdigest(),
        "class": "4" * 64,
        "input": "pose-bbox56.v1",
        "policy": canonical_digest(calibration_document.get("temporal_rule")),
        "members": "6" * 64,
    }
    member_records = [
        {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        for path, content in members.items()
    ]
    payload = {"identities": identities}
    bundle_sha256 = hashlib.sha256(
        json.dumps(
            {"members": member_records, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evaluation = {
        "bundle_sha256": bundle_sha256,
        "bundle_members_digest": identities["members"],
        "dataset_payload_digest": identities["dataset"],
        "calibration_digest": identities["calibration"],
        "conformance_digest": identities["conformance"],
        "input_observation_schema": identities["input"],
        "output_class_count": 2,
        "output_class_semantics_digest": identities["class"],
        "policy_digest": identities["policy"],
    }
    field = {
        **evaluation,
        "evaluation_receipt_digest": canonical_digest(evaluation),
        "status": "green",
    }
    root = tmp_path / "models" / "bundles" / bundle_sha256
    root.mkdir(parents=True)
    for path, content in members.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content)
    receipts = []
    for path, document in (
        ("evaluation-receipt.json", evaluation),
        ("field-evaluation-receipt.json", field),
    ):
        content = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        (root / path).write_bytes(content)
        receipts.append(
            {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        )
    (root / "manifest.json").write_bytes(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_sha256": bundle_sha256,
                "runtime_format": "onnxruntime",
                "members": member_records,
                "receipts": receipts,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    selection = ModelSelection(
        model_publication=ModelPublication("facility/fall", "a" * 40, bundle_sha256),
        bundle_members_digest=identities["members"],
        dataset_publication=DatasetPublication("facility/dataset", "b" * 40, identities["dataset"]),
        evaluation_receipt_digest=canonical_digest(evaluation),
        field_evaluation_receipt_digest=canonical_digest(field),
        calibration_digest=identities["calibration"],
        conformance_digest=identities["conformance"],
        input_observation_schema=identities["input"],
        output_class_count=2,
        output_class_semantics_digest=identities["class"],
        policy_digest=identities["policy"],
        runtime_format="onnxruntime",
        bundle_format="bundle-manifest/proxy-v0",
        preprocessing_identity="coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        transition_threshold=transition_threshold,
        threshold_source=threshold_source,
    )
    return tmp_path / "models", DesiredModelBundle(
        bundle_sha256,
        {
            **identities,
            "evaluation": selection.evaluation_receipt_digest,
            "field": selection.field_evaluation_receipt_digest,
        },
        selection,
    )


def test_packaged_bundle_refuses_conformance_preprocessing_identity(
    tmp_path: Path,
) -> None:
    root = write_pose_bbox56_bundle(tmp_path)
    _rewrite_packaged_json_member(
        root,
        "conformance/pose-bbox56-v1.json",
        lambda document: document.update(preprocessing_identity="replacement-preprocessing-v2"),
    )

    with pytest.raises(
        ModelLoadError,
        match=(
            "bundle 'replacement-preprocessing-v2'.*"
            "runner 'coco17-xyc-plus-pose-head-xyxy-valid-f32-v1'"
        ),
    ):
        OrtPoseBbox56Runner.from_artifact_dir(
            root, session_factory=lambda *_args: _ZeroLogitSession()
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document["vector"].update(length=57), "vector length 57"),
        (
            lambda document: document["temporal"].update(window_frames=31),
            "temporal.window_frames 31",
        ),
    ],
)
def test_packaged_bundle_refuses_incompatible_conformance_shape(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    root = write_pose_bbox56_bundle(tmp_path)
    _rewrite_packaged_json_member(root, "conformance/pose-bbox56-v1.json", mutation)

    with pytest.raises(ModelLoadError, match=message):
        OrtPoseBbox56Runner.from_artifact_dir(
            root, session_factory=lambda *_args: _ZeroLogitSession()
        )


def test_packaged_bundle_refuses_calibration_for_other_preprocessing(
    tmp_path: Path,
) -> None:
    root = write_pose_bbox56_bundle(tmp_path)
    _rewrite_packaged_json_member(
        root,
        "calibration.json",
        lambda document: document.update(preprocessing_identity_digest="f" * 64),
    )

    with pytest.raises(
        ModelLoadError,
        match=(
            "declared 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'.*"
            "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1.*6ab6d816"
        ),
    ):
        OrtPoseBbox56Runner.from_artifact_dir(
            root, session_factory=lambda *_args: _ZeroLogitSession()
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda contract: replace(
                contract, keypoint_order=tuple(reversed(contract.keypoint_order))
            ),
            "keypoint_order",
        ),
        (
            lambda contract: replace(contract, confidence_gate=0.25),
            "confidence.gate",
        ),
        (
            lambda contract: replace(
                contract,
                coordinate_system={
                    **contract.coordinate_system,
                    "xy_normalization_denominators": {
                        "x": "frame_width_minus_one",
                        "y": "frame_height_minus_one",
                    },
                },
            ),
            "coordinate normalization",
        ),
        (
            lambda contract: replace(contract, stride_frames=1),
            "stride/fps",
        ),
        (
            lambda contract: replace(contract, fps=30.0),
            "stride/fps",
        ),
    ],
)
def test_runtime_refuses_conformance_that_differs_from_domain_contract(
    tmp_path: Path,
    mutation: Callable[[object], object],
    message: str,
) -> None:
    runner = OrtPoseBbox56Runner.from_artifact_dir(
        write_pose_bbox56_bundle(tmp_path),
        session_factory=lambda *_args: _ZeroLogitSession(),
    )

    with pytest.raises(ModelLoadError, match=message):
        _validate_fall_bundle_conformance(mutation(runner.conformance))  # type: ignore[arg-type]


def test_create_fall_model_refuses_when_unconfigured_and_never_touches_serving(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_config(), _ForbiddenServingClient(), tmp_path)

    with pytest.raises(RuntimeError, match="fall model must be explicitly configured"):
        runtime._create_fall_model()  # noqa: SLF001


def test_create_fall_model_uses_the_configured_bundle_artifact_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    calls: list[tuple[Path, str]] = []
    sentinel = _FakeRunner("fall")
    sentinel.conformance = OrtPoseBbox56Runner.from_artifact_dir(  # type: ignore[attr-defined]
        artifact_dir, session_factory=lambda *_args: _ZeroLogitSession()
    ).conformance

    def fake_load_packaged_bundle(artifact_dir: Path) -> object:
        calls.append((Path(artifact_dir), "cpu"))
        return type("_Bundle", (), {"runner": sentinel})()

    monkeypatch.setattr(ort_pose_bbox56, "load_packaged_fall_bundle", fake_load_packaged_bundle)
    runtime = _runtime(config, _ForbiddenServingClient(), tmp_path)

    model = runtime._create_fall_model()  # noqa: SLF001

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
        runtime._create_fall_model()  # noqa: SLF001

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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models_root, desired = _selected_onnx_bundle(tmp_path)
    selection = desired.selection
    assert selection is not None
    selected_config = SelectedFallBundleConfig(models_root=models_root, desired=desired)
    selection_path = tmp_path / "model-selection.json"
    selection_path.write_bytes(
        json.dumps(
            selection.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_env, "FALL_SELECTION_PATH", selection_path)
    monkeypatch.setattr(local_env, "FALL_MODELS_ROOT", models_root)
    models = local_env.worker_models_config_from_environment({})
    assert models.fall is None
    assert models.selected == selected_config
    config = _config().model_copy(update={"models": models})
    runtime = _runtime(config, _ForbiddenServingClient(), tmp_path)
    runtime._admit_selected_fall_bundle()  # noqa: SLF001
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


def test_selected_bundle_uses_the_admitted_onnx_member_without_model_pt(tmp_path: Path) -> None:
    models_root, desired = _selected_onnx_bundle(
        tmp_path, transition_threshold=0.31, threshold_source="receipt"
    )
    selection = desired.selection
    assert selection is not None
    proof = admit_model_bundle(models_root, desired)
    artifact_dir = models_root / "bundles" / desired.bundle_sha256

    runner = OrtPoseBbox56Runner.from_admitted_bundle(
        artifact_dir,
        proof,
        selection,
        session_factory=lambda _path, _providers: _ZeroLogitSession(),
    )

    assert not (artifact_dir / "model.pt").exists()
    assert (
        runner.artifact_digest
        == hashlib.sha256((artifact_dir / "model.onnx").read_bytes()).hexdigest()
    )
    assert (runner.receipt_threshold, runner.promotion_eligible) == (0.31, True)
    policy = default_policy_bundle(("camera-a",)).resolve("camera-a", "fall", 2)
    effective = _effective_transition_threshold(runner, policy)
    assert effective.transition_threshold == pytest.approx(0.31)
    assert (effective.transition_votes, effective.transition_window) == (5, 5)
    assert effective.confirmation_rule_source == "receipt"

    definition = DETECTION_MODULE_REGISTRY.get("fall", 2)
    context = CameraModuleContext(
        camera_id="camera-a",
        facility_id="facility-a",
        shared_components={"fall-classifier": runner},
        camera_components={"episode-identity": ("boot", "1", 0)},
        detection_window=None,
        clock=lambda: pytest.fail("fall clock should not be called"),
        diagnostics=None,
        policy=policy,
    )
    module = definition.create_camera_module(context)
    assert (
        module.decider.policy.policy.transition_votes,
        module.decider.policy.policy.transition_window,
    ) == (5, 5)
    assert definition.audit_adapter is not None
    audit = definition.audit_adapter(context)
    assert (audit.transition_votes, audit.transition_window) == (5, 5)
    assert audit.confirmation_rule_source == "receipt"


def test_non_promotable_bundle_keeps_default_confirmation_rule_and_records_declaration(
    tmp_path: Path,
) -> None:
    artifact_dir = write_pose_bbox56_bundle(tmp_path / "bundle")
    runner = OrtPoseBbox56Runner.from_artifact_dir(
        artifact_dir,
        session_factory=lambda _path, _providers: _ZeroLogitSession(),
    )
    policy = default_policy_bundle(("camera-a",)).resolve("camera-a", "fall", 2)

    effective = _effective_transition_threshold(runner, policy)

    assert (effective.transition_votes, effective.transition_window) == (3, 5)
    assert effective.confirmation_rule_source == "default"
    assert (effective.receipt_transition_votes, effective.receipt_transition_window) == (5, 5)
    assert (effective.unapplied_transition_votes, effective.unapplied_transition_window) == (5, 5)
    audit = _audit_snapshot(
        CameraModuleContext(
            camera_id="camera-a",
            facility_id="facility-a",
            shared_components={"fall-classifier": runner},
            camera_components={},
            detection_window=None,
            clock=lambda: pytest.fail("fall clock should not be called"),
            diagnostics=None,
            policy=policy,
        )
    )
    assert (audit.transition_votes, audit.transition_window) == (3, 5)
    assert audit.confirmation_rule_source == "default"
    assert (audit.unapplied_transition_votes, audit.unapplied_transition_window) == (5, 5)


@pytest.mark.parametrize(
    "temporal_rule",
    [
        pytest.param({"m": 6, "n": 5}, id="m-greater-than-n"),
        pytest.param({"m": 5.0, "n": 5}, id="non-integer"),
        pytest.param("missing", id="missing"),
    ],
)
def test_bundle_refuses_malformed_temporal_rule(tmp_path: Path, temporal_rule: object) -> None:
    models_root, desired = _selected_onnx_bundle(tmp_path, temporal_rule=temporal_rule)
    selection = desired.selection
    assert selection is not None
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(ModelLoadError, match="temporal_rule"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            models_root / "bundles" / desired.bundle_sha256,
            proof,
            selection,
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


def test_selected_receipt_without_threshold_refuses_construction(tmp_path: Path) -> None:
    models_root, desired = _selected_onnx_bundle(tmp_path, threshold_source="receipt")
    selection = desired.selection
    assert selection is not None
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(ModelLoadError, match="receipt threshold"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            models_root / "bundles" / desired.bundle_sha256,
            proof,
            replace(selection, transition_threshold=None),  # type: ignore[arg-type]
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


def test_selected_preprocessing_contradiction_refuses_construction(tmp_path: Path) -> None:
    models_root, desired = _selected_onnx_bundle(tmp_path)
    selection = desired.selection
    assert selection is not None
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(
        ModelLoadError, match="selected preprocessing_identity differs from bundle conformance"
    ):
        OrtPoseBbox56Runner.from_admitted_bundle(
            models_root / "bundles" / desired.bundle_sha256,
            proof,
            replace(selection, preprocessing_identity="contradictory-preprocessing"),
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


def test_selected_default_source_with_a_non_default_threshold_refuses(tmp_path: Path) -> None:
    """A selection that says 'default' must declare the default.

    Otherwise the policy honours the source word and runs at 0.5 while the
    declared number the owner read is silently discarded. The contradiction
    refuses at construction, naming both numbers.
    """
    models_root, desired = _selected_onnx_bundle(tmp_path)
    selection = desired.selection
    assert selection is not None
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(ModelLoadError, match="threshold_source is 'default'"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            models_root / "bundles" / desired.bundle_sha256,
            proof,
            replace(selection, threshold_source="default", transition_threshold=0.3),
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


def test_selected_receipt_claim_refuses_when_the_calibration_grants_none(tmp_path: Path) -> None:
    """The word 'receipt' in a deployment document cannot grant what the
    publisher did not: a non-promotable calibration refuses the claim."""
    models_root, desired = _selected_onnx_bundle(
        tmp_path, threshold_source="receipt", calibration_grants=(False, 0.5)
    )
    selection = desired.selection
    assert selection is not None
    bundle_dir = models_root / "bundles" / desired.bundle_sha256
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(ModelLoadError, match="not promotion-eligible"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            bundle_dir,
            proof,
            selection,
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


def test_selected_receipt_claim_refuses_when_the_granted_threshold_differs(
    tmp_path: Path,
) -> None:
    """The receipt is the calibration, not the selection: a declared receipt
    threshold the calibration did not grant refuses naming both."""
    models_root, desired = _selected_onnx_bundle(
        tmp_path,
        threshold_source="receipt",
        transition_threshold=0.3,
        calibration_grants=(True, 0.05),
    )
    selection = desired.selection
    assert selection is not None
    bundle_dir = models_root / "bundles" / desired.bundle_sha256
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(ModelLoadError, match=r"declares receipt threshold 0\.3 .*grants 0\.05"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            bundle_dir,
            proof,
            selection,
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


def test_selected_output_contract_mismatch_refuses_construction(tmp_path: Path) -> None:
    """A selection declares its output class count; this runner implements one.

    A replacement that emits a different class count is a different structure,
    and must refuse here rather than have its logit read as if it were this
    runner's single fall-transition score - a silent wrong answer.
    """
    models_root, desired = _selected_onnx_bundle(tmp_path)
    selection = desired.selection
    assert selection is not None
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(ModelLoadError, match="output_class_count=3"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            models_root / "bundles" / desired.bundle_sha256,
            proof,
            replace(selection, output_class_count=3),
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


def test_selected_bundle_refuses_reversed_calibration_class_order(tmp_path: Path) -> None:
    observed_order = ["fall_transition_proxy", "non_fall"]
    models_root, desired = _selected_onnx_bundle(tmp_path, class_order=observed_order)
    selection = desired.selection
    assert selection is not None
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(ModelLoadError, match=r"\['fall_transition_proxy', 'non_fall'\]"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            models_root / "bundles" / desired.bundle_sha256,
            proof,
            selection,
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


def test_packaged_bundle_refuses_reversed_calibration_class_order(tmp_path: Path) -> None:
    observed_order = ["fall_transition_proxy", "non_fall"]
    artifact_dir = write_pose_bbox56_bundle(tmp_path / "bundle")
    calibration_path = artifact_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["class_order"] = observed_order
    calibration_path.write_text(json.dumps(calibration, sort_keys=True), encoding="utf-8")
    manifest_path = artifact_dir / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration_member = next(
        member for member in manifest["files"] if member["relative_path"] == "calibration.json"
    )
    calibration_member["sha256"] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    calibration_member["size"] = calibration_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    with pytest.raises(ModelLoadError, match=r"\['fall_transition_proxy', 'non_fall'\]"):
        ort_pose_bbox56.load_packaged_fall_bundle(artifact_dir)


def test_selected_bundle_refuses_calibration_class_order_with_wrong_count(tmp_path: Path) -> None:
    models_root, desired = _selected_onnx_bundle(tmp_path, class_order=["non_fall"])
    selection = desired.selection
    assert selection is not None
    proof = admit_model_bundle(models_root, desired)

    with pytest.raises(ModelLoadError, match="must contain exactly 2 entries"):
        OrtPoseBbox56Runner.from_admitted_bundle(
            models_root / "bundles" / desired.bundle_sha256,
            proof,
            selection,
            session_factory=lambda _path, _providers: _ZeroLogitSession(),
        )


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
