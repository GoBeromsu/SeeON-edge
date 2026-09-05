from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.edge_db.compatibility import CURRENT_SCHEMA_RANGE
from shared.detection_policies import default_policy_bundle, make_effective_policy
from tests_support.pose_bbox56_bundle_artifact import write_pose_bbox56_bundle
from worker.adapters.model.ort_pose_bbox56 import load_packaged_fall_bundle
from worker.domains import DETECTION_MODULE_REGISTRY
from worker.domains.module_definition import RuntimeResolvedArtifactDigest, SharedComponentIdentity
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.provenance.manifest import (
    AppliedCameraState,
    AppliedDetectionWindow,
    AppliedRuntimeManifestError,
    RuntimeEnvironmentFacts,
    build_applied_camera_state,
    build_applied_runtime_manifest,
)
from worker.runtime.provenance.models import AppliedBedZone
from worker.runtime.provenance.store import AppliedRuntimeManifestStore
from worker.tools.fetch_models.manifest import load_manifest

_BUILD_REVISION = "1" * 40


def _fall_preprocessing_identity() -> str:
    definition = DETECTION_MODULE_REGISTRY.get("fall", 2)
    [binding] = [
        binding
        for binding in definition.component_bindings
        if binding.component_id == "fall-classifier"
    ]
    assert binding.preprocessing_identity is not None
    return binding.preprocessing_identity


_FALL_ARTIFACT_DIGEST = "a" * 64
_FALL_PREPROCESSING_IDENTITY = _fall_preprocessing_identity()
_ARTIFACTS = {
    "pose": "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
    "person": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "bed": "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5",
    "fall-classifier": _FALL_ARTIFACT_DIGEST,
}
_PREPROCESSING = {
    "pose": "rgb24-to-coco17.v1",
    "person": "rgb24-to-person-boxes.v1",
    "bed": "rgb24-to-bed-regions.v1",
    "fall-classifier": _FALL_PREPROCESSING_IDENTITY,
}


def _facts(*, nvidia: bool = False) -> RuntimeEnvironmentFacts:
    return RuntimeEnvironmentFacts(
        worker_build_revision=_BUILD_REVISION,
        os_name="Linux",
        architecture="x86_64",
        python_version="3.12.11",
        model_runtime="torch",
        model_runtime_version="2.13.0",
        accelerator_runtime="CUDA 13.0" if nvidia else None,
        driver_version="580.65" if nvidia else None,
        device_name="NVIDIA RTX PRO 6000" if nvidia else None,
    )


def _boot(profile: str) -> BootContext:
    spec = PROFILE_REGISTRY[profile]
    return BootContext(
        profile=spec,
        device=spec.device,
        decode=spec.decode,
        encode=spec.encode,
        requested_profile=profile,
    )


def _identities(
    *, runtime: str = "onnxruntime", device: str = "cuda"
) -> tuple[SharedComponentIdentity, ...]:
    return tuple(
        SharedComponentIdentity(
            component_id=component_id,
            artifact_digest=_ARTIFACTS[component_id],
            runtime=runtime,
            device=device,
            preprocessing_identity=_PREPROCESSING[component_id],
        )
        for component_id in ("pose", "bed", "fall-classifier")
    )


def _cameras(*, threshold: float = 0.5) -> tuple[AppliedCameraState, ...]:
    bundle = default_policy_bundle(("camera:opaque/a", "camera:opaque/b"))
    fall = make_effective_policy(
        module_id="fall",
        module_version=2,
        values=replace(
            bundle.resolve("camera:opaque/a", "fall", 2).values,
            transition_threshold=threshold,
        ),
        source="facility-default",
        facility_revision_id=17,
        camera_revision_id=None,
    )
    cameras = []
    for camera_id in ("camera:opaque/a", "camera:opaque/b"):
        cameras.append(
            build_applied_camera_state(
                camera_id=camera_id,
                effective_decode_backend="opencv",
                ingest_target_fps=5.0,
                module_qualified_ids=("bed_exit.v1", "fall.v2"),
                schedule={"pose": 2, "bed": 30},
                detection_windows={"fall": None, "bed_exit": None},
                policies={
                    "fall": fall,
                    "bed_exit": bundle.resolve(camera_id, "bed_exit", 1),
                },
                bed_zone_polygon=None,
                bed_zone_image_width=None,
                bed_zone_image_height=None,
            )
        )
    return tuple(cameras)


def _persisted_bed_camera() -> AppliedCameraState:
    return build_applied_camera_state(
        camera_id="camera:opaque/a",
        effective_decode_backend="opencv",
        ingest_target_fps=5.0,
        module_qualified_ids=("bed_exit.v1", "fall.v2"),
        schedule={"pose": 2, "bed": 30},
        detection_windows={"fall": None, "bed_exit": None},
        policies=_cameras()[0].policies,
        bed_zone_polygon=((1, 2), (9, 2), (9, 8), (1, 8)),
        bed_zone_image_width=640,
        bed_zone_image_height=480,
    )


def _manifest(
    *,
    profile: str = "flow",
    identities: tuple[SharedComponentIdentity, ...] | None = None,
    cameras: tuple[AppliedCameraState, ...] | None = None,
    facts: RuntimeEnvironmentFacts | None = None,
):
    boot = _boot(profile)
    applied_cameras = cameras or tuple(
        replace(camera, effective_decode_backend=boot.decode) for camera in _cameras()
    )
    return build_applied_runtime_manifest(
        boot=boot,
        module_registry=DETECTION_MODULE_REGISTRY,
        module_versions={"fall": 2, "bed_exit": 1},
        component_identities=identities
        or _identities(
            runtime=boot.runtime_profile.effective_inference_backend,
            device=boot.device,
        ),
        cameras=applied_cameras,
        config_version=42,
        restart_generation=7,
        detector_version="worker-domain-detectors-v1",
        environment=facts or _facts(nvidia=True),
        edge_database_schema_version=5,
    )


def test_manifest_is_canonical_hashable_and_repeatable() -> None:
    first = _manifest()
    second = build_applied_runtime_manifest(
        boot=_boot("flow"),
        module_registry=DETECTION_MODULE_REGISTRY,
        module_versions={"bed_exit": 1, "fall": 2},
        component_identities=tuple(reversed(_identities())),
        cameras=tuple(
            reversed(
                tuple(
                    replace(camera, effective_decode_backend=_boot("flow").decode)
                    for camera in _cameras()
                )
            )
        ),
        config_version=42,
        restart_generation=7,
        detector_version="worker-domain-detectors-v1",
        environment=_facts(nvidia=True),
        edge_database_schema_version=5,
    )

    assert first.sha256 == second.sha256
    assert first.canonical_json == second.canonical_json
    assert (
        json.dumps(
            json.loads(first.canonical_json),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        == first.canonical_json
    )


def test_camera_projection_records_canonical_effective_applied_semantics() -> None:
    camera = build_applied_camera_state(
        camera_id="camera:opaque/a",
        effective_decode_backend="cpu",
        ingest_target_fps=7.5,
        module_qualified_ids=("fall.v2", "bed_exit.v1"),
        schedule={"pose": 3, "bed": 30},
        detection_windows={
            "fall": AppliedDetectionWindow(start="21:00", end="06:00", timezone="UTC"),
            "bed_exit": AppliedDetectionWindow(start="22:15", end="05:45", timezone="Asia/Seoul"),
        },
        policies=_cameras()[0].policies,
        bed_zone_polygon=((1, 2), (9, 2), (9, 8), (1, 8)),
        bed_zone_image_width=640,
        bed_zone_image_height=480,
    )
    content = json.loads(_manifest(cameras=(camera,)).canonical_json)["cameras"][0]

    assert content == {
        "camera_id": "camera:opaque/a",
        "effective_decode_backend": "cpu",
        "timing": {
            "ingest_target_fps": 7.5,
            "schedule_interval_basis": "ingested-frame-index",
        },
        "modules": ["bed_exit.v1", "fall.v2"],
        "schedule": {"bed": 30, "pose": 3},
        "detection_windows": {
            "bed_exit": {"end": "05:45", "start": "22:15", "timezone": "Asia/Seoul"},
            "fall": {"end": "06:00", "start": "21:00", "timezone": "UTC"},
        },
        "bed_zone": {
            "authority": "persisted-polygon",
            "coordinate_schema_version": 1,
            "coordinate_space": "source-image-pixels",
            "polygon": [[1, 2], [1, 8], [9, 8], [9, 2]],
            "source_dimensions": {"height": 480, "width": 640},
        },
        "policies": content["policies"],
    }


def test_absent_bed_zone_explicitly_records_live_segmentation_semantics() -> None:
    content = json.loads(_manifest().canonical_json)["cameras"][0]

    assert content["bed_zone"] == {
        "authority": "live-segmentation",
        "coordinate_schema_version": 1,
        "coordinate_space": None,
        "polygon": None,
        "source_dimensions": None,
    }


@pytest.mark.parametrize(
    "changed_camera",
    (
        replace(_cameras()[0], effective_decode_backend="cpu"),
        replace(_cameras()[0], ingest_target_fps=7.5),
        replace(_cameras()[0], schedule={"pose": 4, "bed": 30}),
        replace(
            _cameras()[0],
            detection_windows={
                "fall": AppliedDetectionWindow("21:00", "06:00", "UTC"),
                "bed_exit": None,
            },
        ),
        replace(
            _cameras()[0],
            detection_windows={
                "fall": AppliedDetectionWindow("21:00", "06:00", "Asia/Seoul"),
                "bed_exit": None,
            },
        ),
        _persisted_bed_camera(),
    ),
)
def test_each_effective_camera_semantic_changes_manifest_hash(
    changed_camera: AppliedCameraState,
) -> None:
    baseline = _manifest(cameras=(_cameras()[0],))
    changed = _manifest(cameras=(changed_camera,))

    assert changed.sha256 != baseline.sha256


@pytest.mark.parametrize(
    "changed_bed_zone",
    (
        replace(
            _persisted_bed_camera().bed_zone,
            polygon=((1, 2), (10, 2), (9, 8), (1, 8)),
        ),
        replace(_persisted_bed_camera().bed_zone, source_width=1280),
        replace(_persisted_bed_camera().bed_zone, source_height=960),
        replace(_persisted_bed_camera().bed_zone, coordinate_schema_version=2),
    ),
)
def test_each_persisted_bed_zone_identity_change_changes_manifest_hash(
    changed_bed_zone: AppliedBedZone,
) -> None:
    baseline = _persisted_bed_camera()
    changed = replace(baseline, bed_zone=changed_bed_zone)

    assert _manifest(cameras=(baseline,)).sha256 != _manifest(cameras=(changed,)).sha256


def test_equivalent_polygon_start_and_winding_order_do_not_change_hash() -> None:
    baseline = _persisted_bed_camera()
    rotated = build_applied_camera_state(
        camera_id=baseline.camera_id,
        effective_decode_backend=baseline.effective_decode_backend,
        ingest_target_fps=baseline.ingest_target_fps,
        module_qualified_ids=baseline.module_qualified_ids,
        schedule=baseline.schedule,
        detection_windows=baseline.detection_windows,
        policies=baseline.policies,
        bed_zone_polygon=((9, 8), (1, 8), (1, 2), (9, 2)),
        bed_zone_image_width=640,
        bed_zone_image_height=480,
    )
    reversed_winding = replace(
        rotated,
        bed_zone=replace(rotated.bed_zone, polygon=tuple(reversed(rotated.bed_zone.polygon or ()))),
    )
    closed_ring = replace(
        baseline,
        bed_zone=replace(
            baseline.bed_zone,
            polygon=(*tuple(baseline.bed_zone.polygon or ()), (1, 2)),
        ),
    )

    expected = _manifest(cameras=(baseline,)).sha256
    assert _manifest(cameras=(rotated,)).sha256 == expected
    assert _manifest(cameras=(reversed_winding,)).sha256 == expected
    assert _manifest(cameras=(closed_ring,)).sha256 == expected


def test_camera_semantic_mapping_and_selection_order_do_not_change_hash() -> None:
    camera = _cameras()[0]
    reordered = replace(
        camera,
        module_qualified_ids=tuple(reversed(camera.module_qualified_ids)),
        schedule=dict(reversed(tuple(camera.schedule.items()))),
        detection_windows=dict(reversed(tuple(camera.detection_windows.items()))),
        policies=dict(reversed(tuple(camera.policies.items()))),
    )

    assert _manifest(cameras=(camera,)).sha256 == _manifest(cameras=(reordered,)).sha256


def test_persisted_bed_zone_requires_source_dimensions() -> None:
    with pytest.raises(AppliedRuntimeManifestError, match="source dimensions"):
        build_applied_camera_state(
            camera_id="camera:opaque/a",
            effective_decode_backend="opencv",
            ingest_target_fps=5.0,
            module_qualified_ids=("bed_exit.v1", "fall.v2"),
            schedule={"pose": 2, "bed": 30},
            detection_windows={"fall": None, "bed_exit": None},
            policies=_cameras()[0].policies,
            bed_zone_polygon=((1, 2), (9, 2), (9, 8), (1, 8)),
            bed_zone_image_width=None,
            bed_zone_image_height=None,
        )


def test_policy_content_change_changes_only_applied_identity() -> None:
    baseline = _manifest()
    changed = _manifest(cameras=_cameras(threshold=0.72))

    assert baseline.sha256 != changed.sha256
    baseline_content = json.loads(baseline.canonical_json)
    changed_content = json.loads(changed.canonical_json)
    assert baseline_content["profile"] == changed_content["profile"]
    assert baseline_content["components"] == changed_content["components"]
    assert (
        baseline_content["cameras"][0]["policies"]["fall"]["effective_policy_id"]
        != changed_content["cameras"][0]["policies"]["fall"]["effective_policy_id"]
    )


def test_unresolved_or_contradictory_component_identity_refuses_manifest() -> None:
    with pytest.raises(AppliedRuntimeManifestError, match="unresolved applied identity.*bed"):
        _manifest(
            identities=tuple(
                identity
                for identity in _identities(
                    runtime="onnxruntime",
                    device="cuda",
                )
                if identity.component_id != "bed"
            )
        )

    contradictory = tuple(
        replace(identity, artifact_digest="f" * 64) if identity.component_id == "pose" else identity
        for identity in _identities(runtime="onnxruntime", device="cuda")
    )
    with pytest.raises(AppliedRuntimeManifestError, match="contradictory artifact identity.*pose"):
        _manifest(identities=contradictory)


def test_runtime_resolved_fall_identity_names_loaded_bundle_digest(tmp_path: Path) -> None:
    definition = DETECTION_MODULE_REGISTRY.get("fall", 2)
    binding = next(
        binding
        for binding in definition.component_bindings
        if binding.component_id == "fall-classifier"
    )
    assert isinstance(binding.artifact_digest, RuntimeResolvedArtifactDigest)

    bundle = load_packaged_fall_bundle(write_pose_bbox56_bundle(tmp_path / "fall-bundle"))
    loaded_digest = bundle.published_weights_digest
    shipped_digest = next(
        artifact.sha256
        for artifact in load_manifest().artifacts
        if artifact.path == "fall/pose-bbox56-gru/model.pt"
    )
    assert loaded_digest != shipped_digest
    manifest = _manifest(
        identities=tuple(
            replace(identity, artifact_digest=loaded_digest)
            if identity.component_id == "fall-classifier"
            else identity
            for identity in _identities(runtime="onnxruntime", device="cuda")
        )
    )

    components = json.loads(manifest.canonical_json)["components"]
    fall = next(
        component for component in components if component["component_id"] == "fall-classifier"
    )
    assert fall["artifact_sha256"] == loaded_digest


def test_secret_url_and_absolute_path_are_never_serialized() -> None:
    unsafe = replace(_facts(), device_name="rtsp://admin:secret@camera.local/live")
    with pytest.raises(AppliedRuntimeManifestError, match="unsafe provenance value"):
        _manifest(facts=unsafe)

    manifest = _manifest()
    assert "rtsp://" not in manifest.canonical_json
    assert "admin:secret" not in manifest.canonical_json
    assert "/var/lib/" not in manifest.canonical_json
    assert "facility_id" not in manifest.canonical_json
    assert "relay" not in manifest.canonical_json


def test_local_manifest_record_preserves_opaque_camera_identity_bytes(
    tmp_path: Path,
) -> None:
    camera_id = " \u2007camera/\u1100\u1161/e\u0301 "
    camera = replace(_cameras()[0], camera_id=camera_id)
    manifest = _manifest(cameras=(camera,))
    content = json.loads(manifest.canonical_json)

    serialized_id = content["cameras"][0]["camera_id"]
    assert serialized_id.encode("utf-8") == camera_id.encode("utf-8")

    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    store = AppliedRuntimeManifestStore(database)
    store.persist(
        manifest,
        boot_instance_id="boot:opaque-camera",
        applied_at="2026-08-13T00:00:00Z",
    )
    record = next((database.parent / "runtime-provenance").glob("*.json"))
    persisted_camera = json.loads(json.loads(record.read_text())["canonical_json"])["cameras"][0][
        "camera_id"
    ]
    assert persisted_camera.encode() == camera_id.encode()


def test_store_publishes_idempotent_manifest_envelopes_without_runtime_ddl(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    manifest = _manifest()
    store = AppliedRuntimeManifestStore(database)

    first = store.persist(manifest, boot_instance_id="boot:one", applied_at="2026-08-13T00:00:00Z")
    repeated = store.persist(
        manifest, boot_instance_id="boot:one", applied_at="2026-08-13T00:00:00Z"
    )
    second = store.persist(manifest, boot_instance_id="boot:two", applied_at="2026-08-13T00:01:00Z")

    assert (
        first.manifest_sha256
        == repeated.manifest_sha256
        == second.manifest_sha256
        == manifest.sha256
    )
    assert len(tuple((database.parent / "runtime-provenance").glob("*.json"))) == 2
    assert not (database.parent / "delivery-queue").exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (
            CURRENT_SCHEMA_RANGE.maximum,
        )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert "runtime_manifest_contents" not in tables
        readback = json.loads((database.parent / "applied-runtime-manifest.json").read_text())
        assert readback["manifest_sha256"] == manifest.sha256
        assert readback["canonical_json"] == manifest.canonical_json
        assert (database.parent / "applied-runtime-manifest.json").stat().st_mode & 0o777 == 0o600


def test_manifest_readback_contains_canonical_manifest_reference(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    manifest = _manifest()
    AppliedRuntimeManifestStore(database).persist(
        manifest,
        boot_instance_id="boot:event",
        applied_at="2026-08-13T00:00:00Z",
    )
    values = json.loads((database.parent / "applied-runtime-manifest.json").read_text())
    assert values["manifest_sha256"] == manifest.sha256
    assert values["canonical_json"] == manifest.canonical_json


def test_event_stager_rejects_noncanonical_runtime_manifest_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="runtime_manifest_sha256"):
        DurableEvidenceStager(
            queue_directory=tmp_path / "delivery-queue",
            camera_id="camera:opaque/a",
            facility_id="facility-1",
            resident_id=None,
            config_version=42,
            clock=lambda: 1.0,
            runtime_manifest_sha256="B" * 64,
        )


def test_store_refuses_boot_identity_reuse_for_different_manifest(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    store = AppliedRuntimeManifestStore(database)
    store.persist(_manifest(), boot_instance_id="boot:fixed", applied_at="2026-08-13T00:00:00Z")

    with pytest.raises(AppliedRuntimeManifestError, match="boot identity conflicts"):
        store.persist(
            _manifest(cameras=_cameras(threshold=0.72)),
            boot_instance_id="boot:fixed",
            applied_at="2026-08-13T00:01:00Z",
        )
