from __future__ import annotations

from worker.pipeline.output.overlay_scene import AppliedCameraProvenance, OverlaySceneBuilder
from worker.pipeline.trace.models import (
    AnalysisTrace,
    DecisionTrace,
    OptionalNumber,
    TraceBed,
    TraceComponent,
    TraceKeypoint,
    TracePerson,
)
from worker.types import DecisionTraceSnapshot
from worker.types.overlay_scene import ObservationSemantics, fit_scene_transform


def _analysis() -> AnalysisTrace:
    return AnalysisTrace(
        trace_id="a" * 64,
        frame_key=("boot-a", "camera-a", 3, 7),
        pts=OptionalNumber(12.5),
        source_time=OptionalNumber(100.25),
        frame_width=640,
        frame_height=360,
        bed_region_provenance="cached",
        persons=(
            TracePerson(
                ordinal=0,
                track_id=OptionalNumber(42),
                box=(64, 36, 192, 324),
                confidence=0.91,
                keypoints=(
                    TraceKeypoint(0, 100, 60, 0.99),
                    TraceKeypoint(1, 120, 80, 0.1),
                ),
            ),
        ),
        beds=(
            TraceBed(
                ordinal=0,
                box=(32, 180, 320, 350),
                confidence=0.88,
                provenance="cached",
                polygon=((32, 200), (300, 180), (320, 340), (40, 350)),
            ),
        ),
        components=(
            TraceComponent(0, f"pose.sha256.{'b' * 64}", "observed"),
            TraceComponent(1, f"bed.sha256.{'c' * 64}", "not-scheduled"),
        ),
        schema_version=2,
    )


def _decision(module: str, snapshot: DecisionTraceSnapshot, index: int) -> DecisionTrace:
    return DecisionTrace(
        trace_id=f"{index + 1:x}" * 64,
        analysis_trace_id="a" * 64,
        identity_index=index,
        module_qualified_id=f"{module}.v1",
        policy_qualified_id=f"{module}.policy.v1",
        effective_policy_id="d" * 64,
        runtime_manifest_sha256="e" * 64,
        snapshot=snapshot,
    )


def test_scene_is_typed_versioned_hardware_neutral_and_complete() -> None:
    scene = OverlaySceneBuilder().from_traces(
        _analysis(),
        (
            _decision(
                "fall",
                DecisionTraceSnapshot(
                    reason="fall-onset",
                    previous_state="clear",
                    current_state="fall",
                    triggered=True,
                    track_id=42,
                    bed_id=None,
                    values={"fall_probability": 0.82, "operating_threshold": 0.7},
                ),
                0,
            ),
            _decision(
                "bed_exit",
                DecisionTraceSnapshot(
                    reason="live-grace",
                    previous_state="contained",
                    current_state="live-grace",
                    triggered=False,
                    track_id=42,
                    bed_id=0,
                    values={"containment_ratio": 0.31, "min_containment": 0.5},
                ),
                1,
            ),
        ),
        provenance=AppliedCameraProvenance(
            runtime_manifest_sha256="e" * 64,
            camera_configuration_id="camera-config.v7",
        ),
        transform=fit_scene_transform(640, 360, 1280, 720),
    )

    assert scene.schema_version == 1
    assert len(scene.scene_id) == 64
    assert scene.coordinate_space == "source-pixels"
    assert scene.source_dimensions == (640, 360)
    assert scene.transform.scale_x == scene.transform.scale_y == 2.0
    assert scene.frame.worker_boot_id == "boot-a"
    assert scene.frame.stream_epoch == 3
    assert scene.frame.seq == 7
    assert scene.frame.pts.value == 12.5
    assert scene.frame.camera_configuration_id == "camera-config.v7"
    assert scene.persons[0].track_id.value == 42
    assert scene.persons[0].keypoints[0].confidence == 0.99
    assert scene.persons[0].keypoints[1].semantics is ObservationSemantics.MISSING
    assert scene.beds[0].polygon[0] == (32.0, 200.0)
    assert scene.beds[0].semantics is ObservationSemantics.STALE
    assert scene.beds[0].containments[0].ratio.value == 0.31
    assert scene.decisions[0].module_qualified_id == "fall.v1"
    assert scene.decisions[0].score.name == "fall_probability"
    assert scene.decisions[0].threshold.name == "operating_threshold"
    assert scene.decisions[0].runtime_manifest_sha256 == "e" * 64
    assert scene.decisions[1].score.name == "containment_ratio"
    assert scene.decisions[1].threshold.name == "min_containment"
    assert tuple(item.z_order for item in scene.labels) == tuple(
        sorted(item.z_order for item in scene.labels)
    )


def test_scene_identity_and_canonical_bytes_are_deterministic() -> None:
    builder = OverlaySceneBuilder()
    first = builder.from_traces(
        _analysis(),
        (),
        provenance=AppliedCameraProvenance("e" * 64, "camera-config.v7"),
    )
    second = builder.from_traces(
        _analysis(),
        (),
        provenance=AppliedCameraProvenance("e" * 64, "camera-config.v7"),
    )

    assert first == second
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.scene_id == second.scene_id


def test_missing_and_not_evaluated_are_never_rendered_as_zero() -> None:
    analysis = _analysis()
    missing = _decision(
        "fall",
        DecisionTraceSnapshot(
            reason="score-missing",
            previous_state="unknown",
            current_state="not-evaluated",
            triggered=False,
            track_id=42,
            bed_id=None,
            missing_values={"fall_probability": "no-live-classified-track"},
            values={"operating_threshold": 0.7},
        ),
        0,
    )

    scene = OverlaySceneBuilder().from_traces(
        analysis,
        (missing,),
        provenance=AppliedCameraProvenance("e" * 64, "camera-config.v7"),
    )

    assert scene.decisions[0].score.value is None
    assert scene.decisions[0].score.semantics is ObservationSemantics.MISSING
    assert scene.decisions[0].semantics is ObservationSemantics.NOT_EVALUATED
    assert scene.components[1].semantics is ObservationSemantics.NOT_EVALUATED
