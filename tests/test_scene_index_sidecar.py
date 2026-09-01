from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from contracts.observation import BoundingBox, FrameObservation
from worker.pipeline.output.evidence.scene_index import (
    SCENE_INDEX_FILENAME,
    SceneIndexHeader,
    write_scene_index,
)
from worker.pipeline.output.evidence.scene_wire import (
    SCENE_FRAME_MAX_BYTES,
    SceneIndexWriteError,
    encode_scene_frame,
    encode_scene_record,
)
from worker.pipeline.output.overlay_scene import (
    BED_COLOR,
    DANGER_COLOR,
    NEUTRAL_COLOR,
    PERSON_COLOR,
    POSE_COLOR,
    POSE_DOT_COLOR,
    POSE_EDGES,
    OverlaySceneBuilder,
)
from worker.pipeline.trace.models import DecisionTrace
from worker.types import DecisionTraceSnapshot, SceneRecord
from worker.types.overlay_scene import (
    ObservationSemantics,
    SceneFrameIdentity,
    SceneKeypoint,
    SceneLabel,
    SceneValue,
)
from worker.types.trace import DecisionTraceValueName


def _scene(*, counters: dict[str, float] | None = None):
    decision = DecisionTrace(
        trace_id="a" * 64,
        analysis_trace_id="b" * 64,
        identity_index=0,
        module_qualified_id="fall.v1",
        policy_qualified_id="fall-policy.v1",
        effective_policy_id="c" * 64,
        runtime_manifest_sha256="d" * 64,
        snapshot=DecisionTraceSnapshot(
            reason="fall-onset",
            previous_state="clear",
            current_state="fall",
            triggered=True,
            track_id=7,
            bed_id=None,
            values={"fall_probability": 0.9876, "operating_threshold": 0.6234, **(counters or {})},
        ),
    )
    return OverlaySceneBuilder().from_observation(
        identity=SceneFrameIdentity(
            "boot",
            "camera",
            1,
            2,
            SceneValue(2.0, ObservationSemantics.PRESENT),
            SceneValue(2.0, ObservationSemantics.PRESENT),
            "config.v1",
        ),
        observation=FrameObservation(
            detections=((BoundingBox(1, 2, 300, 400, 0.9876),), ()),
            poses=(((10, 20, 0.1), (30, 40, 0.9876)),),
            regions=((BoundingBox(3, 4, 500, 600, 0.8765, ((3, 4), (5, 6))),), ()),
            track_ids=(None,),
        ),
        source_width=640,
        source_height=480,
        decisions=(decision,),
    )


def _header() -> SceneIndexHeader:
    return SceneIndexHeader(
        "clip",
        "camera",
        "boot",
        1,
        0,
        Fraction(1),
        Fraction(2),
        Fraction(0),
        Fraction(3),
        (640, 480),
    )


def _record(scene, *, seq: int = 2, pts: Fraction = Fraction(2)) -> SceneRecord:
    payload = encode_scene_frame(scene)
    return SceneRecord("boot", "camera", 1, 0, pts, seq, payload, len(payload), False)


def test_canonical_compact_payload_preserves_edge_contracts(tmp_path) -> None:
    counters = {
        item.value: index + 0.1234
        for index, item in enumerate(DecisionTraceValueName)
        if item.value not in {"fall_probability", "operating_threshold"}
    }
    scene = _scene(counters=counters)
    first = encode_scene_frame(scene)
    assert first == encode_scene_frame(scene)
    facts = write_scene_index(tmp_path, (_record(scene),), header=_header())
    assert facts is not None
    encoded = (tmp_path / SCENE_INDEX_FILENAME).read_bytes()
    assert facts.sha256 == hashlib.sha256(encoded).hexdigest()
    payload = json.loads(encoded)
    frame = payload["frames"][0]
    assert frame["t"] == 1000 and frame["p"] == 2.0
    person = frame["ps"][0]
    assert person["tr"] is None and person["tr_r"] == "tracker-unmatched"
    assert person["k"][0] == [0, None, None, 0.1]
    assert frame["dc"][0]["sc"] == 0.99 and frame["dc"][0]["th"] == 0.62
    assert len(frame["dc"][0]["cn"]) == 16 and frame["dc"][0]["cn_t"] is True
    assert "kd" not in frame["lb"][0] and "d" not in frame["lb"][0]
    assert payload["style"] == {
        "palette": {
            "bed": list(BED_COLOR),
            "danger": list(DANGER_COLOR),
            "neutral": list(NEUTRAL_COLOR),
            "person": list(PERSON_COLOR),
            "pose": list(POSE_COLOR),
            "pose_dot": list(POSE_DOT_COLOR),
        },
        "skeleton": {"edges": [list(edge) for edge in POSE_EDGES]},
        "z_order": {"bed": 10, "decision": 40, "person": 20},
    }


def test_writer_matches_cross_language_golden_fixture(tmp_path) -> None:
    """The dashboard fixture is produced by this writer, not hand-authored JSON."""
    facts = write_scene_index(tmp_path, (_record(_scene()),), header=_header())
    assert facts is not None
    encoded = (tmp_path / SCENE_INDEX_FILENAME).read_bytes()
    fixture = (
        Path(__file__).parents[1] / "front/src/shared/api/scene-index.golden.json"
    ).read_bytes()

    assert encoded == fixture
    assert facts.sha256 == hashlib.sha256(fixture).hexdigest()


def test_writer_rejects_unsorted_or_duplicate_pts_and_marks_size_limit(tmp_path) -> None:
    scene = _scene()
    with pytest.raises(SceneIndexWriteError, match="PROVENANCE_CONFLICT"):
        write_scene_index(
            tmp_path,
            (_record(scene, pts=Fraction(2)), _record(scene, seq=3, pts=Fraction(1))),
            header=_header(),
        )
    huge = replace(
        _scene(),
        labels=(SceneLabel("x" * SCENE_FRAME_MAX_BYTES, (0, 0), (1, 2, 3), 1),),
    )
    # An oversized scene without keypoints cannot be safely shed.
    with pytest.raises(SceneIndexWriteError, match="SIZE_LIMIT"):
        encode_scene_frame(huge)


def test_large_pose_sheds_keypoints_and_marks_record() -> None:
    scene = _scene()
    points = tuple(
        SceneKeypoint(index, (index, index), 0.99, ObservationSemantics.PRESENT, None)
        for index in range(400)
    )
    scene = replace(scene, persons=(replace(scene.persons[0], keypoints=points),))

    record = encode_scene_record(
        scene,
        worker_boot_id="boot",
        camera_id="camera",
        stream_epoch=1,
        generation=0,
        source_pts_sec=Fraction(2),
        seq=2,
    )

    assert record.detail_shed is True
    assert record.size_bytes <= SCENE_FRAME_MAX_BYTES
    assert "k" not in json.loads(record.payload)["ps"][0]
