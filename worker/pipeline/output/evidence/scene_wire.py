"""Compact canonical wire encoding for scene sidecar frames.

Pure serialization only: the quantized wire representation produced here is the
canonical value domain (parity comparisons happen after quantization). File
staging and header assembly live in ``scene_index``.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Final

from worker.pipeline.output.overlay_scene import (
    BED_COLOR,
    DANGER_COLOR,
    NEUTRAL_COLOR,
    PERSON_COLOR,
    POSE_COLOR,
    POSE_DOT_COLOR,
    POSE_EDGES,
)
from worker.types import SceneRecord
from worker.types.overlay_scene import OverlayScene, SceneValue

SCENE_FRAME_MAX_BYTES: Final = 3072


class SceneIndexWriteError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def encode_scene_frame(scene: OverlayScene) -> bytes:
    """Encode one compact scene body; callers create SceneRecord outside ring locks."""
    return _encode_scene_frame(scene)[0]


def _encode_scene_frame(scene: OverlayScene) -> tuple[bytes, bool]:
    payload = _frame_payload(scene, include_keypoints=True)
    encoded = canonical_json(payload)
    if len(encoded) <= SCENE_FRAME_MAX_BYTES:
        return encoded, False
    shed = _frame_payload(scene, include_keypoints=False)
    encoded = canonical_json(shed)
    if len(encoded) > SCENE_FRAME_MAX_BYTES:
        raise SceneIndexWriteError("SIZE_LIMIT")
    return encoded, True


def encode_scene_record(
    scene: OverlayScene,
    *,
    worker_boot_id: str,
    camera_id: str,
    stream_epoch: int,
    generation: int,
    source_pts_sec: Fraction,
    seq: int,
) -> SceneRecord:
    """Create the pre-serialized record and retain whether keypoint shedding occurred."""
    payload, detail_shed = _encode_scene_frame(scene)
    return SceneRecord(
        worker_boot_id,
        camera_id,
        stream_epoch,
        generation,
        source_pts_sec,
        seq,
        payload,
        len(payload),
        detail_shed,
    )


def _frame_payload(scene: OverlayScene, *, include_keypoints: bool) -> dict[str, object]:
    return {
        "bd": [_bed(value) for value in scene.beds],
        "dc": [_decision(value) for value in scene.decisions],
        "lb": [
            {
                "c": list(value.color),
                "t": value.text,
                "x": _qi(value.anchor[0]),
                "y": _qi(value.anchor[1]),
                "z": value.z_order,
            }
            for value in scene.labels
        ],
        "ps": [_person(value, include_keypoints) for value in scene.persons],
    }


def _person(value, include_keypoints: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "b": [_qi(item) for item in value.box],
        "c": _q(value.confidence),
        "i": value.ordinal,
        "tr": _value(value.track_id),
    }
    if value.track_id.value is None:
        result["tr_r"] = value.track_id.reason
    if include_keypoints:
        result["k"] = [
            [item.index, *(_point(item)), _q(item.confidence)] for item in value.keypoints
        ]
    return result


def _point(value) -> tuple[int | None, int | None]:
    if value.point is None:
        return (None, None)
    return (_qi(value.point[0]), _qi(value.point[1]))


def _bed(value) -> dict[str, object]:
    return {
        "b": [_qi(item) for item in value.box],
        "c": _q(value.confidence),
        "ct": [
            {
                "r": _value(item.ratio),
                "rs": item.reason,
                "s": item.state,
                "th": _value(item.threshold),
                "tr": _value(item.track_id),
            }
            for item in value.containments
        ],
        "i": value.ordinal,
        "pg": [[_qi(x), _qi(y)] for x, y in value.polygon],
        "pv": value.provenance,
        "sm": value.semantics.value,
    }


def _decision(value) -> dict[str, object]:
    counters = [[name, _q(number)] for name, number in value.counters.items()]
    result: dict[str, object] = {
        "bd": _value(value.bed_id),
        "m": value.module_qualified_id,
        "p": value.policy_qualified_id,
        "ps": value.previous_state,
        "rs": value.reason,
        "rm": value.runtime_manifest_sha256,
        "s": value.current_state,
        "sc": _value(value.score),
        "th": _value(value.threshold),
        "tg": value.triggered,
        "tr": _value(value.track_id),
        "e": value.effective_policy_id,
        "cn": counters[:16],
    }
    if len(counters) > 16:
        result["cn_t"] = True
    return result


def _value(value: SceneValue) -> int | float | None:
    return _q(value.value)


def _qi(value: int | float) -> int:
    return round(value)


def _q(value: int | float | None) -> int | float | None:
    return None if value is None else round(value, 2)


def style_payload() -> dict[str, object]:
    """Header style block generated from the single public token owner."""
    return {
        "palette": {
            "bed": list(BED_COLOR),
            "danger": list(DANGER_COLOR),
            "neutral": list(NEUTRAL_COLOR),
            "person": list(PERSON_COLOR),
            "pose": list(POSE_COLOR),
            "pose_dot": list(POSE_DOT_COLOR),
        },
        "skeleton": {"edges": [list(item) for item in POSE_EDGES]},
        "z_order": {"bed": 10, "decision": 40, "person": 20},
    }


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")


__all__ = [
    "SCENE_FRAME_MAX_BYTES",
    "SceneIndexWriteError",
    "canonical_json",
    "encode_scene_frame",
    "encode_scene_record",
    "style_payload",
]
