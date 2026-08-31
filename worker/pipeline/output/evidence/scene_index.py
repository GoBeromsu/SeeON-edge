"""Canonical compact scene-index sidecar encoding and staging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
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

SCENE_INDEX_FILENAME: Final = "scene-index.json"
SCENE_INDEX_SCHEMA_VERSION: Final = 1
SCENE_INDEX_MAX_BYTES: Final = 8 * 1024 * 1024
SCENE_FRAME_MAX_BYTES: Final = 3072
LOGGER: Final = logging.getLogger(__name__)


class SceneIndexWriteError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SceneIndexHeader:
    clip_id: str
    camera_id: str
    worker_boot_id: str
    stream_epoch: int
    generation: int
    media_origin_pts_sec: Fraction
    event_pts_sec: Fraction
    requested_start_pts_sec: Fraction
    requested_end_pts_sec: Fraction
    source_dimensions: tuple[int, int]
    components: tuple[tuple[str, str], ...] = ()


def encode_scene_frame(scene: OverlayScene) -> bytes:
    """Encode one compact scene body; callers create SceneRecord outside ring locks."""
    return _encode_scene_frame(scene)[0]


def _encode_scene_frame(scene: OverlayScene) -> tuple[bytes, bool]:
    payload = _frame_payload(scene, include_keypoints=True)
    encoded = _canonical(payload)
    if len(encoded) <= SCENE_FRAME_MAX_BYTES:
        return encoded, False
    shed = _frame_payload(scene, include_keypoints=False)
    encoded = _canonical(shed)
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


def write_scene_index(
    staging_dir: Path,
    records: Iterable[SceneRecord],
    *,
    header: SceneIndexHeader,
):
    """Write a bounded canonical sidecar, returning durable facts or ``None`` on size limit."""
    ordered = tuple(records)
    if not ordered:
        raise SceneIndexWriteError("SELECTION_EMPTY")
    if any(record.camera_id != header.camera_id for record in ordered):
        raise SceneIndexWriteError("PROVENANCE_CONFLICT")
    if any(
        record.worker_boot_id != header.worker_boot_id
        or record.stream_epoch != header.stream_epoch
        or record.generation != header.generation
        for record in ordered
    ):
        raise SceneIndexWriteError("PROVENANCE_CONFLICT")
    if tuple(sorted(ordered, key=lambda item: item.source_pts_sec)) != ordered:
        raise SceneIndexWriteError("PROVENANCE_CONFLICT")
    if len({record.source_pts_sec for record in ordered}) != len(ordered):
        raise SceneIndexWriteError("PROVENANCE_CONFLICT")
    frames = []
    provenance: dict[str, dict[str, str]] = {}
    for record in ordered:
        try:
            scene = json.loads(record.payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SceneIndexWriteError("WRITE_FAILED") from exc
        frames.append(
            {
                "p": float(record.source_pts_sec),
                "q": record.seq,
                "sd": record.detail_shed,
                "t": round(float((record.source_pts_sec - header.media_origin_pts_sec) * 1000)),
                **scene,
            }
        )
        for decision in scene.get("dc", []):
            if not isinstance(decision, dict):
                raise SceneIndexWriteError("WRITE_FAILED")
            module = decision.get("m")
            identity = {
                "effective_policy_id": decision.get("e"),
                "policy": decision.get("p"),
                "runtime_manifest_sha256": decision.get("rm"),
            }
            if not isinstance(module, str) or not all(
                isinstance(value, str) for value in identity.values()
            ):
                raise SceneIndexWriteError("WRITE_FAILED")
            previous = provenance.setdefault(module, identity)
            if previous != identity:
                raise SceneIndexWriteError("PROVENANCE_CONFLICT")
    payload = {
        "camera_id": header.camera_id,
        "clip_id": header.clip_id,
        "components": [{"id": item[0], "sm": item[1]} for item in header.components],
        "coordinate_space": "source-pixels",
        "decision_provenance": [
            {"m": module, **identity} for module, identity in sorted(provenance.items())
        ],
        "detail_shed_frame_count": sum(record.detail_shed for record in ordered),
        "frame_count": len(frames),
        "frames": frames,
        "scene_index_schema_version": SCENE_INDEX_SCHEMA_VERSION,
        "scene_schema_version": 1,
        "source_dimensions": list(header.source_dimensions),
        "stream_identity": {
            "generation": header.generation,
            "stream_epoch": header.stream_epoch,
            "worker_boot_id": header.worker_boot_id,
        },
        "style": _style(),
        "time_origin": {
            "event_pts_sec": float(header.event_pts_sec),
            "media_origin_pts_sec": float(header.media_origin_pts_sec),
            "requested_end_pts_sec": float(header.requested_end_pts_sec),
            "requested_start_pts_sec": float(header.requested_start_pts_sec),
        },
    }
    encoded = _canonical(payload)
    if len(encoded) > SCENE_INDEX_MAX_BYTES:
        LOGGER.warning(
            "clip scene index not written: camera_id=%s clip_id=%s reason=SIZE_LIMIT",
            header.camera_id,
            header.clip_id,
        )
        return None
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / SCENE_INDEX_FILENAME
    with path.open("wb") as output:
        _ = output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    from worker.pipeline.output.evidence.manifest_media_models import SceneIndexFacts

    return SceneIndexFacts(
        path=SCENE_INDEX_FILENAME,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        schema=SCENE_INDEX_SCHEMA_VERSION,
        count=len(frames),
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


def _style() -> dict[str, object]:
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


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")


__all__ = [
    "SCENE_FRAME_MAX_BYTES",
    "SCENE_INDEX_FILENAME",
    "SCENE_INDEX_MAX_BYTES",
    "SCENE_INDEX_SCHEMA_VERSION",
    "SceneIndexHeader",
    "SceneIndexWriteError",
    "encode_scene_frame",
    "encode_scene_record",
    "write_scene_index",
]
