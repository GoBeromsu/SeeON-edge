"""Bounded canonical scene-index sidecar staging.

The compact per-frame wire encoding lives in ``scene_wire``; this module owns
header assembly, provenance validation, and the staged file write.
"""

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

from worker.pipeline.output.evidence.scene_wire import (
    SceneIndexWriteError,
    canonical_json,
    style_payload,
)
from worker.types import SceneRecord

SCENE_INDEX_FILENAME: Final = "scene-index.json"
SCENE_INDEX_SCHEMA_VERSION: Final = 1
SCENE_INDEX_MAX_BYTES: Final = 8 * 1024 * 1024
LOGGER: Final = logging.getLogger(__name__)


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
        "style": style_payload(),
        "time_origin": {
            "event_pts_sec": float(header.event_pts_sec),
            "media_origin_pts_sec": float(header.media_origin_pts_sec),
            "requested_end_pts_sec": float(header.requested_end_pts_sec),
            "requested_start_pts_sec": float(header.requested_start_pts_sec),
        },
    }
    encoded = canonical_json(payload)
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


__all__ = [
    "SCENE_INDEX_FILENAME",
    "SCENE_INDEX_MAX_BYTES",
    "SCENE_INDEX_SCHEMA_VERSION",
    "SceneIndexHeader",
    "SceneIndexWriteError",
    "write_scene_index",
]
