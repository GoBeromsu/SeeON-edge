"""Strict wire types for re-analysis of one immutable clip.

This module is a value contract only: it does not open clips, run inference,
or turn a result with no frame evidence into a successful-looking result.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import gcd
from typing import Final, Literal, TypeAlias

MAX_CLIP_ANALYSIS_FRAMES: Final = 5_400
MAX_CLIP_ANALYSIS_OUTPUT_BYTES: Final = 32 * 1024 * 1024
MAX_IMAGE_WIDTH: Final = 3_840
MAX_IMAGE_HEIGHT: Final = 2_160
_MAX_INT64: Final = (1 << 63) - 1
_MAX_SAFE_PTS: Final = (1 << 53) - 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE: Final = "clip_reanalysis"
FrameStatus: TypeAlias = Literal["available", "no_evidence", "ambiguous_timestamp"]


class ClipAnalysisWireError(ValueError):
    """The clip-analysis payload is not a complete, safe wire value."""


def _int(value: object, field: str) -> int:
    if type(value) is not int or not -_MAX_INT64 <= value <= _MAX_INT64:
        raise ClipAnalysisWireError(f"{field} must be a signed 64-bit integer")
    return value


def _pts(value: object, field: str) -> int:
    parsed = _int(value, field)
    if not -_MAX_SAFE_PTS <= parsed <= _MAX_SAFE_PTS:
        raise ClipAnalysisWireError(f"{field} exceeds lossless JSON integer range")
    return parsed


def _number(value: object, field: str) -> float:
    if type(value) is int:
        number = float(value)
    elif type(value) is float:
        number = value
    else:
        raise ClipAnalysisWireError(f"{field} must be finite")
    if not math.isfinite(number):
        raise ClipAnalysisWireError(f"{field} must be finite")
    return number


def _text(value: object, field: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ClipAnalysisWireError(f"{field} must be a non-empty bounded string")
    return value


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ClipAnalysisWireError(f"{field} must be lowercase SHA-256 hex")
    return value


def _source(value: object) -> Literal["clip_reanalysis"]:
    if value != _SOURCE:
        raise ClipAnalysisWireError("source must be clip_reanalysis")
    return "clip_reanalysis"


def _status(value: object) -> FrameStatus:
    if value == "available":
        return "available"
    if value == "no_evidence":
        return "no_evidence"
    if value == "ambiguous_timestamp":
        return "ambiguous_timestamp"
    raise ClipAnalysisWireError("frame status is invalid")


@dataclass(frozen=True, slots=True)
class ClipAnalysisTimeBase:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        numerator, denominator = (
            _int(self.numerator, "time_base.numerator"),
            _int(self.denominator, "time_base.denominator"),
        )
        if numerator <= 0 or denominator <= 0 or gcd(numerator, denominator) != 1:
            raise ClipAnalysisWireError("time base must be positive and reduced")


@dataclass(frozen=True, slots=True)
class ClipAnalysisBox:
    """A person box in source-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = (
            _number(v, f"box.{n}")
            for n, v in (("x1", self.x1), ("y1", self.y1), ("x2", self.x2), ("y2", self.y2))
        )
        confidence = _number(self.confidence, "box.confidence")
        if not (0 <= x1 < x2 <= MAX_IMAGE_WIDTH and 0 <= y1 < y2 <= MAX_IMAGE_HEIGHT):
            raise ClipAnalysisWireError("box coordinates are out of bounds")
        if not 0 <= confidence <= 1:
            raise ClipAnalysisWireError("box confidence is out of bounds")


@dataclass(frozen=True, slots=True)
class ClipAnalysisFrame:
    pts: int
    status: FrameStatus
    boxes: tuple[ClipAnalysisBox, ...] = ()

    def __post_init__(self) -> None:
        _pts(self.pts, "frame.pts")
        if self.status not in ("available", "no_evidence", "ambiguous_timestamp"):
            raise ClipAnalysisWireError("frame status is invalid")
        if not isinstance(self.boxes, tuple) or any(
            not isinstance(box, ClipAnalysisBox) for box in self.boxes
        ):
            raise ClipAnalysisWireError("frame boxes must be typed box values")
        if self.status != "available" and self.boxes:
            raise ClipAnalysisWireError("unavailable or ambiguous frames cannot carry boxes")


@dataclass(frozen=True, slots=True)
class ClipAnalysisBedGeometry:
    """One clip-local bed polygon and its supplying frame PTS."""

    points: tuple[tuple[float, float], ...]
    provenance_pts: int

    def __post_init__(self) -> None:
        _pts(self.provenance_pts, "bed_geometry.provenance_pts")
        if not isinstance(self.points, tuple) or not 3 <= len(self.points) <= 64:
            raise ClipAnalysisWireError("bed geometry needs 3 to 64 points")
        for point in self.points:
            if not isinstance(point, tuple) or len(point) != 2:
                raise ClipAnalysisWireError("bed geometry points must be pairs")
            x, y = _number(point[0], "bed_geometry.x"), _number(point[1], "bed_geometry.y")
            if not (0 <= x <= MAX_IMAGE_WIDTH and 0 <= y <= MAX_IMAGE_HEIGHT):
                raise ClipAnalysisWireError("bed geometry coordinates are out of bounds")


@dataclass(frozen=True, slots=True)
class ClipAnalysisResult:
    source: Literal["clip_reanalysis"]
    clip_id: str
    clip_sha256: str
    pose_model_sha256: str
    bed_model_sha256: str
    decoder_identity: str
    analysis_profile_sha256: str
    image_width: int
    image_height: int
    time_base: ClipAnalysisTimeBase
    frames: tuple[ClipAnalysisFrame, ...]
    bed_geometries: tuple[ClipAnalysisBedGeometry, ...] = ()

    def __post_init__(self) -> None:
        if self.source != _SOURCE:
            raise ClipAnalysisWireError("source must be clip_reanalysis")
        _text(self.clip_id, "clip_id")
        _sha(self.clip_sha256, "clip_sha256")
        _sha(self.pose_model_sha256, "pose_model_sha256")
        _sha(self.bed_model_sha256, "bed_model_sha256")
        _text(self.decoder_identity, "decoder_identity", MAX_CLIP_ANALYSIS_OUTPUT_BYTES * 2)
        _sha(self.analysis_profile_sha256, "analysis_profile_sha256")
        width, height = (
            _int(self.image_width, "image_width"),
            _int(self.image_height, "image_height"),
        )
        if not (1 <= width <= MAX_IMAGE_WIDTH and 1 <= height <= MAX_IMAGE_HEIGHT):
            raise ClipAnalysisWireError("image dimensions are out of bounds")
        if not isinstance(self.time_base, ClipAnalysisTimeBase):
            raise ClipAnalysisWireError("time_base must be a typed value")
        if (
            not isinstance(self.frames, tuple)
            or not 1 <= len(self.frames) <= MAX_CLIP_ANALYSIS_FRAMES
        ):
            raise ClipAnalysisWireError("frames must contain 1 to 5400 records")
        if any(not isinstance(frame, ClipAnalysisFrame) for frame in self.frames):
            raise ClipAnalysisWireError("frames must be typed frame values")
        if len({frame.pts for frame in self.frames}) != len(self.frames):
            raise ClipAnalysisWireError("frame PTS values must be unique")
        for frame in self.frames:
            for box in frame.boxes:
                if box.x2 > width or box.y2 > height:
                    raise ClipAnalysisWireError("box coordinates exceed image dimensions")
        if not isinstance(self.bed_geometries, tuple) or any(
            not isinstance(geometry, ClipAnalysisBedGeometry) for geometry in self.bed_geometries
        ):
            raise ClipAnalysisWireError("bed geometries must be typed values")
        frame_by_pts = {frame.pts: frame for frame in self.frames}
        for geometry in self.bed_geometries:
            provenance_frame = frame_by_pts.get(geometry.provenance_pts)
            if provenance_frame is None or provenance_frame.status != "available":
                raise ClipAnalysisWireError("bed provenance must be an available frame PTS")
            if any(x > width or y > height for x, y in geometry.points):
                raise ClipAnalysisWireError("bed geometry exceeds image dimensions")

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "clip_id": self.clip_id,
            "clip_sha256": self.clip_sha256,
            "pose_model_sha256": self.pose_model_sha256,
            "bed_model_sha256": self.bed_model_sha256,
            "decoder_identity": self.decoder_identity,
            "analysis_profile_sha256": self.analysis_profile_sha256,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "time_base": {
                "numerator": self.time_base.numerator,
                "denominator": self.time_base.denominator,
            },
            "frames": [
                {
                    "pts": frame.pts,
                    "status": frame.status,
                    "boxes": [
                        {
                            "x1": box.x1,
                            "y1": box.y1,
                            "x2": box.x2,
                            "y2": box.y2,
                            "confidence": box.confidence,
                        }
                        for box in frame.boxes
                    ],
                }
                for frame in self.frames
            ],
            "bed_geometries": [
                {
                    "points": [[x, y] for x, y in geometry.points],
                    "provenance_pts": geometry.provenance_pts,
                }
                for geometry in self.bed_geometries
            ],
        }

    def canonical_json(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if len(encoded) > MAX_CLIP_ANALYSIS_OUTPUT_BYTES:
            raise ClipAnalysisWireError("clip-analysis output exceeds 32 MiB")
        return encoded.decode()


_TOP = frozenset(
    {
        "source",
        "clip_id",
        "clip_sha256",
        "pose_model_sha256",
        "bed_model_sha256",
        "decoder_identity",
        "analysis_profile_sha256",
        "image_width",
        "image_height",
        "time_base",
        "frames",
        "bed_geometries",
    }
)
_BOX = frozenset({"x1", "y1", "x2", "y2", "confidence"})
_FRAME = frozenset({"pts", "status", "boxes"})
_GEOMETRY = frozenset({"points", "provenance_pts"})


def _obj(value: object, fields: frozenset[str], where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ClipAnalysisWireError(f"{where} has missing or forbidden fields")
    return value


def _parse_box(value: object) -> ClipAnalysisBox:
    item = _obj(value, _BOX, "box")
    return ClipAnalysisBox(
        _number(item["x1"], "box.x1"),
        _number(item["y1"], "box.y1"),
        _number(item["x2"], "box.x2"),
        _number(item["y2"], "box.y2"),
        _number(item["confidence"], "box.confidence"),
    )


def _parse_frame(value: object) -> ClipAnalysisFrame:
    item = _obj(value, _FRAME, "frame")
    boxes = item["boxes"]
    if not isinstance(boxes, list):
        raise ClipAnalysisWireError("frame.boxes must be a list")
    return ClipAnalysisFrame(
        _pts(item["pts"], "frame.pts"),
        _status(item["status"]),
        tuple(_parse_box(box) for box in boxes),
    )


def _parse_geometry(value: object) -> ClipAnalysisBedGeometry:
    item = _obj(value, _GEOMETRY, "bed geometry")
    points = item["points"]
    if not isinstance(points, list):
        raise ClipAnalysisWireError("bed_geometry.points must be a list")
    parsed: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ClipAnalysisWireError("bed geometry points must be pairs")
        parsed.append(
            (
                _number(point[0], "bed_geometry.x"),
                _number(point[1], "bed_geometry.y"),
            )
        )
    return ClipAnalysisBedGeometry(
        tuple(parsed),
        _pts(item["provenance_pts"], "bed_geometry.provenance_pts"),
    )


def _decode_mapping(payload: Mapping[str, object]) -> ClipAnalysisResult:
    item = _obj(payload, _TOP, "clip-analysis payload")
    clock = _obj(item["time_base"], frozenset({"numerator", "denominator"}), "time_base")
    frames, geometries = item["frames"], item["bed_geometries"]
    if not isinstance(frames, list) or not isinstance(geometries, list):
        raise ClipAnalysisWireError("frames and bed_geometries must be lists")
    return ClipAnalysisResult(
        _source(item["source"]),
        _text(item["clip_id"], "clip_id"),
        _sha(item["clip_sha256"], "clip_sha256"),
        _sha(item["pose_model_sha256"], "pose_model_sha256"),
        _sha(item["bed_model_sha256"], "bed_model_sha256"),
        _text(item["decoder_identity"], "decoder_identity", MAX_CLIP_ANALYSIS_OUTPUT_BYTES * 2),
        _sha(item["analysis_profile_sha256"], "analysis_profile_sha256"),
        _int(item["image_width"], "image_width"),
        _int(item["image_height"], "image_height"),
        ClipAnalysisTimeBase(
            _int(clock["numerator"], "time_base.numerator"),
            _int(clock["denominator"], "time_base.denominator"),
        ),
        tuple(_parse_frame(frame) for frame in frames),
        tuple(_parse_geometry(geometry) for geometry in geometries),
    )


def encode_clip_analysis(result: ClipAnalysisResult) -> bytes:
    """Serialize a validated result, enforcing the byte-level output cap."""
    if not isinstance(result, ClipAnalysisResult):
        raise ClipAnalysisWireError("result must be a ClipAnalysisResult")
    return result.canonical_json().encode()


def decode_clip_analysis(payload: object) -> ClipAnalysisResult:
    """Decode JSON bytes/text or a JSON-shaped mapping with strict fields."""
    if isinstance(payload, bytes):
        if len(payload) > MAX_CLIP_ANALYSIS_OUTPUT_BYTES:
            raise ClipAnalysisWireError("clip-analysis input exceeds 32 MiB")
        try:
            payload = payload.decode()
        except UnicodeDecodeError as error:
            raise ClipAnalysisWireError("clip-analysis input is not UTF-8") from error
    if isinstance(payload, str):
        if len(payload.encode()) > MAX_CLIP_ANALYSIS_OUTPUT_BYTES:
            raise ClipAnalysisWireError("clip-analysis input exceeds 32 MiB")
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ClipAnalysisWireError("clip-analysis input is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ClipAnalysisWireError("clip-analysis payload must be an object")
    result = _decode_mapping(payload)
    result.canonical_json()
    return result


__all__ = [
    "MAX_CLIP_ANALYSIS_FRAMES",
    "MAX_CLIP_ANALYSIS_OUTPUT_BYTES",
    "ClipAnalysisBedGeometry",
    "ClipAnalysisBox",
    "ClipAnalysisFrame",
    "ClipAnalysisResult",
    "ClipAnalysisTimeBase",
    "ClipAnalysisWireError",
    "FrameStatus",
    "decode_clip_analysis",
    "encode_clip_analysis",
]
