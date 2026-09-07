"""Bounded CPU re-analysis of one sealed evidence clip."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import av

from contracts.runner import Image
from shared.events.clip_analysis_wire import (
    ClipAnalysisBedGeometry,
    ClipAnalysisBox,
    ClipAnalysisFrame,
    ClipAnalysisResult,
    ClipAnalysisTimeBase,
)
from worker.adapters.model.ort_bed_seg import OrtBedSegRunner
from worker.adapters.model.ort_clip_pose import OrtClipPoseRunner

_MAX_INPUT_BYTES: Final = 128 * 1024 * 1024
_MAX_PIXELS: Final = 3840 * 2160
_MAX_DURATION_S: Final = 180.0
_MAX_FRAMES: Final = 5400


class ClipAnalysisRejected(ValueError):
    """The immutable input cannot be analyzed within its declared bounds."""


class ClipAnalysisFailed(RuntimeError):
    """A bounded analysis failed after preflight."""


@dataclass(frozen=True, slots=True)
class ClipAnalysisProfile:
    person_threshold: float = 0.25
    bed_confidence: float = 0.25
    max_frames: int = _MAX_FRAMES
    max_duration_s: float = _MAX_DURATION_S
    max_pixels: int = _MAX_PIXELS
    max_input_bytes: int = _MAX_INPUT_BYTES

    def __post_init__(self) -> None:
        if not 0.0 <= self.person_threshold <= 1.0 or not 0.0 <= self.bed_confidence <= 1.0:
            raise ValueError("analysis confidence must be in [0, 1]")
        if not 1 <= self.max_frames <= _MAX_FRAMES:
            raise ValueError("analysis max_frames is out of bounds")
        if not 0.0 < self.max_duration_s <= _MAX_DURATION_S:
            raise ValueError("analysis max_duration_s is out of bounds")
        if not 1 <= self.max_pixels <= _MAX_PIXELS:
            raise ValueError("analysis max_pixels is out of bounds")
        if not 1 <= self.max_input_bytes <= _MAX_INPUT_BYTES:
            raise ValueError("analysis max_input_bytes is out of bounds")


@dataclass(frozen=True, slots=True)
class ClipAnalysisRequest:
    clip_path: Path
    clip_sha256: str
    pose_model_path: Path
    bed_model_path: Path
    analysis_profile: ClipAnalysisProfile


def profile_sha256(profile: ClipAnalysisProfile) -> str:
    payload = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def analyze_clip(request: ClipAnalysisRequest, *, decoder_identity: str) -> ClipAnalysisResult:
    """Analyze exact clip bytes, retaining only PTS-addressed evidence."""
    _validate_request(request)
    try:
        size = request.clip_path.stat().st_size
    except OSError as exc:
        raise ClipAnalysisRejected("input_unreadable") from exc
    if size > request.analysis_profile.max_input_bytes:
        raise ClipAnalysisRejected("input_bytes")
    if not hmac.compare_digest(_sha256_file(request.clip_path), request.clip_sha256):
        raise ClipAnalysisRejected("clip_sha256")
    try:
        container = av.open(str(request.clip_path))
    except Exception as exc:
        raise ClipAnalysisRejected("container_open") from exc
    try:
        stream = container.streams.video[0]
        _preflight_stream(stream, request.analysis_profile)
        pose = OrtClipPoseRunner(request.pose_model_path, request.analysis_profile.person_threshold)
        bed = OrtBedSegRunner(
            str(request.bed_model_path), confidence=request.analysis_profile.bed_confidence
        )
        frames, target_pts = _decode_frames(container, stream, pose, request.analysis_profile)
        representative = _read_representative(request.clip_path, target_pts)
        geometries = _bed_geometries(bed, representative, target_pts)
        time_base = _stream_time_base(stream)
        return ClipAnalysisResult(
            source="clip_reanalysis",
            clip_id=request.clip_path.name,
            clip_sha256=request.clip_sha256,
            pose_model_sha256=pose.artifact_digest,
            bed_model_sha256=bed.artifact_digest,
            decoder_identity=decoder_identity,
            analysis_profile_sha256=profile_sha256(request.analysis_profile),
            image_width=stream.width,
            image_height=stream.height,
            time_base=ClipAnalysisTimeBase(time_base.numerator, time_base.denominator),
            frames=frames,
            bed_geometries=geometries,
        )
    except ClipAnalysisRejected:
        raise
    except ClipAnalysisFailed:
        raise
    except Exception as exc:
        raise ClipAnalysisFailed("analysis_failed") from exc
    finally:
        container.close()


def _validate_request(request: ClipAnalysisRequest) -> None:
    if not isinstance(request, ClipAnalysisRequest):
        raise ClipAnalysisRejected("request")
    if not isinstance(request.analysis_profile, ClipAnalysisProfile):
        raise ClipAnalysisRejected("profile")
    if len(request.clip_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in request.clip_sha256
    ):
        raise ClipAnalysisRejected("clip_sha256")


def _preflight_stream(stream: av.video.stream.VideoStream, profile: ClipAnalysisProfile) -> None:
    if stream.width <= 0 or stream.height <= 0 or stream.width * stream.height > profile.max_pixels:
        raise ClipAnalysisRejected("resolution")
    if stream.duration is None or stream.time_base is None:
        raise ClipAnalysisRejected("duration")
    if float(stream.duration * stream.time_base) > profile.max_duration_s:
        raise ClipAnalysisRejected("duration")


def _decode_frames(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    pose: OrtClipPoseRunner,
    profile: ClipAnalysisProfile,
) -> tuple[tuple[ClipAnalysisFrame, ...], int]:
    stream.thread_type = "NONE"
    stream.thread_count = 1
    records: dict[int, ClipAnalysisFrame] = {}
    order: list[int] = []
    decoded = 0
    none_pts = 0
    for frame in container.decode(stream):
        decoded += 1
        if decoded > profile.max_frames:
            raise ClipAnalysisRejected("frame_count")
        if frame.pts is None:
            none_pts += 1
            continue
        pts = frame.pts
        if pts in records:
            records[pts] = ClipAnalysisFrame(pts, "ambiguous_timestamp")
            continue
        image = frame.to_ndarray(format="rgb24")
        boxes = tuple(ClipAnalysisBox(*box) for box in pose.detect_persons(image))
        records[pts] = ClipAnalysisFrame(pts, "available", boxes)
        order.append(pts)
    if none_pts and not records:
        raise ClipAnalysisRejected("timestamp")
    available = [pts for pts in order if records[pts].status == "available"]
    if not available:
        raise ClipAnalysisRejected("timestamp")
    after_lookback = [pts for pts in available if float(pts * stream.time_base) >= 15.0]
    return tuple(records[pts] for pts in order), (
        after_lookback[0] if after_lookback else available[-1]
    )


def _read_representative(clip_path: Path, target_pts: int) -> Image:
    try:
        with av.open(str(clip_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            stream.thread_count = 1
            for frame in container.decode(stream):
                if frame.pts == target_pts:
                    return frame.to_ndarray(format="rgb24")
    except Exception as exc:
        raise ClipAnalysisFailed("representative_decode") from exc
    raise ClipAnalysisFailed("representative_missing")


def _bed_geometries(
    runner: OrtBedSegRunner, image: Image, provenance_pts: int
) -> tuple[ClipAnalysisBedGeometry, ...]:
    geometries: list[ClipAnalysisBedGeometry] = []
    for instance in runner.detect_beds(image).boxes:
        if len(instance) < 6:
            continue
        polygon = instance[5]
        if not isinstance(polygon, (tuple, list)):
            raise ClipAnalysisFailed("bed_polygon")
        points = tuple((float(point[0]), float(point[1])) for point in polygon)
        if 3 <= len(points) <= 64:
            geometries.append(ClipAnalysisBedGeometry(points, provenance_pts))
    return tuple(geometries)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ClipAnalysisRejected("input_unreadable") from exc
    return digest.hexdigest()


__all__ = [
    "ClipAnalysisFailed",
    "ClipAnalysisProfile",
    "ClipAnalysisRejected",
    "ClipAnalysisRequest",
    "analyze_clip",
    "profile_sha256",
]


def _stream_time_base(stream: av.video.stream.VideoStream) -> Fraction:
    time_base = stream.time_base
    if not isinstance(time_base, Fraction):
        raise ClipAnalysisFailed("stream_time_base")
    return time_base
