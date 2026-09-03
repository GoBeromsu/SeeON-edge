"""Versioned frame-level replay trace contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Literal

REPLAY_TRACE_VERSION = "replay-trace-v2"
ReplaySource = Literal["legacy-association", "nvdcf"]
SourceEvent = Literal["open", "frame", "reconnect", "lost"]
TrackLifecycle = Literal["new", "tracked", "shadow", "lost"]


@dataclass(frozen=True, slots=True)
class ReplayTrack:
    track_id: int
    lifecycle: TrackLifecycle
    bbox: tuple[float, float, float, float, float]
    keypoints: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise TypeError("track_id must be an integer")
        if self.lifecycle not in ("new", "tracked", "shadow", "lost"):
            raise ValueError("invalid lifecycle")
        if (
            len(self.bbox) != 5
            or len(self.keypoints) != 17
            or any(len(point) != 3 for point in self.keypoints)
        ):
            raise ValueError("track requires bbox5 and COCO-17 keypoints")
        _unit_box(self.bbox)
        for point in self.keypoints:
            _unit_coordinates(point)


@dataclass(frozen=True, slots=True)
class ReplayTraceHeader:
    version: str = REPLAY_TRACE_VERSION

    def __post_init__(self) -> None:
        if self.version != REPLAY_TRACE_VERSION:
            raise ValueError("unsupported replay trace version")


@dataclass(frozen=True, slots=True)
class ReplayRow:
    camera_id: str
    seq: int
    pts_ns: int
    epoch: int
    source_event: SourceEvent
    source: ReplaySource
    tracks: tuple[ReplayTrack, ...]
    bed_polygon_id: str | None
    bed_polygon: tuple[tuple[float, float], ...] | None
    bed_polygon_image_size: tuple[int, int] | None
    night_window_active: bool
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.frame_width, self.frame_height)
        ):
            raise ValueError("frame_width and frame_height must be positive integers")
        if (
            not isinstance(self.camera_id, str)
            or not self.camera_id
            or isinstance(self.seq, bool)
            or not isinstance(self.seq, int)
            or isinstance(self.pts_ns, bool)
            or not isinstance(self.pts_ns, int)
            or isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or self.seq < 0
            or self.pts_ns < 0
            or self.epoch < 0
        ):
            raise ValueError("camera_id and non-negative seq/pts_ns/epoch are required")
        if self.source_event not in ("open", "frame", "reconnect", "lost"):
            raise ValueError("invalid source_event")
        if self.source not in ("legacy-association", "nvdcf"):
            raise ValueError("invalid source")
        if not (
            self.bed_polygon_id is None
            and self.bed_polygon is None
            and self.bed_polygon_image_size is None
        ) and not (
            self.bed_polygon_id is not None
            and self.bed_polygon is not None
            and self.bed_polygon_image_size is not None
        ):
            raise ValueError("bed polygon fields must be present exactly together")
        if self.bed_polygon_id is not None and (
            not isinstance(self.bed_polygon_id, str) or not self.bed_polygon_id
        ):
            raise ValueError("invalid bed_polygon_id")
        if self.bed_polygon is not None and len(self.bed_polygon) < 3:
            raise ValueError("bed_polygon must contain at least three xy points")
        if self.bed_polygon is not None:
            for point in self.bed_polygon:
                if len(point) != 2:
                    raise ValueError("bed_polygon must contain at least three xy points")
                _unit_coordinates(point)
        if self.bed_polygon_image_size is not None and (
            len(self.bed_polygon_image_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.bed_polygon_image_size
            )
        ):
            raise ValueError("bed_polygon_image_size must contain positive integers")
        if self.source_event != "frame" and self.tracks:
            raise ValueError("control rows must not contain tracks")
        if not isinstance(self.night_window_active, bool):
            raise TypeError("night_window_active must be boolean")
        if len({track.track_id for track in self.tracks}) != len(self.tracks):
            raise ValueError("track_id values must be unique per row")


def _unit_box(box: tuple[float, float, float, float, float]) -> None:
    _unit_coordinates(box)
    if box[0] > box[2] or box[1] > box[3]:
        raise ValueError("bbox corners must be ordered")


def _unit_coordinates(values: tuple[float, ...]) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, float)
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("coordinates and confidence must be finite unit floats")


def encode_jsonl(header: ReplayTraceHeader, rows: list[ReplayRow]) -> str:
    return "".join(
        json.dumps(asdict(item), separators=(",", ":")) + "\n" for item in (header, *rows)
    )


def decode_jsonl(text: str) -> tuple[ReplayTraceHeader, tuple[ReplayRow, ...]]:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError("missing replay trace header")
    try:
        header = ReplayTraceHeader(**json.loads(lines[0]))
        rows = []
        for line in lines[1:]:
            data = json.loads(line)
            data["tracks"] = tuple(
                ReplayTrack(
                    track_id=item["track_id"],
                    lifecycle=item["lifecycle"],
                    bbox=tuple(item["bbox"]),
                    keypoints=tuple(tuple(point) for point in item["keypoints"]),
                )
                for item in data["tracks"]
            )
            if data["bed_polygon"] is not None:
                data["bed_polygon"] = tuple(tuple(point) for point in data["bed_polygon"])
            rows.append(ReplayRow(**data))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid replay trace JSONL") from exc
    return header, tuple(rows)


def encode_document(header: ReplayTraceHeader, rows: list[ReplayRow]) -> str:
    """Encode a repository-safe replay fixture document."""
    return json.dumps(
        {"header": asdict(header), "rows": [asdict(row) for row in rows]},
        separators=(",", ":"),
    )


def decode_document(text: str) -> tuple[ReplayTraceHeader, tuple[ReplayRow, ...]]:
    """Decode a repository fixture document without changing capture JSONL."""
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return _invalid_document()
        return decode_jsonl(
            "\n".join(
                (
                    json.dumps(payload["header"]),
                    *(json.dumps(row) for row in payload["rows"]),
                )
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid replay trace document") from exc


def _invalid_document() -> tuple[ReplayTraceHeader, tuple[ReplayRow, ...]]:
    raise TypeError("replay document must be an object")


__all__ = [
    "REPLAY_TRACE_VERSION",
    "ReplayRow",
    "ReplaySource",
    "ReplayTraceHeader",
    "ReplayTrack",
    "SourceEvent",
    "TrackLifecycle",
    "decode_document",
    "decode_jsonl",
    "encode_document",
    "encode_jsonl",
]
