"""Versioned frame-level replay trace contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class ReplayTraceHeader:
    version: str = REPLAY_TRACE_VERSION

    def __post_init__(self) -> None:
        if self.version != REPLAY_TRACE_VERSION:
            raise ValueError("unsupported replay trace version")


@dataclass(frozen=True, slots=True)
class ReplayRow:
    camera_id: str
    pts_ns: int
    epoch: int
    source_event: SourceEvent
    source: ReplaySource
    tracks: tuple[ReplayTrack, ...]
    bed_polygon_id: str | None
    bed_polygon: tuple[tuple[float, float], ...] | None
    night_window_active: bool

    def __post_init__(self) -> None:
        if not self.camera_id or self.pts_ns < 0 or self.epoch < 0:
            raise ValueError("camera_id and non-negative pts_ns/epoch are required")
        if self.source_event not in ("open", "frame", "reconnect", "lost"):
            raise ValueError("invalid source_event")
        if self.source not in ("legacy-association", "nvdcf"):
            raise ValueError("invalid source")
        if self.bed_polygon is not None and (
            len(self.bed_polygon) < 3 or any(len(point) != 2 for point in self.bed_polygon)
        ):
            raise ValueError("bed_polygon must contain at least three xy points")


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
