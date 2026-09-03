"""Versioned JSON-lines contract for media-derived replay traces."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass
from typing import Literal

REPLAY_TRACE_VERSION = "replay-trace-v2"
ReplaySource = Literal["legacy", "nvdcf"]
TrackLifecycle = Literal["new", "tracked", "shadow", "lost"]


@dataclass(frozen=True, slots=True)
class ReplayTraceHeader:
    version: str = REPLAY_TRACE_VERSION


@dataclass(frozen=True, slots=True)
class ReplayTraceRow:
    source: ReplaySource
    camera_id: str
    stream_epoch: int
    seq: int
    pts_ns: int
    track_id: int
    track_lifecycle: TrackLifecycle
    pose_bbox56: tuple[float, ...]
    bed_contained: bool | None = None

    def __post_init__(self) -> None:
        if self.source not in ("legacy", "nvdcf"):
            raise ValueError("source must be 'legacy' or 'nvdcf'")
        if not self.camera_id or self.stream_epoch < 0 or self.seq < 0 or self.pts_ns < 0:
            raise ValueError("camera_id and non-negative epoch, seq, pts_ns are required")
        if self.track_lifecycle not in ("new", "tracked", "shadow", "lost"):
            raise ValueError("invalid track_lifecycle")
        if len(self.pose_bbox56) != 56:
            raise ValueError("pose_bbox56 must contain 56 values")
        values = tuple(_float32(value) for value in self.pose_bbox56)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose_bbox56 values must be finite")
        object.__setattr__(self, "pose_bbox56", values)


def encode_jsonl(header: ReplayTraceHeader, rows: list[ReplayTraceRow]) -> str:
    """Encode a complete trace as newline-delimited canonical JSON."""
    if header.version != REPLAY_TRACE_VERSION:
        raise ValueError("unsupported replay trace version")
    payloads = (asdict(header), *(asdict(row) for row in rows))
    return "".join(
        json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n" for item in payloads
    )


def decode_jsonl(text: str) -> tuple[ReplayTraceHeader, tuple[ReplayTraceRow, ...]]:
    """Decode and validate a complete replay-trace-v2 document."""
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError("missing replay trace header")
    try:
        header = ReplayTraceHeader(**json.loads(lines[0]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid replay trace header") from exc
    if header.version != REPLAY_TRACE_VERSION:
        raise ValueError("unsupported replay trace version")
    rows: list[ReplayTraceRow] = []
    for line in lines[1:]:
        try:
            data = json.loads(line)
            data["pose_bbox56"] = tuple(data["pose_bbox56"])
            rows.append(ReplayTraceRow(**data))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid replay trace row") from exc
    return header, tuple(rows)


def _float32(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("pose_bbox56 values must be numbers")
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


__all__ = [
    "REPLAY_TRACE_VERSION",
    "ReplaySource",
    "ReplayTraceHeader",
    "ReplayTraceRow",
    "TrackLifecycle",
    "decode_jsonl",
    "encode_jsonl",
]
