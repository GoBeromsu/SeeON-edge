"""Strict, image-free wire representation for backend-owned replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Final


class ReplayWireError(ValueError):
    """The replay request is incomplete or not a faithful trace representation."""


@dataclass(frozen=True, slots=True)
class ReplayTrace:
    camera_id: str
    frames: tuple[dict[str, object], ...]
    truncation: dict[str, object]

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ReplayWireError("camera_id is required")
        if not self.frames:
            raise ReplayWireError("replay requires at least one captured frame")
        required_truncation = {
            "handoff_dropped_frames",
            "pruned_frames",
            "persistence_failed_frames",
            "retention_blocked_frames",
            "oldest_retained_seq",
            "newest_retained_seq",
            "oldest_retained_key",
            "newest_retained_key",
            "detail_unavailable_reason",
        }
        if set(self.truncation) != required_truncation:
            raise ReplayWireError("truncation fields are incomplete")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def as_dict(self) -> dict[str, object]:
        decoded: dict[str, object] = json.loads(self.canonical_json())
        return decoded


def decode_replay_trace(payload: object) -> ReplayTrace:
    if not isinstance(payload, dict) or set(payload) != {"camera_id", "frames", "truncation"}:
        raise ReplayWireError("replay payload has undeclared or missing fields")
    camera_id = payload["camera_id"]
    frames = payload["frames"]
    truncation = payload["truncation"]
    if (
        not isinstance(camera_id, str)
        or not isinstance(frames, list)
        or not isinstance(truncation, dict)
    ):
        raise ReplayWireError("replay payload has invalid field types")
    normalized_frames: list[dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ReplayWireError("frame must be an object")
        normalized_frames.append(frame)
    return ReplayTrace(camera_id, tuple(normalized_frames), truncation)


#: Frames a single camera may retain, mirroring `DEFAULT_TRACE_RETENTION_POLICY`.
#: Declared here because both the worker sender and the backend receiver must
#: agree, and neither may import the other.
MAX_TRACE_FRAMES: Final = 3_000

#: Measured upper bound for one serialized frame. A frame carrying two persons
#: with seventeen keypoints each, one bed with a four-point polygon, and three
#: components serializes to about 2.9 KiB; this rounds up and leaves room for a
#: denser scene without inviting unbounded growth.
MAX_TRACE_FRAME_BYTES: Final = 6 * 1024

#: Transfer bound, DERIVED rather than chosen. It previously was a bare
#: 4 MiB constant while retention permitted 3,000 frames, and a full retained
#: timeline measures 8.36 MiB -- so exactly the long window a fall investigation
#: needs was refused at the boundary while short traces succeeded. A cap that
#: silently excludes the interesting inputs is a fake capability, so this is
#: computed from the retention bound and cannot drift away from it.
MAX_REPLAY_BODY_BYTES: Final = MAX_TRACE_FRAMES * MAX_TRACE_FRAME_BYTES

__all__ = [
    "MAX_REPLAY_BODY_BYTES",
    "MAX_TRACE_FRAMES",
    "MAX_TRACE_FRAME_BYTES",
    "ReplayTrace",
    "ReplayWireError",
    "decode_replay_trace",
]

