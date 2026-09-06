"""Vendor-neutral association observation (P1b, G8b).

The media plane's tracker assigns identities; the worker consumes them as an
``AssociationObservation`` per accepted frame and never re-tracks. The P1a
episode module and V2 decider read ``PerceptionFrameV1.association`` built from
this observation; replay traces carry ``source: "nvdcf"``.

Row-to-track linkage is deterministic and one-to-one: a tracked object either
owns exactly one pose row for this frame or is explicitly ``pose_row=None``
(shadow/coasted track, missed detection). A row is never shared between two
tracks; ambiguity is a visible ``unmatched`` count, not a borrowed pose.
"""

from __future__ import annotations

from dataclasses import dataclass

from worker.types.perception_frame import PerceptionFrameIdentity

NVDCF_STRATEGY = "nvdcf"


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    confidence: float
    pose_row: int | None  # index into the frame's pose rows, or None


@dataclass(frozen=True, slots=True)
class AssociationObservation:
    identity: PerceptionFrameIdentity
    strategy: str
    tracks: tuple[TrackedObject, ...]
    live_track_ids: tuple[int, ...]
    unmatched_tracks: int
    rows_available: int
