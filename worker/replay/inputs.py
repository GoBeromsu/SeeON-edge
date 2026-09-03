"""Reconstruct image-free decider inputs from persisted analysis traces.

Replay never re-runs extraction (pose/person/bed) -- those already ran once,
on the real frame, and their numeric result is what ``AnalysisTrace`` froze.
Re-running them here would require a network serving client (forbidden for
replay) and would not be "deterministic replay of captured traces" but a
second, independent inference pass. Replay instead reconstructs the exact
``FrameObservation``/``DecisionInput`` the original decider saw and re-runs
only the camera-local decider (classifier + latch/monitor) against a chosen
policy revision, so any difference is attributable to the module/policy
under test, not to nondeterministic extraction.
"""

from __future__ import annotations

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from contracts.replay_trace import ReplayRow
from worker.pipeline.trace.models import AnalysisTrace
from worker.types import DecisionInput


def analysis_trace_to_decision_input(
    trace: AnalysisTrace, *, live_track_ids: tuple[int, ...]
) -> DecisionInput:
    """Rebuild an old persisted analysis frame for the surviving HTTP replay."""
    boxes = tuple(
        BoundingBox(*person.box, confidence=person.confidence) for person in trace.persons
    )
    poses = tuple(
        tuple((point.x, point.y, point.confidence) for point in person.keypoints)
        for person in trace.persons
    )
    bed_boxes = tuple(
        BoundingBox(*bed.box, confidence=bed.confidence, polygon=bed.polygon or None)
        for bed in trace.beds
    )
    observation = FrameObservation(
        detections=(boxes, ()),
        poses=poses,
        regions=(bed_boxes, ()),
        track_ids=tuple(replayed_track_id(person.track_id.value) for person in trace.persons),
    )
    return DecisionInput(
        observation=observation,
        frame_width=trace.frame_width,
        frame_height=trace.frame_height,
        live_track_ids=live_track_ids,
        time_sec=trace.source_time.value,
        frame_index=trace.frame_key[3],
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState(trace.bed_region_provenance)),
    )


def replayed_track_id(value: int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("persisted track_id must be an integer")
    return value


def replay_trace_to_decision_input(
    row: ReplayRow | None,
    *,
    pts_ns: int | None = None,
    seq: int = 0,
) -> DecisionInput:
    """Build a decider input from one declared replay-trace-v2 frame.

    V2 stores normalized pose/bbox56 rows, so replay uses a fixed virtual
    1000x1000 frame.  Feature math receives the same normalized geometry
    without requiring source pixels or an inference adapter.
    """
    width = height = 1000
    boxes = []
    poses = []
    track_ids = []
    live_ids = []
    for track in row.tracks if row is not None else ():
        x1, y1, x2, y2, confidence = track.bbox
        boxes.append(
            BoundingBox(
                round(x1),
                round(y1),
                round(x2),
                round(y2),
                confidence=confidence,
            )
        )
        poses.append(track.keypoints)
        track_ids.append(track.track_id)
        if track.lifecycle in ("new", "tracked"):
            live_ids.append(track.track_id)
    bed_boxes = ()
    if row is not None and row.bed_polygon is not None:
        polygon = tuple((round(x), round(y)) for x, y in row.bed_polygon)
        xs, ys = zip(*polygon, strict=True)
        bed_boxes = (
            BoundingBox(
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                confidence=1.0,
                polygon=polygon,
            ),
        )
    observation = FrameObservation(
        detections=(tuple(boxes), ()),
        poses=tuple(poses),
        regions=(bed_boxes, ()),
        track_ids=tuple(track_ids),
    )
    return DecisionInput(
        observation=observation,
        frame_width=width,
        frame_height=height,
        live_track_ids=tuple(sorted(live_ids)),
        time_sec=(row.pts_ns if row is not None else pts_ns) / 1_000_000_000,
        frame_index=seq,
        bed_region=BedRegionDebugSnapshot(
            source=BedRegionCacheState.FRESH if bed_boxes else BedRegionCacheState.EMPTY
        ),
    )


__all__ = [
    "analysis_trace_to_decision_input",
    "replay_trace_to_decision_input",
    "replayed_track_id",
]
