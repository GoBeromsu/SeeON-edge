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
from contracts.replay_trace import ReplayTraceRow
from worker.pipeline.trace.models import AnalysisTrace
from worker.types import DecisionInput


def analysis_trace_to_decision_input(
    trace: AnalysisTrace,
    *,
    live_track_ids: tuple[int, ...],
) -> DecisionInput:
    """Rebuild the exact ``DecisionInput`` a compiled decider originally saw.

    Inverse of ``worker.pipeline.trace.capture.TraceCapture._analysis``: every
    field it read off ``FrameObservation``/``DecisionInput`` is reconstructed
    here from the persisted, image-free row. ``track_ids`` reconstructs
    ``OptionalNumber(None, "tracker-unmatched")`` persons as unmatched (``None``)
    entries, exactly mirroring the original capture.
    """
    boxes = tuple(
        BoundingBox(*person.box, confidence=person.confidence) for person in trace.persons
    )
    labels: tuple[object, ...] = ()
    poses = tuple(
        tuple((point.x, point.y, point.confidence) for point in person.keypoints)
        for person in trace.persons
    )
    bed_boxes = tuple(
        BoundingBox(*bed.box, confidence=bed.confidence, polygon=bed.polygon or None)
        for bed in trace.beds
    )
    track_ids = tuple(replayed_track_id(person.track_id.value) for person in trace.persons)
    observation = FrameObservation(
        detections=(boxes, labels),
        poses=poses,
        regions=(bed_boxes, ()),
        track_ids=track_ids,
    )
    bed_region = BedRegionDebugSnapshot(source=BedRegionCacheState(trace.bed_region_provenance))
    return DecisionInput(
        observation=observation,
        frame_width=trace.frame_width,
        frame_height=trace.frame_height,
        live_track_ids=live_track_ids,
        time_sec=trace.source_time.value,
        frame_index=trace.frame_key[3],
        bed_region=bed_region,
    )


def replayed_track_id(value: int | float | None) -> int | None:
    """Narrow a persisted numeric track_id back to its stored integer type."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("persisted track_id must be an integer")
    return value


def replay_trace_to_decision_input(
    rows: tuple[ReplayTraceRow, ...],
    *,
    pts_ns: int,
    seq: int,
) -> DecisionInput:
    """Build a decider input from one declared replay-trace-v2 frame.

    V2 stores normalized pose/bbox56 rows, so replay uses a fixed virtual
    1000x1000 frame.  Feature math receives the same normalized geometry
    without requiring source pixels or an inference adapter.
    """
    if not rows:
        raise ValueError("a replay frame requires at least one track row")
    width = height = 1000
    boxes = []
    poses = []
    track_ids = []
    live_ids = []
    for row in rows:
        feature = row.pose_bbox56
        x1, y1, x2, y2 = (value * width for value in feature[51:55])
        boxes.append(BoundingBox(x1, y1, x2, y2, confidence=feature[55]))
        poses.append(tuple((feature[index] * width, feature[index + 1] * height, feature[index + 2])
                           for index in range(0, 51, 3)))
        track_ids.append(row.track_id)
        if row.track_lifecycle in ("new", "tracked"):
            live_ids.append(row.track_id)
    observation = FrameObservation(
        detections=(tuple(boxes), ()),
        poses=tuple(poses),
        regions=((), ()),
        track_ids=tuple(track_ids),
    )
    return DecisionInput(
        observation=observation,
        frame_width=width,
        frame_height=height,
        live_track_ids=tuple(sorted(live_ids)),
        time_sec=pts_ns / 1_000_000_000,
        frame_index=seq,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.EMPTY),
    )


__all__ = [
    "analysis_trace_to_decision_input",
    "replay_trace_to_decision_input",
    "replayed_track_id",
]
