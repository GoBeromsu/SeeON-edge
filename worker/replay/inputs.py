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
from worker.pipeline.perception import SceneState, build_decision_input, build_frame_observation
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

    V2 stores unit coordinates plus the real source frame size, so replay
    projects every fact back into the exact pixel space production observed:
    integer boxes and polygons for the bed rasteriser, float keypoints for the
    pose head. pose+bbox56 rows therefore round-trip byte for byte.
    """
    width = row.frame_width if row is not None else 1
    height = row.frame_height if row is not None else 1
    boxes = []
    poses = []
    track_ids = []
    live_ids = []
    for track in row.tracks if row is not None else ():
        if track.lifecycle not in ("new", "tracked"):
            if track.lifecycle == "shadow":
                live_ids.append(track.track_id)
            continue
        x1, y1, x2, y2, confidence = track.bbox
        boxes.append(
            BoundingBox(
                round(x1 * width),
                round(y1 * height),
                round(x2 * width),
                round(y2 * height),
                confidence=confidence,
            )
        )
        poses.append(tuple((x * width, y * height, score) for x, y, score in track.keypoints))
        track_ids.append(track.track_id)
        live_ids.append(track.track_id)
    bed_boxes = ()
    if row is not None and row.bed_polygon is not None:
        assert row.bed_polygon_image_size is not None
        polygon_width, polygon_height = row.bed_polygon_image_size
        polygon = tuple(
            (round(x * polygon_width), round(y * polygon_height)) for x, y in row.bed_polygon
        )
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
    observation = build_frame_observation(
        boxes=tuple(boxes),
        poses=tuple(poses),
        bed_boxes=(),
        track_ids=tuple(track_ids),
    )
    scene = SceneState(
        camera_id=row.camera_id if row is not None else "replay",
        persisted_bed_regions=bed_boxes,
        bed_zone_image_width=polygon_width if bed_boxes else None,
        bed_zone_image_height=polygon_height if bed_boxes else None,
    )
    return build_decision_input(
        observation,
        frame_width=width,
        frame_height=height,
        live_track_ids=tuple(sorted(live_ids)),
        time_sec=(row.pts_ns if row is not None else pts_ns) / 1_000_000_000,
        frame_index=seq,
        scene_state=scene,
        bed_scheduled=False,
        bed_interval=1,
    )


__all__ = [
    "analysis_trace_to_decision_input",
    "replay_trace_to_decision_input",
    "replayed_track_id",
]
