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

from dataclasses import dataclass

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.pipeline.trace.models import AnalysisTrace
from worker.types import DecisionInput

_MAX_TRACKER_MISSES = 30


@dataclass(slots=True)
class _LiveTrackWindow:
    """Deterministic replay of ``GreedyIouTracker.live_ids`` membership only.

    The real tracker assigns ids from raw pixel IoU; replay does not have
    (and must not need) pixel data. But every persisted ``TracePerson`` already
    carries its assigned id, so only *liveness* (has this id been seen within
    ``max_misses`` frames) needs to be replayed, not assignment.
    """

    max_misses: int = _MAX_TRACKER_MISSES
    _misses: dict[int, int] | None = None

    def __post_init__(self) -> None:
        self._misses = {}

    def update(self, seen_ids: frozenset[int]) -> tuple[int, ...]:
        assert self._misses is not None
        for track_id in seen_ids:
            self._misses[track_id] = 0
        for track_id in tuple(self._misses):
            if track_id not in seen_ids:
                self._misses[track_id] += 1
                if self._misses[track_id] > self.max_misses:
                    del self._misses[track_id]
        return tuple(sorted(self._misses))


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


__all__ = ["_LiveTrackWindow", "analysis_trace_to_decision_input", "replayed_track_id"]
