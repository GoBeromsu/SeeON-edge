"""Bounded worker-to-backend delivery of image-free analysis traces."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Final

from shared.events.evidence_http_transport import (
    bounded_request,
    encode_json,
    join_http_url,
    normalize_http_base,
)
from shared.events.replay_wire import ReplayTrace
from worker.pipeline.trace.models import AnalysisTrace, TraceFrame, TraceTruncation

LOGGER: Final = logging.getLogger(__name__)
ANALYSIS_TRACE_PATH: Final = "/api/v1/relay/analysis-traces"
MAX_ANALYSIS_TRACE_FRAMES_PER_TRANSFER: Final = 16


class AnalysisTraceTransferBoundError(ValueError):
    """A caller attempted to send more than the declared relay window."""


def _analysis_wire(analysis: AnalysisTrace) -> dict[str, object]:
    return {
        "trace_id": analysis.trace_id,
        "frame_key": list(analysis.frame_key),
        "pts": {"value": analysis.pts.value, "missing_reason": analysis.pts.missing_reason},
        "source_time": {
            "value": analysis.source_time.value,
            "missing_reason": analysis.source_time.missing_reason,
        },
        "frame_width": analysis.frame_width,
        "frame_height": analysis.frame_height,
        "bed_region_provenance": analysis.bed_region_provenance,
        "persons": [
            {
                "ordinal": person.ordinal,
                "track_id": {
                    "value": person.track_id.value,
                    "missing_reason": person.track_id.missing_reason,
                },
                "box": list(person.box),
                "confidence": person.confidence,
                "keypoints": [
                    {
                        "index": point.index,
                        "x": point.x,
                        "y": point.y,
                        "confidence": point.confidence,
                    }
                    for point in person.keypoints
                ],
            }
            for person in analysis.persons
        ],
        "beds": [
            {
                "ordinal": bed.ordinal,
                "box": list(bed.box),
                "confidence": bed.confidence,
                "provenance": bed.provenance,
                "polygon": [list(point) for point in bed.polygon],
            }
            for bed in analysis.beds
        ],
        "components": [
            {
                "ordinal": component.ordinal,
                "qualified_id": component.qualified_id,
                "observation_state": component.observation_state,
            }
            for component in analysis.components
        ],
        "schema_version": analysis.schema_version,
    }


class AnalysisTraceSender:
    """Post one explicit, bounded analysis window over the existing relay."""

    def __init__(
        self, relay_url: str, relay_token: str, *, request: Callable[..., object] = bounded_request
    ) -> None:
        self._url = join_http_url(normalize_http_base(relay_url), ANALYSIS_TRACE_PATH)
        self._relay_token = relay_token
        self._request = request

    def send(
        self,
        frames: tuple[TraceFrame, ...],
        truncation_by_camera: Mapping[str, TraceTruncation],
    ) -> None:
        by_camera: dict[str, list[AnalysisTrace]] = {}
        for frame in frames:
            by_camera.setdefault(frame.analysis.frame_key[1], []).append(frame.analysis)
        for camera_id, analyses in by_camera.items():
            if len(analyses) > MAX_ANALYSIS_TRACE_FRAMES_PER_TRANSFER:
                raise AnalysisTraceTransferBoundError(
                    "analysis trace transfer exceeds "
                    f"{MAX_ANALYSIS_TRACE_FRAMES_PER_TRANSFER} frames"
                )
            truncation = truncation_by_camera[camera_id]
            trace = ReplayTrace(
                camera_id=camera_id,
                frames=tuple(_analysis_wire(analysis) for analysis in analyses),
                truncation={
                    "handoff_dropped_frames": truncation.handoff_dropped_frames,
                    "pruned_frames": truncation.pruned_frames,
                    "persistence_failed_frames": truncation.persistence_failed_frames,
                    "retention_blocked_frames": truncation.retention_blocked_frames,
                    "oldest_retained_seq": truncation.oldest_retained_seq,
                    "newest_retained_seq": truncation.newest_retained_seq,
                    "oldest_retained_key": (
                        list(truncation.oldest_retained_key)
                        if truncation.oldest_retained_key is not None
                        else None
                    ),
                    "newest_retained_key": (
                        list(truncation.newest_retained_key)
                        if truncation.newest_retained_key is not None
                        else None
                    ),
                    "detail_unavailable_reason": (
                        truncation.detail_unavailable_reason.value
                        if truncation.detail_unavailable_reason is not None
                        else None
                    ),
                },
            )
            result = self._request(
                self._url,
                "POST",
                {
                    "Authorization": f"Bearer {self._relay_token}",
                    "Content-Type": "application/json",
                },
                encode_json(
                    {
                        "camera_id": trace.camera_id,
                        "frames": list(trace.frames),
                        "truncation": trace.truncation,
                    }
                ),
                2.0,
            )
            if not isinstance(result, tuple) or result[0] // 100 != 2:
                raise RuntimeError("analysis trace relay delivery failed")


__all__ = [
    "ANALYSIS_TRACE_PATH",
    "AnalysisTraceSender",
    "AnalysisTraceTransferBoundError",
    "MAX_ANALYSIS_TRACE_FRAMES_PER_TRANSFER",
]
