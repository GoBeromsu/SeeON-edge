"""Shared valid replay-trace payload.

Extracted so the HTTP suite and the ingest-conflict suite exercise the same
known-good shape. A hand-built variant drifts from the schema's CHECK
constraints and ends up testing the fixture rather than the code.
"""

from __future__ import annotations


def valid_trace_payload() -> dict[str, object]:
    return {
        "camera_id": "camera-replay-http",
        "frames": [
            {
                "trace_id": "a" * 64,
                "frame_key": ["boot-replay-http", "camera-replay-http", 1, 1],
                "pts": {"value": 1.0, "missing_reason": None},
                "source_time": {"value": 1.0, "missing_reason": None},
                "frame_width": 640,
                "frame_height": 480,
                "bed_region_provenance": "empty",
                "persons": [],
                "beds": [],
                "components": [],
                "schema_version": 1,
            }
        ],
        "truncation": {
            "handoff_dropped_frames": 0,
            "pruned_frames": 0,
            "persistence_failed_frames": 0,
            "retention_blocked_frames": 0,
            "oldest_retained_seq": 1,
            "newest_retained_seq": 1,
            "oldest_retained_key": ["boot-replay-http", "camera-replay-http", 1, 1],
            "newest_retained_key": ["boot-replay-http", "camera-replay-http", 1, 1],
            "detail_unavailable_reason": None,
        },
    }


def valid_trace_payload_with_children() -> dict[str, object]:
    """The same shape, populated with children.

    The minimal payload carries no persons, beds or components, so it cannot
    exercise the child-merge hazard at all: an ingest that merged new child
    ordinals into a stored frame would look perfectly correct against it.
    """
    payload = valid_trace_payload()
    frame = payload["frames"][0]  # type: ignore[index]
    frame["persons"] = [  # type: ignore[index]
        {
            "ordinal": 0,
            "track_id": {"value": 7, "missing_reason": None},
            "box": [10.0, 20.0, 30.0, 40.0],
            "confidence": 0.91,
            "keypoints": [{"index": 0, "x": 1.0, "y": 2.0, "confidence": 0.8}],
        }
    ]
    return payload
