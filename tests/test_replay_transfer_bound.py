"""A full retained timeline must actually transfer, not just a small fixture.

The replay body cap was a bare 4 MiB constant while trace retention permitted
3,000 frames per camera. A fully retained timeline measures 8.36 MiB, so the
long window a fall investigation needs was refused at the boundary while short
traces sailed through. A cap that silently excludes the interesting inputs is a
fake capability: the tests pass, the feature demos, and the one case it exists
for fails in production.

The bound is now derived from the retention limit and both ends read the same
value, so the two cannot drift. These tests pin that a maximum-size timeline
fits, and that the derivation is real rather than a coincidentally larger
literal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shared.events.replay_wire import (
    MAX_REPLAY_BODY_BYTES,
    MAX_TRACE_FRAME_BYTES,
    MAX_TRACE_FRAMES,
    ReplayTrace,
)
from worker.pipeline.output._mjpeg_http import MAX_REPLAY_BODY_BYTES as WORKER_CAP

_TRUNCATION: dict[str, Any] = {
    "handoff_dropped_frames": 0,
    "pruned_frames": 0,
    "persistence_failed_frames": 0,
    "retention_blocked_frames": 0,
    "oldest_retained_seq": 0,
    "newest_retained_seq": MAX_TRACE_FRAMES - 1,
    "oldest_retained_key": None,
    "newest_retained_key": None,
    "detail_unavailable_reason": None,
}


def _frame(index: int) -> dict[str, Any]:
    """A densely populated frame: two tracked people, a bed, three components."""
    return {
        "trace_id": f"trace-{index:06d}",
        "frame_key": f"key-{index:06d}",
        "pts": index * 33_333,
        "source_time": {"value": 1_787_000_000.0 + index * 0.033, "missing_reason": None},
        "frame_width": 1920,
        "frame_height": 1080,
        "bed_region_provenance": "persisted",
        "persons": [
            {
                "ordinal": person,
                "track_id": {"value": person, "missing_reason": None},
                "box": [10.0, 20.0, 30.0, 40.0],
                "confidence": 0.91,
                "keypoints": [
                    {
                        "ordinal": point,
                        "name": f"kp{point}",
                        "x": 1.0 * point,
                        "y": 2.0 * point,
                        "confidence": 0.8,
                    }
                    for point in range(17)
                ],
            }
            for person in range(2)
        ],
        "beds": [
            {
                "ordinal": 0,
                "box": [0.0, 0.0, 100.0, 100.0],
                "confidence": 0.99,
                "provenance": "persisted",
                "polygon": [[float(point), float(point)] for point in range(4)],
            }
        ],
        "components": [
            {
                "ordinal": component,
                "component_id": f"comp-{component}",
                "observation_state": "observed",
            }
            for component in range(3)
        ],
    }


def _timeline(frames: int) -> ReplayTrace:
    return ReplayTrace(
        camera_id="camera-1",
        frames=tuple(_frame(index) for index in range(frames)),
        truncation=dict(_TRUNCATION),
    )


def test_a_fully_retained_timeline_fits_the_transfer_bound() -> None:
    """The maximum retention window must transfer, or replay cannot see a fall."""
    encoded = _timeline(MAX_TRACE_FRAMES).canonical_json().encode()

    assert len(encoded) <= MAX_REPLAY_BODY_BYTES, (
        f"a full {MAX_TRACE_FRAMES}-frame timeline serializes to {len(encoded):,} bytes "
        f"but the transfer bound is {MAX_REPLAY_BODY_BYTES:,}; the long windows replay "
        f"exists for would be refused while short traces succeed"
    )


def test_the_bound_is_derived_from_retention_not_chosen() -> None:
    """A literal would drift the moment retention changes."""
    assert MAX_REPLAY_BODY_BYTES == MAX_TRACE_FRAMES * MAX_TRACE_FRAME_BYTES


def test_both_ends_read_the_same_bound() -> None:
    """A worker that sends more than the backend accepts loses the trace."""
    assert WORKER_CAP == MAX_REPLAY_BODY_BYTES


@pytest.mark.parametrize("frames", [1, 100, 1_000, MAX_TRACE_FRAMES])
def test_timelines_across_the_retention_range_all_fit(frames: int) -> None:
    """Not just the extremes: nothing in the permitted range may be refused."""
    encoded = _timeline(frames).canonical_json().encode()

    assert len(encoded) <= MAX_REPLAY_BODY_BYTES


def test_the_per_frame_bound_is_not_optimistic() -> None:
    """Guard the derivation: a too-small per-frame figure makes the cap a lie."""
    encoded = _timeline(MAX_TRACE_FRAMES).canonical_json().encode()
    observed_per_frame = len(encoded) / MAX_TRACE_FRAMES

    assert observed_per_frame <= MAX_TRACE_FRAME_BYTES, (
        f"a dense frame serializes to {observed_per_frame:,.0f} bytes, above the "
        f"declared {MAX_TRACE_FRAME_BYTES:,}; the derived cap would then be too "
        f"small for a real timeline"
    )


def test_persisted_analysis_recovery_is_retired() -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.qa.runtime_trace_store")


def test_schema18_does_not_grow_runtime_analysis_tables(tmp_path: Path) -> None:
    import sqlite3

    from backend.app.edge_db.migrator import migrate_database

    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert not any(name.startswith("runtime_analysis_") for name in tables)
