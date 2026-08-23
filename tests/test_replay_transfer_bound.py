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

from backend.app.features.relay.router import MAX_RELAY_ANALYSIS_TRACE_BODY_BYTES
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
    assert MAX_RELAY_ANALYSIS_TRACE_BODY_BYTES == MAX_REPLAY_BODY_BYTES


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


def test_recovery_is_bounded_to_the_retention_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery must not return unbounded history.

    `recover()` selected every row for a camera with no limit. On a long-running
    deployment that returns far more than the retention window and more than the
    transfer bound permits, so the replay request would be refused at the
    boundary for a camera whose recent history was perfectly recoverable.

    The limit is patched down rather than seeding three thousand rows: the
    property under test is that the query is bounded by the declared constant,
    not the size of the constant.
    """
    import sqlite3

    from replay_fixtures import valid_trace_payload

    from backend.app.edge_db.migrator import migrate_database
    from backend.app.features.qa import runtime_trace_store as store_module
    from shared.events.replay_wire import decode_replay_trace

    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    store = store_module.RuntimeAnalysisStore(database)

    payload = valid_trace_payload()
    camera_id = str(payload["camera_id"])
    template = payload["frames"][0]  # type: ignore[index]
    for index in range(5):
        frame = {**template}
        frame["trace_id"] = f"{index:064d}"
        frame["frame_key"] = [*template["frame_key"][:3], index + 1]  # type: ignore[index]
        store.ingest(
            decode_replay_trace({**payload, "frames": [frame]})
        )

    with sqlite3.connect(database) as connection:
        seeded = connection.execute(
            "SELECT COUNT(*) FROM runtime_analysis_traces WHERE camera_id = ?", (camera_id,)
        ).fetchone()[0]
    assert seeded == 5, "the fixture did not seed distinct frames"

    monkeypatch.setattr(store_module, "MAX_TRACE_FRAMES", 2)
    recovered = store.recover(camera_id)

    assert len(recovered.frames) == 2, (
        f"recovery returned {len(recovered.frames)} frames against a declared "
        f"window of 2; an unbounded query returns more than the transfer bound "
        f"can carry and the replay request is refused at the boundary"
    )


def test_backend_trace_storage_is_bounded_not_unbounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingest must prune, or the edge database grows forever.

    `runtime_analysis_*` carried no DELETE at all, so every camera publishing
    analysis traces grew the edge database without bound -- unbounded growth of
    the very database this ownership change centres on, on a device with finite
    storage. Recovery is limited to the retention window, so anything older is
    storage nobody can read.
    """
    import sqlite3

    from replay_fixtures import valid_trace_payload

    from backend.app.edge_db.migrator import migrate_database
    from backend.app.features.qa import runtime_trace_store as store_module
    from shared.events.replay_wire import decode_replay_trace

    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    monkeypatch.setattr(store_module, "MAX_TRACE_FRAMES", 3)
    store = store_module.RuntimeAnalysisStore(database)

    payload = valid_trace_payload()
    camera_id = str(payload["camera_id"])
    template = payload["frames"][0]  # type: ignore[index]
    for index in range(8):
        frame = {**template}
        frame["trace_id"] = f"{index:064d}"
        frame["frame_key"] = [*template["frame_key"][:3], index + 1]  # type: ignore[index]
        store.ingest(decode_replay_trace({**payload, "frames": [frame]}))

    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT COUNT(*) FROM runtime_analysis_traces WHERE camera_id = ?", (camera_id,)
        ).fetchone()[0]

    assert stored == 3, (
        f"{stored} frames retained against a window of 3; ingest is not pruning "
        f"and the edge database grows without bound"
    )

    # Pruning must trim the front, never the middle: a late-starting window is
    # replayable, a holed one is refused.
    recovered = store.recover(camera_id)
    sequences = [frame["frame_key"][3] for frame in recovered.frames]  # type: ignore[index]
    assert sequences == [6, 7, 8], (
        f"retained {sequences}; pruning must keep the newest contiguous window"
    )
