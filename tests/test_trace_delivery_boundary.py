from __future__ import annotations

import json
from base64 import b64decode
from pathlib import Path

from shared.events.delivery_queue import DeliveryQueue, EventEntry
from worker.pipeline.trace import AnalysisTrace, DetailUnavailableReason, OptionalNumber, TraceFrame
from worker.pipeline.trace.store import TraceStore


def _frame(sequence: int) -> TraceFrame:
    analysis = AnalysisTrace(
        trace_id=f"analysis-{sequence}",
        frame_key=("boot-a", "camera-a", 1, sequence),
        pts=OptionalNumber(float(sequence)),
        source_time=OptionalNumber(float(sequence)),
        frame_width=4,
        frame_height=4,
        bed_region_provenance="fresh",
        persons=(),
        beds=(),
        components=(),
    )
    return TraceFrame(analysis, ())


def test_decision_basis_is_admitted_with_event_while_detail_drop_is_observable(
    tmp_path: Path,
) -> None:
    decision_trace = json.dumps({"reason": "fall-onset"}, sort_keys=True).encode()
    values = json.dumps({"fall_probability": 0.98}, sort_keys=True).encode()
    queue = DeliveryQueue(tmp_path / "delivery")

    admitted = queue.try_admit(
        EventEntry(
            edge_event_id="event-a",
            event_type="fall",
            detected_at="2026-08-21T00:00:00Z",
            camera_id="camera-a",
            facility_id="facility-a",
            decision_trace=decision_trace,
            values=values,
        )
    )

    assert admitted.accepted
    entry = next(queue.entries())
    assert b64decode(str(entry["decision_trace_b64"])) == decision_trace
    assert b64decode(str(entry["values_b64"])) == values

    store = TraceStore(tmp_path / "detail-cache")
    store.persist_batch(
        (_frame(1), _frame(2)),
        max_frames_per_camera=1,
        max_age_seconds=60.0,
        max_cameras=1,
        max_total_frames=1,
        max_total_rows=8,
        max_total_bytes=8_000,
        dropped_by_camera={},
    )

    recovered = store.recover_camera("camera-a")
    assert [frame.frame_key[-1] for frame in recovered.frames] == [2]
    assert recovered.truncation.detail_unavailable_reason is DetailUnavailableReason.RETENTION_BOUND
