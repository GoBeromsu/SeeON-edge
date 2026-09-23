"""Decision-path invariance with execution-record sink on versus off."""

from __future__ import annotations

from dataclasses import replace

from test_execution_record_wiring import _metadata, _pump

from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.types.business_event import BusinessEvent


def _comparable_event(event: object) -> BusinessEvent:
    assert isinstance(event, BusinessEvent)
    audit = None
    if event.audit is not None:
        audit = {
            key: value for key, value in dict(event.audit).items() if key != "decision_trace_id"
        }
    return replace(event, identity="*", audit=audit or None)


def test_d1_identical_frames_emit_equal_events_and_snapshots() -> None:
    off_events: list[object] = []
    on_events: list[object] = []
    off_pump = _pump(None, emitted=off_events, fall_transition=0.9)
    on_lanes = ExecutionRecordLanes(lane_capacity=64)
    on_pump = _pump(on_lanes, emitted=on_events, fall_transition=0.9)
    for seq in range(3):
        pts = 100 + seq * 66_666_667
        off_pump._process(_metadata(child=off_pump._child, seq=seq, pts=pts))
        on_pump._process(_metadata(child=on_pump._child, seq=seq, pts=pts))
    assert [_comparable_event(event) for event in off_events] == [
        _comparable_event(event) for event in on_events
    ]
    assert off_pump._decision.last_trace_snapshots == on_pump._decision.last_trace_snapshots
    assert on_lanes.queued() > 0
