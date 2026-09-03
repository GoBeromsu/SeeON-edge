from pathlib import Path

from contracts.replay_trace import decode_document
from shared.detection_policies import BedExitPolicyV1, make_effective_policy
from worker.replay.engine import replay, replay_trace_frames

FIXTURES = Path("tests/fixtures/replay")


def _rows(name: str):
    return decode_document((FIXTURES / f"{name}.json").read_text())[1]


def _policy():
    return make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.5, hold_frames=1, grace_frames=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _run(name: str):
    return replay(camera_id="fixture", rows=_rows(name), module_id="bed_exit", policy=_policy())


def _alerts(run):
    return [(frame.pts_ns, event.event_type) for frame in run.frames for event in frame.events]


def test_pts_gap_changes_the_full_replay_episode_outcome() -> None:
    control = _run("gap-control-v2")
    gap = _run("gap-axis-v2")
    assert _alerts(control) == [(200_000_001, "bed-exit")]
    assert _alerts(gap) == []
    assert any(not frame.valid for frame in gap.frames)
    assert [frame.valid for frame in replay_trace_frames(_rows("gap-axis-v2"))] == [1, 1, 0, 0, 1]


def test_reconnect_keeps_camera_local_decider_state_within_one_boot() -> None:
    control = _run("reconnect-control-v2")
    reconnected = _run("reconnect-axis-v2")
    assert _alerts(control) == [(200_000_001, "bed-exit")]
    assert _alerts(reconnected) == _alerts(control)
    assert [frame.stream_epoch for frame in reconnected.frames] == [0, 0, 1, 1, 1]


def test_control_rows_are_not_replay_frames() -> None:
    _, rows = decode_document(
        Path("tests/fixtures/replay/reconnect-axis-v2.json").read_text(encoding="utf-8")
    )
    frames = replay_trace_frames(rows)
    assert all(frame.row is None or frame.row.source_event == "frame" for frame in frames)


def test_id_switch_reports_churn_and_changes_declared_episode_outcome() -> None:
    control = _run("id-switch-control-v2")
    switched = _run("id-switch-axis-v2")
    assert _alerts(control) == [(200_000_001, "bed-exit")]
    assert _alerts(switched) == [(266_666_668, "bed-exit")]
    assert switched.track_id_switch_total == 1
