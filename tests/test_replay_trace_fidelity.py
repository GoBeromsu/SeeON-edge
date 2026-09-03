from pathlib import Path

from contracts.replay_trace import decode_jsonl
from shared.detection_policies import BedExitPolicyV1, make_effective_policy
from worker.replay.engine import replay_trace, replay_trace_frames

FIXTURES = Path("tests_support/fixtures")


def _rows(name: str):
    return decode_jsonl((FIXTURES / name).read_text())[1]


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
    rows = _rows(name)
    return replay_trace(camera_id="fixture", rows=rows, module_id="bed_exit", policy=_policy())


def test_pts_gap_fixture_is_filled_by_replay_resampler() -> None:
    frames = replay_trace_frames(_rows("replay-pts-gap-v2.jsonl"))
    assert [frame.valid for frame in frames] == [1, 0, 0, 1]
    assert [frame.valid for frame in _run("replay-pts-gap-v2.jsonl").frames] == [1, 0, 0, 1]


def test_reconnect_epoch_resets_the_replay_vehicle() -> None:
    run = _run("replay-reconnect-v2.jsonl")
    assert [frame.stream_epoch for frame in run.frames] == [0, 1]
    assert [frame.seq for frame in run.frames] == [0, 0]
    assert run.boot_ids == ("epoch-0", "epoch-1")


def test_id_switch_fixture_preserves_declared_new_track() -> None:
    frames = replay_trace_frames(_rows("replay-id-switch-v2.jsonl"))
    assert [frame.rows[0].track_id for frame in frames] == [1, 2]
    assert [frame.rows[0].track_lifecycle for frame in frames] == ["new", "new"]
    assert [frame.seq for frame in _run("replay-id-switch-v2.jsonl").frames] == [0, 1]
