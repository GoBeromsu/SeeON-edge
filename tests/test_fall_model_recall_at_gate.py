from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from contracts.replay_trace import ReplayRow, ReplayTraceHeader, ReplayTrack, encode_jsonl
from worker.replay.engine import ReplayFrameResult, ReplayRun
from worker.types import BusinessEvent, DecisionTraceSnapshot


@pytest.fixture(scope="module")
def recall_script() -> ModuleType:
    path = Path("scripts/qa/fall_model_recall_at_gate.py")
    spec = importlib.util.spec_from_file_location("fall_model_recall_at_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track() -> ReplayTrack:
    return ReplayTrack(
        track_id=7,
        lifecycle="tracked",
        bbox=(0.1, 0.1, 0.5, 0.9, 0.9),
        keypoints=tuple((0.3, 0.3, 0.9) for _ in range(17)),
    )


def _row(camera_id: str, seq: int, pts_sec: int) -> ReplayRow:
    return ReplayRow(
        camera_id=camera_id,
        seq=seq,
        pts_ns=pts_sec * 1_000_000_000,
        epoch=0,
        source_event="frame",
        source="nvdcf",
        tracks=(_track(),),
        bed_polygon_id=None,
        bed_polygon=None,
        bed_polygon_image_size=None,
        night_window_active=False,
        frame_width=640,
        frame_height=360,
    )


def _write_trace(path: Path, camera_id: str, last_pts_sec: int) -> None:
    rows = [_row(camera_id, seq, seq) for seq in range(last_pts_sec + 1)]
    path.write_text(encode_jsonl(ReplayTraceHeader(), rows), encoding="utf-8")


def _snapshot(*, scored: float | None) -> DecisionTraceSnapshot:
    if scored is not None:
        return DecisionTraceSnapshot(
            reason="below-threshold",
            previous_state="clear",
            current_state="clear",
            triggered=False,
            track_id=7,
            bed_id=None,
            values={"fall_transition_probability": scored},
        )
    return DecisionTraceSnapshot(
        reason="below-threshold",
        previous_state="clear",
        current_state="clear",
        triggered=False,
        track_id=7,
        bed_id=None,
        missing_values={"fall_transition_probability": "classifier-warmup"},
    )


def _frame(
    pts_sec: int, *, camera_id: str, event_prob: float | None, snapshot_score: float | None
) -> ReplayFrameResult:
    events = ()
    if event_prob is not None:
        events = (
            BusinessEvent(
                domain="fall",
                event_type="fall",
                identity=f"ep-{pts_sec}",
                camera_id=camera_id,
                facility_id="replay",
                time_sec=float(pts_sec),
                probability=event_prob,
            ),
        )
    return ReplayFrameResult(
        frame_key=("replay-trace-v2:boot-0", camera_id, 0, pts_sec),
        analysis_trace_id=f"v2:0:0:{pts_sec}",
        events=events,
        snapshots=(_snapshot(scored=snapshot_score),),
        stream_epoch=0,
        seq=pts_sec,
        pts_ns=pts_sec * 1_000_000_000,
        valid=1,
    )


class _FakeRunner:
    receipt_threshold = 0.42
    promotion_eligible = True


def _fake_replay_factory(runs_by_camera: dict[str, ReplayRun]):
    def _replay(*, camera_id: str, rows, module_id: str, policy, fall_model) -> ReplayRun:
        assert module_id == "fall"
        assert fall_model is not None
        return runs_by_camera[camera_id]

    return _replay


def test_owner_fall_hits_false_positive_rate_and_window_fraction(
    tmp_path: Path, recall_script: ModuleType
) -> None:
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir / "positive.jsonl", "cam-positive", 60)
    _write_trace(traces_dir / "negative.jsonl", "cam-negative", 30)

    # Positive trace: two real owner falls (hits, at 20s and 40s) plus one
    # spurious "fall" far outside both +/-10s exclusion windows (a false
    # positive on the ground-truth trace itself).
    positive_run = ReplayRun(
        camera_id="cam-positive",
        module_qualified_id="fall.v2",
        policy_qualified_id="fall.v2",
        effective_policy_id="policy-1",
        frames=(
            _frame(0, camera_id="cam-positive", event_prob=0.9, snapshot_score=None),
            _frame(20, camera_id="cam-positive", event_prob=0.9, snapshot_score=0.95),
            _frame(40, camera_id="cam-positive", event_prob=0.7, snapshot_score=0.6),
        ),
        resample_gap_rows_total=3,
        buffer_lineage_adopted_total=1,
    )
    negative_run = ReplayRun(
        camera_id="cam-negative",
        module_qualified_id="fall.v2",
        policy_qualified_id="fall.v2",
        effective_policy_id="policy-1",
        frames=(
            _frame(5, camera_id="cam-negative", event_prob=0.8, snapshot_score=0.8),
            _frame(15, camera_id="cam-negative", event_prob=0.8, snapshot_score=None),
        ),
        resample_gap_rows_total=2,
        buffer_lineage_adopted_total=0,
    )

    receipt = recall_script.score_traces(
        bundle=tmp_path / "bundle",
        traces_dir=traces_dir,
        positive_trace="positive.jsonl",
        owner_fall_offsets_sec=[20.0, 40.0],
        hit_window_sec=10.0,
        exclusion_window_sec=10.0,
        runner_factory=lambda _: _FakeRunner(),
        replay_factory=_fake_replay_factory(
            {"cam-positive": positive_run, "cam-negative": negative_run}
        ),
    )

    assert receipt["owner_fall_hits"] == [
        {"offset_sec": 20.0, "hit": True, "peak_fall_transition_score": 0.95},
        {"offset_sec": 40.0, "hit": True, "peak_fall_transition_score": 0.6},
    ]
    # One false positive on the positive trace (its 0s event, outside both
    # +/-10s exclusion windows) plus two on the negative trace.
    assert receipt["false_positive_episode_count"] == 3
    assert receipt["exposed_camera_hours"] == pytest.approx(50 / 3600)
    assert receipt["false_positive_episodes_per_camera_hour"] == pytest.approx(216.0)
    # 2 scored snapshots (t=20 positive, t=5 negative) vs. 2 classifier-warmup
    # snapshots (t=0 positive, t=15 negative); t=40's 0.6 is also scored.
    assert receipt["live_track_frames_scored"] == 3
    assert receipt["live_track_frames_classifier_warmup"] == 2
    assert receipt["fraction_live_track_frames_with_full_window"] == pytest.approx(3 / 5)
    assert receipt["resample_gap_rows_total"] == 5
    assert receipt["buffer_lineage_adopted_total"] == 1
    assert receipt["model_receipt_threshold"] == 0.42
    assert receipt["model_promotion_eligible"] is True


def test_missing_positive_trace_is_rejected(tmp_path: Path, recall_script: ModuleType) -> None:
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir / "negative.jsonl", "cam-negative", 5)

    with pytest.raises(ValueError, match="positive trace"):
        recall_script.score_traces(
            bundle=tmp_path / "bundle",
            traces_dir=traces_dir,
            positive_trace="missing.jsonl",
            owner_fall_offsets_sec=[1.0],
            runner_factory=lambda _: _FakeRunner(),
            replay_factory=_fake_replay_factory({}),
        )


def test_write_receipt_round_trips_json(tmp_path: Path, recall_script: ModuleType) -> None:
    out = tmp_path / "receipt.json"
    recall_script.write_receipt(out, {"status": "measured"})
    assert '"status": "measured"' in out.read_text(encoding="utf-8")
