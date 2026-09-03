import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from contracts.replay_trace import (
    ReplayRow,
    ReplayTraceHeader,
    ReplayTrack,
    decode_document,
    encode_jsonl,
)
from shared.detection_policies import FallPolicyV2, make_effective_policy
from tests_support import episode_metric
from tests_support.episode_metric import _id_churn_allowance, _load_rows, evaluate
from tests_support.golden_episodes import GoldenEpisode
from worker.interfaces.fall_model import FallV2Probabilities


@dataclass(frozen=True)
class _FallModel:
    artifact_digest = "test-fall-model"

    def predict(self, features: object) -> FallV2Probabilities:
        del features
        return FallV2Probabilities(background=1.0, fall_transition=0.0, fallen=0.0)


class _HighFallModel(_FallModel):
    def predict(self, features: object) -> FallV2Probabilities:
        del features
        return FallV2Probabilities(background=0.01, fall_transition=0.99, fallen=0.0)


class _RecoveringFallModel(_FallModel):
    """High, then clear, then high again.

    A second episode may only open after a *scored* confirmed recovery -- track
    loss is never recovery -- so a two-onset corpus needs a run of clear scores
    between the onsets. The window count matches the policy's
    ``recovery_consecutive`` (5) with headroom.
    """

    def __init__(self, clear_from: int, clear_until: int) -> None:
        self._calls = 0
        self._clear_from = clear_from
        self._clear_until = clear_until

    def predict(self, features: object) -> FallV2Probabilities:
        del features
        self._calls += 1
        if self._clear_from <= self._calls <= self._clear_until:
            return FallV2Probabilities(background=0.99, fall_transition=0.01, fallen=0.0)
        return FallV2Probabilities(background=0.01, fall_transition=0.99, fallen=0.0)


def _golden() -> GoldenEpisode:
    return GoldenEpisode(
        "episode",
        "fixture",
        "bed_exit",
        0,
        300_000_000,
        {"a": "real"},
        "real",
        "single",
        1,
    )


def test_metric_runs_canonical_replay_rows_and_reports_gap_counter() -> None:
    _, rows = decode_document(Path("tests/fixtures/replay/gap-axis-v2.json").read_text())
    result = evaluate(rows, (_golden(),), fall_model=_FallModel())
    assert result["exact"] is False
    assert result["resample_gap_rows_total"] == 2
    assert result["incident_cooldown_suppressed_total"] == 0


def test_metric_accepts_pinned_fall_policy() -> None:
    _, rows = decode_document(Path("tests/fixtures/replay/gap-axis-v2.json").read_text())
    fall = GoldenEpisode(
        "episode", "fixture", "fall", 0, 300_000_000, {"a": "real"}, "real", "single", 1
    )
    policy = make_effective_policy(
        module_id="fall",
        module_version=2,
        values=FallPolicyV2(transition_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    result = evaluate(rows, (fall,), policy, fall_model=_FallModel())
    assert result["domains"]["fall"]["effective_policy_id"] == policy.effective_policy_id


def _row(pts_ns: int, source_event: str) -> ReplayRow:
    return ReplayRow(
        camera_id="fixture",
        seq=0 if source_event == "open" else pts_ns,
        pts_ns=pts_ns,
        epoch=0,
        source_event=source_event,  # type: ignore[arg-type]
        source="legacy-association",
        tracks=(
            ReplayTrack(
                1,
                "tracked",
                (0.1, 0.1, 0.2, 0.2, 0.9),
                tuple((0.15, 0.15, 0.9) for _ in range(17)),
            ),
        )
        if source_event == "frame"
        else (),
        bed_polygon_id="bed",
        bed_polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),
        bed_polygon_image_size=(1000, 1000),
        night_window_active=True,
        frame_width=1000,
        frame_height=1000,
    )


def test_metric_loader_reads_rotations_oldest_to_newest(tmp_path: Path) -> None:
    header = ReplayTraceHeader()
    (tmp_path / "trace.jsonl.1").write_text(encode_jsonl(header, [_row(0, "open")]))
    (tmp_path / "trace.jsonl").write_text(encode_jsonl(header, [_row(100, "frame")]))

    rows, truncated = _load_rows(tmp_path)

    assert [row.pts_ns for row in rows] == [0, 100]
    assert truncated is False


def test_metric_loader_refuses_ambiguous_retained_start_unless_allowed(tmp_path: Path) -> None:
    (tmp_path / "trace.jsonl").write_text(encode_jsonl(ReplayTraceHeader(), [_row(0, "frame")]))

    with pytest.raises(ValueError, match="allow-truncated-start"):
        _load_rows(tmp_path)
    _, truncated = _load_rows(tmp_path, allow_truncated_start=True)
    assert truncated is True


def _write_trace(directory: Path, rows: tuple[ReplayRow, ...]) -> None:
    directory.mkdir(exist_ok=True)
    (directory / "trace.jsonl").write_text(encode_jsonl(ReplayTraceHeader(), list(rows)))


def _recovering_model() -> _RecoveringFallModel:
    """Clear scores for the recovery run that separates the two onsets."""
    return _RecoveringFallModel(_ONSET_WINDOWS + 1, _ONSET_WINDOWS + _RECOVERY_ROWS // 5)


def _cli_golden(event_type: str, start_ns: int, end_ns: int) -> GoldenEpisode:
    return GoldenEpisode(
        f"{event_type}-episode",
        "fixture",
        event_type,
        start_ns,
        end_ns,
        {"a": "real"},
        "real",
        "single",
        1,
    )


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rows: tuple[ReplayRow, ...],
    goldens: tuple[GoldenEpisode, ...],
    *,
    provisional: bool = False,
    allow_provisional: bool = False,
    fall_model_loader: object = _HighFallModel,
) -> tuple[int, dict[str, object] | None]:
    traces = tmp_path / "traces"
    _write_trace(traces, rows)
    golden = tmp_path / "golden.json"
    golden.write_text("{}")
    out = tmp_path / "metric.json"
    monkeypatch.setattr(episode_metric, "load_golden_episodes", lambda _: goldens)
    monkeypatch.setattr(episode_metric, "_is_provisional", lambda _: provisional)
    monkeypatch.setattr(episode_metric, "_resolve_fall_model", fall_model_loader)
    argv = [
        "episode_metric",
        "--traces",
        str(traces),
        "--golden",
        str(golden),
        "--out",
        str(out),
    ]
    if allow_provisional:
        argv.append("--allow-provisional-golden")
    monkeypatch.setattr(sys, "argv", argv)
    status = episode_metric.main()
    return status, json.loads(out.read_text()) if out.exists() else None


_FRAME_NS = 66_666_667
# A V2 fall alert needs a 30-row pose+bbox56 window plus three predictions at
# stride 5 (rows 30, 35, 40): 41 contiguous 15 fps rows per onset.
_ONSET_ROWS = 41
# Rows without the track after an onset: past the 45-frame track TTL the V2
# state is evicted, so the next appearance is a fresh episode.
_ABSENT_ROWS = 50
# 30 s of 15 fps frames clears the IncidentManager's admission cooldown.
_COOLDOWN_CLEAR_ROWS = 15 * 31


def _run(
    start_ns: int,
    count: int,
    *,
    first_seq: int,
    tracked: bool = True,
    bbox: tuple[float, float, float, float, float] = (0.1, 0.1, 0.2, 0.2, 0.9),
) -> tuple[ReplayRow, ...]:
    return tuple(
        replace(
            _row(start_ns + index * _FRAME_NS, "frame"),
            seq=first_seq + index,
            tracks=(replace(_row(0, "frame").tracks[0], bbox=bbox),) if tracked else (),
        )
        for index in range(count)
    )


# Predictions land every stride-5 row once the 30-row window is full, so a
# 41-row onset run scores 3 windows. The recovery run below must score at least
# `recovery_consecutive` (5) clear windows before the next onset may open.
_RECOVERY_ROWS = 30
_ONSET_WINDOWS = 3


def _two_onsets(gap_rows: int = 0) -> tuple[ReplayRow, ...]:
    """One onset, a scored recovery run, then a second onset on the same track.

    The track stays live throughout: the episode authority only re-arms on a
    confirmed recovery, and a track that merely disappears resolves without
    ever alerting again.
    """
    first = _run(0, _ONSET_ROWS, first_seq=1)
    recovery_start = first[-1].pts_ns + _FRAME_NS
    recovery = _run(recovery_start, _RECOVERY_ROWS, first_seq=len(first) + 1)
    second_start = recovery[-1].pts_ns + (gap_rows + 1) * _FRAME_NS
    second = _run(second_start, _ONSET_ROWS, first_seq=len(first) + len(recovery) + 1)
    return (_row(0, "open"), *first, *recovery, *second)


def _fall_rows() -> tuple[ReplayRow, ...]:
    """Two recovery-separated onsets far enough apart that both are admitted."""
    return _two_onsets(gap_rows=_COOLDOWN_CLEAR_ROWS)


def _cooldown_rows() -> tuple[ReplayRow, ...]:
    """Two recovery-separated onsets inside the 30 s admission cooldown."""
    return _two_onsets()


def test_metric_cli_exact_returns_zero_for_both_domains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One person: in the bed for 20 rows, then out of it while the fall window
    # keeps filling -- both domains alert exactly once inside their windows.
    inside = _run(0, 20, first_seq=1)
    outside = _run(20 * _FRAME_NS, _ONSET_ROWS - 20, first_seq=21, bbox=(0.7, 0.7, 0.8, 0.8, 0.9))
    rows = (_row(0, "open"), *inside, *outside)
    status, result = _run_cli(
        monkeypatch,
        tmp_path,
        rows,
        (
            _cli_golden("fall", 0, 3_000_000_000),
            _cli_golden("bed_exit", 0, 3_000_000_000),
        ),
    )
    assert status == 0
    assert result is not None
    assert result["exact"] is True
    assert result["domains"]["fall"]["exact"] is True  # type: ignore[index]
    assert result["domains"]["bed_exit"]["exact"] is True  # type: ignore[index]


def test_metric_cli_multi_alert_and_outside_window_return_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = _fall_rows()
    status, result = _run_cli(
        monkeypatch,
        tmp_path,
        rows,
        (_cli_golden("fall", 0, 62_000_000_000),),
        fall_model_loader=_recovering_model,
    )
    assert status == 1
    assert result is not None
    assert result["episodes"][0]["alerts"] == 2  # type: ignore[index]

    status, result = _run_cli(
        monkeypatch,
        tmp_path,
        rows,
        (_cli_golden("fall", 62_000_000_000, 63_000_000_000),),
        fall_model_loader=_recovering_model,
    )
    assert status == 1
    assert result is not None
    assert result["alerts_outside_golden_windows"] == 2


def test_metric_cli_empty_golden_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, result = _run_cli(
        monkeypatch,
        tmp_path,
        (_row(0, "open"),),
        (),
    )
    assert status == 2
    assert result is None


@pytest.mark.parametrize("allow_provisional", [False, True])
def test_metric_cli_provisional_golden_returns_two_with_or_without_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, allow_provisional: bool
) -> None:
    status, result = _run_cli(
        monkeypatch,
        tmp_path,
        (_row(0, "open"),),
        (_cli_golden("fall", 0, 1),),
        provisional=True,
        allow_provisional=allow_provisional,
    )
    assert status == 2
    if allow_provisional:
        assert result is not None
        assert result["owner_decision_required"] is True
    else:
        assert result is None


def test_metric_cli_bad_header_returns_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "trace.jsonl").write_text('{"version":"not-a-trace"}\n')
    golden = tmp_path / "golden.json"
    golden.write_text("{}")
    monkeypatch.setattr(
        episode_metric, "load_golden_episodes", lambda _: (_cli_golden("fall", 0, 1),)
    )
    monkeypatch.setattr(episode_metric, "_is_provisional", lambda _: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "episode_metric",
            "--traces",
            str(traces),
            "--golden",
            str(golden),
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    assert episode_metric.main() == 2


def test_metric_cli_refuses_unavailable_fall_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _ = _run_cli(
        monkeypatch,
        tmp_path,
        (_row(0, "open"),),
        (_cli_golden("fall", 0, 1),),
        fall_model_loader=lambda: (_ for _ in ()).throw(
            ValueError("fall model unavailable: missing model.pt")
        ),
    )
    assert status == 2
    assert "fall model unavailable" in capsys.readouterr().out


def test_two_recovery_separated_onsets_are_both_admitted_without_cooldown_suppression() -> None:
    """P1a-AC2: a genuine second episode is never suppressed.

    The episode authority is the lifecycle owner, so two onsets separated by a
    scored confirmed recovery are two distinct episodes even inside the 30 s
    admission window. The cooldown is overload protection keyed on the emitted
    identity, so every suppression it reports is a defect signal -- and here
    there must be none.
    """
    result = evaluate(
        _cooldown_rows(),
        (_cli_golden("fall", 0, 62_000_000_000),),
        fall_model=_recovering_model(),
    )
    assert result["incident_cooldown_suppressed_total"] == 0
    assert result["episodes"][0]["alerts"] == 2  # type: ignore[index]


def _churn_rows(*, switch_pts_ns: tuple[int, ...]) -> tuple[ReplayRow, ...]:
    rows: list[ReplayRow] = []
    for index, pts_ns in enumerate(switch_pts_ns):
        rows.extend(
            (
                replace(
                    _row(pts_ns - _FRAME_NS, "frame"),
                    seq=index * 2 + 1,
                    tracks=(replace(_row(0, "frame").tracks[0], track_id=index * 2 + 1),),
                ),
                replace(
                    _row(pts_ns, "frame"),
                    seq=index * 2 + 2,
                    tracks=(
                        replace(
                            _row(0, "frame").tracks[0],
                            track_id=index * 2 + 2,
                            lifecycle="new",
                        ),
                    ),
                ),
            )
        )
    return tuple(rows)


def _failed_episode(*, start_ns: int = 0, end_ns: int = 10_000_000_000) -> list[dict[str, object]]:
    return [
        {
            "camera_id": "fixture",
            "event_type": "fall",
            "start_ns": start_ns,
            "end_ns": end_ns,
            "alerts": 0,
        }
    ]


def test_id_churn_allowance_requires_a_switch_beyond_the_reassociation_window() -> None:
    rows = (
        replace(_row(0, "frame"), seq=1),
        replace(
            _row(5_000_000_001, "frame"),
            seq=2,
            tracks=(replace(_row(0, "frame").tracks[0], track_id=2, lifecycle="new"),),
        ),
    )
    allowance = _id_churn_allowance(
        rows,
        _failed_episode(),
        outside_alerts=[],
    )

    assert allowance[0]["track_id"] == 2
    assert allowance[0]["episode_start_ns"] == 0
    assert allowance[0]["track_id_switch_total"] == 1


def test_id_churn_allowance_rejects_switch_inside_the_five_second_window() -> None:
    allowance = _id_churn_allowance(
        _churn_rows(switch_pts_ns=(4_999_999_999,)),
        _failed_episode(),
        outside_alerts=[],
    )

    assert allowance == []


def test_id_churn_allowance_rejects_an_unrelated_failure() -> None:
    allowance = _id_churn_allowance(
        _churn_rows(switch_pts_ns=(1_000_000_000,)),
        [
            *_failed_episode(),
            {"camera_id": "other", "event_type": "fall", "start_ns": 0, "alerts": 0},
        ],
        outside_alerts=[],
    )

    assert allowance == []


def _nonexact_result(*, affected: int, outside: int = 0) -> dict[str, object]:
    episodes = [
        {
            "camera_id": "fixture",
            "event_type": "fall",
            "start_ns": index,
            "end_ns": index,
            "alerts": 0,
        }
        for index in range(affected)
    ]
    return {
        "exact": False,
        "episodes": episodes,
        "alerts_outside_golden_windows": outside,
        "id_churn_allowance": [
            {
                "camera_id": "fixture",
                "event_type": "fall",
                "episode_start_ns": index,
                "track_id_switch_total": 1,
                "elapsed_ns": 5_000_000_001,
            }
            for index in range(affected)
        ],
        "id_churn_allowance_limit_exceeded": affected > 10,
    }


@pytest.mark.parametrize(("affected", "expected_status"), [(1, 0), (10, 0), (11, 1)])
def test_metric_cli_applies_the_bounded_churn_residual_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, affected: int, expected_status: int
) -> None:
    monkeypatch.setattr(
        episode_metric, "evaluate", lambda *_args: _nonexact_result(affected=affected)
    )
    status, result = _run_cli(
        monkeypatch,
        tmp_path,
        (_row(0, "open"),),
        (_cli_golden("fall", 0, 1),),
    )

    assert status == expected_status
    assert result is not None
    assert result["ac1_passed"] is (affected <= 10)


@pytest.mark.parametrize(
    "result",
    [
        _nonexact_result(affected=0),
        _nonexact_result(affected=1, outside=1),
    ],
)
def test_metric_cli_rejects_non_churn_and_outside_window_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, result: dict[str, object]
) -> None:
    monkeypatch.setattr(episode_metric, "evaluate", lambda *_args: result)
    status, written = _run_cli(
        monkeypatch,
        tmp_path,
        (_row(0, "open"),),
        (_cli_golden("fall", 0, 1),),
    )

    assert status == 1
    assert written is not None
    assert written["ac1_passed"] is False


def test_id_churn_allowance_rejects_an_unrelated_new_id_after_candidate_expiry() -> None:
    rows = (
        replace(_row(0, "frame"), seq=1),
        replace(_row(1_000_000_000, "frame"), seq=2, tracks=()),
        replace(
            _row(6_000_000_001, "frame"),
            seq=3,
            tracks=(replace(_row(0, "frame").tracks[0], track_id=2, lifecycle="new"),),
        ),
    )

    assert _id_churn_allowance(rows, _failed_episode(), outside_alerts=[]) == []


def test_id_churn_allowance_ignores_an_ancient_stale_id_for_a_recent_split() -> None:
    rows = (
        replace(_row(0, "frame"), seq=1),
        replace(_row(1_000_000_000, "frame"), seq=2, tracks=()),
        replace(
            _row(6_000_000_000, "frame"),
            seq=3,
            tracks=(replace(_row(0, "frame").tracks[0], track_id=2),),
        ),
        replace(
            _row(11_000_000_001, "frame"),
            seq=4,
            tracks=(replace(_row(0, "frame").tracks[0], track_id=3, lifecycle="new"),),
        ),
    )

    allowance = _id_churn_allowance(
        rows,
        _failed_episode(end_ns=12_000_000_000),
        outside_alerts=[],
    )

    assert allowance[0]["previous_track_id"] == 2
    assert allowance[0]["track_id"] == 3


def test_id_churn_allowance_rejects_two_new_ids_for_one_disappearance() -> None:
    rows = (
        replace(_row(0, "frame"), seq=1),
        replace(
            _row(5_000_000_001, "frame"),
            seq=2,
            tracks=(
                replace(_row(0, "frame").tracks[0], track_id=2, lifecycle="new"),
                replace(_row(0, "frame").tracks[0], track_id=3, lifecycle="new"),
            ),
        ),
    )

    assert _id_churn_allowance(rows, _failed_episode(), outside_alerts=[]) == []


def test_metric_cli_rejects_unattributed_zero_alert_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _nonexact_result(affected=1)
    result["id_churn_allowance"] = []
    result["id_churn_allowance_limit_exceeded"] = False
    monkeypatch.setattr(episode_metric, "evaluate", lambda *_args: result)

    status, written = _run_cli(
        monkeypatch,
        tmp_path,
        (_row(0, "open"),),
        (_cli_golden("fall", 0, 1),),
    )

    assert status == 1
    assert written is not None
    assert written["ac1_passed"] is False
