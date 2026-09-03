import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

from contracts.replay_trace import (
    ReplayRow,
    ReplayTraceHeader,
    ReplayTrack,
    decode_document,
    encode_jsonl,
)
from shared.detection_policies import FallPolicyV1, make_effective_policy
from tests_support import episode_metric
from tests_support.episode_metric import _load_rows, evaluate
from tests_support.golden_episodes import GoldenEpisode
from worker.types import FallModelInput


@dataclass(frozen=True)
class _FallMetadata:
    window: int = 1
    stride: int = 1
    mode: Literal["features"] = "features"


class _FallModel:
    metadata = _FallMetadata()
    operating_threshold = 0.7
    artifact_digest = "test-fall-model"

    def predict(self, features: FallModelInput) -> float:
        del features
        return 0.0


class _HighFallModel(_FallModel):
    def predict(self, features: FallModelInput) -> float:
        del features
        return 0.99


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
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    result = evaluate(rows, (fall,), policy, fall_model=_FallModel())
    assert result["domains"]["fall"]["effective_policy_id"] == policy.effective_policy_id


def _row(pts_ns: int, source_event: str) -> ReplayRow:
    return ReplayRow(
        camera_id="fixture",
        seq=pts_ns,
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
        ),
        bed_polygon_id="bed",
        bed_polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),
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


def _fall_rows() -> tuple[ReplayRow, ...]:
    return (
        _row(0, "open"),
        replace(_row(1_000_000_000, "frame"), tracks=()),
        _row(61_000_000_000, "frame"),
    )


def _cooldown_rows() -> tuple[ReplayRow, ...]:
    return (
        _row(0, "open"),
        replace(_row(1_000_000_000, "frame"), tracks=()),
        _row(2_000_000_000, "frame"),
    )


def test_metric_cli_exact_returns_zero_for_both_domains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, fixture_rows = decode_document(
        Path("tests/fixtures/replay/gap-control-v2.json").read_text()
    )
    rows = (replace(fixture_rows[0], source_event="open"), *fixture_rows[1:])
    status, result = _run_cli(
        monkeypatch,
        tmp_path,
        rows,
        (
            _cli_golden("fall", 0, 1_000_000_000),
            _cli_golden("bed_exit", 0, 1_000_000_000),
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
        monkeypatch, tmp_path, rows, (_cli_golden("fall", 0, 62_000_000_000),)
    )
    assert status == 1
    assert result is not None
    assert result["episodes"][0]["alerts"] == 2  # type: ignore[index]

    status, result = _run_cli(
        monkeypatch,
        tmp_path,
        rows,
        (_cli_golden("fall", 62_000_000_000, 63_000_000_000),),
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


def test_metric_cli_bad_header_returns_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "trace.jsonl").write_text('{"version":"not-a-trace"}\n')
    golden = tmp_path / "golden.json"
    golden.write_text("{}")
    monkeypatch.setattr(
        episode_metric, "load_golden_episodes", lambda _: (_cli_golden("fall", 0, 1),)
    )
    monkeypatch.setattr(episode_metric, "_is_provisional", lambda _: False)
    monkeypatch.setattr(sys, "argv", [
        "episode_metric", "--traces", str(traces), "--golden", str(golden),
        "--out", str(tmp_path / "out.json"),
    ])
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


def test_metric_reports_cooldown_suppression_for_repeated_fall_onsets() -> None:
    result = evaluate(
        _cooldown_rows(),
        (_cli_golden("fall", 0, 62_000_000_000),),
        fall_model=_HighFallModel(),
    )
    assert result["incident_cooldown_suppressed_total"] > 0
