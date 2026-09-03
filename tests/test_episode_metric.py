from pathlib import Path

from contracts.replay_trace import decode_document
from tests_support.episode_metric import evaluate
from tests_support.golden_episodes import GoldenEpisode


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
    result = evaluate(rows, (_golden(),))
    assert result["exact"] is False
    assert result["resample_gap_rows_total"] == 2
    assert result["incident_cooldown_suppressed_total"] == 0


def test_metric_rejects_non_bed_exit_without_a_pinned_policy() -> None:
    _, rows = decode_document(Path("tests/fixtures/replay/gap-axis-v2.json").read_text())
    fall = GoldenEpisode(
        "episode", "fixture", "fall", 0, 300_000_000, {"a": "real"}, "real", "single", 1
    )
    try:
        evaluate(rows, (fall,))
    except ValueError as exc:
        assert "--policy" in str(exc)
    else:
        raise AssertionError("fall metrics must require a pinned policy")
