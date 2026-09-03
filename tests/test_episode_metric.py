from tests_support.episode_metric import evaluate
from tests_support.golden_episodes import GoldenEpisode


def test_multiple_alerts_in_one_episode_are_not_exact(tmp_path) -> None:
    (tmp_path / "alerts.jsonl").write_text(
        '{"camera_id":"cam","event_type":"fall","pts_ns":2}\n'
        '{"camera_id":"cam","event_type":"fall","pts_ns":3}\n'
    )
    result = evaluate(tmp_path, (GoldenEpisode("cam", "fall", 1, 4, "real"),))
    assert result["exact"] is False
    assert result["recall"] == 0.0


def test_replay_run_document_supplies_real_cooldown_counter(tmp_path) -> None:
    (tmp_path / "run.jsonl").write_text(
        '{"incident_cooldown_suppressed_total":2,"frames":['
        '{"pts_ns":2,"events":[{"camera_id":"cam","event_type":"fall"}]}]}\n'
    )
    result = evaluate(tmp_path, (GoldenEpisode("cam", "fall", 1, 4, "real"),))
    assert result["exact"] is True
    assert result["incident_cooldown_suppressed_total"] == 2
