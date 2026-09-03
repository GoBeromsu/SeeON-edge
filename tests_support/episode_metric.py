"""Exact golden-episode metric command for replay alert outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests_support.golden_episodes import GoldenEpisode, load_golden_episodes


def evaluate(traces: Path, goldens: tuple[GoldenEpisode, ...]) -> dict[str, object]:
    """Measure alerts against real labelled episodes (``unsure`` is excluded)."""
    episodes = tuple(item for item in goldens if item.label == "real")
    alerts, cooldown_suppressed = _load_alerts(traces)
    per_episode = []
    matched: set[int] = set()
    for episode in episodes:
        matches = [
            alert for alert in alerts
            if alert["camera_id"] == episode.camera_id
            and alert["event_type"] == episode.event_type
            and episode.start_ns <= alert["pts_ns"] <= episode.end_ns
        ]
        matched.update(id(alert) for alert in matches)
        per_episode.append({"camera_id": episode.camera_id, "event_type": episode.event_type,
                            "start_ns": episode.start_ns, "alerts": len(matches)})
    outside = [alert for alert in alerts if id(alert) not in matched]
    exact = all(item["alerts"] == 1 for item in per_episode) and not outside
    return {
        "recall": (
            sum(item["alerts"] == 1 for item in per_episode) / len(per_episode)
            if per_episode
            else 0.0
        ),
        "precision": 1.0 if not outside else 0.0,
        "alerts_per_episode": len(alerts) / len(episodes) if episodes else 0.0,
        "incident_cooldown_suppressed_total": cooldown_suppressed,
        "id_churn_allowance": [],
        "episodes": per_episode,
        "alerts_outside_golden_windows": len(outside),
        "exact": exact,
    }


def _load_alerts(directory: Path) -> tuple[list[dict[str, object]], int]:
    """Read alert JSONL or a serialized ``ReplayRun`` JSON document.

    Replay output documents expose ``incident_cooldown_suppressed_total`` and
    frames containing admitted event dictionaries.  Flat alert JSONL remains
    supported for independently generated traces and has a counter of zero.
    """
    if not directory.is_dir():
        raise ValueError("traces must be a directory")
    alerts = []
    cooldown_suppressed = 0
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            if "frames" in row:
                cooldown_suppressed += int(row.get("incident_cooldown_suppressed_total", 0))
                for frame in row["frames"]:
                    for event in frame.get("events", ()):
                        alerts.append(
                            {
                                "camera_id": str(event["camera_id"]),
                                "event_type": str(event["event_type"]),
                                "pts_ns": int(frame["pts_ns"]),
                            }
                        )
                continue
            if "event_type" in row and ("pts_ns" in row or "detected_at_ns" in row):
                alerts.append(
                    {
                        "camera_id": str(row["camera_id"]),
                        "event_type": str(row["event_type"]),
                        "pts_ns": int(row.get("pts_ns", row.get("detected_at_ns"))),
                    }
                )
    return alerts, cooldown_suppressed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.traces, load_golden_episodes(args.golden))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":")))
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
