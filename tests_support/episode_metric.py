"""Exact golden-episode metrics from canonical frame-level replay traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from contracts.replay_trace import ReplayRow, decode_jsonl
from shared.detection_policies import (
    BedExitPolicyV1,
    EffectivePolicy,
    make_effective_policy,
    parse_effective_policy,
)
from tests_support.golden_episodes import GoldenEpisode, load_golden_episodes
from worker.replay.engine import replay


def _default_policy(episodes: tuple[GoldenEpisode, ...]) -> EffectivePolicy:
    event_types = {episode.event_type for episode in episodes}
    if event_types != {"bed_exit"}:
        raise ValueError("--policy is required unless every golden episode is bed_exit")
    return make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.5, hold_frames=1, grace_frames=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _load_rows(directory: Path) -> tuple[ReplayRow, ...]:
    if not directory.is_dir():
        raise ValueError("traces must be a directory")
    rows: list[ReplayRow] = []
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise ValueError("traces directory contains no JSONL files")
    for path in paths:
        _, decoded = decode_jsonl(path.read_text())
        rows.extend(decoded)
    if not rows:
        raise ValueError("traces contain no ReplayRow entries")
    return tuple(rows)


def _is_provisional(path: Path) -> bool:
    payload = json.loads(path.read_text())
    return bool(payload.get("provisional", False))


def _validate_golden_input(
    goldens: tuple[GoldenEpisode, ...], *, provisional: bool, allow_provisional: bool
) -> None:
    if not goldens:
        raise ValueError("golden fixture contains no episodes")
    if provisional and not allow_provisional:
        raise ValueError("golden fixture is provisional")


def evaluate(
    rows: tuple[ReplayRow, ...],
    goldens: tuple[GoldenEpisode, ...],
    policy: EffectivePolicy | None = None,
) -> dict[str, object]:
    """Run canonical replay then compare admitted alerts to labelled windows."""
    episodes = tuple(item for item in goldens if item.resolved == "real")
    if not episodes:
        raise ValueError("golden fixture contains no real episodes")
    selected_policy = policy or _default_policy(episodes)
    runs = [
        replay(
            camera_id=camera_id,
            rows=tuple(camera_rows),
            module_id=selected_policy.module_id,
            policy=selected_policy,
        )
        for camera_id, camera_rows in _rows_by_camera(rows).items()
    ]
    alerts = [
        {
            "camera_id": event.camera_id,
            "event_type": event.event_type.replace("-", "_"),
            "pts_ns": frame.pts_ns,
        }
        for run in runs
        for frame in run.frames
        for event in frame.events
        if frame.pts_ns is not None
    ]
    per_episode: list[dict[str, object]] = []
    matched: set[int] = set()
    for episode in episodes:
        matches = [
            index
            for index, alert in enumerate(alerts)
            if alert["camera_id"] == episode.camera_id
            and alert["event_type"] == episode.event_type
            and episode.start_ns <= alert["pts_ns"] <= episode.end_ns
        ]
        matched.update(matches)
        per_episode.append(
            {
                "camera_id": episode.camera_id,
                "event_type": episode.event_type,
                "start_ns": episode.start_ns,
                "alerts": len(matches),
            }
        )
    outside = [alert for index, alert in enumerate(alerts) if index not in matched]
    start_ns, end_ns = min(row.pts_ns for row in rows), max(row.pts_ns for row in rows)
    duration_hours = max((end_ns - start_ns) / 3_600_000_000_000, 1 / 3_600_000_000_000)
    gap_rows = sum(1 for run in runs for frame in run.frames if not frame.valid)
    exact = all(item["alerts"] == 1 for item in per_episode) and not outside
    return {
        "recall": sum(item["alerts"] == 1 for item in per_episode) / len(per_episode),
        "precision": 1.0 if not outside else 0.0,
        "alerts_per_episode": len(alerts) / len(episodes),
        "alerts_per_hour": len(alerts) / duration_hours,
        "incident_cooldown_suppressed_total": sum(
            run.incident_cooldown_suppressed_total for run in runs
        ),
        "resample_gap_rows_total": gap_rows,
        "id_churn_allowance": _id_churn_allowance(rows),
        "episodes": per_episode,
        "alerts_outside_golden_windows": len(outside),
        "exact": exact,
    }


def _rows_by_camera(rows: tuple[ReplayRow, ...]) -> dict[str, list[ReplayRow]]:
    grouped: dict[str, list[ReplayRow]] = defaultdict(list)
    for row in rows:
        grouped[row.camera_id].append(row)
    return grouped


def _id_churn_allowance(rows: tuple[ReplayRow, ...]) -> list[dict[str, int | str]]:
    """List at most ten new ids that follow a prior visible id in an epoch."""
    result: list[dict[str, int | str]] = []
    previous: dict[tuple[str, int], set[int]] = {}
    for row in sorted(rows, key=lambda item: (item.camera_id, item.epoch, item.pts_ns)):
        key = (row.camera_id, row.epoch)
        current = {track.track_id for track in row.tracks if track.lifecycle in ("new", "tracked")}
        old = previous.get(key, set())
        for track in row.tracks:
            if track.lifecycle == "new" and old and track.track_id not in old and len(result) < 10:
                result.append(
                    {
                        "camera_id": row.camera_id,
                        "epoch": row.epoch,
                        "pts_ns": row.pts_ns,
                        "track_id": track.track_id,
                    }
                )
        previous[key] = current
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--allow-provisional-golden", action="store_true")
    args = parser.parse_args()
    try:
        goldens = load_golden_episodes(args.golden)
        provisional = _is_provisional(args.golden)
        _validate_golden_input(
            goldens,
            provisional=provisional,
            allow_provisional=args.allow_provisional_golden,
        )
        policy = (
            parse_effective_policy(json.loads(args.policy.read_text())) if args.policy else None
        )
        result = evaluate(_load_rows(args.traces), goldens, policy)
        if provisional:
            result["owner_decision_required"] = True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"input error: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":")))
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
