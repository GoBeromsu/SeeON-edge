"""Exact golden-episode metrics from canonical frame-level replay traces."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from contracts.replay_trace import ReplayRow, decode_jsonl
from shared.detection_policies import (
    BedExitPolicyV1,
    EffectivePolicy,
    FallPolicyV1,
    make_effective_policy,
    parse_effective_policy,
)
from tests_support.golden_episodes import GoldenEpisode, load_golden_episodes
from worker.adapters.model.fall_family_registry import DEFAULT_FALL_MODEL_FAMILY_REGISTRY
from worker.domains.fall import FallModelProtocol
from worker.replay.engine import replay
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.local_env import fall_model_config_from_environment


def _default_bed_exit_policy() -> EffectivePolicy:
    return make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.5, hold_frames=1, grace_frames=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _default_fall_policy() -> EffectivePolicy:
    config = fall_model_config_from_environment()
    return make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=config.operating_threshold),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _resolve_fall_model() -> FallModelProtocol:
    """Use the worker's packaged-default config and registered CPU loader."""
    try:
        config = fall_model_config_from_environment()
        return DEFAULT_FALL_MODEL_FAMILY_REGISTRY.create(config.type, config, "cpu")
    except (OSError, RuntimeError, ValueError, TypeError, WorkerConfigError) as exc:
        raise ValueError(f"fall model unavailable: {exc}") from exc


_ROTATION_SUFFIX = re.compile(r"^(?P<base>.+\.jsonl)(?:\.(?P<index>[1-9]\d*))?$")


def _load_rows(
    directory: Path, *, allow_truncated_start: bool = False
) -> tuple[tuple[ReplayRow, ...], bool]:
    """Read every trace rotation chain oldest-to-newest with continuity checks."""
    if not directory.is_dir():
        raise ValueError("traces must be a directory")
    rows: list[ReplayRow] = []
    chains: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in directory.iterdir():
        match = _ROTATION_SUFFIX.fullmatch(path.name)
        if path.is_file() and match is not None:
            chains[match.group("base")].append((int(match.group("index") or 0), path))
    if not chains:
        raise ValueError("traces directory contains no JSONL files")
    truncated_start = False
    for _, segments in sorted(chains.items()):
        previous_seq: int | None = None
        previous_epoch: int | None = None
        previous_pts: int | None = None
        first = True
        for _, path in sorted(segments, key=lambda item: item[0], reverse=True):
            _, decoded = decode_jsonl(path.read_text())
            if first:
                if not decoded:
                    raise ValueError(f"trace segment {path.name} contains no ReplayRow entries")
                if decoded[0].source_event not in ("open", "reconnect"):
                    if not allow_truncated_start:
                        raise ValueError(
                            f"trace segment {path.name} has an ambiguous retained start; "
                            "use --allow-truncated-start to evaluate it"
                        )
                    truncated_start = True
                first = False
            for row in decoded:
                if previous_seq is not None and row.seq <= previous_seq:
                    raise ValueError(f"trace seq is not strictly increasing in {path.name}")
                if previous_epoch is not None and row.epoch < previous_epoch:
                    raise ValueError(f"trace epoch decreases in {path.name}")
                if (
                    row.epoch == previous_epoch
                    and previous_pts is not None
                    and row.pts_ns < previous_pts
                    and row.source_event == "frame"
                ):
                    raise ValueError(f"trace pts decreases within epoch in {path.name}")
                previous_seq = row.seq
                previous_epoch = row.epoch
                if row.source_event == "frame":
                    previous_pts = row.pts_ns
                rows.append(row)
    if not rows:
        raise ValueError("traces contain no ReplayRow entries")
    return tuple(rows), truncated_start


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
    *,
    fall_model: FallModelProtocol | None = None,
) -> dict[str, object]:
    """Run canonical replay then compare admitted alerts to labelled windows."""
    episodes = tuple(item for item in goldens if item.resolved == "real")
    if not episodes:
        raise ValueError("golden fixture contains no real episodes")
    policies = {
        "bed_exit": _default_bed_exit_policy(),
        "fall": _default_fall_policy(),
    }
    if policy is not None:
        policies[policy.module_id] = policy
    selected_fall_model = fall_model or _resolve_fall_model()
    runs = [
        (
            event_type,
            replay(
                camera_id=camera_id,
                rows=tuple(camera_rows),
                module_id=event_type,
                policy=policies[event_type],
                fall_model=selected_fall_model if event_type == "fall" else None,
            ),
        )
        for camera_id, camera_rows in _rows_by_camera(rows).items()
        for event_type in ("fall", "bed_exit")
    ]
    alerts = [
        {
            "camera_id": event.camera_id,
            "event_type": event.event_type.replace("-", "_"),
            "pts_ns": frame.pts_ns,
        }
        for _, run in runs
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
    gap_rows = sum(1 for _, run in runs for frame in run.frames if not frame.valid) // 2
    exact = all(item["alerts"] == 1 for item in per_episode) and not outside
    domains = {}
    for event_type in ("fall", "bed_exit"):
        domain_episodes = [item for item in per_episode if item["event_type"] == event_type]
        domain_outside = [alert for alert in outside if alert["event_type"] == event_type]
        domains[event_type] = {
            "episodes": len(domain_episodes),
            "alerts": sum(alert["event_type"] == event_type for alert in alerts),
            "alerts_outside_golden_windows": len(domain_outside),
            "exact": all(item["alerts"] == 1 for item in domain_episodes)
            and not domain_outside,
            "effective_policy_id": policies[event_type].effective_policy_id,
        }
    return {
        "recall": sum(item["alerts"] == 1 for item in per_episode) / len(per_episode),
        "precision": 1.0 if not outside else 0.0,
        "alerts_per_episode": len(alerts) / len(episodes),
        "alerts_per_hour": len(alerts) / duration_hours,
        "incident_cooldown_suppressed_total": sum(
            run.incident_cooldown_suppressed_total for _, run in runs
        ),
        "resample_gap_rows_total": gap_rows,
        "id_churn_allowance": _id_churn_allowance(rows),
        "episodes": per_episode,
        "alerts_outside_golden_windows": len(outside),
        "domains": domains,
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
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fall and bed_exit golden episodes. --policy is the "
            "production-resolved EffectivePolicy document; absent policy uses "
            "the packaged bed_exit default and the worker's packaged fall "
            "model resolved by fall_model_config_from_environment() then "
            "DEFAULT_FALL_MODEL_FAMILY_REGISTRY on CPU. --policy replaces "
            "the effective policy only for its module."
        )
    )
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--allow-provisional-golden", action="store_true")
    parser.add_argument("--allow-truncated-start", action="store_true")
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
        rows, truncated_start = _load_rows(
            args.traces, allow_truncated_start=args.allow_truncated_start
        )
        result = evaluate(rows, goldens, policy)
        result["truncated_start_allowed"] = truncated_start
        if provisional:
            result["owner_decision_required"] = True
    except WorkerConfigError as exc:
        print(f"fall model unavailable: {exc}")
        return 2
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        if str(exc).startswith("fall model unavailable:"):
            print(str(exc))
            return 2
        print(f"input error: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":")))
    return 2 if provisional else 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
