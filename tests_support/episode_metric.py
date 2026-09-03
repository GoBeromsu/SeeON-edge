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
    FallPolicyV2,
    make_effective_policy,
    parse_effective_policy,
)
from tests_support.golden_episodes import GoldenEpisode, load_golden_episodes
from worker.adapters.model.fall_family_registry import DEFAULT_FALL_MODEL_FAMILY_REGISTRY
from worker.interfaces.fall_model import FallV2ModelProtocol
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
        module_version=2,
        values=FallPolicyV2(transition_threshold=config.operating_threshold),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _resolve_fall_model() -> FallV2ModelProtocol:
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
                if row.source_event == "open":
                    if row.seq != 0:
                        raise ValueError(f"trace open seq must be zero in {path.name}")
                    previous_seq = None
                    previous_epoch = None
                    previous_pts = None
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
    fall_model: FallV2ModelProtocol | None = None,
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
                "end_ns": episode.end_ns,
                "alerts": len(matches),
            }
        )
    outside = [alert for index, alert in enumerate(alerts) if index not in matched]
    start_ns, end_ns = min(row.pts_ns for row in rows), max(row.pts_ns for row in rows)
    duration_hours = max((end_ns - start_ns) / 3_600_000_000_000, 1 / 3_600_000_000_000)
    # The decider's resampler is the single cadence owner, so its own count is
    # the only gap authority; counting invalid frames here double-counted the
    # per-domain runs and needed a fudge factor to compensate.
    gap_rows = sum(run.resample_gap_rows_total for _, run in runs)
    exact = all(item["alerts"] == 1 for item in per_episode) and not outside
    id_churn_allowance = _id_churn_allowance(
        rows,
        per_episode,
        outside_alerts=outside,
    )
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
        "track_id_switch_total": sum(run.track_id_switch_total for _, run in runs),
        "track_id_switch_absorbed_total": sum(
            run.track_id_switch_absorbed_total for _, run in runs
        ),
        "resample_gap_rows_total": gap_rows,
        "id_churn_allowance": id_churn_allowance,
        "id_churn_allowance_limit_exceeded": len(id_churn_allowance) > 10,
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


def _id_churn_allowance(
    rows: tuple[ReplayRow, ...],
    episodes: list[dict[str, object]],
    *,
    outside_alerts: list[dict[str, object]],
) -> list[dict[str, int | str]]:
    """List failed episodes caused solely by beyond-window legacy id splits."""
    if outside_alerts:
        return []
    switches: list[dict[str, int | str]] = []
    live: dict[tuple[str, int], dict[int, tuple[int, int]]] = {}
    candidates: dict[tuple[str, int], dict[int, tuple[int, int]]] = {}
    frame_counts: dict[tuple[str, int], int] = defaultdict(int)
    reassociation_ns = 5_000_000_000
    for row in sorted(rows, key=lambda item: (item.camera_id, item.epoch, item.pts_ns, item.seq)):
        if row.source_event != "frame":
            continue
        key = (row.camera_id, row.epoch)
        frame_counts[key] += 1
        frame_index = frame_counts[key]
        current = {track.track_id for track in row.tracks if track.lifecycle in ("new", "tracked")}
        previous_live = live.get(key, {})
        pending = {
            track_id: seen
            for track_id, seen in candidates.get(key, {}).items()
            if row.pts_ns - seen[1] <= reassociation_ns
        }
        disappeared = {
            track_id: seen
            for track_id, seen in previous_live.items()
            if track_id not in current
        }
        # A directly preceding live id is evaluated even when sparse trace
        # sampling crosses the window in one step. Older ids are only retained
        # for the bounded re-association window above.
        predecessors = {**pending, **disappeared}
        new_ids = [
            track.track_id
            for track in row.tracks
            if track.lifecycle == "new" and track.track_id not in previous_live
        ]
        paired_predecessor: int | None = None
        if len(predecessors) == 1 and len(new_ids) == 1:
            previous_track_id, (previous_frame, previous_pts_ns) = next(
                iter(predecessors.items())
            )
            paired_predecessor = previous_track_id
            elapsed_ns = row.pts_ns - previous_pts_ns
            if elapsed_ns > reassociation_ns:
                switches.append(
                    {
                        "camera_id": row.camera_id,
                        "epoch": row.epoch,
                        "pts_ns": row.pts_ns,
                        "previous_track_id": previous_track_id,
                        "track_id": new_ids[0],
                        "frames_since_previous": frame_index - previous_frame,
                        "elapsed_ns": elapsed_ns,
                    }
                )
        # A new id consumes the entire contemporaneous candidacy. Retaining a
        # candidate after an ambiguous arrival would let a later id turn an
        # already ambiguous split into an allowance.
        candidates[key] = (
            {}
            if new_ids
            else {
                track_id: seen
                for track_id, seen in {**pending, **disappeared}.items()
                if track_id != paired_predecessor
                and row.pts_ns - seen[1] <= reassociation_ns
            }
        )
        live[key] = dict.fromkeys(current, (frame_index, row.pts_ns))
    failed = [episode for episode in episodes if episode["alerts"] == 0]
    if len(failed) != sum(episode["alerts"] != 1 for episode in episodes):
        return []
    result: list[dict[str, int | str]] = []
    for episode in failed:
        camera_id = episode["camera_id"]
        start_ns = episode["start_ns"]
        end_ns = episode.get("end_ns")
        if (
            not isinstance(camera_id, str)
            or not isinstance(start_ns, int)
            or not isinstance(end_ns, int)
        ):
            raise TypeError("episode identity and window bounds must be typed")
        evidence = [
            switch
            for switch in switches
            if switch["camera_id"] == camera_id and start_ns <= switch["pts_ns"] <= end_ns
        ]
        if len(evidence) != 1:
            return []
        result.append(
            {
                "camera_id": camera_id,
                "event_type": episode["event_type"],
                "episode_start_ns": start_ns,
                "track_id_switch_total": len(evidence),
                **evidence[0],
            }
        )
    return result


def _ac1_passed(result: dict[str, object]) -> bool:
    if result["exact"] is True:
        return True
    allowance = result["id_churn_allowance"]
    episodes = result["episodes"]
    outside = result["alerts_outside_golden_windows"]
    if (
        not isinstance(allowance, list)
        or not isinstance(episodes, list)
        or outside != 0
        or result.get("id_churn_allowance_limit_exceeded") is not False
        or any(not isinstance(episode, dict) or "alerts" not in episode for episode in episodes)
    ):
        return False
    failed = [episode for episode in episodes if episode["alerts"] != 1]
    failed_ids = {
        (episode.get("camera_id"), episode.get("event_type"), episode.get("start_ns"))
        for episode in failed
    }
    allowance_ids = {
        (item.get("camera_id"), item.get("event_type"), item.get("episode_start_ns"))
        for item in allowance
        if isinstance(item, dict)
        and item.get("track_id_switch_total") == 1
        and isinstance(item.get("elapsed_ns"), int)
        and item["elapsed_ns"] > 5_000_000_000
    }
    return (
        1 <= len(allowance) <= 10
        and len(allowance) == len(failed)
        and len(allowance_ids) == len(allowance)
        and allowance_ids == failed_ids
    )


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
    result["ac1_passed"] = _ac1_passed(result)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":")))
    return (
        2
        if provisional
        else 0
        if result["ac1_passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
