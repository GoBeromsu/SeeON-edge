"""Create a stratified human-labelling worksheet of candidate episodes.

An *episode* merges consecutive incidents on one camera and event type whose
``detected_at`` gaps stay inside the owner-confirmed horizon (fall 120 s,
bed-exit 60 s). Only episodes whose first incident has a clip are candidates,
because the labeller judges the clip. Candidates are spread evenly across each
camera/type bucket's time span (deterministic, no randomness) so the worksheet
is not the first hour of the corpus.

Run with ``python -m scripts.qa.golden_worksheet``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tests_support.golden_episodes import EPISODE_HORIZON_SEC

LABELS = ("real", "false", "unsure")


def _time(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


def episodes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge one bucket's time-ordered incidents into episodes."""
    merged: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item["detected_at"])):
        event_type = str(row["event_type"])
        horizon = EPISODE_HORIZON_SEC.get(event_type.replace("-", "_"), 60)
        if merged:
            last = merged[-1]
            gap = (_time(row["detected_at"]) - _time(last["last_detected_at"])).total_seconds()
            if gap <= horizon:
                last["last_detected_at"] = row["detected_at"]
                last["incident_count"] = int(str(last["incident_count"])) + 1
                last["edge_event_ids"] = f"{last['edge_event_ids']} {row['edge_event_id']}"
                if not last.get("clip_path") and row.get("clip_path"):
                    last["clip_path"] = row["clip_path"]
                    last["clip_started_at"] = row.get("clip_started_at")
                    last["clip_duration_s"] = row.get("clip_duration_s")
                continue
        merged.append(
            {
                "episode_id": f"{row['camera_id']}:{row['event_type']}:{row['detected_at']}",
                "camera_id": row["camera_id"],
                "event_type": row["event_type"],
                "detected_at": row["detected_at"],
                "last_detected_at": row["detected_at"],
                "incident_count": 1,
                "edge_event_ids": str(row["edge_event_id"]),
                "clip_path": row.get("clip_path"),
                "clip_started_at": row.get("clip_started_at"),
                "clip_duration_s": row.get("clip_duration_s"),
            }
        )
    return merged


def _spread(items: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def _roster(value: str) -> tuple[str, ...]:
    path = Path(value)
    raw = path.read_text(encoding="utf-8") if "," not in value and path.is_file() else value
    roster = tuple(camera.strip() for camera in raw.replace("\n", ",").split(",") if camera.strip())
    if not roster or len(set(roster)) != len(roster):
        raise ValueError("roster must contain unique camera ids")
    return roster


def build(manifest: Path, output: Path, limit: int, roster: tuple[str, ...]) -> int:
    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if str(row["camera_id"]) not in roster:
            continue
        buckets[(str(row["camera_id"]), str(row["event_type"]))].append(row)
    candidates: dict[tuple[str, str], list[dict[str, object]]] = {
        key: [episode for episode in episodes(rows) if episode.get("clip_path")]
        for key, rows in buckets.items()
    }
    by_camera: dict[str, list[dict[str, object]]] = defaultdict(list)
    for items in candidates.values():
        for episode in items:
            by_camera[str(episode["camera_id"])].append(episode)
    shortfall = [camera for camera in roster if len(by_camera[camera]) < 5]
    if shortfall:
        raise ValueError(f"roster cameras have fewer than five candidates: {', '.join(shortfall)}")
    selected = []
    # A golden corpus is only representative when every included camera has
    # at least five independently reviewable episodes.
    for camera_id in roster:
        selected.extend(_spread(by_camera[camera_id], 5))
    selected_ids = {str(episode["episode_id"]) for episode in selected}
    keys = sorted(key for key, items in candidates.items() if items)
    if keys:
        per_bucket = max(1, limit // len(keys))
        for key in keys:
            selected.extend(
                episode
                for episode in _spread(candidates[key], per_bucket)
                if str(episode["episode_id"]) not in selected_ids
            )
            selected_ids = {str(episode["episode_id"]) for episode in selected}
        remaining = limit - len(selected)
        for key in keys:
            if remaining <= 0:
                break
            chosen = {episode["episode_id"] for episode in selected}
            extra = [episode for episode in candidates[key] if episode["episode_id"] not in chosen]
            take = _spread(extra, min(remaining, 1))
            selected.extend(take)
            remaining -= len(take)
    selected = selected[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id",
        "clip_path",
        "clip_started_at",
        "clip_duration_s",
        "camera_roster",
        "thumbnail_path",
        "camera_id",
        "event_type",
        "detected_at",
        "last_detected_at",
        "incident_count",
        "edge_event_ids",
        "label",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in selected:
            clip = str(episode["clip_path"])
            writer.writerow(
                {
                    **{name: episode.get(name, "") for name in fieldnames},
                    "thumbnail_path": str(Path(clip).with_name("thumbnail.jpg")),
                    "camera_roster": ",".join(roster),
                    "label": "",
                }
            )
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Stratified golden worksheet; fill `label` with one of {LABELS}."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--roster", required=True, help="Comma-separated camera ids or a roster file."
    )
    args = parser.parse_args()
    try:
        print(f"episodes={build(args.manifest, args.out, args.limit, _roster(args.roster))}")
    except (OSError, ValueError) as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
