"""Validation and loading for the versioned golden episode corpus."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from scripts.qa.golden_worksheet import EPISODE_HORIZON_SEC

ALLOWED_LABELS = frozenset(("real", "false", "unsure"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class GoldenEpisode:
    episode_id: str
    camera_id: str
    event_type: str
    start_ns: int
    end_ns: int
    labels: dict[str, str]
    resolved: str
    resolution: str
    corroborating_overlap_s: float


def load_golden_episodes(path: Path) -> tuple[GoldenEpisode, ...]:
    """Load a complete golden-episodes-v1 JSON fixture or raise ``ValueError``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid golden fixture {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError("golden fixture must be an object")
    labellers = _header(payload)
    provisional = payload["provisional"]
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise TypeError("episodes must be a list")
    episodes = tuple(_episode(row, labellers, provisional) for row in rows)
    _validate_windows(episodes)
    if not provisional:
        if len(episodes) != 100:
            raise ValueError("non-provisional golden fixture must contain exactly 100 episodes")
        cameras = {episode.camera_id for episode in episodes}
        if any(sum(item.camera_id == camera for item in episodes) < 5 for camera in cameras):
            raise ValueError(
                "non-provisional golden fixture needs at least five episodes per camera"
            )
    return episodes


def _header(payload: dict[str, Any]) -> tuple[str, ...]:
    if payload.get("schema") != "golden-episodes-v1":
        raise ValueError("unsupported golden schema")
    if payload.get("horizons") != EPISODE_HORIZON_SEC:
        raise ValueError("golden horizons do not match the canonical horizon table")
    digest = payload.get("corpus_sha256")
    provisional_header = payload.get("provisional") is True and payload.get("episodes") == []
    if digest is None and provisional_header:
        pass  # empty provisional placeholder: no corpus has been labelled yet
    elif not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("invalid corpus_sha256")
    labellers = payload.get("labellers")
    if (
        not isinstance(labellers, list)
        or not all(isinstance(item, str) and item for item in labellers)
        or len(set(labellers)) != len(labellers)
    ):
        raise ValueError("invalid labellers")
    provisional = payload.get("provisional")
    if not isinstance(provisional, bool):
        raise TypeError("provisional must be boolean")
    if len(labellers) < 2 and not provisional:
        raise ValueError("fewer than two labellers requires provisional=true")
    return tuple(labellers)


def _episode(row: Any, labellers: tuple[str, ...], provisional: bool) -> GoldenEpisode:
    if not isinstance(row, dict):
        raise TypeError("episode must be an object")
    labels = row.get("labels")
    if (
        not isinstance(labels, dict)
        or set(labels) != set(labellers)
        or any(value not in ALLOWED_LABELS for value in labels.values())
    ):
        raise ValueError("invalid episode labels")
    try:
        episode = GoldenEpisode(
            episode_id=row["episode_id"],
            camera_id=row["camera_id"],
            event_type=row["event_type"],
            start_ns=row["start_ns"],
            end_ns=row["end_ns"],
            labels=labels,
            resolved=row["resolved"],
            resolution=row["resolution"],
            corroborating_overlap_s=row["corroborating_overlap_s"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid episode fields") from exc
    if (
        not isinstance(episode.episode_id, str)
        or not episode.episode_id
        or not isinstance(episode.camera_id, str)
        or not episode.camera_id
        or episode.event_type not in EPISODE_HORIZON_SEC
        or not isinstance(episode.start_ns, int)
        or not isinstance(episode.end_ns, int)
        or episode.start_ns >= episode.end_ns
        or episode.resolved not in ALLOWED_LABELS
        or episode.resolution not in {"agree", "third-pass", "single"}
        or not isinstance(episode.corroborating_overlap_s, (int, float))
        or episode.corroborating_overlap_s < 1
    ):
        raise ValueError("invalid episode")
    values = tuple(labels[labeller] for labeller in labellers)
    if episode.resolution == "single":
        if not provisional or len(values) != 1 or episode.resolved != values[0]:
            raise ValueError("invalid single-labeller resolution")
    elif episode.resolution == "agree":
        if len(values) < 2 or values[0] != values[1] or episode.resolved != values[0]:
            raise ValueError("invalid agreeing resolution")
    elif len(values) < 3 or values[0] == values[1] or episode.resolved != values[2]:
        raise ValueError("disagreement requires a third-pass label")
    return episode


def _validate_windows(episodes: tuple[GoldenEpisode, ...]) -> None:
    seen: set[str] = set()
    by_stream: dict[tuple[str, str], list[GoldenEpisode]] = {}
    for episode in episodes:
        if episode.episode_id in seen:
            raise ValueError("duplicate episode_id")
        seen.add(episode.episode_id)
        by_stream.setdefault((episode.camera_id, episode.event_type), []).append(episode)
    for stream in by_stream.values():
        ordered = sorted(stream, key=lambda item: item.start_ns)
        if any(left.end_ns > right.start_ns for left, right in pairwise(ordered)):
            raise ValueError("overlapping golden windows within camera/event type")


__all__ = ["ALLOWED_LABELS", "EPISODE_HORIZON_SEC", "GoldenEpisode", "load_golden_episodes"]
