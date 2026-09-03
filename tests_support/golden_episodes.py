"""Strict owner-supplied golden episode fixture reader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ALLOWED_LABELS = frozenset(("real", "false", "unsure"))


@dataclass(frozen=True, slots=True)
class GoldenEpisode:
    camera_id: str
    event_type: str
    start_ns: int
    end_ns: int
    label: str

    def __post_init__(self) -> None:
        if not self.camera_id or self.event_type not in ("fall", "bed_exit"):
            raise ValueError("invalid camera_id or event_type")
        if self.start_ns > self.end_ns or self.label not in ALLOWED_LABELS:
            raise ValueError("invalid golden episode")


def load_golden_episodes(path: Path) -> tuple[GoldenEpisode, ...]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            try:
                rows.append(GoldenEpisode(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid golden fixture {path}") from exc
    return tuple(rows)


__all__ = ["ALLOWED_LABELS", "GoldenEpisode", "load_golden_episodes"]
