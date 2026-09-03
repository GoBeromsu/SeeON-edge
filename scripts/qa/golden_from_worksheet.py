"""Build a versioned golden episode fixture from independent worksheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from scripts.qa.golden_worksheet import EPISODE_HORIZON_SEC

_EVENT_TYPE = {"fall": "fall", "bed-exit": "bed_exit", "bed_exit": "bed_exit"}
_LABELS = {"real", "false", "unsure"}


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1_000_000_000)


def _read(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not all(row.get("labeller") and row.get("label") in _LABELS for row in rows):
        raise ValueError(f"{path} must contain labelled rows and a labeller column")
    labellers = {str(row["labeller"]) for row in rows}
    if len(labellers) != 1:
        raise ValueError(f"{path} contains more than one labeller")
    keyed = {str(row.get("episode_id")): row for row in rows}
    if "" in keyed or len(keyed) != len(rows):
        raise ValueError(f"{path} has missing or duplicate episode_id")
    return labellers.pop(), keyed


def convert(
    worksheets: list[Path], output: Path, corpus: Path, third_pass: Path | None, lead_sec: float
) -> int:
    if not 1 <= len(worksheets) <= 2:
        raise ValueError("supply one or two worksheets")
    loaded = [_read(path) for path in worksheets]
    if len({labeller for labeller, _ in loaded}) != len(loaded):
        raise ValueError("worksheets must be independently labelled")
    ids = set(loaded[0][1])
    if any(set(rows) != ids for _, rows in loaded[1:]):
        raise ValueError("independent worksheets must cover identical episode_ids")
    third = _read(third_pass) if third_pass is not None else None
    if third is not None and third[0] in {labeller for labeller, _ in loaded}:
        raise ValueError("third pass must be independently labelled")
    labellers = [labeller for labeller, _ in loaded] + ([] if third is None else [third[0]])
    episodes = []
    disputed_ids: set[str] = set()
    for episode_id in sorted(ids):
        rows = [items[episode_id] for _, items in loaded]
        labels = {labeller: row["label"] for (labeller, _), row in zip(loaded, rows, strict=True)}
        if len(rows) == 1:
            resolved, resolution = rows[0]["label"], "single"
        elif rows[0]["label"] == rows[1]["label"]:
            resolved, resolution = rows[0]["label"], "agree"
            if third is not None and episode_id in third[1]:
                raise ValueError("third pass supplied for an agreeing episode")
        else:
            disputed_ids.add(episode_id)
            if third is None or episode_id not in third[1]:
                raise ValueError(f"{episode_id} disagrees and needs a third pass")
            labels[third[0]] = third[1][episode_id]["label"]
            resolved, resolution = labels[third[0]], "third-pass"
        event_type = _EVENT_TYPE.get(rows[0]["event_type"])
        if event_type is None or any(row["camera_id"] != rows[0]["camera_id"] for row in rows):
            raise ValueError(f"{episode_id} has inconsistent episode identity")
        starts = [_ns(row["detected_at"]) for row in rows]
        ends = [_ns(row["last_detected_at"]) for row in rows]
        episodes.append(
            {
                "episode_id": episode_id,
                "camera_id": rows[0]["camera_id"],
                "event_type": event_type,
                "start_ns": min(starts) - int(lead_sec * 1_000_000_000),
                "end_ns": max(ends) + EPISODE_HORIZON_SEC[event_type] * 1_000_000_000,
                "labels": labels,
                "resolved": resolved,
                "resolution": resolution,
                "corroborating_overlap_s": 1,
            }
        )
    if third is not None and set(third[1]) != disputed_ids:
        raise ValueError("third pass must contain exactly the disputed episode_ids")
    payload = {
        "schema": "golden-episodes-v1",
        "horizons": EPISODE_HORIZON_SEC,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "labellers": labellers,
        "provisional": len(loaded) < 2,
        "episodes": episodes,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(episodes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worksheet", type=Path, action="append", required=True)
    parser.add_argument("--third-pass", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lead", type=float, default=5.0)
    args = parser.parse_args()
    try:
        count = convert(args.worksheet, args.out, args.corpus, args.third_pass, args.lead)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"episodes={count} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
