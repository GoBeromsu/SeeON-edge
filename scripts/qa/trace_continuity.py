"""Per-track continuity statistics from a worker replay trace (`/traces/*.jsonl`).

Diagnostic only. Answers "are tracked people observed on consecutive frames?"
without touching production: presence ratio per NvDCF id, gap histogram between
appearances, seq parity of object rows, rebirths of the same id, and the
people-per-frame histogram. A healthy serving path shows gap-1 share >= 0.9 and
no parity skew; the #503 defect showed gap-1 ~0.10, gap-2 ~0.42 and 90 % of
object rows on one seq parity.

Usage:
    python scripts/qa/trace_continuity.py /tmp/p1b-flow/traces/<trace>.jsonl [--last-rows 27000]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
from itertools import pairwise
from pathlib import Path


def analyze(path: Path, *, last_rows: int | None, tail_bytes: int) -> dict[str, object]:
    with path.open("rb") as handle:
        handle.seek(max(0, os.path.getsize(path) - tail_bytes))
        lines = handle.read().decode(errors="ignore").splitlines()[1:]
    if last_rows is not None:
        lines = lines[-last_rows:]
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    present: collections.Counter[int] = collections.Counter()
    births: collections.Counter[int] = collections.Counter()
    parity: collections.Counter[int] = collections.Counter()
    people: collections.Counter[int] = collections.Counter()
    appearances: dict[int, list[int]] = collections.defaultdict(list)
    scores: list[float] = []
    camera = None
    for index, line in enumerate(lines):
        row = json.loads(line)
        camera = row.get("camera_id", camera)
        live = [track for track in row["tracks"] if track["lifecycle"] != "lost"]
        people[len(live)] += 1
        if live:
            parity[row["seq"] % 2] += 1
        for track in live:
            track_id = track["track_id"]
            first.setdefault(track_id, index)
            last[track_id] = index
            present[track_id] += 1
            appearances[track_id].append(index)
            scores.append(track["bbox"][4])
            if track["lifecycle"] == "new":
                births[track_id] += 1
    gaps: collections.Counter[int] = collections.Counter()
    for indices in appearances.values():
        for start, end in pairwise(indices):
            gaps[end - start] += 1
    ratios = [present[t] / (last[t] - first[t] + 1) for t in first if last[t] - first[t] + 1 >= 30]
    total_gaps = sum(gaps.values()) or 1
    return {
        "trace": str(path),
        "camera_id": camera,
        "rows": len(lines),
        "people_per_frame": {str(k): v for k, v in sorted(people.items())},
        "ids": len(first),
        "ids_with_span_ge_1s": len(ratios),
        "presence_median": round(statistics.median(ratios), 3) if ratios else None,
        "presence_p10": round(sorted(ratios)[int(len(ratios) * 0.1)], 3) if ratios else None,
        "max_rebirths_one_id": max(births.values()) if births else 0,
        "gap_share": {str(k): round(gaps[k] / total_gaps, 3) for k in (1, 2, 3, 4)},
        "object_row_seq_parity": {str(k): v for k, v in sorted(parity.items())},
        "score_p50": round(statistics.median(scores), 3) if scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--last-rows", type=int, default=None, help="analyze only the last N rows")
    parser.add_argument("--tail-bytes", type=int, default=60_000_000)
    args = parser.parse_args()
    for trace in args.trace:
        print(json.dumps(analyze(trace, last_rows=args.last_rows, tail_bytes=args.tail_bytes)))


if __name__ == "__main__":
    main()
