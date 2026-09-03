"""Export a de-identified incident/clip corpus from an edge SQLite snapshot.

Clips are bound to incidents through the clip manifests' ``event_refs`` (the
worker-side truth), not through ``artifacts`` rows: on pre-P0 databases every
incident's primary-clip binding is missing because the backend never completed
the lifecycle, so an ``artifacts`` join yields no clips at all.

Only identity, camera, type, time and clip location leave the snapshot.
Credentials, audit rows, RTSP URLs and review notes are never read.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def _clip_index(clip_store: Path) -> dict[str, dict[str, object]]:
    """Map every ``event_refs`` entry to its clip location and duration."""
    index: dict[str, dict[str, object]] = {}
    clips_dir = clip_store / "clips"
    if not clips_dir.is_dir():
        return index
    for manifest_path in sorted(clips_dir.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        clip_path = manifest_path.parent / "clip.mp4"
        if not clip_path.is_file():
            continue
        refs = manifest.get("event_refs") or [manifest.get("event_ref")]
        record = {
            "clip_id": manifest_path.parent.name,
            "clip_path": str(clip_path),
            "duration_s": manifest.get("duration_s"),
        }
        for ref in refs:
            if isinstance(ref, str) and ref not in index:
                index[ref] = record
    return index


def export(snapshot: Path, clip_store: Path, output: Path) -> tuple[int, int, float]:
    """Write the corpus and return (incidents, incidents_with_clip, alerts/hour)."""
    connection = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT incident_id, edge_event_id, camera_id, event_type, detected_at "
        "FROM incidents ORDER BY detected_at"
    ).fetchall()
    clips = _clip_index(clip_store)
    records: list[dict[str, object]] = []
    with_clip = 0
    for row in rows:
        clip = clips.get(str(row["edge_event_id"]))
        if clip is not None:
            with_clip += 1
        records.append(
            {key: row[key] for key in row.keys()}  # noqa: SIM118 - sqlite3.Row
            | {
                "clip_id": clip["clip_id"] if clip else None,
                "clip_path": clip["clip_path"] if clip else None,
                "duration_s": clip["duration_s"] if clip else None,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    times = [str(row["detected_at"]) for row in rows if row["detected_at"]]
    if len(times) < 2:
        return len(records), with_clip, 0.0
    span = (_parse_time(times[-1]) - _parse_time(times[0])).total_seconds() / 3600
    return len(records), with_clip, len(records) / span if span else 0.0


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--clip-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count, with_clip, alerts_per_hour = export(args.snapshot, args.clip_store, args.out)
    print(f"incidents={count} with_clip={with_clip} alerts_per_hour={alerts_per_hour:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
