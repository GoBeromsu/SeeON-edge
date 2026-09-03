"""Export a de-identified incident/clip corpus from an edge SQLite snapshot.

Clips are bound to incidents through the clip manifests' ``event_refs`` (the
worker-side truth), not through ``artifacts`` rows: on pre-P0 databases every
incident's primary-clip binding is missing because the backend never completed
the lifecycle, so an ``artifacts`` join yields no clips at all.

Only identity, camera, type, time and clip location leave the snapshot.
Credentials, audit rows, RTSP URLs and review notes are never read.

Run with ``python -m scripts.qa.export_incident_corpus``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

_CLIP_ID = re.compile(r"^[A-Za-z0-9:_-]{1,128}$")


class CorpusValidationError(ValueError):
    """The local clip store cannot safely support a corpus export."""


def _clip_index(clip_store: Path) -> dict[str, dict[str, object]]:
    """Map claimed event refs from the canonical clip layout to local media."""
    index: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    unavailable: list[str] = []
    clips_dir = clip_store / "clips"
    if not clips_dir.is_dir():
        raise CorpusValidationError(f"missing canonical clips tree: {clips_dir}")
    for manifest_path in sorted(clips_dir.glob("*/manifest.json")):
        clip_id = manifest_path.parent.name
        if manifest_path.parent.is_symlink() or manifest_path.is_symlink():
            errors.append(f"{clip_id}: manifest must be a regular contained file")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            errors.append(f"{clip_id}: malformed manifest")
            continue
        if (
            not _CLIP_ID.fullmatch(clip_id)
            or not isinstance(manifest, dict)
            or manifest.get("clip_id") != clip_id
        ):
            errors.append(f"{clip_id}: invalid clip_id")
            continue
        refs = manifest.get("event_refs")
        if (
            not isinstance(refs, list)
            or not refs
            or len(set(refs)) != len(refs)
            or any(not isinstance(ref, str) or not ref for ref in refs)
        ):
            errors.append(f"{clip_id}: invalid event_refs")
            continue
        started_at = manifest.get("started_at")
        duration_s = manifest.get("duration_s")
        if (
            not isinstance(started_at, str)
            or not started_at.endswith("Z")
            or not isinstance(duration_s, (int, float))
            or isinstance(duration_s, bool)
            or not math.isfinite(duration_s)
            or duration_s < 0
        ):
            errors.append(f"{clip_id}: invalid clip timing")
            continue
        try:
            datetime.fromisoformat(started_at.removesuffix("Z") + "+00:00")
        except ValueError:
            errors.append(f"{clip_id}: invalid clip timing")
            continue
        clip_path = manifest_path.parent / "clip.mp4"
        if manifest.get("video_available") is False:
            # Declared by the worker (e.g. STREAM_EPOCH_MISMATCH): not a corpus
            # defect, just no media to label. Recorded, never a candidate.
            unavailable.append(clip_id)
            continue
        if not clip_path.is_file() or clip_path.is_symlink():
            errors.append(f"{clip_id}: missing clip.mp4")
            continue
        record = {
            "clip_id": clip_id,
            "clip_path": str(clip_path),
            "clip_started_at": started_at,
            "clip_duration_s": duration_s,
        }
        for ref in refs:
            existing = index.get(ref)
            if existing is not None:
                errors.append(
                    f"{ref}: duplicate event_ref claimed by {existing['clip_id']} and {clip_id}"
                )
            else:
                index[ref] = record
    if errors:
        raise CorpusValidationError("\n".join(errors))
    if unavailable:
        print(f"declared_unavailable_clips={len(unavailable)}")
    return index


def export(snapshot: Path, clip_store: Path, output: Path) -> tuple[int, int, float]:
    """Write the corpus and return (incidents, incidents_with_clip, alerts/hour)."""
    with sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT incident_id, edge_event_id, camera_id, event_type, detected_at "
            "FROM incidents ORDER BY detected_at"
        ).fetchall()
    clips = _clip_index(clip_store)
    records: list[dict[str, object]] = []
    times: list[datetime] = []
    with_clip = 0
    for row in rows:
        detected_at = row["detected_at"]
        if not isinstance(detected_at, str) or not detected_at:
            raise CorpusValidationError("incident has missing detected_at timestamp")
        try:
            times.append(_parse_time(detected_at))
        except ValueError as exc:
            raise CorpusValidationError(f"invalid detected_at timestamp: {detected_at}") from exc
        required_values = (
            row["incident_id"],
            row["edge_event_id"],
            row["camera_id"],
            row["event_type"],
        )
        if not all(isinstance(value, str) and value for value in required_values):
            raise CorpusValidationError("incident contains invalid required fields")
        clip = clips.get(row["edge_event_id"])
        if clip is not None:
            with_clip += 1
        records.append(
            {key: row[key] for key in row.keys()}  # noqa: SIM118 - sqlite3.Row
            | {
                "clip_id": clip["clip_id"] if clip else None,
                "clip_path": clip["clip_path"] if clip else None,
                "clip_started_at": clip["clip_started_at"] if clip else None,
                "clip_duration_s": clip["clip_duration_s"] if clip else None,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.writelines(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    os.replace(temporary, output)
    if len(times) < 2:
        return len(records), with_clip, 0.0
    span = (times[-1] - times[0]).total_seconds() / 3600
    return len(records), with_clip, len(records) / span if span else 0.0


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--clip-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        count, with_clip, alerts_per_hour = export(args.snapshot, args.clip_store, args.out)
    except (CorpusValidationError, OSError, sqlite3.Error) as exc:
        print(exc)
        return 1
    print(f"incidents={count} with_clip={with_clip} alerts_per_hour={alerts_per_hour:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
