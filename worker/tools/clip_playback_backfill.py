"""Create missing browser playback renditions for immutable evidence clips."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from worker.pipeline.output.evidence.playback_rendition import (
    PLAYBACK_NAME,
    PlaybackRenditionError,
    probe_video_codec,
    write_playback_rendition,
)

LOGGER = logging.getLogger(__name__)


def backfill(clip_store: Path, *, dry_run: bool = False) -> dict[str, int | bool]:
    """Backfill missing renditions and return a machine-readable summary."""
    summary: dict[str, int | bool] = {
        "scanned": 0,
        "h264": 0,
        "created": 0,
        "skipped": 0,
        "failed": 0,
        "dry_run": dry_run,
    }
    for clip_path in sorted((clip_store / "clips").glob("*/clip.mp4")):
        summary["scanned"] += 1
        clip_id = clip_path.parent.name
        if (clip_path.parent / PLAYBACK_NAME).exists():
            summary["skipped"] += 1
            continue
        try:
            codec = probe_video_codec(clip_path)
            if codec in {"avc1", "h264"}:
                summary["h264"] += 1
                continue
            if dry_run:
                summary["skipped"] += 1
                continue
            if write_playback_rendition(clip_path) is not None:
                summary["created"] += 1
        except PlaybackRenditionError as exc:
            summary["failed"] += 1
            LOGGER.warning(
                "playback rendition backfill failed stage=transcode clip_id=%s exception_class=%s",
                clip_id,
                type(exc).__name__,
            )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_store", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    print(json.dumps(backfill(arguments.clip_store, dry_run=arguments.dry_run), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
