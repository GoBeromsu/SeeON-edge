"""Create missing browser playback renditions for immutable evidence clips."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

from worker.adapters.media.ffmpeg_thumbnail import (
    FfmpegThumbnailGenerator,
    ThumbnailUnavailable,
)
from worker.pipeline.output.evidence.playback_rendition import (
    PLAYBACK_NAME,
    PLAYBACK_TIMING_NAME,
    PlaybackRenditionError,
    probe_video_codec,
    write_playback_rendition,
)

LOGGER = logging.getLogger(__name__)


def backfill(
    clip_store: Path,
    *,
    dry_run: bool = False,
    thumbnails: bool = False,
) -> dict[str, int | bool]:
    """Backfill missing renditions and return a machine-readable summary."""
    if thumbnails:
        return _backfill_thumbnails(clip_store, dry_run=dry_run)
    summary: dict[str, int | bool] = {
        "scanned": 0,
        "h264": 0,
        "created": 0,
        "skipped": 0,
        "pending": 0,
        "failed": 0,
        "dry_run": dry_run,
    }
    for clip_path in sorted((clip_store / "clips").glob("*/clip.mp4")):
        summary["scanned"] += 1
        clip_id = clip_path.parent.name
        rendition = clip_path.parent / PLAYBACK_NAME
        if rendition.exists() and _timing_is_identical(clip_path.parent / PLAYBACK_TIMING_NAME):
            summary["skipped"] += 1
            continue
        try:
            codec = probe_video_codec(clip_path)
            if codec in {"avc1", "h264"}:
                summary["h264"] += 1
                continue
            if dry_run:
                summary["pending"] += 1
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


def _timing_is_identical(sidecar: Path) -> bool:
    try:
        payload = json.loads(sidecar.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("pts_identical") is True


def _backfill_thumbnails(clip_store: Path, *, dry_run: bool) -> dict[str, int | bool]:
    summary: dict[str, int | bool] = {
        "scanned": 0,
        "created": 0,
        "skipped": 0,
        "pending": 0,
        "failed": 0,
        "dry_run": dry_run,
    }
    generator = FfmpegThumbnailGenerator()
    for clip_path in sorted((clip_store / "clips").glob("*/clip.mp4")):
        summary["scanned"] += 1
        clip_id = clip_path.parent.name
        thumbnail_path = clip_path.parent / "thumbnail.jpg"
        if thumbnail_path.is_file():
            summary["skipped"] += 1
            continue
        if dry_run:
            summary["pending"] += 1
            continue
        try:
            generator.generate(clip_path, thumbnail_path, _manifest_duration(clip_path.parent))
            summary["created"] += 1
        except (ThumbnailUnavailable, OSError, KeyError, ValueError) as exc:
            summary["failed"] += 1
            LOGGER.warning(
                "thumbnail backfill failed stage=thumbnail clip_id=%s exception_class=%s",
                clip_id,
                type(exc).__name__,
            )
    return summary


def _manifest_duration(clip_dir: Path) -> float:
    payload = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
    duration_s = payload["duration_s"]
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, int | float)
        or not math.isfinite(duration_s)
        or duration_s < 0
    ):
        raise ValueError("manifest duration is invalid")
    return float(duration_s)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_store", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--renditions", action="store_true")
    parser.add_argument("--thumbnails", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.renditions and arguments.thumbnails:
        parser.error("--renditions and --thumbnails are mutually exclusive")
    print(
        json.dumps(
            backfill(
                arguments.clip_store,
                dry_run=arguments.dry_run,
                thumbnails=arguments.thumbnails,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
