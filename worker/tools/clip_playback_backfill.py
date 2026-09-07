"""Create missing browser playback renditions for immutable evidence clips."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path

from worker.adapters.media.ffmpeg_thumbnail import (
    FfmpegThumbnailGenerator,
    ThumbnailUnavailable,
)
from worker.pipeline.output.evidence.clip_identity import is_clip_id
from worker.pipeline.output.evidence.playback_rendition import (
    PLAYBACK_MANIFEST_NAME,
    PLAYBACK_RENDITION_PREFIX,
    PlaybackRenditionError,
    probe_video_codec,
    write_playback_rendition,
)

LOGGER = logging.getLogger(__name__)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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
    for clip_path in _clip_paths(clip_store):
        summary["scanned"] += 1
        clip_id = clip_path.parent.name
        if _rendition_is_identical(clip_path):
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


def _rendition_is_identical(clip_path: Path) -> bool:
    try:
        payload = json.loads(
            (clip_path.parent / PLAYBACK_MANIFEST_NAME).read_text(encoding="ascii")
        )
        source_digest = _manifest_source_sha256(clip_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    rendition_name = payload.get("rendition")
    rendition_digest = payload.get("rendition_sha256")
    if not _valid_attestation(payload, source_digest, rendition_name, rendition_digest):
        return False
    try:
        actual_digest = _sha256(clip_path.parent / rendition_name)
    except OSError:
        return False
    return (
        payload.get("pts_identical") is True
        and payload.get("source_sha256") == source_digest
        and rendition_digest == actual_digest
    )


def _manifest_source_sha256(clip_path: Path) -> str | None:
    payload = json.loads(clip_path.with_name("manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    digest = payload.get("sha256")
    return digest if isinstance(digest, str) and _SHA256_RE.fullmatch(digest) else None


def _valid_attestation(
    payload: dict[object, object],
    source_digest: str | None,
    rendition_name: object,
    rendition_digest: object,
) -> bool:
    frames = payload.get("frames")
    source_frames = payload.get("source_frames")
    time_base = payload.get("time_base")
    return (
        isinstance(rendition_name, str)
        and isinstance(rendition_digest, str)
        and _SHA256_RE.fullmatch(rendition_digest) is not None
        and rendition_name == f"{PLAYBACK_RENDITION_PREFIX}{rendition_digest[:16]}.mp4"
        and payload.get("source_sha256") == source_digest
        and payload.get("pts_identical") is True
        and isinstance(time_base, str)
        and re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", time_base) is not None
        and isinstance(frames, int)
        and not isinstance(frames, bool)
        and frames >= 0
        and isinstance(source_frames, int)
        and not isinstance(source_frames, bool)
        and source_frames >= 0
    )


def _clip_paths(clip_store: Path) -> tuple[Path, ...]:
    paths: dict[str, Path] = {}
    for clips_root in _bounded_clip_roots(clip_store):
        try:
            candidates = tuple(clips_root.iterdir())
        except OSError:
            continue
        for candidate in candidates:
            clip_path = candidate / "clip.mp4"
            if (
                is_clip_id(candidate.name)
                and candidate.is_dir()
                and clip_path.is_file()
                and candidate.name not in paths
            ):
                paths[candidate.name] = clip_path
    return tuple(sorted(paths.values()))


def _bounded_clip_roots(clip_store: Path) -> tuple[Path, ...]:
    roots = [clip_store / "clips"]
    try:
        first_level = tuple(clip_store.iterdir())
    except OSError:
        return tuple(roots)
    for first in first_level:
        if first.name == "clips" or not first.is_dir():
            continue
        first_clips = first / "clips"
        if first_clips.is_dir():
            roots.append(first_clips)
        try:
            second_level = tuple(first.iterdir())
        except OSError:
            continue
        for second in second_level:
            if second.name == "clips" or not second.is_dir():
                continue
            second_clips = second / "clips"
            if second_clips.is_dir():
                roots.append(second_clips)
    return tuple(roots)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for clip_path in _clip_paths(clip_store):
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
