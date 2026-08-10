"""Idempotent historical thumbnail backfill over bounded clip layouts."""

from __future__ import annotations

import logging
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from worker.adapters.encode.adapter_errors import ThumbnailGenerationError
from worker.adapters.encode.thumbnail import THUMBNAIL_FILENAME, is_valid_thumbnail
from worker.interfaces import ThumbnailGenerator
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLock

LOGGER: Final = logging.getLogger(__name__)
_MANIFEST_PAYLOAD: Final = TypeAdapter(dict[str, JsonValue])
_CLIP_ID_RE: Final = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MEDIA_SUFFIXES: Final = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv"})


@dataclass(frozen=True, slots=True)
class BackfillReport:
    scanned: int
    playable: int
    generated: int
    skipped: int
    failed: int
    missing: int

    def summary(self) -> str:
        return (
            f"thumbnail backfill: scanned={self.scanned} playable={self.playable} "
            f"generated={self.generated} skipped={self.skipped} "
            f"failed={self.failed} missing={self.missing}"
        )


@dataclass(frozen=True, slots=True)
class _PlayableClip:
    manifest_path: Path
    video_path: Path
    duration_s: float


def backfill_thumbnails(root: Path, generator: ThumbnailGenerator) -> BackfillReport:
    recording_roots = _recording_roots(root)
    with ExitStack() as locks:
        for recording_root in recording_roots:
            locks.enter_context(ClipStoreLock.acquire(recording_root))
        return _backfill_locked(root, recording_roots, generator)


def _backfill_locked(
    root: Path,
    recording_roots: tuple[Path, ...],
    generator: ThumbnailGenerator,
) -> BackfillReport:
    manifest_paths = _manifest_paths(recording_roots)
    playable = tuple(
        clip
        for manifest_path in manifest_paths
        if (clip := _playable_clip(root, manifest_path)) is not None
    )
    generated = 0
    skipped = 0
    failed = 0
    for clip in playable:
        thumbnail_path = clip.manifest_path.parent / THUMBNAIL_FILENAME
        if is_valid_thumbnail(thumbnail_path):
            skipped += 1
            continue
        try:
            _ = generator.generate(clip.video_path, thumbnail_path, clip.duration_s)
        except ThumbnailGenerationError as exc:
            failed += 1
            LOGGER.warning(
                "thumbnail backfill failed manifest=%r error_type=%s",
                str(clip.manifest_path),
                type(exc).__name__,
            )
            continue
        generated += 1
    missing = sum(
        not is_valid_thumbnail(clip.manifest_path.parent / THUMBNAIL_FILENAME)
        for clip in playable
    )
    return BackfillReport(
        scanned=len(manifest_paths),
        playable=len(playable),
        generated=generated,
        skipped=skipped,
        failed=failed,
        missing=missing,
    )


def _recording_roots(root: Path) -> tuple[Path, ...]:
    roots = {root}
    first_level = _directories(root)
    for first in first_level:
        if _is_recording_root(first):
            roots.add(first)
        for second in _directories(first):
            if _is_recording_root(second):
                roots.add(second)
    return tuple(sorted(roots, key=lambda path: str(path.absolute())))


def _directories(root: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(path for path in root.iterdir() if _is_directory(path)))
    except OSError:
        return ()


def _is_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _is_recording_root(path: Path) -> bool:
    return _is_directory(path / "clips") or _is_regular_file(path / ".worker.lock")


def _manifest_paths(recording_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    paths = {
        manifest_path
        for recording_root in recording_roots
        for manifest_path in recording_root.glob("clips/*/manifest.json")
    }
    return tuple(sorted(paths))


def _playable_clip(root: Path, manifest_path: Path) -> _PlayableClip | None:
    try:
        manifest_stat = manifest_path.lstat()
        resolved_parent = manifest_path.parent.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError:
        return None
    if not stat.S_ISREG(manifest_stat.st_mode) or (
        resolved_parent != resolved_root and resolved_root not in resolved_parent.parents
    ):
        return None
    try:
        payload = _MANIFEST_PAYLOAD.validate_json(manifest_path.read_bytes())
    except (OSError, ValidationError):
        return None
    clip_id = payload.get("clip_id")
    if (
        not isinstance(clip_id, str)
        or not _CLIP_ID_RE.fullmatch(clip_id)
        or clip_id != manifest_path.parent.name
        or payload.get("finalized") is not True
    ):
        return None
    video_available = payload.get("video_available")
    if video_available is False or (video_available is None and payload.get("path") is None):
        return None
    video_path = _adjacent_video(manifest_path.parent, clip_id)
    if video_path is None:
        return None
    raw_duration = payload.get("duration_s")
    duration_s = (
        float(raw_duration)
        if isinstance(raw_duration, int | float) and not isinstance(raw_duration, bool)
        else 0.0
    )
    return _PlayableClip(manifest_path, video_path, max(0.0, duration_s))


def _adjacent_video(directory: Path, clip_id: str) -> Path | None:
    preferred = (
        directory / f"{clip_id}.mp4",
        directory / "clip.mp4",
        directory / "video.mp4",
        directory / "final.mp4",
    )
    for candidate in preferred:
        if _is_regular_file(candidate):
            return candidate
    try:
        media = tuple(
            path
            for path in directory.iterdir()
            if path.suffix.lower() in _MEDIA_SUFFIXES and _is_regular_file(path)
        )
    except OSError:
        return None
    return media[0] if len(media) == 1 else None


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


__all__ = ["BackfillReport", "backfill_thumbnails"]
