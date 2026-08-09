"""Read-only contained access to clip-local thumbnail files."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backend.app.features.clips.descriptor_files import read_bounded_regular_file

MAX_THUMBNAIL_BYTES: Final = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ThumbnailFileState:
    mtime_ns: int | None
    size_bytes: int | None
    available: bool


def bounded_clip_roots(root: Path) -> tuple[Path, ...]:
    roots = [root / "clips"]
    if not root.is_dir():
        return tuple(roots)
    try:
        first_level = tuple(root.iterdir())
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


def contained_thumbnail_path(root: Path, manifest_path: Path) -> Path | None:
    candidate = manifest_path.parent / "thumbnail.jpg"
    return candidate if thumbnail_file_state(root, manifest_path).available else None


def thumbnail_file_state(root: Path, manifest_path: Path) -> ThumbnailFileState:
    candidate = manifest_path.parent / "thumbnail.jpg"
    try:
        candidate_stat = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        resolved_root = root.resolve(strict=False)
    except OSError:
        return ThumbnailFileState(None, None, False)
    available = (
        stat.S_ISREG(candidate_stat.st_mode)
        and candidate_stat.st_size <= MAX_THUMBNAIL_BYTES
        and (resolved == resolved_root or resolved_root in resolved.parents)
    )
    return ThumbnailFileState(candidate_stat.st_mtime_ns, candidate_stat.st_size, available)


def read_regular_file(root: Path, path: Path) -> bytes:
    return read_bounded_regular_file(root, path, MAX_THUMBNAIL_BYTES)


__all__ = [
    "MAX_THUMBNAIL_BYTES",
    "ThumbnailFileState",
    "bounded_clip_roots",
    "contained_thumbnail_path",
    "read_regular_file",
    "thumbnail_file_state",
]
