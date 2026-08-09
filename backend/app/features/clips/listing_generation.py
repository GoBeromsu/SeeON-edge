"""Filesystem preparation for immutable clip listing generations."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from backend.app.features.clips.listing import EventTypeFacet, effective_event_type
from backend.app.features.clips.manifest import discover_manifest_paths, read_manifest_file
from backend.app.features.clips.store import ClipManifest, ClipStore
from backend.app.features.clips.thumbnail_files import (
    ThumbnailFileState,
    thumbnail_file_state,
)

SqlParameter: TypeAlias = str | int | float | None
GLOBAL_CAMERA = ""
TOTAL_FACET = ""


@dataclass(frozen=True, slots=True)
class ReconcileStats:
    scanned: int
    read: int
    upserted: int
    deleted: int
    unchanged: int


@dataclass(frozen=True, slots=True)
class IndexedClip:
    manifest_path: str
    manifest_mtime_ns: int
    manifest_size_bytes: int
    clip_id: str
    camera_id: str
    event_ref: str
    event_type: str | None
    event_facet: EventTypeFacet
    started_at: str
    duration_s: float
    codec: str
    media_path: str | None
    video_available: bool
    video_error: str | None
    finalized: bool
    size_bytes: int | None
    thumbnail_mtime_ns: int | None
    thumbnail_size_bytes: int | None
    thumbnail_available: bool

    def base_sql_values(self, generation: int) -> tuple[SqlParameter, ...]:
        return (
            generation,
            self.clip_id,
            self.manifest_path,
            self.manifest_mtime_ns,
            self.manifest_size_bytes,
            self.camera_id,
            self.event_ref,
            self.event_type,
            self.event_facet,
            self.started_at,
            self.duration_s,
            self.codec,
            self.media_path,
            int(self.video_available),
            self.video_error,
            int(self.finalized),
            self.size_bytes,
        )

    def thumbnail_sql_values(self, generation: int) -> tuple[SqlParameter, ...]:
        return (
            generation,
            self.clip_id,
            self.thumbnail_mtime_ns,
            self.thumbnail_size_bytes,
            int(self.thumbnail_available),
        )

    def to_manifest(self) -> ClipManifest:
        return ClipManifest(
            clip_id=self.clip_id,
            camera_id=self.camera_id,
            event_ref=self.event_ref,
            event_type=self.event_type,
            started_at=self.started_at,
            duration_s=self.duration_s,
            codec=self.codec,
            path=self.media_path,
            video_available=self.video_available,
            video_error=self.video_error,
            finalized=self.finalized,
            size_bytes=self.size_bytes,
            thumbnail_available=self.thumbnail_available,
        )


@dataclass(frozen=True, slots=True)
class ListingSummary:
    camera_id: str
    event_facet: str
    count: int

    def sql_values(self, generation: int) -> tuple[SqlParameter, ...]:
        return generation, self.camera_id, self.event_facet, self.count


@dataclass(frozen=True, slots=True)
class PreparedGeneration:
    clips: tuple[IndexedClip, ...]
    summaries: tuple[ListingSummary, ...]
    stats: ReconcileStats


class ClipListingPreparationError(ValueError):
    pass


def prepare_generation(
    clip_store: ClipStore,
    existing: Mapping[str, IndexedClip],
) -> PreparedGeneration:
    paths = sorted(discover_manifest_paths(clip_store.root))
    current: dict[str, IndexedClip] = {}
    read_count = 0
    unchanged = 0
    upserted = 0
    for manifest_path in paths:
        try:
            file_stat = manifest_path.stat()
        except FileNotFoundError:
            continue
        path_text = str(manifest_path)
        previous = existing.get(path_text)
        thumbnail_state = thumbnail_file_state(clip_store.root, manifest_path)
        if previous is not None and _same_fingerprint(previous, file_stat, thumbnail_state):
            current[path_text] = previous
            unchanged += 1
            continue
        read_count += 1
        manifest = read_manifest_file(manifest_path)
        if (
            manifest is None
            or not manifest.finalized
            or manifest.clip_id != manifest_path.parent.name
        ):
            continue
        size_bytes = None
        if manifest.video_available:
            try:
                size_bytes = clip_store.resolve_video_path(manifest).stat().st_size
            except (ValueError, FileNotFoundError, OSError):
                size_bytes = None
        current[path_text] = _indexed_clip(
            manifest_path,
            file_stat,
            manifest,
            size_bytes,
            thumbnail_state,
        )
        upserted += 1
    clips = tuple(current.values())
    if len({clip.clip_id for clip in clips}) != len(clips):
        raise ClipListingPreparationError("duplicate clip_id across manifest layouts")
    summaries = _summaries(clips)
    stats = ReconcileStats(
        scanned=len(paths),
        read=read_count,
        upserted=upserted,
        deleted=len(set(existing) - set(current)),
        unchanged=unchanged,
    )
    return PreparedGeneration(clips, summaries, stats)


def _same_fingerprint(
    clip: IndexedClip,
    file_stat: os.stat_result,
    thumbnail_state: ThumbnailFileState,
) -> bool:
    return (
        clip.manifest_mtime_ns == file_stat.st_mtime_ns
        and clip.manifest_size_bytes == file_stat.st_size
        and clip.thumbnail_mtime_ns == thumbnail_state.mtime_ns
        and clip.thumbnail_size_bytes == thumbnail_state.size_bytes
        and clip.thumbnail_available == thumbnail_state.available
    )


def _indexed_clip(
    path: Path,
    file_stat: os.stat_result,
    manifest: ClipManifest,
    size_bytes: int | None,
    thumbnail_state: ThumbnailFileState,
) -> IndexedClip:
    return IndexedClip(
        manifest_path=str(path),
        manifest_mtime_ns=file_stat.st_mtime_ns,
        manifest_size_bytes=file_stat.st_size,
        clip_id=manifest.clip_id,
        camera_id=manifest.camera_id,
        event_ref=manifest.event_ref,
        event_type=manifest.event_type,
        event_facet=effective_event_type(manifest),
        started_at=manifest.started_at,
        duration_s=manifest.duration_s,
        codec=manifest.codec,
        media_path=manifest.path,
        video_available=manifest.video_available,
        video_error=manifest.video_error,
        finalized=manifest.finalized,
        size_bytes=size_bytes,
        thumbnail_mtime_ns=thumbnail_state.mtime_ns,
        thumbnail_size_bytes=thumbnail_state.size_bytes,
        thumbnail_available=thumbnail_state.available,
    )


def _summaries(clips: tuple[IndexedClip, ...]) -> tuple[ListingSummary, ...]:
    counts: Counter[tuple[str, str]] = Counter()
    for clip in clips:
        counts[(GLOBAL_CAMERA, TOTAL_FACET)] += 1
        counts[(GLOBAL_CAMERA, clip.event_facet)] += 1
        counts[(clip.camera_id, TOTAL_FACET)] += 1
        counts[(clip.camera_id, clip.event_facet)] += 1
    return tuple(
        ListingSummary(camera_id, event_facet, count)
        for (camera_id, event_facet), count in sorted(counts.items())
    )


__all__ = [
    "ClipListingPreparationError",
    "GLOBAL_CAMERA",
    "IndexedClip",
    "ListingSummary",
    "PreparedGeneration",
    "ReconcileStats",
    "TOTAL_FACET",
    "prepare_generation",
]
