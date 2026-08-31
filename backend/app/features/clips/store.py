"""Read-only manifest access for clip playback."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import final

from backend.app.features.clips.descriptor_files import (
    OpenedRegularFile,
    open_contained_regular_file,
)
from backend.app.features.clips.manifest import (
    ClipManifest,
    discover_manifest_paths,
    is_valid_clip_id,
    read_manifest_file,
    video_file_from_dir,
)
from backend.app.features.clips.scene_files import resolve_scene_index
from backend.app.features.clips.thumbnail_files import (
    bounded_clip_roots,
    contained_thumbnail_path,
    read_regular_file,
)
from backend.app.shared.state_dir import resolve_state_dir

CLIP_STORE_DIR_ENV = "CLIP_STORE_DIR"
API_LABEL_STORE_ENV = "API_LABEL_STORE"
DEFAULT_CLIP_STORE_DIR = "/var/lib/clip-store"

@dataclass(frozen=True, slots=True)
class LocatedClip:
    manifest: ClipManifest
    manifest_path: Path

    @property
    def recording_root(self) -> Path:
        return self.manifest_path.parent.parent.parent


@dataclass(frozen=True, slots=True)
class ScannedManifest:
    """One ``manifest.json`` found by a single walk of the store, unparsed.

    Only ``stat`` facts are carried so a listing can decide, against the
    catalogue, whether the file needs to be read at all.
    """

    clip_id: str
    manifest_path: Path
    size_bytes: int
    mtime_ns: int

    def located(self) -> LocatedClip | None:
        manifest = read_manifest_file(self.manifest_path)
        if manifest is None or not manifest.finalized or manifest.clip_id != self.clip_id:
            return None
        return LocatedClip(manifest, self.manifest_path)


@final
class DuplicateClipIdError(RuntimeError):
    def __init__(self, clip_id: str, manifest_paths: tuple[Path, ...]) -> None:
        self.clip_id = clip_id
        self.manifest_paths = manifest_paths
        super().__init__(f"duplicate clip_id across manifest layouts: {clip_id}")


class ClipStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def from_env(cls) -> ClipStore:
        return cls(os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))

    def list_manifests(self, *, camera_id: str | None = None) -> list[ClipManifest]:
        manifests: list[ClipManifest] = []
        for manifest_path in self._manifest_paths():
            manifest = self._read_manifest_file(manifest_path)
            if (
                manifest is None
                or not manifest.finalized
                or manifest.clip_id != manifest_path.parent.name
            ):
                continue
            if camera_id is not None and manifest.camera_id != camera_id:
                continue
            manifests.append(manifest)
        return sorted(manifests, key=lambda manifest: manifest.started_at, reverse=True)

    def _manifest_paths(self) -> list[Path]:
        """Manifests can live directly under the store root
        (``root/clips/*/manifest.json`` -- the layout before any storage
        location was ever selected) or nested one or two levels down under a
        chosen ``clip_store_subdir`` (``root/<sub>/clips/*/manifest.json``,
        ``root/<sub2>/<sub1>/clips/*/manifest.json`` -- ``store_subdir`` may
        itself be a multi-segment relative path, see
        ``ClipRecordingConfig.store_subdir``). Listing must keep finding
        clips recorded under any past selection, not just the current one, so
        all three layouts are always checked -- bounded to two subdir levels
        rather than an unbounded recursive walk, since a clip store can
        accumulate many unrelated directories over time.
        """
        return discover_manifest_paths(self.root)

    def scan_manifests(self) -> list[ScannedManifest]:
        """Walk every bounded ``clips`` root once and ``stat`` each manifest.

        This is the whole filesystem cost of a listing request: one directory
        listing per clips root plus one ``stat`` per clip, no manifest parsing.
        A clip id present under more than one layout is read to decide which
        copy is the finalized one; two finalized copies are the same
        ``DuplicateClipIdError`` that ``locate_manifest`` raises.
        """
        by_id: dict[str, list[ScannedManifest]] = {}
        for clips_root in bounded_clip_roots(self.root):
            try:
                entries = list(os.scandir(clips_root))
            except OSError:
                continue
            for entry in entries:
                if not is_valid_clip_id(entry.name):
                    continue
                manifest_path = clips_root / entry.name / "manifest.json"
                try:
                    manifest_stat = manifest_path.stat()
                except OSError:
                    continue
                if not stat.S_ISREG(manifest_stat.st_mode):
                    continue
                by_id.setdefault(entry.name, []).append(
                    ScannedManifest(
                        entry.name,
                        manifest_path,
                        manifest_stat.st_size,
                        manifest_stat.st_mtime_ns,
                    )
                )
        scanned: list[ScannedManifest] = []
        for clip_id, candidates in by_id.items():
            if len(candidates) == 1:
                scanned.append(candidates[0])
                continue
            finalized = [item for item in candidates if item.located() is not None]
            if len(finalized) > 1:
                raise DuplicateClipIdError(
                    clip_id, tuple(item.manifest_path for item in finalized)
                )
            scanned.extend(finalized)
        return scanned

    def get_manifest(self, clip_id: str) -> ClipManifest | None:
        located = self.locate_manifest(clip_id)
        return None if located is None else located.manifest

    def locate_manifest(self, clip_id: str) -> LocatedClip | None:
        if not is_valid_clip_id(clip_id):
            raise ValueError("invalid clip_id")
        located: list[LocatedClip] = []
        for clips_root in bounded_clip_roots(self.root):
            manifest_path = clips_root / clip_id / "manifest.json"
            manifest = read_manifest_file(manifest_path)
            if manifest is None or not manifest.finalized or manifest.clip_id != clip_id:
                continue
            located.append(LocatedClip(manifest, manifest_path))
        if len(located) > 1:
            raise DuplicateClipIdError(
                clip_id,
                tuple(item.manifest_path for item in located),
            )
        if not located:
            return None
        return located[0]

    def thumbnail_available(self, clip: str | LocatedClip) -> bool:
        if isinstance(clip, LocatedClip):
            return contained_thumbnail_path(self.root, clip.manifest_path) is not None
        clip_id = clip
        if not is_valid_clip_id(clip_id):
            raise ValueError("invalid clip_id")
        manifest_paths = tuple(
            clips_root / clip_id / "manifest.json"
            for clips_root in bounded_clip_roots(self.root)
            if (clips_root / clip_id / "manifest.json").is_file()
        )
        if len(manifest_paths) > 1:
            raise DuplicateClipIdError(clip_id, manifest_paths)
        return bool(manifest_paths) and contained_thumbnail_path(
            self.root, manifest_paths[0]
        ) is not None

    def read_thumbnail(self, located: LocatedClip) -> bytes:
        thumbnail_path = located.manifest_path.parent / "thumbnail.jpg"
        return read_regular_file(self.root, thumbnail_path)

    def scene_available(self, located: LocatedClip) -> bool:
        opened = resolve_scene_index(
            self.root, located.manifest_path, located.manifest.scene_index
        )
        if opened is None:
            return False
        opened.handle.close()
        return True

    def resolve_scene_index(self, located: LocatedClip) -> OpenedRegularFile | None:
        return resolve_scene_index(
            self.root, located.manifest_path, located.manifest.scene_index
        )

    def resolve_video_path(self, manifest: ClipManifest) -> Path:
        located = self.locate_manifest(manifest.clip_id)
        recording_root = self.root if located is None else located.recording_root
        return self._resolve_video_path(manifest, recording_root)

    def resolve_located_video_path(self, located: LocatedClip) -> Path:
        return self._resolve_video_path(located.manifest, located.recording_root)

    def open_located_video(self, located: LocatedClip) -> OpenedRegularFile:
        path = self.resolve_located_video_path(located)
        return open_contained_regular_file(self.root, path)

    def _resolve_video_path(self, manifest: ClipManifest, recording_root: Path) -> Path:
        if manifest.path is None:
            raise FileNotFoundError(str(self.root))
        raw_path = Path(manifest.path)
        if raw_path.is_absolute():
            candidate = raw_path
        else:
            recording_prefix = recording_root.relative_to(self.root).parts
            worker_relative = raw_path.parts[:1] == ("clips",)
            legacy_relative = bool(recording_prefix) and raw_path.parts[
                : len(recording_prefix)
            ] == recording_prefix
            anchor = self.root if legacy_relative and not worker_relative else recording_root
            candidate = anchor / raw_path
        resolved = candidate.resolve(strict=False)
        root = self.root.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise ValueError("manifest path escapes clip store")
        if resolved.is_dir():
            resolved = video_file_from_dir(resolved, manifest.clip_id)
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        return resolved

    def _read_manifest_file(self, path: Path) -> ClipManifest | None:
        return read_manifest_file(path)

def default_label_store_dir() -> Path:
    """Default root for clip labels + the audit log, absent ``API_LABEL_STORE``.

    Was a hardcoded ``/var/lib/ml-api-labels`` -- a container-root-only path
    that a native (non-container) dev process cannot ``mkdir`` into (issue
    #152: ``GET /clips`` 500s from the audit-log append's ``PermissionError``
    before it ever reaches the read). Following ``resolve_state_dir``'s single
    rule (``backend/app/shared/state_dir.py``) instead gives every runtime --
    container or native -- a location its own user can already write.
    """
    return resolve_state_dir("ml-api") / "labels"


__all__ = [
    "API_LABEL_STORE_ENV",
    "CLIP_STORE_DIR_ENV",
    "ClipManifest",
    "ClipStore",
    "DuplicateClipIdError",
    "LocatedClip",
    "ScannedManifest",
    "default_label_store_dir",
    "is_valid_clip_id",
]
