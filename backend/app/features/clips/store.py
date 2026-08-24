"""Read-only manifest access and API-owned clip labels."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

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
from backend.app.features.clips.thumbnail_files import (
    bounded_clip_roots,
    contained_thumbnail_path,
    read_regular_file,
)
from backend.app.shared.state_dir import resolve_state_dir

logger = logging.getLogger(__name__)

CLIP_STORE_DIR_ENV = "CLIP_STORE_DIR"
API_LABEL_STORE_ENV = "API_LABEL_STORE"
DEFAULT_CLIP_STORE_DIR = "/var/lib/clip-store"

LabelValue = Literal["TRUE_POSITIVE", "FALSE_POSITIVE"] | None

@dataclass(frozen=True)
class LabelRecord:
    clip_id: str
    label: LabelValue
    reviewer: str
    reviewed_at: str

    def as_response(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "label": self.label,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
        }


@dataclass(frozen=True, slots=True)
class LocatedClip:
    manifest: ClipManifest
    manifest_path: Path

    @property
    def recording_root(self) -> Path:
        return self.manifest_path.parent.parent.parent


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


class LabelStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def from_env(cls) -> LabelStore:
        configured = os.environ.get(API_LABEL_STORE_ENV)
        return cls(configured) if configured else cls(default_label_store_dir())

    def save(self, record: LabelRecord) -> bool:
        """Best-effort persist; an unwritable state dir must not crash the caller.

        Mirrors catalog-store graceful-degradation
        pattern (``backend/app/features/status/runtime_status_store.py``): a
        label write that cannot land durably is dropped (with a warning)
        rather than crashing the labeling request. Returns ``True`` only when
        the record actually landed durably, so the caller (``label_clip`` in
        ``router.py``) can surface a degraded save to the client instead of
        silently reporting success.
        """
        if not is_valid_clip_id(record.clip_id):
            raise ValueError("invalid clip_id")
        labels_dir = self.root / "labels"
        try:
            labels_dir.mkdir(parents=True, exist_ok=True)
            target = labels_dir / f"{record.clip_id}.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(record.as_response(), ensure_ascii=False, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, target)
        except OSError as exc:
            logger.warning("label store unavailable at %s: %s", labels_dir, exc)
            return False
        return True

    def get(self, clip_id: str) -> LabelRecord | None:
        if not is_valid_clip_id(clip_id):
            raise ValueError("invalid clip_id")
        target = self.root / "labels" / f"{clip_id}.json"
        try:
            parsed = json.loads(target.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(parsed, dict):
            return None
        return LabelRecord(
            clip_id=str(parsed.get("clip_id", "")),
            label=_label_value(parsed.get("label")),
            reviewer=str(parsed.get("reviewer", "")),
            reviewed_at=str(parsed.get("reviewed_at", "")),
        )


def _label_value(value: object) -> LabelValue:
    if value is None or value in ("TRUE_POSITIVE", "FALSE_POSITIVE"):
        return value
    raise ValueError("invalid label")


__all__ = [
    "API_LABEL_STORE_ENV",
    "CLIP_STORE_DIR_ENV",
    "ClipManifest",
    "ClipStore",
    "DuplicateClipIdError",
    "LabelRecord",
    "LabelStore",
    "LabelValue",
    "LocatedClip",
    "default_label_store_dir",
    "is_valid_clip_id",
]
