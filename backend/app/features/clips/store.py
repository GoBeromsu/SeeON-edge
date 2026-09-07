"""Read-only manifest access for clip playback."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import final

from backend.app.features.clips.descriptor_files import (
    OpenedRegularFile,
    open_contained_regular_file,
    read_bounded_regular_file,
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

CLIP_STORE_DIR_ENV = "CLIP_STORE_DIR"
API_LABEL_STORE_ENV = "API_LABEL_STORE"
DEFAULT_CLIP_STORE_DIR = "/var/lib/clip-store"
PLAYBACK_H264_MANIFEST_FILENAME = "clip.playback-h264.json"
_PLAYBACK_RENDITION_NAME_RE = re.compile(r"clip\.playback-h264\.[0-9a-f]{16}\.mp4\Z")
# Immutable, identity-named artifacts published by the worker
# (worker/pipeline/output/evidence/clip_analysis_artifact.py); the newest
# digest-verified one for the clip is served.
CLIP_ANALYSIS_GLOB = "clip.analysis.*.json"
_CLIP_ANALYSIS_NAME_RE = re.compile(r"^clip\.analysis\.[0-9a-f]{16}\.json$")
_SHA256_SIDECAR_BYTES = 65
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PLAYBACK_MANIFEST_BYTES = 1024


@dataclass(frozen=True, slots=True)
class LocatedClip:
    manifest: ClipManifest
    manifest_path: Path

    @property
    def recording_root(self) -> Path:
        return self.manifest_path.parent.parent.parent


@dataclass(frozen=True, slots=True)
class OpenedPlaybackIdentity:
    """Opened served media with immutable-source and rendition timing identity."""

    opened: OpenedRegularFile
    original_sha256: str | None
    served_media_sha256: str | None
    served_pts_identical: bool | None


@dataclass(frozen=True, slots=True)
class PlaybackAttestation:
    """A validated immutable rendition pointer."""

    rendition: str
    rendition_sha256: str
    pts_identical: bool


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
        self._playback_digest_cache: dict[tuple[Path, int, int], str] = {}
        self._playback_digest_cache_lock = Lock()

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
                raise DuplicateClipIdError(clip_id, tuple(item.manifest_path for item in finalized))
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
        return (
            bool(manifest_paths)
            and contained_thumbnail_path(self.root, manifest_paths[0]) is not None
        )

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

    def open_located_playback(self, located: LocatedClip) -> OpenedRegularFile:
        """Open a verified browser-safe rendition, or the immutable original."""
        return self.open_located_playback_identity(located).opened

    def open_located_playback_identity(self, located: LocatedClip) -> OpenedPlaybackIdentity:
        """Open served media and expose its immutable-source and timing identity."""
        original = self.open_located_video(located)
        original_sha256 = self.manifest_video_sha256(located)
        playback = self._open_verified_playback(original.path, original_sha256)
        if playback is None:
            return OpenedPlaybackIdentity(original, original_sha256, original_sha256, None)
        opened, attestation = playback
        original.handle.close()
        return OpenedPlaybackIdentity(
            opened,
            original_sha256,
            attestation.rendition_sha256,
            attestation.pts_identical,
        )

    def served_media_sha256(self, located: LocatedClip) -> str | None:
        """Return the digest of the bytes the video route would serve now."""
        identity = self.open_located_playback_identity(located)
        try:
            return identity.served_media_sha256
        finally:
            identity.opened.handle.close()

    def playback_codec(self, located: LocatedClip) -> str:
        """Return the codec an operator will receive from the video endpoint."""
        try:
            original_path = self.resolve_located_video_path(located)
        except (ValueError, FileNotFoundError):
            return located.manifest.codec
        playback = self._open_verified_playback(
            original_path,
            self.manifest_video_sha256(located),
        )
        if playback is None:
            return located.manifest.codec
        playback[0].handle.close()
        return "h264"

    def manifest_video_sha256(self, located: LocatedClip) -> str | None:
        """Return the immutable original-media identity declared by the manifest."""
        try:
            payload = json.loads(
                read_bounded_regular_file(self.root, located.manifest_path, 1024 * 1024)
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        digest = payload.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            return None
        return digest

    def read_clip_analysis(self, located: LocatedClip) -> bytes | None:
        """Read the newest digest-verified analysis artifact beside the clip."""
        clip_dir = located.manifest_path.parent
        candidates = [
            path
            for path in clip_dir.glob(CLIP_ANALYSIS_GLOB)
            if _CLIP_ANALYSIS_NAME_RE.fullmatch(path.name) is not None
        ]
        if not candidates:
            return None
        artifact = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
        sidecar = artifact.with_name(f"{artifact.name}.sha256")
        try:
            expected = read_bounded_regular_file(self.root, sidecar, _SHA256_SIDECAR_BYTES)
            payload = read_bounded_regular_file(
                self.root,
                artifact,
                32 * 1024 * 1024,
            )
        except FileNotFoundError:
            return None
        try:
            expected_digest = expected[:-1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("clip analysis digest is not ASCII") from exc
        if len(expected) != _SHA256_SIDECAR_BYTES or expected[-1:] != b"\n":
            raise ValueError("clip analysis digest is invalid")
        if _SHA256_RE.fullmatch(expected_digest) is None:
            raise ValueError("clip analysis digest is invalid")
        if not hmac.compare_digest(
            hashlib.sha256(payload).hexdigest(),
            expected_digest,
        ):
            raise ValueError("clip analysis digest does not match")
        return payload

    def _open_verified_playback(
        self,
        original_path: Path,
        source_sha256: str | None,
    ) -> tuple[OpenedRegularFile, PlaybackAttestation] | None:
        attestation = self._read_playback_attestation(original_path, source_sha256)
        if attestation is None:
            return None
        try:
            playback = open_contained_regular_file(
                self.root,
                original_path.with_name(attestation.rendition),
            )
        except FileNotFoundError:
            return None
        try:
            digest = self._playback_digest(playback)
            if hmac.compare_digest(digest, attestation.rendition_sha256):
                return playback, attestation
        except OSError:
            pass
        playback.handle.close()
        return None

    def _read_playback_attestation(
        self,
        original_path: Path,
        source_sha256: str | None,
    ) -> PlaybackAttestation | None:
        if source_sha256 is None:
            return None
        try:
            raw = read_bounded_regular_file(
                self.root,
                original_path.with_name(PLAYBACK_H264_MANIFEST_FILENAME),
                _PLAYBACK_MANIFEST_BYTES,
            )
            payload = json.loads(raw.decode("ascii"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        rendition = payload.get("rendition")
        rendition_sha256 = payload.get("rendition_sha256")
        pts_identical = payload.get("pts_identical")
        time_base = payload.get("time_base")
        frames = payload.get("frames")
        source_frames = payload.get("source_frames")
        if (
            not isinstance(rendition, str)
            or _PLAYBACK_RENDITION_NAME_RE.fullmatch(rendition) is None
            or not isinstance(rendition_sha256, str)
            or _SHA256_RE.fullmatch(rendition_sha256) is None
            or rendition != f"clip.playback-h264.{rendition_sha256[:16]}.mp4"
            or payload.get("source_sha256") != source_sha256
            or not isinstance(pts_identical, bool)
            or not isinstance(time_base, str)
            or re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", time_base) is None
            or isinstance(frames, bool)
            or not isinstance(frames, int)
            or frames < 0
            or isinstance(source_frames, bool)
            or not isinstance(source_frames, int)
            or source_frames < 0
        ):
            return None
        return PlaybackAttestation(rendition, rendition_sha256, pts_identical)

    def _playback_digest(self, opened: OpenedRegularFile) -> str:
        file_stat = os.fstat(opened.handle.fileno())
        key = (opened.path, file_stat.st_size, file_stat.st_mtime_ns)
        with self._playback_digest_cache_lock:
            cached = self._playback_digest_cache.get(key)
        if cached is not None:
            return cached
        digest = hashlib.file_digest(opened.handle, "sha256").hexdigest()
        opened.handle.seek(0)
        with self._playback_digest_cache_lock:
            self._playback_digest_cache[key] = digest
        return digest

    def _resolve_video_path(self, manifest: ClipManifest, recording_root: Path) -> Path:
        if manifest.path is None:
            raise FileNotFoundError(str(self.root))
        raw_path = Path(manifest.path)
        if raw_path.is_absolute():
            candidate = raw_path
        else:
            recording_prefix = recording_root.relative_to(self.root).parts
            worker_relative = raw_path.parts[:1] == ("clips",)
            legacy_relative = (
                bool(recording_prefix)
                and raw_path.parts[: len(recording_prefix)] == recording_prefix
            )
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
    "CLIP_ANALYSIS_GLOB",
    "CLIP_STORE_DIR_ENV",
    "PLAYBACK_H264_MANIFEST_FILENAME",
    "ClipManifest",
    "ClipStore",
    "DuplicateClipIdError",
    "LocatedClip",
    "ScannedManifest",
    "default_label_store_dir",
    "is_valid_clip_id",
]
