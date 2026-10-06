"""Receipt file proof shared by persistence owners; no database or acknowledgement.

Manifest facts and their digest come from the same descriptor read. Rechecks
compare that captured content and pathname identity as well as the caller's
live media descriptor. Temporary descriptors close within each check; ownership
of the supplied media descriptor never transfers here.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.app.features.clips.descriptor_files import (
    OpenedRegularFile,
    open_contained_regular_file,
)
from backend.app.features.clips.manifest import ClipManifest, parse_manifest_bytes
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptVerificationError,
    ClipProjection,
    VerifiedArtifact,
    verified_artifact,
)


@dataclass(frozen=True, slots=True)
class ReceiptHooks:
    """Deterministic race hooks; production leaves both callbacks absent."""

    after_preflight: Callable[[], None] | None = None
    before_final_check: Callable[[], None] | None = None


def open_receipt_media(root: Path, path: Path) -> OpenedRegularFile:
    """Return an owned regular descriptor; the caller must close it."""
    return _open_regular(root, path)


def _open_regular(root: Path, path: Path) -> OpenedRegularFile:
    try:
        path_stat = os.lstat(path)
        if not stat.S_ISREG(path_stat.st_mode):
            raise ArtifactReceiptVerificationError("receipt pathname is not regular")
        opened = open_contained_regular_file(root, path)
    except (OSError, ValueError) as error:
        raise ArtifactReceiptVerificationError("receipt file is unavailable") from error
    try:
        opened_stat = os.fstat(opened.handle.fileno())
    except BaseException:
        opened.handle.close()
        raise
    if (path_stat.st_dev, path_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
        opened.handle.close()
        raise ArtifactReceiptVerificationError("receipt pathname changed")
    return opened


def _verify_current_path(root: Path, path: Path, identity: tuple[int, int]) -> None:
    opened = _open_regular(root, path)
    with opened.handle:
        current = os.fstat(opened.handle.fileno())
        if (current.st_dev, current.st_ino) != identity:
            raise ArtifactReceiptVerificationError("receipt pathname identity changed")


def _manifest_facts(root: Path, path: Path) -> tuple[ClipManifest, str, int, tuple[int, int]]:
    opened = _open_regular(root, path)
    with opened.handle:
        try:
            content = opened.handle.read()
            current = os.fstat(opened.handle.fileno())
        except OSError as error:
            raise ArtifactReceiptVerificationError("clip manifest cannot be verified") from error
        if len(content) != opened.size_bytes or len(content) != current.st_size:
            raise ArtifactReceiptVerificationError("clip manifest changed during verification")
        identity = (current.st_dev, current.st_ino)
        _verify_current_path(root, path, identity)
        manifest = parse_manifest_bytes(content)
        if manifest is None or not manifest.finalized:
            raise ArtifactReceiptVerificationError("clip manifest is invalid")
        return manifest, hashlib.sha256(content).hexdigest(), len(content), identity


@dataclass(frozen=True, slots=True)
class ReceiptManifest:
    root: Path
    path: Path
    manifest: ClipManifest
    sha256: str
    size_bytes: int
    identity: tuple[int, int]

    @classmethod
    def capture(cls, store: ClipStore, clip_id: str) -> ReceiptManifest:
        located = store.locate_manifest(clip_id)
        if located is None:
            raise ArtifactReceiptVerificationError("clip manifest is missing")
        manifest, digest, size, identity = _manifest_facts(store.root, located.manifest_path)
        if manifest.clip_id != clip_id or manifest != located.manifest:
            raise ArtifactReceiptVerificationError("clip manifest changed after location")
        return cls(store.root, located.manifest_path, manifest, digest, size, identity)

    def verify(self) -> None:
        if _manifest_facts(self.root, self.path) != (
            self.manifest,
            self.sha256,
            self.size_bytes,
            self.identity,
        ):
            raise ArtifactReceiptVerificationError("clip manifest changed during receipt")

    @property
    def media_path(self) -> Path:
        return self.path.parent / "clip.mp4"


@dataclass(frozen=True, slots=True)
class ReceiptFiles:
    receipt: ArtifactReceipt
    route_verified: VerifiedArtifact
    manifest: ReceiptManifest

    @classmethod
    def capture(
        cls, store: ClipStore, receipt: ArtifactReceipt, route_verified: VerifiedArtifact
    ) -> ReceiptFiles:
        result = cls(receipt, route_verified, ReceiptManifest.capture(store, receipt.artifact_id))
        result._verify_media()
        return result

    def _verify_media(self) -> VerifiedArtifact:
        current = verified_artifact(self.route_verified.handle)
        if current.identity != self.route_verified.identity:
            raise ArtifactReceiptVerificationError("verified descriptor identity changed")
        if (current.sha256, current.size_bytes) != (self.receipt.sha256, self.receipt.size_bytes):
            raise ArtifactReceiptVerificationError("declared receipt differs from media bytes")
        _verify_current_path(self.manifest.root, self.manifest.media_path, current.identity)
        return current

    def verify(self) -> ClipProjection:
        current = self._verify_media()
        self.manifest.verify()
        return ClipProjection(
            self.receipt,
            current,
            self.manifest.manifest,
            self.manifest.path.relative_to(self.manifest.root).as_posix(),
            self.manifest.media_path.relative_to(self.manifest.root).as_posix(),
            self.manifest.sha256,
            self.manifest.size_bytes,
        )


__all__ = ["ReceiptFiles", "ReceiptHooks", "ReceiptManifest", "open_receipt_media"]
