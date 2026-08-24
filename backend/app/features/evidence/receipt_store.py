"""Backend receipt persistence contract for locally produced clip artifacts.

The schema migrator owns the production table.  This module deliberately owns
no DDL: deployments inject the migrated persistence implementation through
``app.state.artifact_receipt_store``.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, runtime_checkable

from backend.app.features.clips.catalog import CatalogConflictError, CatalogStore

if TYPE_CHECKING:
    from fastapi import FastAPI

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactReceiptConflictError(RuntimeError):
    """A retry changed immutable identity, hash, or size fields."""


class ArtifactReceiptVerificationError(RuntimeError):
    """Declared artifact identity does not match the local regular file."""


class ArtifactReceiptPersistenceError(RuntimeError):
    """No durable backend receipt store is available."""


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    artifact_id: str
    sha256: str
    size_bytes: int
    accepted: bool = True

    def __post_init__(self) -> None:
        if not self.artifact_id or "\x00" in self.artifact_id:
            raise ValueError("invalid artifact identity")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("invalid artifact hash")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("invalid artifact size")


@runtime_checkable
class ArtifactReceiptStore(Protocol):
    """Durable compare-or-insert receipt port.

    ``commit`` must return only after its transaction is durable.  It inserts a
    first receipt, returns an identical existing receipt, and raises
    ``ArtifactReceiptConflictError`` when immutable fields differ.
    """

    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt: ...

    def get(self, artifact_id: str) -> ArtifactReceipt | None: ...


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    handle: BinaryIO
    sha256: str
    size_bytes: int
    device: int
    inode: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode


def verified_artifact(handle: BinaryIO) -> VerifiedArtifact:
    """Hash one open regular descriptor and preserve its identity and position."""
    try:
        descriptor_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise ArtifactReceiptVerificationError("artifact descriptor is not regular")
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
    except OSError as error:
        raise ArtifactReceiptVerificationError("artifact descriptor cannot be verified") from error
    return VerifiedArtifact(
        handle=handle,
        sha256=digest.hexdigest(),
        size_bytes=descriptor_stat.st_size,
        device=descriptor_stat.st_dev,
        inode=descriptor_stat.st_ino,
    )


class CatalogArtifactReceiptStore:
    """Receipt adapter over the existing API-owned ``clips`` catalog table."""

    def __init__(self, catalog: CatalogStore) -> None:
        self._catalog = catalog

    @classmethod
    def from_app(cls, app: FastAPI) -> CatalogArtifactReceiptStore:
        from backend.app.features.clips.catalog import get_catalog_store

        catalog = get_catalog_store(app)
        if catalog is None:
            raise ArtifactReceiptPersistenceError("clip catalog is unavailable")
        return cls(catalog)

    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt:
        try:
            sha256, size_bytes, accepted = self._catalog.commit_artifact_receipt(
                receipt.artifact_id, receipt.sha256, receipt.size_bytes
            )
        except CatalogConflictError as exc:
            raise ArtifactReceiptConflictError(str(exc)) from exc
        return ArtifactReceipt(receipt.artifact_id, sha256, size_bytes, accepted)

    def get(self, artifact_id: str) -> ArtifactReceipt | None:
        row = self._catalog.artifact_receipt(artifact_id)
        return None if row is None else ArtifactReceipt(artifact_id, *row)


def verify_artifact(path: Path, receipt: ArtifactReceipt) -> None:
    """Require a current regular-file size and SHA-256 match before use."""
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise ArtifactReceiptVerificationError("artifact is missing") from exc
    if not path.is_file() or stat_result.st_size != receipt.size_bytes:
        raise ArtifactReceiptVerificationError("artifact size does not match receipt")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactReceiptVerificationError("artifact cannot be verified") from exc
    if digest.hexdigest() != receipt.sha256:
        raise ArtifactReceiptVerificationError("artifact hash does not match receipt")


__all__ = [
    "ArtifactReceipt",
    "ArtifactReceiptConflictError",
    "CatalogArtifactReceiptStore",
    "VerifiedArtifact",
    "verified_artifact",
    "ArtifactReceiptPersistenceError",
    "ArtifactReceiptStore",
    "ArtifactReceiptVerificationError",
    "verify_artifact",
]
