"""Transactional read boundary for evidence runtime-manifest references."""

from __future__ import annotations

import sqlite3
from enum import StrEnum


class RuntimeManifestReferenceFailure(StrEnum):
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


class RuntimeManifestReferenceError(RuntimeError):
    """A staged event cannot prove its referenced runtime manifest exists."""

    def __init__(
        self,
        manifest_sha256: str,
        failure: RuntimeManifestReferenceFailure,
    ) -> None:
        self.manifest_sha256 = manifest_sha256
        self.failure = failure
        super().__init__(f"runtime manifest reference {manifest_sha256} is {failure.value}")


def require_runtime_manifest_contents(
    connection: sqlite3.Connection,
    manifest_sha256: str,
) -> None:
    """Require an immutable contents row on the caller's active transaction."""
    try:
        row = connection.execute(
            "SELECT 1 FROM runtime_manifest_contents WHERE manifest_sha256 = ?",
            (manifest_sha256,),
        ).fetchone()
    except sqlite3.Error as error:
        raise RuntimeManifestReferenceError(
            manifest_sha256,
            RuntimeManifestReferenceFailure.UNAVAILABLE,
        ) from error
    if row is None:
        raise RuntimeManifestReferenceError(
            manifest_sha256,
            RuntimeManifestReferenceFailure.MISSING,
        )


__all__ = [
    "RuntimeManifestReferenceError",
    "RuntimeManifestReferenceFailure",
    "require_runtime_manifest_contents",
]
