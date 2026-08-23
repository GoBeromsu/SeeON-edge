"""Transactional read boundary for evidence runtime-manifest references."""

from __future__ import annotations

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


def require_runtime_manifest_contents(_connection: object, manifest_sha256: str) -> None:
    """Local runtime-manifest catalogs no longer exist in this slot."""
    raise RuntimeManifestReferenceError(
        manifest_sha256, RuntimeManifestReferenceFailure.UNAVAILABLE
    )


__all__ = [
    "RuntimeManifestReferenceError",
    "RuntimeManifestReferenceFailure",
    "require_runtime_manifest_contents",
]
