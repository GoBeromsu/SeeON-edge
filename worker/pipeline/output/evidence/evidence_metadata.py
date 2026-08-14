from __future__ import annotations

from collections.abc import Mapping
from typing import Final

RUNTIME_MANIFEST_SHA256_KEY: Final = "runtime_manifest_sha256"


def validate_runtime_manifest_sha256(value: object | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("runtime_manifest_sha256 must be a lowercase SHA-256")
    return value


def runtime_manifest_sha256_from_audit(
    audit: Mapping[str, object] | None,
) -> str | None:
    if audit is None:
        return None
    return validate_runtime_manifest_sha256(audit.get(RUNTIME_MANIFEST_SHA256_KEY))


__all__ = [
    "RUNTIME_MANIFEST_SHA256_KEY",
    "runtime_manifest_sha256_from_audit",
    "validate_runtime_manifest_sha256",
]
