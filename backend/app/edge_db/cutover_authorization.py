"""Capability for applying schema 18 only to a Task 6 candidate copy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.app.edge_db.compatibility import EdgeDatabaseError


class HeldDeploymentLock(Protocol):
    """The live exclusive lock that may issue and redeem a cutover capability."""

    _active: bool


class CompactCutoverRequiredError(EdgeDatabaseError):
    """Ordinary migrate_database cannot apply schema 18 to a live v17 file."""

    def __str__(self) -> str:
        return "EDGE_DB_CUTOVER_UNAUTHORIZED"


class CompactCutoverSourceError(EdgeDatabaseError):
    """The candidate provenance tuple is not a usable cutover source."""

    def __str__(self) -> str:
        return "EDGE_DB_CUTOVER_SOURCE_INVALID"


class _CutoverCapability:
    """Unforgeable proof issued only while a deployment lock is held."""

    __slots__ = ("_lock",)

    def __init__(self, lock: HeldDeploymentLock) -> None:
        if not lock._active:
            raise CompactCutoverRequiredError
        self._lock = lock


def _require_hex64(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CompactCutoverSourceError


@dataclass(frozen=True, slots=True)
class CompactCutoverSource:
    source_schema_version: int
    source_db_sha256: str
    reconciliation_sha256: str

    def __post_init__(self) -> None:
        if self.source_schema_version != 17:
            raise CompactCutoverSourceError
        _require_hex64(self.source_db_sha256)
        _require_hex64(self.reconciliation_sha256)
        if self.source_db_sha256 == self.reconciliation_sha256:
            raise CompactCutoverSourceError


@dataclass(frozen=True, slots=True)
class CompactCutoverAuthorization:
    source: CompactCutoverSource
    _capability: object

    def bind(self, lock: HeldDeploymentLock) -> CompactCutoverSource:
        """Return provenance only when this capability was issued for *lock*."""
        capability = self._capability
        if not isinstance(capability, _CutoverCapability):
            raise CompactCutoverRequiredError
        if capability._lock is not lock or not lock._active:
            raise CompactCutoverRequiredError
        return self.source


def issue_compact_cutover_authorization(
    lock: HeldDeploymentLock,
    source: CompactCutoverSource,
) -> CompactCutoverAuthorization:
    """Issue candidate-only authorization while *lock* is held."""
    return CompactCutoverAuthorization(source=source, _capability=_CutoverCapability(lock))


__all__ = [
    "CompactCutoverAuthorization",
    "CompactCutoverRequiredError",
    "CompactCutoverSource",
    "CompactCutoverSourceError",
    "issue_compact_cutover_authorization",
]
