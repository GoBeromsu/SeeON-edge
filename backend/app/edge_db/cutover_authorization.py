"""Capability for applying schema 18 only to a Task 6 candidate copy."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.app.edge_db.compatibility import EdgeDatabaseError


class HeldDeploymentLock(Protocol):
    """The live exclusive lock that may issue and redeem a cutover capability."""

    _active: bool
    _proof: object

    def require_for(self, database: Path) -> None:
        """Refuse when this lock does not authentically cover *database*."""


class CompactCutoverRequiredError(EdgeDatabaseError):
    """Ordinary migrate_database cannot apply schema 18 to a live v17 file."""

    def __init__(self, reason: str = "EDGE_DB_CUTOVER_UNAUTHORIZED") -> None:
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


class CompactCutoverSourceError(EdgeDatabaseError):
    """The candidate provenance tuple is not a usable cutover source."""

    def __str__(self) -> str:
        return "EDGE_DB_CUTOVER_SOURCE_INVALID"


@dataclass(frozen=True, slots=True)
class CompactCutoverSource:
    source_schema_version: int
    source_db_sha256: str
    reconciliation_sha256: str


class _CutoverTicket:
    """Single-use binding of a verified candidate to one authentic lock."""

    __slots__ = (
        "_candidate",
        "_consumed",
        "_dev",
        "_digest",
        "_ino",
        "_lock",
        "_recon_hash",
        "_source_hash",
    )

    def __init__(
        self,
        lock: HeldDeploymentLock,
        candidate: Path,
        digest: str,
        source_hash: str,
        recon_hash: str,
        dev: int,
        ino: int,
    ) -> None:
        self._lock = lock
        self._candidate = candidate
        self._dev = dev
        self._ino = ino
        self._digest = digest
        self._source_hash = source_hash
        self._recon_hash = recon_hash
        self._consumed = False


@dataclass(frozen=True, slots=True)
class CompactCutoverAuthorization:
    _ticket: object

    def redeem(self, lock: HeldDeploymentLock, candidate: Path) -> CompactCutoverSource:
        """Consume this capability for *candidate* under the issuing lock."""
        ticket = self._ticket
        if not isinstance(ticket, _CutoverTicket):
            raise CompactCutoverRequiredError("FORGED_LOCK")
        try:
            lock.require_for(candidate)
        except EdgeDatabaseError as error:
            reason = str(error)
            if reason == "EXPIRED_LOCK":
                raise CompactCutoverRequiredError("EXPIRED_LOCK") from error
            raise CompactCutoverRequiredError("FORGED_LOCK") from error
        if ticket._lock is not lock:
            raise CompactCutoverRequiredError("FORGED_LOCK")
        if ticket._consumed:
            raise CompactCutoverRequiredError("REUSED")
        resolved = candidate.resolve()
        if resolved != ticket._candidate:
            raise CompactCutoverRequiredError("WRONG_CANDIDATE")
        stat = resolved.stat()
        if (stat.st_dev, stat.st_ino) != (ticket._dev, ticket._ino):
            raise CompactCutoverRequiredError("WRONG_CANDIDATE")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != ticket._digest:
            raise CompactCutoverRequiredError("CANDIDATE_CHANGED")
        ticket._consumed = True
        return CompactCutoverSource(
            source_schema_version=17,
            source_db_sha256=ticket._source_hash,
            reconciliation_sha256=ticket._recon_hash,
        )


def _candidate_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reconciliation_digest(reconciliation: Path | bytes) -> str:
    payload = reconciliation.read_bytes() if isinstance(reconciliation, Path) else reconciliation
    return hashlib.sha256(payload).hexdigest()


def _require_schema17(path: Path) -> None:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = connection.execute("PRAGMA user_version").fetchone()
    finally:
        connection.close()
    if row is None or int(row[0]) != 17:
        raise CompactCutoverSourceError


def issue_compact_cutover_authorization(
    lock: HeldDeploymentLock,
    *,
    source: Path,
    candidate: Path,
    reconciliation: Path | bytes,
) -> CompactCutoverAuthorization:
    """Issue candidate-only authorization from verified source/candidate bytes."""
    try:
        lock.require_for(candidate)
    except EdgeDatabaseError as error:
        reason = str(error)
        if reason == "EXPIRED_LOCK":
            raise CompactCutoverRequiredError("EXPIRED_LOCK") from error
        raise CompactCutoverRequiredError("FORGED_LOCK") from error
    source_bytes = source.read_bytes()
    candidate_bytes = candidate.read_bytes()
    if source_bytes != candidate_bytes:
        raise CompactCutoverSourceError
    _require_schema17(candidate)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    recon_hash = _reconciliation_digest(reconciliation)
    if source_hash == recon_hash:
        raise CompactCutoverSourceError
    resolved = candidate.resolve()
    stat = resolved.stat()
    ticket = _CutoverTicket(
        lock=lock,
        candidate=resolved,
        digest=_candidate_digest(resolved),
        source_hash=source_hash,
        recon_hash=recon_hash,
        dev=stat.st_dev,
        ino=stat.st_ino,
    )
    return CompactCutoverAuthorization(_ticket=ticket)


__all__ = [
    "CompactCutoverAuthorization",
    "CompactCutoverRequiredError",
    "CompactCutoverSource",
    "CompactCutoverSourceError",
    "issue_compact_cutover_authorization",
]
