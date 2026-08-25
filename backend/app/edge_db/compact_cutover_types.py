"""Typed state and phase events for compact cutover."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from backend.app.edge_db.compatibility import EdgeDatabaseError


class CutoverPhase(StrEnum):
    BEFORE_CHECKPOINT = "before_checkpoint"
    AFTER_CHECKPOINT = "after_checkpoint"
    ARCHIVE_WRITTEN = "archive_written"
    ARCHIVE_SYNCED = "archive_synced"
    RECEIPT_WRITTEN = "receipt_written"
    RECEIPT_SYNCED = "receipt_synced"
    CANDIDATE_WRITTEN = "candidate_written"
    CANDIDATE_SYNCED = "candidate_synced"
    BEFORE_V18_TRANSACTION = "before_v18_transaction"
    V18_COMMITTED = "v18_committed"
    BEFORE_RECONCILIATION = "before_reconciliation"
    RECONCILED = "reconciled"
    CANDIDATE_FILE_SYNCED = "candidate_file_synced"
    BEFORE_PRE_RENAME_DIRECTORY_SYNC = "before_pre_rename_directory_sync"
    PRE_RENAME_DIRECTORY_SYNCED = "pre_rename_directory_synced"
    RENAMED = "renamed"
    FINAL_DIRECTORY_SYNCED = "final_directory_synced"
    BEFORE_MANIFEST_VERIFY = "before_manifest_verify"
    MANIFEST_VERIFIED = "manifest_verified"


CutoverProgress = Callable[[CutoverPhase], None]


@dataclass(frozen=True, slots=True)
class CompactCutoverRequest:
    source: Path
    live: Path
    archive: Path
    candidate: Path
    receipt: Path
    clip_store: Path
    worker_state: Path
    expected_source_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CompactCutoverResult:
    live: Path
    archive: Path
    receipt: Path
    source_sha256: str
    receipt_sha256: str
    source_rows: int


class CompactCutoverError(EdgeDatabaseError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)

    def __str__(self) -> str:
        return self.reason


__all__ = [
    "CompactCutoverError",
    "CompactCutoverRequest",
    "CompactCutoverResult",
    "CutoverPhase",
    "CutoverProgress",
]
