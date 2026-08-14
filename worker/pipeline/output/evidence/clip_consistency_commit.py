"""Classification of SQLite commit calls that raise with an ambiguous outcome."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_database import (
    relation_state_sha256,
    validate_database,
)
from worker.pipeline.output.evidence.clip_consistency_journal import (
    ApplyJournal,
    mark_journal,
)
from worker.pipeline.output.evidence.clip_consistency_preimage import (
    non_relation_preimage_sha256,
)
from worker.pipeline.output.evidence.clip_consistency_storage import (
    open_write_exclusion,
    required_mutation_paths,
    restore_quarantine,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    JournalState,
    RepairRequest,
)


def commit_connection(connection: sqlite3.Connection) -> None:
    connection.commit()


def abort_precommit(
    request: RepairRequest,
    connection: sqlite3.Connection,
    moved: list[tuple[Path, Path]],
    journal: ApplyJournal | None,
    error: BaseException,
) -> None:
    if connection.in_transaction:
        connection.rollback()
    restore_quarantine(moved)
    root, path = required_mutation_paths(request)
    if journal is not None and path.exists():
        mark_journal(
            journal,
            "ABORTED",
            path=path,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=None,
            error=type(error).__name__,
        )


def classify_ambiguous_commit(
    request: RepairRequest,
    journal: ApplyJournal,
    moved: list[tuple[Path, Path]],
    error: BaseException,
) -> None:
    root, path = required_mutation_paths(request)
    inspection = open_write_exclusion(request.state_db)
    try:
        relation_state, non_relation_state = _durable_state(inspection)
        outcome: JournalState
        if (
            relation_state == journal.relations_before_sha256
            and non_relation_state == journal.non_relation_state_sha256
        ):
            restore_quarantine(moved)
            outcome = "ABORTED"
        elif (
            relation_state == journal.relations_after_sha256
            and non_relation_state == journal.non_relation_state_sha256
        ):
            outcome = "DB_COMMITTED"
        else:
            outcome = "UNKNOWN"
        mark_journal(
            journal,
            outcome,
            path=path,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=None,
            error=f"ambiguous_commit:{type(error).__name__}",
        )
    finally:
        if inspection.in_transaction:
            inspection.rollback()
        inspection.close()


def _durable_state(connection: sqlite3.Connection) -> tuple[str, str]:
    try:
        validate_database(connection, now=time.time())
        return (
            relation_state_sha256(connection),
            non_relation_preimage_sha256(connection),
        )
    except (ClipConsistencyError, sqlite3.Error):
        return "", ""


__all__ = ["abort_precommit", "classify_ambiguous_commit", "commit_connection"]
