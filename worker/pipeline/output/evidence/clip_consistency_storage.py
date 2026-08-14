"""SQLite mutation and staging quarantine operations for clip consistency repair."""

from __future__ import annotations

import os
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_database import (
    RelationPlan,
    relation_state_sha256,
)
from worker.pipeline.output.evidence.clip_consistency_io import FaultHook, checkpoint
from worker.pipeline.output.evidence.clip_consistency_journal import (
    ApplyJournal,
    mark_journal,
)
from worker.pipeline.output.evidence.clip_consistency_quarantine import (
    absolute_quarantine_rows,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.durability import fsync_directory


def execute_plan(connection: sqlite3.Connection, plan: RelationPlan) -> None:
    if relation_state_sha256(connection) != plan.before_sha256:
        raise ClipConsistencyError("source_changed", "relation state changed before mutation")
    deleted = sum(
        connection.execute(
            "DELETE FROM clip_events WHERE edge_event_id = ?", (event_id,)
        ).rowcount
        for event_id in plan.delete_event_ids
    )
    connection.executemany(
        "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, ?)",
        plan.insert_rows,
    )
    if deleted != len(plan.delete_event_ids):
        raise ClipConsistencyError("counter_mismatch", "SQL delete count differs from plan")
    if relation_state_sha256(connection) != plan.after_sha256:
        raise ClipConsistencyError("counter_mismatch", "SQL result differs from plan")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ClipConsistencyError("foreign_key_drift", "repair introduced FK drift")


def plan_staging_quarantine(
    clip_store: Path,
    clip_ids: tuple[str, ...],
    plan_sha256: str,
) -> list[tuple[Path, Path]]:
    rows = absolute_quarantine_rows(clip_store, clip_ids, plan_sha256)
    for _, held in rows:
        _require_available_quarantine(held)
    return rows


def quarantine_staging(
    rows: list[tuple[Path, Path]],
    hook: FaultHook | None,
) -> None:
    moved: list[tuple[Path, Path]] = []
    try:
        for original, held in rows:
            os.replace(original, held)
            moved.append((original, held))
            checkpoint(hook, "quarantine:rename")
        if moved:
            fsync_directory(moved[0][0].parent)
            checkpoint(hook, "quarantine:rename_fsync")
    except BaseException:
        restore_quarantine(moved)
        raise


def restore_quarantine(moved: list[tuple[Path, Path]]) -> None:
    for original, held in reversed(moved):
        if held.exists() and not original.exists():
            os.replace(held, original)
    if moved:
        fsync_directory(moved[0][0].parent)


def ensure_quarantine(clip_store: Path, rows: tuple[tuple[str, str], ...]) -> None:
    moved: list[tuple[Path, Path]] = []
    try:
        for original_relative, held_relative in rows:
            original = clip_store / original_relative
            held = clip_store / held_relative
            if held.exists() and not original.exists():
                continue
            if original.exists() and not held.exists():
                os.replace(original, held)
                moved.append((original, held))
                continue
            _raise_quarantine_state(original, held)
        if rows:
            fsync_directory(clip_store / "clips" / ".staging")
    except BaseException:
        restore_quarantine(moved)
        raise


def open_write_exclusion(state_db: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{state_db}?mode=rw",
        uri=True,
        isolation_level=None,
        timeout=0,
    )
    try:
        connection.execute("PRAGMA busy_timeout = 0")
        connection.execute("PRAGMA foreign_keys = ON")
        # Normalize quiescent WAL bytes before taking the raw source identity.
        # A pinned reader may leave frames in WAL; BEGIN IMMEDIATE still excludes
        # every writer and the receipt then binds that exact WAL-aware state.
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        connection.execute("BEGIN IMMEDIATE")
    except BaseException:
        connection.close()
        raise
    return connection


def finish_cleanup(request: RepairRequest, journal: ApplyJournal) -> ApplyJournal:
    root, path = required_mutation_paths(request)
    for _, held_relative in journal.quarantine:
        held = request.clip_store / held_relative
        if held.exists():
            checkpoint(request.fault_hook, "quarantine:before_remove")
            shutil.rmtree(held)
            checkpoint(request.fault_hook, "quarantine:after_remove")
    if journal.quarantine:
        fsync_directory(request.clip_store / "clips" / ".staging")
        checkpoint(request.fault_hook, "quarantine:fsync_directory")
    done = replace(
        journal,
        counters=replace(
            journal.counters,
            staging_after=0,
            staging_to_delete=0,
            staging_deleted=journal.counters.staging_before,
        ),
    )
    return mark_journal(
        done,
        "DONE",
        path=path,
        maintenance_root=root,
        expected_uid=request.expected_owner_uid,
        hook=request.fault_hook,
    )


def required_mutation_paths(request: RepairRequest) -> tuple[Path, Path]:
    assert request.maintenance_root is not None
    assert request.journal_path is not None
    return request.maintenance_root, request.journal_path


def _require_available_quarantine(held: Path) -> None:
    if held.exists():
        raise ClipConsistencyError("quarantine_conflict", "quarantine already exists")


def _raise_quarantine_state(original: Path, held: Path) -> None:
    if not original.exists() and not held.exists():
        raise ClipConsistencyError("quarantine_missing", "prepared staging is missing")
    raise ClipConsistencyError("quarantine_conflict", "both staging paths exist")


__all__ = [
    "ensure_quarantine",
    "execute_plan",
    "finish_cleanup",
    "open_write_exclusion",
    "plan_staging_quarantine",
    "quarantine_staging",
    "required_mutation_paths",
    "restore_quarantine",
]
