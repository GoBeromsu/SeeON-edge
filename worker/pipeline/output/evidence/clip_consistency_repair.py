"""Recoverable authoritative-manifest repair for clip relations."""

from __future__ import annotations

import sqlite3
import time

from worker.pipeline.output.evidence.clip_consistency_apply import (
    apply_repair,
    resume_repair,
)
from worker.pipeline.output.evidence.clip_consistency_database import (
    plan_relations,
    validate_database,
)
from worker.pipeline.output.evidence.clip_consistency_io import (
    validate_directory,
    validate_regular,
    validate_under_root,
)
from worker.pipeline.output.evidence.clip_consistency_manifest import (
    scan_manifest_authority,
)
from worker.pipeline.output.evidence.clip_consistency_quiescence import (
    validate_quiescence_receipt,
)
from worker.pipeline.output.evidence.clip_consistency_receipt import (
    RECEIPT_VERSION,
    build_counters,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairReceipt,
    RepairRequest,
)
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLock


def repair_clip_consistency(request: RepairRequest) -> RepairReceipt:
    _validate_request_paths(request)
    if request.apply and request.resume:
        raise ClipConsistencyError("invalid_mode", "apply and resume are exclusive")
    if not request.apply and not request.resume:
        return _dry_run(request)
    _validate_mutating_request(request)
    assert request.maintenance_root is not None
    assert request.quiescence_receipt is not None
    validate_quiescence_receipt(
        request.quiescence_receipt,
        state_db=request.state_db,
        clip_store=request.clip_store,
        maintenance_root=request.maintenance_root,
        expected_uid=request.expected_owner_uid,
    )
    try:
        with ClipStoreLock.acquire(request.clip_store):
            return resume_repair(request) if request.resume else apply_repair(request)
    except sqlite3.OperationalError as exc:
        raise ClipConsistencyError(
            "database_busy", "worker-state writer exclusion unavailable"
        ) from exc
    except sqlite3.Error as exc:
        raise ClipConsistencyError("database_error", "SQLite operation failed") from exc


def _dry_run(request: RepairRequest) -> RepairReceipt:
    connection = sqlite3.connect(
        f"file:{request.state_db}?mode=ro",
        uri=True,
        isolation_level=None,
    )
    try:
        connection.execute("BEGIN")
        schema_version = validate_database(connection, now=time.time())
        authority = scan_manifest_authority(
            request.clip_store,
            expected_uid=request.expected_owner_uid,
            ffprobe_bin=request.ffprobe_bin,
        )
        plan = plan_relations(
            connection,
            authority.desired,
            quarantine_clip_ids=tuple(path.name for path in authority.staging),
        )
        return RepairReceipt(
            RECEIPT_VERSION,
            "dry-run",
            "DRY_RUN",
            schema_version,
            build_counters(plan, authority),
            None,
            None,
        )
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def _validate_request_paths(request: RepairRequest) -> None:
    validate_regular(
        request.state_db,
        expected_uid=request.expected_owner_uid,
        exact_mode=None,
        label="state database",
    )
    validate_directory(
        request.clip_store,
        expected_uid=request.expected_owner_uid,
        owner_controlled=False,
        label="clip store",
    )


def _validate_mutating_request(request: RepairRequest) -> None:
    if (
        request.maintenance_root is None
        or request.journal_path is None
        or request.quiescence_receipt is None
    ):
        raise ClipConsistencyError(
            "operator_proof_required",
            "apply/resume requires maintenance root, journal, and quiescence receipt",
        )
    validate_directory(
        request.maintenance_root,
        expected_uid=request.expected_owner_uid,
        owner_controlled=True,
        label="maintenance root",
    )
    validate_under_root(
        request.journal_path,
        request.maintenance_root,
        allow_missing_leaf=request.apply,
    )
    if request.prebackup_receipt is not None:
        validate_under_root(
            request.prebackup_receipt,
            request.maintenance_root,
            allow_missing_leaf=False,
        )


__all__ = ["repair_clip_consistency"]
