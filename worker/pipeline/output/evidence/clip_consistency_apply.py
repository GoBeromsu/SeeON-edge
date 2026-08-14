"""Apply and resume state machine for clip consistency repair."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_backup import (
    ensure_prebackup,
    verify_backup_receipt_for_resume,
)
from worker.pipeline.output.evidence.clip_consistency_commit import (
    abort_precommit as _abort_precommit,
)
from worker.pipeline.output.evidence.clip_consistency_commit import (
    classify_ambiguous_commit as _classify_ambiguous_commit,
)
from worker.pipeline.output.evidence.clip_consistency_commit import (
    commit_connection as _commit_connection,
)
from worker.pipeline.output.evidence.clip_consistency_database import (
    plan_relations,
    relation_state_sha256,
    validate_database,
)
from worker.pipeline.output.evidence.clip_consistency_io import checkpoint, sha256_regular
from worker.pipeline.output.evidence.clip_consistency_journal import (
    ApplyJournal,
    mark_journal,
    prepared_journal,
    read_journal,
    write_journal,
)
from worker.pipeline.output.evidence.clip_consistency_manifest import (
    ManifestAuthority,
    scan_manifest_authority,
)
from worker.pipeline.output.evidence.clip_consistency_preimage import (
    non_relation_preimage_sha256,
)
from worker.pipeline.output.evidence.clip_consistency_receipt import (
    build_counters,
    journal_receipt,
)
from worker.pipeline.output.evidence.clip_consistency_storage import (
    ensure_quarantine,
    execute_plan,
    finish_cleanup,
    open_write_exclusion,
    plan_staging_quarantine,
    quarantine_staging,
    required_mutation_paths,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairReceipt,
    RepairRequest,
)


def apply_repair(request: RepairRequest) -> RepairReceipt:
    root, journal_path = required_mutation_paths(request)
    if journal_path.exists():
        raise ClipConsistencyError("journal_exists", "use --resume for existing journal")
    control: sqlite3.Connection | None = open_write_exclusion(request.state_db)
    moved: list[tuple[Path, Path]] = []
    journal: ApplyJournal | None = None
    committed = False
    commit_attempted = False
    try:
        validate_database(control, now=time.time())
        authority = _authority(request)
        plan = plan_relations(
            control,
            authority.desired,
            quarantine_clip_ids=tuple(path.name for path in authority.staging),
        )
        counters = build_counters(plan, authority)
        backup = ensure_prebackup(
            request.state_db,
            control,
            receipt_path=request.prebackup_receipt,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=request.fault_hook,
        )
        moved = plan_staging_quarantine(
            request.clip_store,
            plan.quarantine_clip_ids,
            plan.plan_sha256,
        )
        journal = prepared_journal(
            owner_uid=request.expected_owner_uid,
            state_db=request.state_db,
            clip_store=request.clip_store,
            journal_path=journal_path,
            source_state_sha256=backup.source_state_sha256,
            source_identity_sha256=backup.source_identity_sha256,
            non_relation_state_sha256=non_relation_preimage_sha256(control),
            backup_receipt_path=backup.receipt_path,
            backup_receipt_sha256=sha256_regular(Path(backup.receipt_path)),
            plan=plan,
            counters=counters,
        )
        write_journal(
            journal,
            path=journal_path,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=request.fault_hook,
            stage="journal_prepared",
        )
        quarantine_staging(moved, request.fault_hook)
        checkpoint(request.fault_hook, "apply:before_relations")
        execute_plan(control, plan)
        checkpoint(request.fault_hook, "apply:before_commit")
        commit_attempted = True
        try:
            _commit_connection(control)
        except BaseException as commit_error:
            control.close()
            control = None
            _classify_ambiguous_commit(request, journal, moved, commit_error)
            raise
        committed = True
        checkpoint(request.fault_hook, "apply:after_commit")
        journal = mark_journal(
            journal,
            "DB_COMMITTED",
            path=journal_path,
            maintenance_root=root,
            expected_uid=request.expected_owner_uid,
            hook=request.fault_hook,
        )
        return journal_receipt("apply", finish_cleanup(request, journal))
    except BaseException as exc:
        if not committed and not commit_attempted:
            assert control is not None
            _abort_precommit(request, control, moved, journal, exc)
        raise
    finally:
        if control is not None:
            control.close()


def resume_repair(request: RepairRequest) -> RepairReceipt:
    root, journal_path = required_mutation_paths(request)
    journal = read_journal(
        journal_path,
        maintenance_root=root,
        expected_uid=request.expected_owner_uid,
        state_db=request.state_db,
        clip_store=request.clip_store,
    )
    _verify_journal_backup(journal, root, request.expected_owner_uid)
    if journal.state == "UNKNOWN":
        raise ClipConsistencyError(
            "commit_state_unknown", "ambiguous commit state requires incident recovery"
        )
    if journal.state in {"DONE", "ABORTED"}:
        return journal_receipt("resume", journal)
    control = open_write_exclusion(request.state_db)
    try:
        validate_database(control, now=time.time())
        if non_relation_preimage_sha256(control) != journal.non_relation_state_sha256:
            raise ClipConsistencyError(
                "resume_conflict", "non-relation database preimage changed"
            )
        current = relation_state_sha256(control)
        if journal.state == "PREPARED":
            journal = _resume_prepared(request, journal, control, current)
        else:
            if current != journal.relations_after_sha256:
                raise ClipConsistencyError(
                    "resume_conflict", "committed relation state differs"
                )
            control.rollback()
    finally:
        control.close()
    return journal_receipt("resume", finish_cleanup(request, journal))


def _resume_prepared(
    request: RepairRequest,
    journal: ApplyJournal,
    control: object,
    current: str,
) -> ApplyJournal:
    assert isinstance(control, sqlite3.Connection)
    if current == journal.relations_before_sha256:
        ensure_quarantine(request.clip_store, journal.quarantine)
        execute_plan(control, journal.relation_plan())
        try:
            _commit_connection(control)
        except BaseException as commit_error:
            control.close()
            moved = [
                (request.clip_store / original, request.clip_store / held)
                for original, held in journal.quarantine
            ]
            _classify_ambiguous_commit(request, journal, moved, commit_error)
            raise
        checkpoint(request.fault_hook, "resume:after_commit")
    elif current == journal.relations_after_sha256:
        control.rollback()
    else:
        raise ClipConsistencyError("resume_conflict", "relation state is neither boundary")
    root, path = required_mutation_paths(request)
    return mark_journal(
        journal,
        "DB_COMMITTED",
        path=path,
        maintenance_root=root,
        expected_uid=request.expected_owner_uid,
        hook=request.fault_hook,
    )


def _verify_journal_backup(journal: ApplyJournal, root: Path, uid: int) -> None:
    receipt_path = Path(journal.backup_receipt_path)
    if sha256_regular(receipt_path) != journal.backup_receipt_sha256:
        raise ClipConsistencyError("backup_receipt_invalid", "journal receipt hash differs")
    backup = verify_backup_receipt_for_resume(
        receipt_path,
        maintenance_root=root,
        expected_uid=uid,
    )
    if (
        backup.source_state_sha256 != journal.source_state_sha256
        or backup.source_identity_sha256 != journal.source_identity_sha256
        or backup.source_path != journal.state_db
    ):
        raise ClipConsistencyError("backup_receipt_invalid", "journal source identity differs")


def _authority(request: RepairRequest) -> ManifestAuthority:
    return scan_manifest_authority(
        request.clip_store,
        expected_uid=request.expected_owner_uid,
        ffprobe_bin=request.ffprobe_bin,
    )


__all__ = ["apply_repair", "resume_repair"]
