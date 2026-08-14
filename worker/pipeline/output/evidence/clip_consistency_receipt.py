"""Repair counter and receipt construction."""

from __future__ import annotations

from worker.pipeline.output.evidence.clip_consistency_database import RelationPlan
from worker.pipeline.output.evidence.clip_consistency_journal import ApplyJournal
from worker.pipeline.output.evidence.clip_consistency_manifest import ManifestAuthority
from worker.pipeline.output.evidence.clip_consistency_types import (
    RepairCounters,
    RepairReceipt,
)

RECEIPT_VERSION = 2


def build_counters(plan: RelationPlan, authority: ManifestAuthority) -> RepairCounters:
    staging = len(authority.staging)
    return RepairCounters(
        ready_finals=authority.ready_count,
        unavailable_finals=authority.unavailable_count,
        relations_before=plan.relations_before,
        relations_after=plan.relations_after,
        mismatch_clips=plan.mismatch_clips,
        mismatch_tuples=plan.mismatch_tuples,
        sql_relations_deleted=len(plan.delete_event_ids),
        sql_relations_inserted=len(plan.insert_rows),
        staging_before=staging,
        staging_after=staging,
        staging_to_delete=staging,
        staging_deleted=0,
    )


def journal_receipt(mode: str, journal: ApplyJournal) -> RepairReceipt:
    return RepairReceipt(
        RECEIPT_VERSION,
        mode,
        journal.state,
        journal.schema_version,
        journal.counters,
        journal.backup_receipt_path,
        journal.journal_path,
    )


__all__ = ["RECEIPT_VERSION", "build_counters", "journal_receipt"]
