"""Retired local evidence record catalog."""

from __future__ import annotations

from worker.pipeline.output.evidence.evidence_record_models import (
    ArtifactState,
    EvidenceLifecycle,
    EvidenceRecord,
    EvidenceRecordConflictError,
    PrimaryEvidence,
)


class EvidenceRecordStore:
    def __init__(self, _path: object) -> None:
        raise RuntimeError("local evidence records were removed; use backend evidence")


__all__ = [
    "ArtifactState",
    "EvidenceLifecycle",
    "EvidenceRecord",
    "EvidenceRecordConflictError",
    "EvidenceRecordStore",
    "PrimaryEvidence",
]
