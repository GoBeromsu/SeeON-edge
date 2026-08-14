"""Durable serialization for clip consistency apply journals."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from worker.pipeline.output.evidence.clip_consistency_io import atomic_write_json
from worker.pipeline.output.evidence.clip_consistency_types import FaultHook


class JournalPayload(Protocol):
    def to_dict(self) -> dict[str, object]: ...


def write_journal(
    journal: JournalPayload,
    *,
    path: Path,
    maintenance_root: Path,
    expected_uid: int,
    hook: FaultHook | None,
    stage: str,
) -> None:
    atomic_write_json(
        path,
        journal.to_dict(),
        root=maintenance_root,
        expected_uid=expected_uid,
        hook=hook,
        stage=stage,
    )


__all__ = ["write_journal"]
