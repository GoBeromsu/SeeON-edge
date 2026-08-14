"""Canonical same-ID staging quarantine authority derived from a repair plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError


def canonical_quarantine_rows(
    clip_ids: tuple[str, ...],
    plan_sha256: str,
) -> tuple[tuple[str, str], ...]:
    if clip_ids != tuple(sorted(set(clip_ids))):
        raise ClipConsistencyError("journal_invalid", "quarantine clip IDs are not canonical")
    rows: list[tuple[str, str]] = []
    for clip_id in clip_ids:
        _validate_clip_id(clip_id)
        original = PurePosixPath("clips", ".staging", clip_id).as_posix()
        held = PurePosixPath(
            "clips",
            ".staging",
            f".clip-consistency-{plan_sha256[:16]}-{clip_id}",
        ).as_posix()
        rows.append((original, held))
    return tuple(rows)


def quarantine_rows_sha256(rows: tuple[tuple[str, str], ...]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def absolute_quarantine_rows(
    clip_store: Path,
    clip_ids: tuple[str, ...],
    plan_sha256: str,
) -> list[tuple[Path, Path]]:
    return [
        (clip_store / original, clip_store / held)
        for original, held in canonical_quarantine_rows(clip_ids, plan_sha256)
    ]


def _validate_clip_id(clip_id: str) -> None:
    path = PurePosixPath(clip_id)
    if (
        not clip_id
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != clip_id
        or clip_id in {".", ".."}
    ):
        raise ClipConsistencyError("journal_invalid", "quarantine clip ID is unsafe")


__all__ = [
    "absolute_quarantine_rows",
    "canonical_quarantine_rows",
    "quarantine_rows_sha256",
]
