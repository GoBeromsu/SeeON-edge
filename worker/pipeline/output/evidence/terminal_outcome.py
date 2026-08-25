"""Durable exactly-one terminal clip outcome for every event identity."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import final

from worker.pipeline.output.evidence.durability import fsync_directory, fsync_file


class TerminalClipState(StrEnum):
    READY = "ClipCommitted"
    UNAVAILABLE = "ClipUnavailableCommitted"
    CORRUPT = "ClipCorruptCommitted"


@dataclass(frozen=True, slots=True)
class TerminalClipOutcome:
    clip_id: str
    event_ids: tuple[str, ...]
    state: TerminalClipState
    manifest_sha256: str


@final
class TerminalOutcomeConflictError(Exception):
    pass


def commit_terminal_outcome(clip_dir: Path, outcome: TerminalClipOutcome) -> Path:
    event_ids = tuple(dict.fromkeys(outcome.event_ids))
    if not event_ids:
        raise TerminalOutcomeConflictError("terminal event identities are invalid")
    outcome = TerminalClipOutcome(
        outcome.clip_id, event_ids, outcome.state, outcome.manifest_sha256
    )
    aggregate = _commit_path(clip_dir / "terminal-outcome.json", _aggregate_payload(outcome))
    event_dir = clip_dir / "terminal-outcomes"
    event_dir.mkdir(exist_ok=True)
    fsync_directory(clip_dir)
    for event_id in outcome.event_ids:
        name = hashlib.sha256(event_id.encode()).hexdigest() + ".json"
        _ = _commit_path(event_dir / name, _event_payload(outcome, event_id))
    fsync_directory(event_dir)
    return aggregate


def commit_corrupt_terminal_outcome(clip_dir: Path, outcome: TerminalClipOutcome) -> Path:
    if outcome.state is not TerminalClipState.CORRUPT:
        raise ValueError("corrupt terminal transition requires CORRUPT state")
    outcome = TerminalClipOutcome(
        outcome.clip_id, tuple(dict.fromkeys(outcome.event_ids)),
        outcome.state, outcome.manifest_sha256,
    )
    aggregate = _replace_path(clip_dir / "terminal-outcome.json", _aggregate_payload(outcome))
    event_dir = clip_dir / "terminal-outcomes"
    event_dir.mkdir(exist_ok=True)
    for event_id in outcome.event_ids:
        name = hashlib.sha256(event_id.encode()).hexdigest() + ".json"
        _ = _replace_path(event_dir / name, _event_payload(outcome, event_id))
    fsync_directory(event_dir)
    return aggregate


def _aggregate_payload(outcome: TerminalClipOutcome) -> bytes:
    return _encode(
        {
            "clip_id": outcome.clip_id,
            "event_ids": list(outcome.event_ids),
            "manifest_sha256": outcome.manifest_sha256,
            "state": outcome.state.value,
        }
    )


def _event_payload(outcome: TerminalClipOutcome, event_id: str) -> bytes:
    return _encode(
        {
            "clip_id": outcome.clip_id,
            "event_id": event_id,
            "manifest_sha256": outcome.manifest_sha256,
            "state": outcome.state.value,
        }
    )


def _encode(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def _commit_path(destination: Path, payload: bytes) -> Path:
    if destination.exists():
        if destination.read_bytes() != payload:
            raise TerminalOutcomeConflictError(str(destination.name))
        fsync_file(destination)
        return destination
    _write_replace(destination, payload, exclusive=True)
    return destination


def _replace_path(destination: Path, payload: bytes) -> Path:
    if destination.exists() and destination.read_bytes() == payload:
        fsync_file(destination)
        return destination
    _write_replace(destination, payload, exclusive=False)
    return destination


def _write_replace(destination: Path, payload: bytes, *, exclusive: bool) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    mode = "xb" if exclusive else "wb"
    with temporary.open(mode) as output:
        _ = output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)
    fsync_file(destination)
    fsync_directory(destination.parent)


__all__ = [
    "TerminalClipOutcome", "TerminalClipState", "TerminalOutcomeConflictError",
    "commit_corrupt_terminal_outcome", "commit_terminal_outcome",
]
