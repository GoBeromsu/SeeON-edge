"""Shared schema-17 cutover fixture builders."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from backend.app.edge_db.compact_cutover import CompactCutoverRequest
from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import MIGRATIONS

TS = "2026-08-24T00:00:00Z"


def cutover_request(tmp_path: Path) -> CompactCutoverRequest:
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    source = state / "source.sqlite3"
    migrate_database(source, migrations=MIGRATIONS[:17])
    live = state / "edge.sqlite3"
    shutil.copyfile(source, live)
    clip_store = state / "clip-store"
    worker_state = state / "worker-state"
    clip_store.mkdir()
    worker_state.mkdir()
    return CompactCutoverRequest(
        source=source,
        live=live,
        archive=state / "edge.v17.archive.sqlite3",
        candidate=state / "edge.v18.candidate.sqlite3",
        receipt=state / "edge.v17.reconciliation.jsonl",
        clip_store=clip_store,
        worker_state=worker_state,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["TS", "cutover_request", "sha256"]
