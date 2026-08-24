"""Locked path, inode, checkpoint, and drain preflight."""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

from backend.app.edge_db.compact_cutover_files import (
    require_distinct_inodes,
    require_regular_single_link,
)
from backend.app.edge_db.compact_cutover_types import CompactCutoverError, CompactCutoverRequest
from backend.app.edge_db.schema import MIGRATIONS


def require_paths(request: CompactCutoverRequest) -> None:
    source = require_regular_single_link(request.source, "EDGE_DB_CUTOVER_SOURCE")
    live = require_regular_single_link(request.live, "EDGE_DB_CUTOVER_LIVE")
    require_distinct_inodes(
        (request.source, request.live, request.archive, request.candidate, request.receipt)
    )
    live_wal = Path(f"{request.live}-wal")
    if live_wal.exists() and live_wal.stat().st_size > 0:
        raise CompactCutoverError("EDGE_DB_CUTOVER_LIVE_SIDECAR")
    for directory in (request.clip_store, request.worker_state, request.live.parent):
        mode = directory.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise CompactCutoverError("EDGE_DB_CUTOVER_SYMLINK")
        if not stat.S_ISDIR(mode):
            raise CompactCutoverError("EDGE_DB_CUTOVER_DIRECTORY_INVALID")
    parent = request.live.parent.resolve()
    if request.source.parent.resolve() != parent:
        raise CompactCutoverError("EDGE_DB_CUTOVER_CROSS_FILESYSTEM")
    if any(output.parent.resolve() != parent for output in _outputs(request)):
        raise CompactCutoverError("EDGE_DB_CUTOVER_CROSS_FILESYSTEM")
    devices = {
        source.st_dev,
        live.st_dev,
        request.clip_store.stat().st_dev,
        request.worker_state.stat().st_dev,
        parent.stat().st_dev,
    }
    if len(devices) != 1:
        raise CompactCutoverError("EDGE_DB_CUTOVER_CROSS_FILESYSTEM")
    if source.st_uid != live.st_uid or source.st_uid != os.geteuid():
        raise CompactCutoverError("EDGE_DB_CUTOVER_OWNERSHIP")


def checkpoint_and_preflight(source: Path) -> None:
    connection = sqlite3.connect(source, timeout=5.0, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout=5000")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise CompactCutoverError("EDGE_DB_CUTOVER_CHECKPOINT_BUSY")
    finally:
        connection.close()
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as readonly:
        version = readonly.execute("PRAGMA user_version").fetchone()
        if version != (17,):
            raise CompactCutoverError("EDGE_DB_CUTOVER_SOURCE_VERSION")
        preflight = MIGRATIONS[17].preflight
        assert preflight is not None
        preflight(readonly)


def schema_version(path: Path) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        row = connection.execute("PRAGMA user_version").fetchone()
    return -1 if row is None else int(row[0])


def _outputs(request: CompactCutoverRequest) -> tuple[Path, Path, Path]:
    return request.archive, request.candidate, request.receipt


__all__ = ["checkpoint_and_preflight", "require_paths", "schema_version"]
