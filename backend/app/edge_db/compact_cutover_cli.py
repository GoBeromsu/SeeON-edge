"""CLI boundary for the stopped-runtime compact cutover."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from backend.app.edge_db.compact_cutover import (
    CompactCutoverRequest,
    rollback_compact_cutover,
    run_compact_cutover,
)
from backend.app.edge_db.compact_cutover_files import copy_exclusive
from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.sqlite_runtime import SqliteRuntimeError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and atomically install a verified schema-18 candidate"
    )
    for name in (
        "source",
        "live",
        "archive",
        "candidate",
        "receipt",
        "clip-store",
        "worker-state",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--rollback", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = args.source
        live = args.live
        if source.exists() and live.exists() and source.resolve() == live.resolve():
            snapshot = live.with_name("edge-v17-source.sqlite3")
            if not snapshot.exists():
                copy_exclusive(live, snapshot, mode=0o400)
            source = snapshot
        request = CompactCutoverRequest(
            source=source,
            live=live,
            archive=args.archive,
            candidate=args.candidate,
            receipt=args.receipt,
            clip_store=args.clip_store,
            worker_state=args.worker_state,
            expected_source_sha256=args.expected_source_sha256,
        )
        result = (
            rollback_compact_cutover(request)
            if args.rollback
            else run_compact_cutover(request)
        )
    except (OSError, sqlite3.Error, EdgeDatabaseError, SqliteRuntimeError) as error:
        print(f"EDGE_DB_COMPACT_CUTOVER_FAILED: {error}", file=sys.stderr)
        return 1
    print(
        f"EDGE_DB_COMPACT_CUTOVER_OK live={result.live} rows={result.source_rows} "
        f"source_sha256={result.source_sha256} receipt_sha256={result.receipt_sha256}"
    )
    return 0


__all__ = ["main"]
