"""CLI boundary for the stopped-runtime compact cutover."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from backend.app.edge_db.compact_cutover import CompactCutoverRequest, run_compact_cutover
from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.sqlite_runtime import SqliteRuntimeError, parse_sqlite_version


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
    parser.add_argument("--sqlite-version")
    parser.add_argument("--rollback", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_compact_cutover(
            CompactCutoverRequest(
                source=args.source,
                live=args.live,
                archive=args.archive,
                candidate=args.candidate,
                receipt=args.receipt,
                clip_store=args.clip_store,
                worker_state=args.worker_state,
                expected_source_sha256=args.expected_source_sha256,
            ),
            rollback=args.rollback,
            sqlite_version=(
                None if args.sqlite_version is None else parse_sqlite_version(args.sqlite_version)
            ),
        )
    except (OSError, sqlite3.Error, EdgeDatabaseError, SqliteRuntimeError) as error:
        print(f"EDGE_DB_COMPACT_CUTOVER_FAILED: {error}", file=sys.stderr)
        return 1
    print(
        f"EDGE_DB_COMPACT_CUTOVER_OK live={result.live} current={result.current_version} "
        f"source_sha256={result.source_sha256 or '-'}"
    )
    return 0


__all__ = ["main"]
