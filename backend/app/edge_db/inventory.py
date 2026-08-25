"""Read-only filesystem drain gate for the schema-18 cutover."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db.clip_root import resolve_clip_root
from backend.app.edge_db.paths import EDGE_DATABASE_PATH

DEFAULT_RUNTIME_STATE_DIR = Path("/var/lib/seeon-worker-state")
DEFAULT_CLIP_STORE_DIR = Path("/var/lib/clip-store")
SCHEMA_V18 = 18


@dataclass(frozen=True, slots=True)
class FilesystemInventory:
    pending_delivery_entries: int
    pending_staging_clips: int
    #: Evidence the backend refused or that exhausted delivery. It is retained
    #: rather than deleted, which means it is still undelivered -- so the
    #: pre-cutover gate must count it. Ignoring it let the gate wave through a
    #: deployment holding evidence nobody had reviewed.
    retained_refused_entries: int = 0
    #: Snapshots staged but never published. A crash between stage() and
    #: publish() leaves one behind: no attachment was queued and no disposition
    #: was tagged, so the evidence is neither delivered nor discarded. Nothing
    #: reconciles them, so the gate must at least refuse to migrate over them.
    pending_snapshot_stagings: int = 0

    @property
    def is_empty(self) -> bool:
        return (
            self.pending_delivery_entries == 0
            and self.pending_staging_clips == 0
            and self.retained_refused_entries == 0
            and self.pending_snapshot_stagings == 0
        )

    def describe_pending(self) -> str:
        parts: list[str] = []
        if self.pending_delivery_entries:
            parts.append(f"delivery-queue entries={self.pending_delivery_entries}")
        if self.retained_refused_entries:
            parts.append(f"retained refused entries={self.retained_refused_entries}")
        if self.pending_snapshot_stagings:
            parts.append(f"unpublished snapshots={self.pending_snapshot_stagings}")
        if self.pending_staging_clips:
            parts.append(f"clip staging entries={self.pending_staging_clips}")
        return ", ".join(parts)


def read_schema_version(database: Path) -> int:
    """Read the central schema version without permitting SQLite to create or mutate it."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        row = connection.execute("PRAGMA user_version").fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


def inspect_filesystem(
    runtime_state_dir: Path, clip_store_dir: Path
) -> FilesystemInventory:
    """Count durable delivery envelopes and incomplete clip staging directories."""
    delivery_queue = runtime_state_dir / "delivery-queue"
    staging_root = clip_store_dir / "clips" / ".staging"
    snapshot_staging = clip_store_dir / ".snapshot-staging"
    return FilesystemInventory(
        pending_delivery_entries=sum(1 for path in delivery_queue.glob("*.json") if path.is_file()),
        retained_refused_entries=sum(
            1
            for path in (
                runtime_state_dir / "delivery-queue-dead-letter"
            ).glob("*")
            if path.is_file()
        ),
        pending_staging_clips=sum(1 for path in staging_root.iterdir() if path.is_dir())
        if staging_root.exists()
        else 0,
        pending_snapshot_stagings=sum(
            1 for path in snapshot_staging.rglob("*") if path.is_file()
        )
        if snapshot_staging.exists()
        else 0,
    )


def check_filesystem_drain(
    database: Path, runtime_state_dir: Path, clip_store_dir: Path
) -> tuple[bool, str]:
    """Apply the pre-cutover filesystem drain gate."""
    if not database.exists():
        # A fresh deployment: the migrator has not created the database yet, and
        # this gate runs before it. There is no legacy schema-16 evidence to
        # drain because there is no database at all. Failing closed here made a
        # clean install impossible to start, which is the over-broad direction
        # of this gate and just as damaging as letting a dirty one through.
        return True, "EDGE_FS_INVENTORY_FRESH no database yet; nothing to drain"
    version = read_schema_version(database)
    if version >= SCHEMA_V18:
        return True, f"EDGE_FS_INVENTORY_BYPASS schema={version} gate retired post-cutover"
    # The worker records beneath the operator-selected clip_store_subdir, so
    # scanning the mount root reports zero pending staging for a store it never
    # looked at. Resolve the root the worker actually uses, and fail closed if
    # the selection cannot be read rather than assuming the mount.
    inventory = inspect_filesystem(
        runtime_state_dir, resolve_clip_root(clip_store_dir, database)
    )
    if inventory.is_empty:
        return True, f"EDGE_FS_INVENTORY_OK schema={version}"
    return False, f"EDGE_FS_INVENTORY_PENDING schema={version} {inventory.describe_pending()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check filesystem drain before schema-18 cutover")
    parser.add_argument("--database", type=Path, default=EDGE_DATABASE_PATH)
    parser.add_argument("--runtime-state-dir", type=Path, default=DEFAULT_RUNTIME_STATE_DIR)
    parser.add_argument("--clip-store-dir", type=Path, default=DEFAULT_CLIP_STORE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        passed, message = check_filesystem_drain(
            args.database, args.runtime_state_dir, args.clip_store_dir
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"EDGE_FS_INVENTORY_FAILED: {error}", file=sys.stderr)
        return 1
    print(message, file=None if passed else sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CLIP_STORE_DIR",
    "DEFAULT_RUNTIME_STATE_DIR",
    "FilesystemInventory",
    "check_filesystem_drain",
    "inspect_filesystem",
    "main",
    "read_schema_version",
]
