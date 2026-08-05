#!/usr/bin/env python3
"""One-time migration (issue #155): rewrite legacy string camera floor
values ('2층', 'B1', '2 층', ...) in camera_registry to the canonical
integer (basement negative, e.g. B1 = -1).

Always backs up the catalog file before writing (see --catalog). Safe to
run more than once -- an already-integer (or unset) floor is left
untouched, and a value that can't be parsed -- or parses outside the fixed
B1..10층 catalog -- is never silently dropped; it defaults to 1층 with a
warning log (see backend.app.features.cameras.store.parse_legacy_floor).
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.shared.state_dir import resolve_state_dir

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=resolve_state_dir("ml-api") / "catalog.sqlite3",
    )
    args = parser.parse_args()

    if not args.catalog.exists():
        print(f"no catalog at {args.catalog}, nothing to migrate")
        return

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = args.catalog.with_name(f"{args.catalog.name}.bak-{timestamp}")
    shutil.copy2(args.catalog, backup_path)
    print(f"backed up {args.catalog} to {backup_path}")

    store = CameraRegistryStore(args.catalog)
    changes = store.migrate_legacy_string_floors()
    if not changes:
        print("no legacy string floor values found")
        return
    for change in changes:
        print(f"camera {change['camera_id']}: floor {change['old']!r} -> {change['new']!r}")
    print(f"migrated {len(changes)} camera(s)")


if __name__ == "__main__":
    main()
