#!/usr/bin/env python3
"""Resolve schema-16 evidence states that block schema-17 migration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover legacy schema-16 evidence clips.")
    _ = parser.add_argument("--database", required=True, type=Path)
    _ = parser.add_argument("--clip-store", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository))
    from backend.app.edge_db.clip_root import resolve_clip_root
    from backend.app.edge_db.legacy_clip_recovery import (  # noqa: PLC0415
        LegacyClipRecovery,
        LegacyClipStoreUnavailableError,
    )
    from backend.app.edge_db.legacy_transitional_recovery import (  # noqa: PLC0415
        LegacyTransitionalRecovery,
    )

    arguments = _parser().parse_args(argv)
    try:
        # The worker records beneath the operator-selected clip_store_subdir.
        # Pointed at the mount root while clips live nested, recovery finds a
        # stale .staging marker, proceeds, and classifies every still-present
        # clip as missing -- writing off live evidence and then clearing the
        # schema-17 gate. Resolve the root the worker actually uses.
        clip_root = resolve_clip_root(arguments.clip_store, arguments.database)
        clips = LegacyClipRecovery(arguments.database, clip_root).run()
    except LegacyClipStoreUnavailableError as error:
        # Nothing was written. Exit distinctly from "ran and found work left",
        # so an unreadable or mistyped --clip-store cannot be mistaken for a
        # completed pass.
        print(json.dumps({"error": "clip_store_unavailable", "detail": str(error)}))
        return 3
    try:
        transitional = LegacyTransitionalRecovery(arguments.database, arguments.clip_store).run()
    except LegacyClipStoreUnavailableError as error:
        # Clip recovery has already committed its independently verified
        # classifications. Do not misrepresent that partial completion as a
        # no-write refusal.
        print(
            json.dumps(
                {
                    "corrupt": clips.corrupt,
                    "detail": str(error),
                    "error": "clip_store_unavailable_after_clip_recovery",
                    "publication_terminalized": clips.publication_terminalized,
                    "unavailable": clips.unavailable,
                    "verified": clips.verified,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 4
    print(
        json.dumps(
            {
                "corrupt": clips.corrupt,
                "derivatives_cancelled": transitional.derivatives_cancelled,
                "publication_terminalized": clips.publication_terminalized,
                "retention_failed": transitional.retention_failed,
                "retention_purged": transitional.retention_purged,
                "unavailable": clips.unavailable,
                "unresolved": clips.unresolved + transitional.unresolved,
                "verified": clips.verified,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if clips.unresolved + transitional.unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
