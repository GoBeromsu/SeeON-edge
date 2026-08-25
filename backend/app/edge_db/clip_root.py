"""Resolve the clip-store root the worker is actually writing into.

The worker appends the operator-selected ``clip_store_subdir`` beneath the
mounted clip store (``WorkerRuntime._resolved_clip_store_dir``), so with a
selection configured the finalized clips live at
``<mount>/<subdir>/clips/...`` and not at ``<mount>/clips/...``.

Tools that assumed the mount root were therefore looking in the wrong place. The
pre-cutover inventory would report zero pending staging for a store it never
scanned, and clip recovery pointed at the mount root would find a stale
``.staging`` marker, proceed, and classify every still-present nested clip as
missing -- writing off live evidence and then clearing the schema-17 gate. That
is the same defect class that has already been caught five separate times in
this effort, so the resolution lives in one place and both callers use it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path, PurePosixPath

__all__ = ["ClipRootError", "resolve_clip_root"]


class ClipRootError(RuntimeError):
    """The clip root could not be resolved with confidence."""


def resolve_clip_root(mount: Path, database: Path) -> Path:
    """Return the directory the worker records finalized clips beneath.

    Fails closed: an unreadable selection raises rather than silently falling
    back to the mount root, because that fallback is exactly how a populated
    nested store gets classified as empty.
    """
    if not database.exists():
        raise ClipRootError(
            f"cannot resolve the clip root without the edge database at {database}; "
            f"refusing to assume the mount root, because a wrong root classifies "
            f"present clips as missing"
        )
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        tables = {
            str(item[0])
            for item in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "edge_site" in tables:
            row = connection.execute(
                "SELECT clip_store_subdir FROM edge_site WHERE id=1"
            ).fetchone()
        elif "clip_storage_location" in tables:
            row = connection.execute(
                "SELECT selected_path FROM clip_storage_location WHERE id=1"
            ).fetchone()
        else:
            row = None
    except sqlite3.Error as error:
        raise ClipRootError(
            f"clip storage selection is unreadable in {database}: {error}"
        ) from error
    finally:
        connection.close()

    selected = "" if row is None or row[0] is None else str(row[0])
    if not selected:
        return mount
    candidate = PurePosixPath(selected)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ClipRootError(f"clip storage selection {selected!r} is not a contained relative path")
    return mount / selected
