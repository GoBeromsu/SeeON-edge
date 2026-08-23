"""Tools must look where the worker actually writes clips.

The worker appends the operator-selected `clip_store_subdir` beneath the mounted
clip store, so with a selection configured finalized clips live at
`<mount>/<subdir>/clips/...`. The pre-cutover inventory and legacy clip recovery
both assumed `<mount>/clips/...`.

Two consequences, both live-evidence:

- inventory reports zero pending staging for a store it never scanned, so the
  gate clears while unfinished work sits in the real one;
- recovery pointed at the mount root finds a stale `.staging` marker, proceeds,
  and classifies every still-present nested clip as missing -- writing off real
  evidence and then opening the schema-17 gate.

That is the sixth distinct route by which clip recovery could write off the 1053
live clips, so the resolution lives in one place and both callers use it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.clip_root import ClipRootError, resolve_clip_root
from backend.app.edge_db.migrator import migrate_database


def _database(tmp_path: Path, *, selected: str | None) -> Path:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    if selected is not None:
        with sqlite3.connect(database) as connection:
            connection.execute(
                "INSERT INTO clip_storage_location (id, selected_path) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET selected_path = excluded.selected_path",
                (selected,),
            )
    return database


def test_a_configured_subdirectory_moves_the_clip_root(tmp_path: Path) -> None:
    """The decisive case: the worker is not recording at the mount root."""
    mount = tmp_path / "clip-store"
    database = _database(tmp_path, selected="external-drive")

    assert resolve_clip_root(mount, database) == mount / "external-drive"


def test_no_selection_resolves_to_the_mount_root(tmp_path: Path) -> None:
    """Guard the guard: the ordinary deployment must still resolve."""
    mount = tmp_path / "clip-store"
    database = _database(tmp_path, selected="")

    assert resolve_clip_root(mount, database) == mount


def test_an_absent_selection_table_is_not_treated_as_ambiguity(
    tmp_path: Path,
) -> None:
    """A store that never chose a subdirectory records at the mount root.

    Refusing here would make the gate unclearable for the ordinary case, which
    is its own accident.
    """
    mount = tmp_path / "clip-store"
    database = tmp_path / "bare.sqlite3"
    sqlite3.connect(database).close()

    assert resolve_clip_root(mount, database) == mount


def test_a_missing_database_fails_closed(tmp_path: Path) -> None:
    """Assuming the mount root is how a populated nested store looks empty."""
    with pytest.raises(ClipRootError, match="refusing to assume"):
        resolve_clip_root(tmp_path / "clip-store", tmp_path / "absent.sqlite3")


@pytest.mark.parametrize("selected", ["/etc", "../escape", "a/../../b"])
def test_an_uncontained_selection_is_refused(tmp_path: Path, selected: str) -> None:
    """A selection that escapes the mount is never a clip root."""
    database = _database(tmp_path, selected=selected)

    with pytest.raises(ClipRootError, match="contained relative path"):
        resolve_clip_root(tmp_path / "clip-store", database)


def test_the_inventory_gate_scans_the_resolved_root(tmp_path: Path) -> None:
    """End to end: unfinished work in the real store must block the cutover."""
    from backend.app.edge_db import inventory

    mount = tmp_path / "clip-store"
    database = _database(tmp_path, selected="external-drive")
    # The gate only applies before the cutover; a v17 database bypasses it.
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 16")
    state = tmp_path / "runtime-state"
    (state / "delivery-queue").mkdir(parents=True)

    # Unfinished staging exists only in the nested store the worker uses.
    staging = mount / "external-drive" / "clips" / ".staging" / "clip-in-progress"
    staging.mkdir(parents=True)

    cleared, detail = inventory.check_filesystem_drain(database, state, mount)

    assert not cleared, (
        f"the gate cleared with unfinished staging in the real clip store: {detail}"
    )
    assert "staging" in detail
