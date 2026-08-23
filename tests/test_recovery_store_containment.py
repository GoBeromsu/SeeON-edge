"""Recovery must never reach an object outside the pinned clip store.

Two escapes existed. `evidence_clips.clip_id` is unconstrained TEXT and was
passed straight to `os.stat`/`os.open` with `dir_fd`, where an absolute value
ignores the descriptor entirely and a `../` value walks out of it -- so a
manifest and media planted anywhere on the filesystem could be recorded
`VERIFIED` for a clip in the database. Separately, retention inspection did one
`os.stat` on a whole multi-component path, and `follow_symlinks=False` protects
only the final component, so an intermediate symlink resolved outside the store
and let an external object decide `PURGED`.

Both are resident-safety issues: they decide whether recorded evidence is
declared present or written off.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.legacy_transitional_recovery import LegacyTransitionalRecovery
from backend.app.edge_db.migrator import MIGRATIONS, migrate_database
from backend.app.features.clips.consistency_ops import (
    ClipConsistencyError,
    inspect_finalized_clip,
)

_MEDIA = b"\x00\x00\x00\x18ftypmp42" + b"z" * 200
_SCHEMA_16 = 16


def _planted_clip(directory: Path, clip_id: str) -> None:
    """A structurally perfect clip, deliberately placed outside the store."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "clip.mp4").write_bytes(_MEDIA)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 2,
                "state": "READY",
                "clip_id": clip_id,
                "camera_id": "cam-1",
                "event_refs": ["11111111-1111-4111-8111-111111111111"],
                "event_ref": "11111111-1111-4111-8111-111111111111",
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:00:01Z",
                "sha256": hashlib.sha256(_MEDIA).hexdigest(),
                "size_bytes": len(_MEDIA),
                "mime_type": "video/mp4",
                "codec": "h264",
                "duration_ms": 1000,
                "path": f"clips/{clip_id}/clip.mp4",
            }
        ),
        encoding="utf-8",
    )


def _store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    (store / "clips" / ".staging").mkdir(parents=True, exist_ok=True)
    return store


@pytest.mark.parametrize("shape", ["relative_escape", "absolute"])
def test_an_escaping_clip_id_is_refused_before_any_filesystem_access(
    tmp_path: Path, shape: str
) -> None:
    """A clip id must name one bounded component, or be refused outright."""
    store = _store(tmp_path)
    outside = tmp_path / "outside" / "evil"
    _planted_clip(outside, "evil")

    clip_id = "../../outside/evil" if shape == "relative_escape" else str(outside)

    handle = os.open(store / "clips", os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ClipConsistencyError) as caught:
            inspect_finalized_clip(store, clip_id, clips_dir_fd=handle)
    finally:
        os.close(handle)

    assert caught.value.code == "clip_id_unsafe"


@pytest.mark.parametrize("shape", ["relative_escape", "absolute"])
def test_the_pathname_variant_refuses_the_same_ids(tmp_path: Path, shape: str) -> None:
    """The escape is just as reachable without a descriptor."""
    store = _store(tmp_path)
    outside = tmp_path / "outside" / "evil"
    _planted_clip(outside, "evil")

    clip_id = "../../outside/evil" if shape == "relative_escape" else str(outside)

    with pytest.raises(ClipConsistencyError) as caught:
        inspect_finalized_clip(store, clip_id)

    assert caught.value.code == "clip_id_unsafe"


def test_an_ordinary_clip_id_is_still_accepted(tmp_path: Path) -> None:
    """Guard the guard: refusing everything would also pass the tests above."""
    store = _store(tmp_path)
    _planted_clip(store / "clips" / "clip:ok", "clip:ok")

    assert inspect_finalized_clip(store, "clip:ok").local_state == "VERIFIED"


def test_retention_through_an_intermediate_symlink_is_unverifiable(
    tmp_path: Path,
) -> None:
    """An intermediate symlink must not let an external object decide PURGED."""
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    _planted_clip(outside / "evil", "evil")
    (store / "clips" / "link").symlink_to(outside)

    database = tmp_path / "edge.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:_SCHEMA_16])
    stamp = "2026-08-22T00:00:00Z"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, local_state, media_relpath, state_version) "
            "VALUES ('clip:s', 'VERIFIED', 'clips/link/evil/clip.mp4', 1)"
        )
        connection.execute(
            "INSERT INTO evidence_retention_states "
            "(clip_id, state, revision, requested_at, updated_at) "
            "VALUES ('clip:s', 'PENDING', 1, ?, ?)",
            (stamp, stamp),
        )

    LegacyTransitionalRecovery(database, store).run()

    with sqlite3.connect(database) as connection:
        state, reason = connection.execute(
            "SELECT state, reason FROM evidence_retention_states"
        ).fetchone()

    assert state == "FAILED"
    assert reason == "LEGACY_RETENTION_MEDIA_PATH_UNVERIFIABLE"


# --- exhaustive shape coverage -------------------------------------------------
#
# The cases above cover two clip-id escapes and one symlink position. A reviewer
# correctly pointed out that the guard's real surface is wider than that, so
# every shape probed by hand is pinned here instead of resting on an ad-hoc run.

_UNSAFE_CLIP_IDS = (
    ("empty", ""),
    ("dot", "."),
    ("dotdot", ".."),
    ("relative_escape", "../../outside/evil"),
    ("nested_separator", "a/b"),
    ("backslash", "a\\b"),
    ("trailing_separator", "clip:ok/"),
    ("leading_separator", "/clip:ok"),
)


@pytest.mark.parametrize(
    ("name", "clip_id"), _UNSAFE_CLIP_IDS, ids=[case[0] for case in _UNSAFE_CLIP_IDS]
)
@pytest.mark.parametrize("through_descriptor", [False, True], ids=["pathname", "descriptor"])
def test_every_unsafe_clip_id_shape_is_refused(
    tmp_path: Path, name: str, clip_id: str, through_descriptor: bool
) -> None:
    """Both inspection entries must refuse anything that is not one component."""
    store = _store(tmp_path)

    if not through_descriptor:
        with pytest.raises(ClipConsistencyError) as caught:
            inspect_finalized_clip(store, clip_id)
        assert caught.value.code == "clip_id_unsafe"
        return

    handle = os.open(store / "clips", os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ClipConsistencyError) as caught:
            inspect_finalized_clip(store, clip_id, clips_dir_fd=handle)
    finally:
        os.close(handle)
    assert caught.value.code == "clip_id_unsafe"


def _retention_verdict(tmp_path: Path, store: Path, media_relpath: str) -> tuple[str, str | None]:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database, migrations=MIGRATIONS[:_SCHEMA_16])
    stamp = "2026-08-22T00:00:00Z"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO evidence_clips (clip_id, local_state, media_relpath, state_version) "
            "VALUES ('clip:r', 'VERIFIED', ?, 1)",
            (media_relpath,),
        )
        connection.execute(
            "INSERT INTO evidence_retention_states "
            "(clip_id, state, revision, requested_at, updated_at) "
            "VALUES ('clip:r', 'PENDING', 1, ?, ?)",
            (stamp, stamp),
        )

    LegacyTransitionalRecovery(database, store).run()

    with sqlite3.connect(database) as connection:
        state, reason = connection.execute(
            "SELECT state, reason FROM evidence_retention_states"
        ).fetchone()
    return (str(state), reason)


_LEXICALLY_REJECTED = (
    ("empty", ""),
    ("clips_alone", "clips"),
    ("absolute", "/etc/passwd"),
    ("escape", "../../etc/passwd"),
    ("escape_via_clips", "clips/../../etc/passwd"),
    ("dot_segment", "clips/./evil/clip.mp4"),
    ("double_separator", "clips//evil/clip.mp4"),
    ("trailing_separator", "clips/evil/"),
    ("wrong_root", "media/evil/clip.mp4"),
)


@pytest.mark.parametrize(
    ("name", "media_relpath"),
    _LEXICALLY_REJECTED,
    ids=[case[0] for case in _LEXICALLY_REJECTED],
)
def test_every_malformed_retention_path_is_unverifiable(
    tmp_path: Path, name: str, media_relpath: str
) -> None:
    """A path we cannot canonically bound must never decide a purge."""
    store = _store(tmp_path)

    assert _retention_verdict(tmp_path, store, media_relpath) == (
        "FAILED",
        "LEGACY_RETENTION_MEDIA_PATH_UNVERIFIABLE",
    )


def test_a_symlink_at_the_final_component_is_unverifiable_not_present(
    tmp_path: Path,
) -> None:
    """A link we refused to follow is not evidence that the media is there.

    This reported `LEGACY_RETENTION_MEDIA_STILL_PRESENT` -- the same reason as
    genuinely present media -- because `follow_symlinks=False` returns the link
    inode and the caller read that as existence. An operator would conclude the
    file was sitting there when what was actually found is a pointer nobody
    resolved.
    """
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "clip.mp4").write_bytes(_MEDIA)
    directory = store / "clips" / "real"
    directory.mkdir(parents=True)
    (directory / "clip.mp4").symlink_to(outside / "clip.mp4")

    assert _retention_verdict(tmp_path, store, "clips/real/clip.mp4") == (
        "FAILED",
        "LEGACY_RETENTION_MEDIA_PATH_UNVERIFIABLE",
    )


def test_genuinely_absent_media_still_purges(tmp_path: Path) -> None:
    """Guard the guard: refusing every shape would make PURGED unreachable."""
    store = _store(tmp_path)
    (store / "clips" / "real").mkdir(parents=True)

    assert _retention_verdict(tmp_path, store, "clips/real/clip.mp4") == ("PURGED", None)


def test_genuinely_present_media_is_reported_as_present(tmp_path: Path) -> None:
    """Real media must be distinguishable from a path we could not verify."""
    store = _store(tmp_path)
    directory = store / "clips" / "real"
    directory.mkdir(parents=True)
    (directory / "clip.mp4").write_bytes(_MEDIA)

    state, reason = _retention_verdict(tmp_path, store, "clips/real/clip.mp4")
    assert state == "FAILED"
    assert reason == "LEGACY_RETENTION_MEDIA_STILL_PRESENT"
