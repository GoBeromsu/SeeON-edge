"""The preserved-sidecar archival transaction never destroys an unarchived source.

Each test forces one of the loss paths the transaction exists to close and
asserts the same invariant: on failure, every source is still present and
byte-identical. Hash comparison alone cannot prove that, which is why these
cases inject aliasing, durability, and clobber failures rather than only
checking the happy path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.archive_preserved_sidecars import (
    ArchivalError,
    archive_batch,
    destroy_sources,
    digest_and_size,
    execute,
    load_manifest,
    verify_batch,
)

_FILES = {
    "arch.json": b'{"hidden": 128}\n',
    "metadata.yaml": b"weights_sha256: 72570b\nfps: 15\n",
    "handoff.md": b"# handoff\n\ncontent\n",
}


@pytest.fixture
def sources(tmp_path: Path) -> list[Path]:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    written: list[Path] = []
    for name, payload in _FILES.items():
        path = worktree / name
        path.write_bytes(payload)
        written.append(path)
    return written


@pytest.fixture
def archive_root(tmp_path: Path) -> Path:
    return tmp_path / "archive" / "sidecars"


def _snapshot(paths: list[Path]) -> dict[Path, tuple[str, int]]:
    return {path: digest_and_size(path) for path in paths}


def _assert_sources_untouched(
    paths: list[Path], before: dict[Path, tuple[str, int]]
) -> None:
    for path in paths:
        assert path.is_file(), f"source destroyed despite failure: {path}"
        assert digest_and_size(path) == before[path], f"source mutated: {path}"


def test_happy_path_archives_every_file_and_publishes_a_matching_manifest(
    sources: list[Path], archive_root: Path
) -> None:
    before = _snapshot(sources)

    manifest_path, archived = archive_batch(sources, archive_root)

    assert manifest_path.is_file()
    assert len(archived) == len(sources)
    entries = load_manifest(manifest_path)
    assert len(entries) == len(sources)
    for entry in entries:
        # The archived copy is independent of its source and digests identically.
        assert entry.destination.is_file()
        assert not entry.destination.is_symlink()
        assert digest_and_size(entry.destination) == (entry.sha256, entry.size_bytes)
        assert before[entry.source] == (entry.sha256, entry.size_bytes)
        assert os.stat(entry.destination).st_ino != os.stat(entry.source).st_ino

    # Archival alone never touches a source.
    _assert_sources_untouched(sources, before)
    # And the pre-destruction re-verification passes on an untouched batch.
    assert len(verify_batch(manifest_path)) == len(sources)


def test_symlink_destination_pointing_back_at_the_source_is_refused(
    sources: list[Path], archive_root: Path
) -> None:
    before = _snapshot(sources)
    archive_root.mkdir(parents=True)
    # The classic aliasing trap: the archive entry is a symlink back into the
    # worktree, so a naive copy-then-compare would digest-match its own source.
    (archive_root / sources[0].name).symlink_to(sources[0])

    with pytest.raises(ArchivalError):
        archive_batch(sources, archive_root)

    _assert_sources_untouched(sources, before)


def test_fsync_failure_leaves_every_source_untouched(
    sources: list[Path], archive_root: Path
) -> None:
    before = _snapshot(sources)

    def failing_fsync(_fd: int) -> None:
        raise OSError("simulated fsync failure")

    with pytest.raises(OSError):
        archive_batch(sources, archive_root, fsync_file=failing_fsync)

    _assert_sources_untouched(sources, before)
    assert not list(archive_root.glob(".*.incoming")), "temporary artifact leaked"


def test_publication_failure_leaves_every_source_untouched(
    sources: list[Path], archive_root: Path
) -> None:
    before = _snapshot(sources)

    def failing_publish(_temporary: Path, _final: Path) -> None:
        raise OSError("simulated rename failure")

    with pytest.raises(OSError):
        archive_batch(sources, archive_root, publish=failing_publish)

    _assert_sources_untouched(sources, before)
    assert not list(archive_root.glob(".*.incoming")), "temporary artifact leaked"


def test_source_drift_after_archival_halts_before_any_destruction(
    sources: list[Path], archive_root: Path
) -> None:
    manifest_path, _ = archive_batch(sources, archive_root)
    # Someone edits a source in the window between archival and destruction.
    sources[1].write_bytes(b"edited after the archive was taken\n")
    after_drift = _snapshot(sources)

    with pytest.raises(ArchivalError):
        verify_batch(manifest_path)

    # The drifted source, and every other source, survives untouched.
    _assert_sources_untouched(sources, after_drift)


def test_manifest_name_collision_is_refused_rather_than_clobbered(
    sources: list[Path], archive_root: Path
) -> None:
    before = _snapshot(sources)
    archive_root.mkdir(parents=True)
    prior = archive_root / "manifest.json"
    prior.write_text("prior archive run\n")

    with pytest.raises(ArchivalError):
        archive_batch(sources, archive_root)

    assert prior.read_text() == "prior archive run\n", "earlier manifest clobbered"
    _assert_sources_untouched(sources, before)


def test_existing_archive_entry_is_refused_rather_than_clobbered(
    sources: list[Path], archive_root: Path
) -> None:
    before = _snapshot(sources)
    archive_root.mkdir(parents=True)
    prior = archive_root / sources[0].name
    prior.write_text("earlier archive of the same name\n")

    with pytest.raises(ArchivalError):
        archive_batch(sources, archive_root)

    assert prior.read_text() == "earlier archive of the same name\n"
    _assert_sources_untouched(sources, before)


def test_destruction_only_runs_after_verification_passes(
    sources: list[Path], archive_root: Path, tmp_path: Path
) -> None:
    tracked = [sources[0]]
    untracked = sources[1:]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        # Simulate `git restore` returning the tracked file to HEAD content.
        sources[0].write_bytes(b"HEAD content\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    manifest_path, _ = archive_batch([*tracked, *untracked], archive_root)
    verify_batch(manifest_path)
    destroy_sources(tmp_path, tracked, untracked, run=fake_run)

    assert calls and calls[0][:3] == ["git", "restore", "--"]
    assert "--staged" not in calls[0] and "commit" not in calls[0]
    assert sources[0].read_bytes() == b"HEAD content\n"
    for path in untracked:
        assert not path.exists(), f"untracked source not removed: {path}"
    # The archive still holds every original.
    for entry in load_manifest(manifest_path):
        assert digest_and_size(entry.destination) == (entry.sha256, entry.size_bytes)


def test_execute_dry_run_archives_without_mutating_sources(
    sources: list[Path], archive_root: Path, tmp_path: Path
) -> None:
    before = _snapshot(sources)

    manifest_path = execute(
        tmp_path, archive_root, sources[:1], sources[1:], destroy=False
    )

    assert manifest_path.is_file()
    _assert_sources_untouched(sources, before)


def test_missing_source_refuses_the_whole_batch(
    sources: list[Path], archive_root: Path
) -> None:
    before = _snapshot(sources[:-1])
    sources[-1].unlink()

    with pytest.raises(ArchivalError):
        archive_batch(sources, archive_root)

    _assert_sources_untouched(sources[:-1], before)
