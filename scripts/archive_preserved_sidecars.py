"""Durable, no-clobber archival transfer of owner-preserved working-tree files.

Moves a fixed set of owner-owned files out of the repository into a pinned
non-ephemeral archive root, then restores/removes the in-repo originals -- but
only after the archive is provably durable, independent, and complete.

Why this is not ``cp`` plus ``sha256sum``
-----------------------------------------
Comparing a fresh copy's digest to its source proves the bytes matched at one
instant. It does not prove the copy is durable, independent, or unique:

1. **Aliasing.** A destination that is a symlink (or hardlink) back into the
   worktree digests identically to its source. Destroying the source then
   leaves no independent copy. Closed by ``O_NOFOLLOW`` on create plus an
   explicit ``(st_dev, st_ino)`` distinctness proof.
2. **Page cache.** A just-written destination can be read back out of the page
   cache and digest correctly while nothing has reached stable storage. A power
   failure then loses it. Closed by ``fsync`` on the file *and* on the
   containing directory (the directory entry created by the rename needs its
   own flush).
3. **Silent overwrite.** An unpinned or reused destination name can clobber an
   earlier archive. POSIX ``rename(2)`` overwrites its target silently, so
   publication uses ``link(2)`` -- which fails with ``EEXIST`` rather than
   clobbering -- followed by unlinking the temporary name.

Every source mutation is gated behind the whole batch succeeding. On any
failure at any step the transaction halts having touched zero sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MANIFEST_NAME: Final = "manifest.json"
MANIFEST_SCHEMA: Final = 1
_CHUNK: Final = 1 << 20


class ArchivalError(RuntimeError):
    """The archival transaction refused to proceed. No source was mutated."""


@dataclass(frozen=True, slots=True)
class ArchivedFile:
    source: Path
    destination: Path
    size_bytes: int
    sha256: str

    def as_entry(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def digest_and_size(path: Path) -> tuple[str, int]:
    """Hash *path* by streaming it, returning ``(hex_digest, size_bytes)``."""
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _fsync_directory(directory: Path) -> None:
    """Flush *directory*'s own entries so a rename survives a crash."""
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _reject_aliased_target(
    source_stat: os.stat_result, target_stat: os.stat_result, destination: Path
) -> None:
    """Refuse a destination that resolves to the source's own inode.

    A symlink or hardlink back into the worktree digests identically to its
    source, so hashing alone cannot detect it.
    """
    if (target_stat.st_dev, target_stat.st_ino) == (
        source_stat.st_dev,
        source_stat.st_ino,
    ):
        raise ArchivalError(
            f"archive destination aliases its source inode: {destination}"
        )


def _publish_no_clobber(temporary: Path, final: Path) -> None:
    """Atomically publish *temporary* as *final* without clobbering.

    ``os.rename`` silently replaces an existing target, so it cannot be used.
    ``os.link`` fails with ``FileExistsError`` when the target exists, which is
    exactly the no-clobber publication primitive required here.
    """
    try:
        os.link(temporary, final)
    except FileExistsError as error:
        raise ArchivalError(
            f"refusing to clobber an existing archive entry: {final}"
        ) from error
    os.unlink(temporary)


def _copy_exclusive(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    fsync_file: Callable[[int], None] = os.fsync,
    publish: Callable[[Path, Path], None] = _publish_no_clobber,
    fsync_dir: Callable[[Path], None] = _fsync_directory,
) -> ArchivedFile:
    """Archive one file as a durable, independent, no-clobber copy."""
    if destination.exists() or destination.is_symlink():
        raise ArchivalError(
            f"refusing to clobber an existing archive entry: {destination}"
        )
    directory = destination.parent
    temporary = directory / f".{destination.name}.incoming"
    if temporary.exists() or temporary.is_symlink():
        raise ArchivalError(f"stale archival temporary already present: {temporary}")

    source_stat = os.stat(source)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        _reject_aliased_target(source_stat, os.fstat(fd), destination)
        with source.open("rb") as reader:
            while chunk := reader.read(_CHUNK):
                os.write(fd, chunk)
        fsync_file(fd)
    except BaseException:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    os.close(fd)

    try:
        publish(temporary, destination)
        fsync_dir(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    published_sha256, published_size = digest_and_size(destination)
    if published_size != expected_size:
        raise ArchivalError(
            f"archived size mismatch for {destination}: "
            f"{published_size} != {expected_size}"
        )
    if published_sha256 != expected_sha256:
        raise ArchivalError(
            f"archived digest mismatch for {destination}: "
            f"{published_sha256} != {expected_sha256}"
        )
    return ArchivedFile(source, destination, published_size, published_sha256)


def archive_batch(
    sources: Sequence[Path],
    archive_root: Path,
    *,
    fsync_file: Callable[[int], None] = os.fsync,
    publish: Callable[[Path, Path], None] = _publish_no_clobber,
    fsync_dir: Callable[[Path], None] = _fsync_directory,
) -> tuple[Path, tuple[ArchivedFile, ...]]:
    """Archive every source, then durably publish a no-clobber manifest."""
    if not sources:
        raise ArchivalError("refusing to run an empty archival batch")
    archive_root.mkdir(parents=True, exist_ok=True)

    archived: list[ArchivedFile] = []
    for source in sources:
        if not source.is_file():
            raise ArchivalError(f"source is missing or not a regular file: {source}")
        expected_sha256, expected_size = digest_and_size(source)
        archived.append(
            _copy_exclusive(
                source,
                archive_root / source.name,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                fsync_file=fsync_file,
                publish=publish,
                fsync_dir=fsync_dir,
            )
        )

    manifest_path = archive_root / MANIFEST_NAME
    payload = json.dumps(
        {
            "schema_version": MANIFEST_SCHEMA,
            "entries": [entry.as_entry() for entry in archived],
        },
        indent=2,
        sort_keys=True,
    ).encode()
    _publish_manifest(
        manifest_path,
        payload,
        fsync_file=fsync_file,
        publish=publish,
        fsync_dir=fsync_dir,
    )
    return manifest_path, tuple(archived)


def _publish_manifest(
    manifest_path: Path,
    payload: bytes,
    *,
    fsync_file: Callable[[int], None],
    publish: Callable[[Path, Path], None],
    fsync_dir: Callable[[Path], None],
) -> None:
    """Write the manifest through the same durability sequence as each file."""
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ArchivalError(f"refusing to clobber an existing manifest: {manifest_path}")
    directory = manifest_path.parent
    temporary = directory / f".{manifest_path.name}.incoming"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        os.write(fd, payload)
        fsync_file(fd)
    except BaseException:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    os.close(fd)
    try:
        publish(temporary, manifest_path)
        fsync_dir(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_manifest(manifest_path: Path) -> tuple[ArchivedFile, ...]:
    document = json.loads(manifest_path.read_text())
    if document.get("schema_version") != MANIFEST_SCHEMA:
        raise ArchivalError(f"unsupported manifest schema in {manifest_path}")
    return tuple(
        ArchivedFile(
            Path(str(entry["source"])),
            Path(str(entry["destination"])),
            int(entry["size_bytes"]),
            str(entry["sha256"]),
        )
        for entry in document["entries"]
    )


def verify_batch(manifest_path: Path) -> tuple[ArchivedFile, ...]:
    """Re-verify every source and archived copy against the manifest.

    Run immediately before destruction. Closes the window between archival and
    destruction: a source that drifted or vanished must not be destroyed.
    """
    entries = load_manifest(manifest_path)
    for entry in entries:
        if not entry.destination.is_file():
            raise ArchivalError(f"archived copy is missing: {entry.destination}")
        archived_sha256, archived_size = digest_and_size(entry.destination)
        if (archived_sha256, archived_size) != (entry.sha256, entry.size_bytes):
            raise ArchivalError(f"archived copy drifted: {entry.destination}")

        if not entry.source.is_file():
            raise ArchivalError(f"source vanished after archival: {entry.source}")
        source_sha256, source_size = digest_and_size(entry.source)
        if (source_sha256, source_size) != (entry.sha256, entry.size_bytes):
            raise ArchivalError(f"source drifted after archival: {entry.source}")
    return entries


def destroy_sources(
    repo_root: Path,
    tracked: Sequence[Path],
    untracked: Sequence[Path],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Restore tracked sources to HEAD and unlink untracked ones.

    Working tree only. Never stages, commits, or pushes.
    """
    if tracked:
        result = run(
            ["git", "restore", "--", *[str(path) for path in tracked]],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ArchivalError(f"git restore failed: {result.stderr.strip()}")
    for path in untracked:
        path.unlink(missing_ok=True)


def execute(
    repo_root: Path,
    archive_root: Path,
    tracked: Sequence[Path],
    untracked: Sequence[Path],
    *,
    destroy: bool = True,
) -> Path:
    """Run the whole transaction. Sources are mutated only at the very end."""
    sources = [*tracked, *untracked]
    manifest_path, _ = archive_batch(sources, archive_root)
    verify_batch(manifest_path)
    if destroy:
        destroy_sources(repo_root, tracked, untracked)
    return manifest_path


TRACKED: Final = ()
UNTRACKED: Final = (
    Path("deep-interview-edge-hub-contract-identity-20260817.md"),
    Path("deep-interview-edge-hub-contract-identity-20260817-addendum.md"),
    Path("nvidia-cutover-handoff-20260817.md"),
)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="archive and verify, but leave every source in place",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    tracked = [repo_root / path for path in TRACKED]
    untracked = [repo_root / path for path in UNTRACKED]
    try:
        manifest_path = execute(
            repo_root,
            args.archive_root,
            tracked,
            untracked,
            destroy=not args.dry_run,
        )
    except ArchivalError as error:
        print(f"archival transaction halted, no source mutated: {error}", file=sys.stderr)
        return 1
    print(f"manifest: {manifest_path}")
    print("dry run: sources left in place" if args.dry_run else "sources restored/removed")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "ArchivalError",
    "ArchivedFile",
    "archive_batch",
    "destroy_sources",
    "digest_and_size",
    "execute",
    "load_manifest",
    "verify_batch",
]
