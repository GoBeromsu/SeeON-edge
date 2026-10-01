"""Explicitly prepare ignored test inputs from a caller-supplied pinned archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import stat
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

ARCHIVE_SHA256 = "7b4320238bc5c00ae5e08910fc76979aa9b69b928a4531b3c639a4c93f089b2a"


class InputPreparationError(ValueError):
    """An archive or existing cache cannot be admitted safely."""


def _input_path(member: tarfile.TarInfo) -> PurePosixPath:
    name = member.name
    if not member.isreg():
        raise InputPreparationError(f"Archive member must be a regular file: {name!r}")
    if (
        not name
        or "\\" in name
        or ":" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise InputPreparationError(f"Unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    allowed = (
        path == PurePosixPath("tests/fixtures/synthetic-scene-v1.json")
        or path == PurePosixPath("worker/runtime/rust/tests/fixtures/gpu/manifest.json")
        or (
            path.parts[:3] == ("tests", "fixtures", "worker-wire")
            and len(path.parts) > 3
            and path.suffix in {".json", ".sha256"}
        )
        or (
            path.parts[:4] == ("worker", "rust", "tests", "fixtures")
            and len(path.parts) > 4
            and path.suffix == ".json"
        )
    )
    if not allowed:
        raise InputPreparationError(f"Archive path is not an allowed test input: {name!r}")
    return path


def _read_inputs(archive: Path, expected_digest: str) -> dict[PurePosixPath, bytes]:
    try:
        raw = archive.read_bytes()
    except FileNotFoundError as exc:
        raise InputPreparationError(
            f"Missing input archive: {archive}; supply an existing file with --archive"
        ) from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_digest:
        raise InputPreparationError(
            f"Archive SHA256 mismatch: expected {expected_digest}, got {digest}"
        )
    inputs: dict[PurePosixPath, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as bundle:
            for member in bundle:
                path = _input_path(member)
                if path in inputs:
                    raise InputPreparationError(f"Duplicate archive path: {path}")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise InputPreparationError(f"Missing archive payload: {path}")
                with stream:
                    inputs[path] = stream.read()
    except (tarfile.TarError, EOFError) as exc:
        raise InputPreparationError(f"Invalid input archive: {archive}: {exc}") from exc
    if not inputs:
        raise InputPreparationError(f"Input archive contains no test inputs: {archive}")
    for path in inputs:
        if any(parent in inputs for parent in path.parents):
            raise InputPreparationError(f"Conflicting archive paths: {path}")
    return inputs


@contextmanager
def _directory(path: Path, *, create: bool) -> Iterator[int | None]:
    """Walk absolute paths using directory descriptors; never follow symlinks."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    yield None
                    return
                os.mkdir(part, dir_fd=descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _verify_existing(directory: int, name: str, payload: bytes, target: Path) -> bool:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return False
    with os.fdopen(descriptor, "rb") as existing:
        metadata = os.fstat(existing.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise InputPreparationError(f"Cache path must be a single-link regular file: {target}")
        if existing.read() != payload:
            raise InputPreparationError(f"Conflicting existing cache data: {target}")
    return True


def restore_test_inputs(archive: Path, destination: Path, expected_digest: str) -> int:
    """Restore or verify inputs, returning their count.

    All archive members and existing destinations are validated before mutation.
    Matching files are left untouched; differing files and links are refused.
    This Linux/Unix preparation command holds directory descriptors while writing
    and creates files exclusively, rather than extracting archive paths directly.
    """
    try:
        inputs = _read_inputs(Path(archive), expected_digest)
        root = Path(destination).absolute()
        if ".." in root.parts:
            raise InputPreparationError(f"Unsafe destination path: {destination}")
        for path, payload in inputs.items():
            target = root / path
            with _directory(target.parent, create=False) as directory:
                if directory is not None:
                    _verify_existing(directory, target.name, payload, target)
        for path, payload in inputs.items():
            target = root / path
            with _directory(target.parent, create=True) as directory:
                if directory is None:
                    raise InputPreparationError(f"Missing cache directory: {target.parent}")
                if not _verify_existing(directory, target.name, payload, target):
                    descriptor = os.open(
                        target.name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o644,
                        dir_fd=directory,
                    )
                    with os.fdopen(descriptor, "wb") as output:
                        output.write(payload)
    except OSError as exc:
        raise InputPreparationError(
            f"Cannot prepare test inputs; unsafe or inaccessible archive/cache path: {exc}"
        ) from exc
    return len(inputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Local pinned input archive")
    args = parser.parse_args()
    try:
        count = restore_test_inputs(
            args.archive, Path(__file__).resolve().parents[1], ARCHIVE_SHA256
        )
    except InputPreparationError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Restored or verified {count} test inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
