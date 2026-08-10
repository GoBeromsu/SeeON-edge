"""Descriptor-pinned reads for untrusted clip-store media."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True, slots=True)
class OpenedRegularFile:
    handle: BinaryIO
    path: Path
    size_bytes: int


def open_contained_regular_file(root: Path, path: Path) -> OpenedRegularFile:
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=False)
        relative = resolved_path.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(str(path)) from exc
    if not relative.parts:
        raise FileNotFoundError(str(path))

    directory_fd: int | None = None
    media_fd: int | None = None
    try:
        directory_fd = os.open(
            resolved_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        for component in relative.parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        media_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        opened_stat = os.fstat(media_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise FileNotFoundError(str(path))
        handle = os.fdopen(media_fd, "rb", closefd=True)
        media_fd = None
        return OpenedRegularFile(handle, resolved_path, opened_stat.st_size)
    except OSError as exc:
        raise FileNotFoundError(str(path)) from exc
    finally:
        if media_fd is not None:
            os.close(media_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def read_bounded_regular_file(root: Path, path: Path, max_bytes: int) -> bytes:
    opened = open_contained_regular_file(root, path)
    with opened.handle as source:
        if opened.size_bytes > max_bytes:
            raise FileNotFoundError(str(path))
        content = source.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise FileNotFoundError(str(path))
    return content


__all__ = ["OpenedRegularFile", "open_contained_regular_file", "read_bounded_regular_file"]
