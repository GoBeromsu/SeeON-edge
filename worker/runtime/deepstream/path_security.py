"""Owner-only validation for dark-runner filesystem state paths."""

from __future__ import annotations

import os
import stat
from pathlib import Path


class PrivatePathError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code, detail)
        self.code: str = code
        self.detail: str = detail


def validate_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid():
        raise PrivatePathError("runtime_directory_owner", path.name)
    if stat.S_IMODE(details.st_mode) != 0o700:
        raise PrivatePathError("runtime_directory_mode", oct(stat.S_IMODE(details.st_mode)))


__all__ = ["PrivatePathError", "validate_private_directory"]
