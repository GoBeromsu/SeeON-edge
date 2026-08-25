"""Packaged SQLite version gate for schema-18 cutover."""

from __future__ import annotations

import sqlite3
from typing import Final

MINIMUM_SQLITE_VERSION: Final = (3, 51, 3)
SqliteVersion = tuple[int, int, int]


class SqliteRuntimeError(RuntimeError):
    """Base failure for the packaged SQLite runtime gate."""


class SqliteVersionTooOldError(SqliteRuntimeError):
    def __init__(self, found: SqliteVersion, minimum: SqliteVersion) -> None:
        self.found = found
        self.minimum = minimum
        rendered_found = ".".join(str(part) for part in found)
        rendered_minimum = ".".join(str(part) for part in minimum)
        super().__init__(f"sqlite {rendered_found} is below required {rendered_minimum}")


class MalformedSqliteVersionError(SqliteRuntimeError):
    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__(f"sqlite version {value!r} is malformed")


def parse_sqlite_version(value: str) -> SqliteVersion:
    parts = value.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise MalformedSqliteVersionError(value)
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def require_supported_sqlite(version: SqliteVersion) -> None:
    if version < MINIMUM_SQLITE_VERSION:
        raise SqliteVersionTooOldError(found=version, minimum=MINIMUM_SQLITE_VERSION)


def packaged_sqlite_version() -> SqliteVersion:
    return sqlite3.sqlite_version_info[:3]


__all__ = [
    "MINIMUM_SQLITE_VERSION",
    "MalformedSqliteVersionError",
    "SqliteRuntimeError",
    "SqliteVersion",
    "SqliteVersionTooOldError",
    "packaged_sqlite_version",
    "parse_sqlite_version",
    "require_supported_sqlite",
]
