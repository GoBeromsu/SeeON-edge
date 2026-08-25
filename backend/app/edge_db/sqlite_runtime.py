"""Packaged SQLite version gate for schema-18 cutover."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MINIMUM_SQLITE_VERSION: Final = (3, 51, 3)
SqliteVersion = tuple[int, int, int]


class SqliteRuntimeError(RuntimeError):
    """Base failure for the packaged SQLite runtime gate."""


@dataclass(frozen=True, slots=True)
class SqliteVersionTooOldError(SqliteRuntimeError):
    found: SqliteVersion
    minimum: SqliteVersion

    def __str__(self) -> str:
        found = ".".join(str(part) for part in self.found)
        minimum = ".".join(str(part) for part in self.minimum)
        return f"sqlite {found} is below required {minimum}"


@dataclass(frozen=True, slots=True)
class MalformedSqliteVersionError(SqliteRuntimeError):
    value: str

    def __str__(self) -> str:
        return f"sqlite version {self.value!r} is malformed"


def parse_sqlite_version(value: str) -> SqliteVersion:
    parts = value.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise MalformedSqliteVersionError(value)
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def require_supported_sqlite(version: SqliteVersion) -> None:
    if version < MINIMUM_SQLITE_VERSION:
        raise SqliteVersionTooOldError(found=version, minimum=MINIMUM_SQLITE_VERSION)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refuse schema-18 cutover when packaged SQLite is too old"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sqlite-version")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = Path(str(args.source))
    candidate = Path(str(args.candidate))
    requested = args.sqlite_version
    try:
        version = parse_sqlite_version(
            str(sqlite3.sqlite_version) if requested is None else str(requested)
        )
        require_supported_sqlite(version)
    except SqliteRuntimeError as error:
        print(f"EDGE_DB_SQLITE_RUNTIME_FAILED: {error}", file=sys.stderr)
        return 1
    print(f"EDGE_DB_SQLITE_RUNTIME_OK source={source} candidate={candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MINIMUM_SQLITE_VERSION",
    "MalformedSqliteVersionError",
    "SqliteRuntimeError",
    "SqliteVersionTooOldError",
    "main",
    "parse_sqlite_version",
    "require_supported_sqlite",
]
