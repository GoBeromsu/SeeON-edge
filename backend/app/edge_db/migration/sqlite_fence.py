"""Stamp the legacy SQLite file so an old runtime refuses it once PostgreSQL takes over."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from backend.app.edge_db.migration.authority_file import discard, fsync_directory
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.report import write_report
from backend.app.edge_db.migration.snapshot import (
    copy_database,
    create_private_file,
    discard_database,
    exclusive_deployment_lock,
    sidecar_paths,
    snapshot_sha256,
    temporary_path,
)

FENCE_RECEIPT_FORMAT: Final = "seeon-edge-sqlite-fence/1"
# Far above every schema an old runtime knows, so it refuses the file as newer;
# the remainder names the PostgreSQL generation that took over.
SENTINEL_BASE: Final = 1_000_000
_MAX_USER_VERSION: Final = 2**31 - 1
_USER_VERSION_OFFSET: Final = 60
_HEADER_BYTES: Final = 100
_CHUNK: Final = 1 << 20
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_RECEIPT_KEYS: Final = frozenset(
    {
        "format",
        "generation",
        "user_version",
        "source_present",
        "snapshot_sha256",
        "pre_fence_sha256",
        "fenced_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class FenceReceipt:
    """The bytes the fence proved: the snapshot, the file it stamped, and the result."""

    generation: int
    source_present: bool
    snapshot_sha256: str | None
    pre_fence_sha256: str | None
    fenced_sha256: str

    def __post_init__(self) -> None:
        if not _valid_generation(self.generation):
            raise ValueError("fence generation is out of range")
        if type(self.source_present) is not bool or not _is_sha256(self.fenced_sha256):
            raise ValueError("fence receipt is malformed")
        digests = (self.snapshot_sha256, self.pre_fence_sha256)
        if self.source_present:
            if not all(_is_sha256(digest) for digest in digests):
                raise ValueError("fence receipt is malformed")
        elif digests != (None, None):
            raise ValueError("fence receipt is malformed")

    @property
    def user_version(self) -> int:
        return SENTINEL_BASE + self.generation

    def to_json(self) -> dict[str, object]:
        return {
            "format": FENCE_RECEIPT_FORMAT,
            "generation": self.generation,
            "user_version": self.user_version,
            "source_present": self.source_present,
            "snapshot_sha256": self.snapshot_sha256,
            "pre_fence_sha256": self.pre_fence_sha256,
            "fenced_sha256": self.fenced_sha256,
        }


def sentinel_user_version(generation: int) -> int:
    if not _valid_generation(generation):
        raise MigrationError("fence generation is out of range")
    return SENTINEL_BASE + generation


def read_fence_receipt(path: Path) -> FenceReceipt:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise MigrationError("fence receipt is unreadable") from error
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if (
        not isinstance(payload, dict)
        or set(payload) != _RECEIPT_KEYS
        or payload["format"] != FENCE_RECEIPT_FORMAT
    ):
        raise MigrationError("fence receipt is malformed")
    try:
        receipt = FenceReceipt(
            generation=payload["generation"],
            source_present=payload["source_present"],
            snapshot_sha256=payload["snapshot_sha256"],
            pre_fence_sha256=payload["pre_fence_sha256"],
            fenced_sha256=payload["fenced_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise MigrationError("fence receipt is malformed") from error
    if type(payload["user_version"]) is not int or payload["user_version"] != receipt.user_version:
        raise MigrationError("fence receipt is malformed")
    return receipt


def preserved_path(receipt_path: Path) -> Path:
    """The pre-fence byte copy kept beside the receipt; nothing deletes it."""
    return receipt_path.with_name(f"{receipt_path.stem}.pre-fence.sqlite3")


def fence_sqlite(
    source: Path, *, snapshot: Path | None, generation: int, receipt: Path
) -> FenceReceipt:
    """Stamp the source with the sentinel user_version, or create a stamped empty file.

    A present source must still export to the snapshot's bytes. The receipt is
    durable before the live stamp, so a crash leaves either the untouched file or
    one the receipt names; rerunning the fence finishes either.
    """
    user_version = sentinel_user_version(generation)
    if not source.parent.is_dir() or not receipt.parent.is_dir():
        raise MigrationError("fence directory does not exist")
    with exclusive_deployment_lock(source.parent):
        recorded = read_fence_receipt(receipt) if _exists(receipt) else None
        if _exists(source):
            fence = _fence_present(source, snapshot, generation, user_version, receipt, recorded)
        else:
            fence = _fence_absent(source, snapshot, generation, user_version, receipt, recorded)
        _require_settled(source, fence)
    return fence


def inspect_fence(source: Path, fence: FenceReceipt) -> tuple[dict[str, object], list[str]]:
    """Compare the live file with the receipt by its bytes; SQLite never opens it."""
    wal, shm, journal = sidecar_paths(source)
    reasons: list[str] = []
    live: str | None = None
    user_version: int | None = None
    if not _exists(source):
        reasons.append("sqlite:source_missing")
    elif source.is_symlink() or not source.is_file():
        reasons.append("sqlite:source_not_regular")
    else:
        live = snapshot_sha256(source)
        user_version = _header_user_version(source)
        if live != fence.fenced_sha256:
            reasons.append("sqlite:live_changed")
    wal_bytes, shm_bytes = _size(wal), _size(shm)
    if wal_bytes:
        reasons.append("sqlite:wal_content")
    if shm_bytes:
        reasons.append("sqlite:shm_content")
    has_journal = _exists(journal)
    if has_journal:
        reasons.append("sqlite:journal")
    section: dict[str, object] = {
        "generation": fence.generation,
        "source_present": fence.source_present,
        "fenced_sha256": fence.fenced_sha256,
        "live_sha256": live,
        "user_version": user_version,
        "wal_bytes": wal_bytes,
        "shm_bytes": shm_bytes,
        "journal": has_journal,
    }
    return section, reasons


def descriptor_blocks(descriptor: int) -> Iterator[bytes]:
    """Read a file by offset, so the descriptor stays open for as long as it is needed."""
    offset = 0
    while block := os.pread(descriptor, _CHUNK, offset):
        offset += len(block)
        yield block


def _fence_present(
    source: Path,
    snapshot: Path | None,
    generation: int,
    user_version: int,
    receipt: Path,
    recorded: FenceReceipt | None,
) -> FenceReceipt:
    if source.is_symlink() or not source.is_file():
        raise MigrationError("source database is not a regular file")
    if _exists(sidecar_paths(source)[2]):
        raise MigrationError("source database has a rollback journal")
    # POSIX drops every lock this process holds on a file when any descriptor on it
    # closes, so the source is read only through this one until SQLite lets go.
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with closing(_open_exclusive(source)) as connection:
            live = _descriptor_sha256(descriptor)
            if recorded is not None and recorded.fenced_sha256 == live:
                # A rerun after the stamp committed: the checkpoint above folded it in.
                _require_same_fence(recorded, generation, snapshot)
                discard(sidecar_paths(source)[1])
                return recorded
            if snapshot is None:
                raise MigrationError("a present SQLite source needs its snapshot")
            expected = snapshot_sha256(snapshot)
            _require_snapshot_bytes(connection, snapshot, expected)
            # The export left a -shm for its own read; nothing else can hold it now.
            discard(sidecar_paths(source)[1])
            preserved = preserved_path(receipt)
            _preserve(descriptor, preserved, live)
            fence = FenceReceipt(
                generation=generation,
                source_present=True,
                snapshot_sha256=expected,
                pre_fence_sha256=live,
                fenced_sha256=_stamped_sha256(preserved, user_version),
            )
            if recorded is not None and recorded != fence:
                raise MigrationError("fence receipt already records a different fence")
            write_report(receipt, fence.to_json())
            _stamp(connection, user_version)
            return fence
    finally:
        os.close(descriptor)


def _fence_absent(
    source: Path,
    snapshot: Path | None,
    generation: int,
    user_version: int,
    receipt: Path,
    recorded: FenceReceipt | None,
) -> FenceReceipt:
    if snapshot is not None:
        raise MigrationError("SQLite source is missing after its snapshot")
    if any(_exists(path) for path in sidecar_paths(source)):
        raise MigrationError("SQLite side files exist without their database")
    temp = temporary_path(source)
    try:
        create_private_file(temp)
        _stamp_file(temp, user_version)
        fence = FenceReceipt(
            generation=generation,
            source_present=False,
            snapshot_sha256=None,
            pre_fence_sha256=None,
            fenced_sha256=snapshot_sha256(temp),
        )
        if recorded is not None and recorded != fence:
            raise MigrationError("fence receipt already records a different fence")
        write_report(receipt, fence.to_json())
        try:
            os.link(temp, source)
        except FileExistsError as error:
            raise MigrationError("SQLite source appeared while it was fenced") from error
        temp.unlink()
        fsync_directory(source.parent)
    finally:
        discard_database(temp)
    return fence


def _open_exclusive(source: Path) -> sqlite3.Connection:
    """Hold the only connection: no reader or writer can open the file until it closes."""
    connection = sqlite3.connect(
        source.resolve().as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=0
    )
    try:
        _hold_exclusive(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def _hold_exclusive(connection: sqlite3.Connection) -> None:
    # Exclusive locking keeps the file lock across transactions until close.
    connection.execute("PRAGMA locking_mode = EXCLUSIVE").fetchone()
    try:
        connection.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as error:
        raise MigrationError("source database is in use") from error
    if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
        raise MigrationError("source database is not in WAL mode")
    connection.execute("COMMIT")
    _checkpoint(connection)


def _require_same_fence(recorded: FenceReceipt, generation: int, snapshot: Path | None) -> None:
    expected = None if snapshot is None else snapshot_sha256(snapshot)
    if (recorded.generation, recorded.snapshot_sha256) != (generation, expected):
        raise MigrationError("fence receipt already records a different fence")


def _require_snapshot_bytes(connection: sqlite3.Connection, snapshot: Path, expected: str) -> None:
    # The export is not the live file's bytes, but exporting again the same way is.
    copy = temporary_path(snapshot)
    try:
        create_private_file(copy)
        copy_database(connection, copy)
        if snapshot_sha256(copy) != expected:
            raise MigrationError("source changed after the snapshot")
    finally:
        discard_database(copy)


def _preserve(descriptor: int, preserved: Path, live: str) -> None:
    if _exists(preserved):
        # Left by an interrupted fence before its receipt was written.
        if preserved.is_symlink() or not preserved.is_file() or snapshot_sha256(preserved) != live:
            raise MigrationError("preserved source copy does not match the live source")
        return
    temp = temporary_path(preserved)
    try:
        _write_copy(descriptor, temp)
        if snapshot_sha256(temp) != live:
            raise MigrationError("preserved source copy does not match the live source")
        try:
            os.link(temp, preserved)
        except FileExistsError as error:
            raise MigrationError("preserved source copy already exists") from error
        temp.unlink()
        fsync_directory(preserved.parent)
    finally:
        discard(temp)


def _stamped_sha256(preserved: Path, user_version: int) -> str:
    """Stamp a scratch copy first: the stamp is deterministic, so the receipt names the result."""
    scratch = temporary_path(preserved)
    try:
        descriptor = os.open(preserved, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            _write_copy(descriptor, scratch)
        finally:
            os.close(descriptor)
        _stamp_file(scratch, user_version)
        return snapshot_sha256(scratch)
    finally:
        discard_database(scratch)


def _write_copy(descriptor: int, path: Path) -> None:
    create_private_file(path)
    with path.open("r+b") as handle:
        for block in descriptor_blocks(descriptor):
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())


def _stamp_file(path: Path, user_version: int) -> None:
    # WAL like the source, so an old runtime's WAL switch changes nothing before it refuses.
    with closing(sqlite3.connect(path, isolation_level=None, timeout=0)) as connection:
        if connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
            raise MigrationError("fence file is not in WAL mode")
        _stamp(connection, user_version)


def _stamp(connection: sqlite3.Connection, user_version: int) -> None:
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(f"PRAGMA user_version = {user_version:d}")
    connection.execute("COMMIT")
    _checkpoint(connection)


def _checkpoint(connection: sqlite3.Connection) -> None:
    # Fold every committed frame into the main file, whose bytes alone are compared.
    busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if busy:
        raise MigrationError("SQLite checkpoint was blocked")


def _require_settled(source: Path, fence: FenceReceipt) -> None:
    _, reasons = inspect_fence(source, fence)
    if reasons:
        raise MigrationError("fenced source does not match the fence receipt")


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    for block in descriptor_blocks(descriptor):
        digest.update(block)
    return digest.hexdigest()


def _header_user_version(path: Path) -> int | None:
    with path.open("rb") as handle:
        header = handle.read(_HEADER_BYTES)
    if len(header) < _HEADER_BYTES:
        return None
    return int.from_bytes(header[_USER_VERSION_OFFSET : _USER_VERSION_OFFSET + 4], "big")


def _size(path: Path) -> int:
    try:
        return path.lstat().st_size
    except FileNotFoundError:
        return 0


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _valid_generation(generation: object) -> bool:
    return type(generation) is int and 1 <= generation <= _MAX_USER_VERSION - SENTINEL_BASE


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "FENCE_RECEIPT_FORMAT",
    "SENTINEL_BASE",
    "FenceReceipt",
    "descriptor_blocks",
    "fence_sqlite",
    "inspect_fence",
    "preserved_path",
    "read_fence_receipt",
    "sentinel_user_version",
]
