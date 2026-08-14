"""Strict source/backup identities and prebackup receipt parsing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_backup_fields import (
    boolean as _boolean,
)
from worker.pipeline.output.evidence.clip_consistency_backup_fields import (
    integer as _integer,
)
from worker.pipeline.output.evidence.clip_consistency_backup_fields import (
    string as _string,
)
from worker.pipeline.output.evidence.clip_consistency_database import validate_database
from worker.pipeline.output.evidence.clip_consistency_io import (
    read_strict_json,
    reject_lexical_parent_components,
    sha256_regular,
    validate_regular,
    validate_under_root,
)
from worker.pipeline.output.evidence.clip_consistency_schema import (
    is_supported_schema_version,
)
from worker.pipeline.output.evidence.clip_consistency_types import (
    BackupReceipt,
    ClipConsistencyError,
)

RECEIPT_VERSION = 3
_RECEIPT_KEYS = frozenset(BackupReceipt.__dataclass_fields__)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    source_path: str
    source_mode: int
    source_size: int
    source_file_sha256: str
    source_wal_path: str
    source_wal_present: bool
    source_wal_size: int
    source_wal_sha256: str
    source_state_sha256: str

    @property
    def source_identity_sha256(self) -> str:
        return _source_identity_sha256(self)

    def matches_reusable_source(self, receipt: BackupReceipt) -> bool:
        return self.source_identity_sha256 == receipt.source_identity_sha256


def source_identity(
    source: Path,
    snapshot: sqlite3.Connection,
    expected_uid: int,
) -> SourceIdentity:
    info = validate_regular(
        source,
        expected_uid=expected_uid,
        exact_mode=None,
        label="state database",
    )
    canonical_source = source.resolve(strict=True)
    wal = Path(f"{canonical_source}-wal")
    if wal.exists():
        wal_info = validate_regular(
            wal,
            expected_uid=expected_uid,
            exact_mode=None,
            label="state database WAL",
        )
        wal_present, wal_size, wal_sha = True, wal_info.st_size, sha256_regular(wal)
    else:
        wal_present, wal_size, wal_sha = False, 0, _EMPTY_SHA256
    return SourceIdentity(
        source_path=str(canonical_source),
        source_mode=stat.S_IMODE(info.st_mode),
        source_size=info.st_size,
        source_file_sha256=sha256_regular(source),
        source_wal_path=str(wal.absolute()),
        source_wal_present=wal_present,
        source_wal_size=wal_size,
        source_wal_sha256=wal_sha,
        source_state_sha256=connection_state_sha256(snapshot),
    )


def parse_backup_receipt(
    path: Path,
    *,
    maintenance_root: Path,
    expected_uid: int,
) -> BackupReceipt:
    validate_under_root(path, maintenance_root, allow_missing_leaf=False)
    payload = read_strict_json(
        path,
        expected_uid=expected_uid,
        exact_mode=0o600,
        error_code="backup_receipt_invalid",
    )
    if frozenset(payload) != _RECEIPT_KEYS:
        raise ClipConsistencyError("backup_receipt_invalid", "receipt key set differs")
    receipt = BackupReceipt(
        format_version=_integer(payload, "format_version"),
        schema_version=_integer(payload, "schema_version"),
        owner_uid=_integer(payload, "owner_uid"),
        source_path=_string(payload, "source_path"),
        source_mode=_integer(payload, "source_mode"),
        source_size=_integer(payload, "source_size"),
        source_file_sha256=_string(payload, "source_file_sha256"),
        source_wal_path=_string(payload, "source_wal_path"),
        source_wal_present=_boolean(payload, "source_wal_present"),
        source_wal_size=_integer(payload, "source_wal_size"),
        source_wal_sha256=_string(payload, "source_wal_sha256"),
        source_state_sha256=_string(payload, "source_state_sha256"),
        source_identity_sha256=_string(payload, "source_identity_sha256"),
        backup_path=_string(payload, "backup_path"),
        backup_mode=_integer(payload, "backup_mode"),
        backup_size=_integer(payload, "backup_size"),
        backup_file_sha256=_string(payload, "backup_file_sha256"),
        backup_state_sha256=_string(payload, "backup_state_sha256"),
        receipt_path=_string(payload, "receipt_path"),
    )
    if (
        receipt.format_version != RECEIPT_VERSION
        or not is_supported_schema_version(receipt.schema_version)
        or receipt.owner_uid != expected_uid
        or receipt.receipt_path != str(path.resolve(strict=True))
        or not _valid_receipt_facts(receipt)
    ):
        raise ClipConsistencyError("backup_receipt_invalid", "receipt identity differs")
    return receipt


def verify_backup(receipt: BackupReceipt, expected_uid: int) -> None:
    backup = Path(receipt.backup_path)
    info = validate_regular(
        backup,
        expected_uid=expected_uid,
        exact_mode=0o600,
        label="prebackup",
    )
    if (
        receipt.backup_mode != 0o600
        or receipt.backup_size != info.st_size
        or receipt.backup_file_sha256 != sha256_regular(backup)
        or receipt.backup_state_sha256 != database_state_sha256(backup)
    ):
        raise ClipConsistencyError("backup_verification_failed", "backup facts differ")


def database_state_sha256(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        validate_database(connection, now=0, check_leases=False)
        return connection_state_sha256(connection)
    finally:
        connection.close()


def connection_state_sha256(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for statement in connection.iterdump():
        digest.update(statement.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _valid_receipt_facts(receipt: BackupReceipt) -> bool:
    source = Path(receipt.source_path)
    wal = Path(receipt.source_wal_path)
    backup = Path(receipt.backup_path)
    authority_paths = (source, wal, backup, Path(receipt.receipt_path))
    hashes = (
        receipt.source_file_sha256,
        receipt.source_wal_sha256,
        receipt.source_state_sha256,
        receipt.source_identity_sha256,
        receipt.backup_file_sha256,
        receipt.backup_state_sha256,
    )
    try:
        for path in authority_paths:
            reject_lexical_parent_components(path)
    except ClipConsistencyError:
        return False
    wal_facts = (
        receipt.source_wal_size >= 0
        and (receipt.source_wal_present or receipt.source_wal_size == 0)
        and (
            receipt.source_wal_present
            or receipt.source_wal_sha256 == _EMPTY_SHA256
        )
    )
    return (
        all(path.is_absolute() for path in authority_paths)
        and wal == Path(f"{source}-wal")
        and source != backup
        and receipt.source_size >= 0
        and receipt.backup_size >= 0
        and receipt.backup_mode == 0o600
        and all(_is_sha256(value) for value in hashes)
        and receipt.source_state_sha256 == receipt.backup_state_sha256
        and receipt.source_identity_sha256 == _source_identity_sha256(receipt)
        and wal_facts
    )


def _source_identity_sha256(identity: SourceIdentity | BackupReceipt) -> str:
    payload = {
        field: getattr(identity, field)
        for field in SourceIdentity.__dataclass_fields__
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "RECEIPT_VERSION",
    "SourceIdentity",
    "database_state_sha256",
    "parse_backup_receipt",
    "source_identity",
    "verify_backup",
]
