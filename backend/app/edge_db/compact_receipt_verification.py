"""Parse and verify canonical schema-18 reconciliation receipts."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from backend.app.edge_db.compact_receipts import receipt_lines
from backend.app.edge_db.compact_schema import COMPACT_APPLICATION_TABLES


@dataclass(frozen=True, slots=True)
class ReceiptVerification:
    source: Path
    candidate: Path
    receipt: Path
    expected_rows: int
    inventory_sha256: str
    rebuilt_clip_ids: tuple[str, ...]


class ReceiptSemanticError(ValueError):
    """A parsed receipt has an invalid action/reason/target combination."""


class ReconciliationReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["MAP", "REBUILD", "NONE"]
    inventory_sha256: str
    reason: str | None
    source_pk: list[str]
    source_row_sha256: str
    source_table: str
    target_pks: list[str]

    @model_validator(mode="after")
    def require_semantic_disposition(self) -> ReconciliationReceipt:
        if self.action == "MAP" and (self.reason is not None or not self.target_pks):
            raise ReceiptSemanticError("MAP receipt requires targets and no retirement reason")
        if self.action == "NONE" and (self.reason is None or self.target_pks):
            raise ReceiptSemanticError("NONE receipt requires a reason and no targets")
        if self.action == "REBUILD" and self.reason != "filesystem_manifest_authority":
            raise ReceiptSemanticError("REBUILD receipt requires filesystem authority reason")
        return self


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_candidate_contract(candidate: Path, receipt: Path) -> None:
    """Verify integrity, foreign keys, exact tables, receipt hash, and checkpoint."""
    with sqlite3.connect(candidate, isolation_level=None) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_INTEGRITY")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_FOREIGN_KEY")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM pragma_table_list() WHERE schema='main' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        if tables != COMPACT_APPLICATION_TABLES:
            raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_TABLE_MANIFEST")
        recorded = connection.execute(
            "SELECT reconciliation_sha256 FROM schema_migrations WHERE version=18"
        ).fetchone()
        if recorded != (_file_sha256(receipt),):
            raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_RECEIPT_HASH")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise sqlite3.DatabaseError("EDGE_DB_CUTOVER_CANDIDATE_CHECKPOINT")


def _target_rows(connection: sqlite3.Connection) -> set[str]:
    specifications = (
        ("schema_migrations", ("version",)),
        ("credentials", ("id",)),
        ("edge_site", ("id",)),
        ("locations", ("location_id", "kind")),
        ("cameras", ("camera_id",)),
        ("policies", ("policy_id",)),
        ("clips", ("clip_id",)),
        ("incidents", ("incident_id",)),
        ("artifacts", ("incident_id", "kind")),
        ("audit_events", ("audit_id",)),
    )
    targets: set[str] = set()
    for table, columns in specifications:
        selected = ",".join(columns)
        for row in connection.execute(f"SELECT {selected} FROM {table}"):
            key = ",".join(f"{column}={value}" for column, value in zip(columns, row, strict=True))
            targets.add(f"{table}:{key}")
    return targets


def verify_receipts(verification: ReceiptVerification) -> None:
    """Prove source rows, declared fan-out, and candidate rows in both directions."""
    count = 0
    declared: set[str] = set()
    expected = iter(
        receipt_lines(
            verification.source,
            verification.inventory_sha256,
            verification.rebuilt_clip_ids,
        )
    )
    connection = sqlite3.connect(f"file:{verification.candidate}?mode=ro", uri=True)
    try:
        with verification.receipt.open("rb") as stream:
            for line in stream:
                if next(expected, None) != line:
                    raise sqlite3.DatabaseError("receipt differs from source row inventory")
                record = ReconciliationReceipt.model_validate_json(line)
                if record.inventory_sha256 != verification.inventory_sha256:
                    raise sqlite3.DatabaseError("receipt inventory hash differs")
                count += 1
                for target in record.target_pks:
                    declared.add(target)
            if next(expected, None) is not None:
                raise sqlite3.DatabaseError("receipt omits source rows")
        actual = _target_rows(connection)
    finally:
        connection.close()
    if count != verification.expected_rows:
        raise sqlite3.DatabaseError("reconciliation receipt row count differs from source")
    if declared != actual:
        missing = sorted(actual - declared)
        extra = sorted(declared - actual)
        raise sqlite3.DatabaseError(
            f"bidirectional receipt mismatch missing={missing} extra={extra}"
        )


__all__ = ["ReceiptVerification", "verify_candidate_contract", "verify_receipts"]
