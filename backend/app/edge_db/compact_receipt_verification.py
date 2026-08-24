"""Parse and verify canonical schema-18 reconciliation receipts."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.app.edge_db.compact_schema import COMPACT_APPLICATION_TABLES


class ReconciliationReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["MAP", "REBUILD", "NONE"]
    inventory_sha256: str
    source_pk: list[str]
    source_row_sha256: str
    source_table: str
    target_pks: list[str]


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


def verify_receipts(candidate: Path, receipt: Path, expected_rows: int) -> None:
    """Parse every receipt and prove each declared target primary key exists."""
    count = 0
    connection = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)
    try:
        with receipt.open("rb") as stream:
            for line in stream:
                record = ReconciliationReceipt.model_validate_json(line)
                count += 1
                for target in record.target_pks:
                    table, encoded_key = target.split(":", 1)
                    predicates = encoded_key.split(",")
                    columns, values = zip(
                        *(predicate.split("=", 1) for predicate in predicates), strict=True
                    )
                    where = " AND ".join(f'"{column}"=?' for column in columns)
                    found = connection.execute(
                        f'SELECT 1 FROM "{table}" WHERE {where}', values
                    ).fetchone()
                    if found != (1,):
                        raise sqlite3.DatabaseError(f"missing receipt target {target}")
    finally:
        connection.close()
    if count != expected_rows:
        raise sqlite3.DatabaseError("reconciliation receipt row count differs from source")


__all__ = ["verify_candidate_contract", "verify_receipts"]
