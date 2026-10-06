"""Snapshot-only rollback predicate over every durable PostgreSQL record class."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Final

import psycopg

from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.mapping import build_mappings, require_identifier
from backend.app.edge_db.migration.reconcile import (
    PASS,
    TableResult,
    authority_state,
    compare_tables,
    delivery_state,
    diagnostics_state,
    ledger_rows,
    seed_only_site,
)
from backend.app.edge_db.migration.snapshot import open_snapshot, snapshot_sha256
from backend.app.edge_db.migration.sqlite_fence import inspect_fence, read_fence_receipt
from backend.app.edge_db.postgres import PostgresDatabase

REPORT_FORMAT: Final = "seeon-edge-pg-rollback/1"
ALLOW: Final = "ALLOW"
DENY: Final = "DENY"


def rollback_check(
    database: PostgresDatabase,
    *,
    schema: str,
    snapshot_path: Path,
    source: Path,
    fence_receipt: Path,
) -> dict[str, object]:
    """ALLOW a snapshot restore only when PostgreSQL holds nothing the snapshot lacks.

    A missing new event is not enough: delivery, execution, audit, status and
    diagnostics history written only to PostgreSQL each deny, as does a live
    authority. The fenced SQLite file must also still hold exactly the bytes
    the fence left, compared without opening it.
    """
    require_identifier(schema, "schema")
    fence = read_fence_receipt(fence_receipt)
    source_sha256 = snapshot_sha256(snapshot_path)
    with closing(open_snapshot(snapshot_path)) as snapshot:

        def inspect(connection: psycopg.Connection) -> tuple[object, ...]:
            mappings = build_mappings(snapshot, connection, schema)
            return (
                compare_tables(snapshot, connection, mappings, schema),
                ledger_rows(connection, schema),
                authority_state(connection, schema),
                delivery_state(connection, schema),
                diagnostics_state(connection, schema),
                seed_only_site(connection, schema),
            )

        tables, ledger, authority, delivery, diagnostics, seed_only = database.read_snapshot(
            inspect
        )
    if snapshot_sha256(snapshot_path) != source_sha256:
        raise MigrationError("snapshot changed during the rollback check")

    reasons: list[str] = []
    if authority["accepting"] or authority["egress_enabled"]:
        reasons.append("authority_not_fenced")
    reasons.extend(
        f"delivery_history:{table}" for table, count in delivery["delivery_rows"].items() if count
    )
    if diagnostics["result"] != PASS:
        reasons.append("diagnostics:schema")
    else:
        # Diagnostics are never imported, so every row is history SQLite never saw.
        reasons.extend(
            f"diagnostics_history:{table}" for table, count in diagnostics["rows"].items() if count
        )
    imported = len(ledger) == 1 and ledger[0][4] is not None
    if len(ledger) != 1:
        reasons.append("ledger:entries")
    elif not imported:
        # A failed or skipped import must have left the target empty, but for the
        # edge_site row that activation seeds.
        reasons.extend(
            f"unimported_rows:{result.name}"
            for result in tables
            if result.candidate.rows and not _activation_seed(result, seed_only)
        )
    else:
        if ledger[0][4] != source_sha256:
            reasons.append("ledger:snapshot_mismatch")
        # The seed is history only where the snapshot had no edge_site row to restore.
        reasons.extend(
            f"target_history:{result.name}"
            for result in tables
            if not result.passed
            and not (result.reference.rows == 0 and _activation_seed(result, seed_only))
        )
    if fence.source_present:
        sqlite, fence_reasons = inspect_fence(source, fence)
        if fence.snapshot_sha256 != source_sha256:
            reasons.append("sqlite:receipt_snapshot_mismatch")
        reasons.extend(fence_reasons)
    else:
        # unfence refuses a receipt with no source (unfence.py:41-44), so there is nothing
        # to restore and PostgreSQL stays authoritative.
        sqlite = {
            "generation": fence.generation,
            "source_present": False,
            "fenced_sha256": fence.fenced_sha256,
        }
        reasons.append("sqlite:source_absent")
    return {
        "format": REPORT_FORMAT,
        "result": DENY if reasons else ALLOW,
        "reasons": reasons,
        "snapshot": {"sha256": source_sha256},
        "sqlite": sqlite,
        "imported": imported,
        "authority": authority,
        "pending": delivery,
        "diagnostics": diagnostics,
        "tables": [result.to_json(("snapshot", "target")) for result in tables],
    }


def _activation_seed(result: TableResult, seed_only: bool) -> bool:
    return result.name == "edge_site" and seed_only


__all__ = ["ALLOW", "DENY", "REPORT_FORMAT", "rollback_check"]
