"""Per-table reconciliation of a SQLite snapshot against the PostgreSQL product schema."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import psycopg
from psycopg import sql

from backend.app.edge_db.migration.compatibility import verify_runtime_schema
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.mapping import (
    DELIVERY_TABLES,
    DIAGNOSTICS_TABLES,
    DIAGNOSTICS_TARGET_TABLES,
    TableMapping,
    build_mappings,
    decision_table,
    diagnostics_schema_name,
    postgres_table_names,
    require_identifier,
    sqlite_table_names,
)
from backend.app.edge_db.migration.provision import (
    DIAGNOSTICS_SCHEMA_NAME,
    DIAGNOSTICS_SCHEMA_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    diagnostics_checksum,
    schema_checksum,
)
from backend.app.edge_db.migration.report import write_report
from backend.app.edge_db.migration.snapshot import (
    open_fenced_source,
    open_snapshot,
    snapshot_sha256,
)
from backend.app.edge_db.migration.sqlite_fence import inspect_fence, read_fence_receipt
from backend.app.edge_db.migration.worker_state import queue_digest
from backend.app.edge_db.postgres import PostgresDatabase

REPORT_FORMAT: Final = "seeon-edge-pg-reconcile/1"
PASS: Final = "PASS"
FAIL: Final = "FAIL"
_BATCH: Final = 2000
# Status values are closed enums; anything else is reported by digest only.
_STATUS_LABEL: Final = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def canonical_value(value: object) -> object:
    """One JSON form per stored value, identical for SQLite and PostgreSQL drivers."""
    if value is None or type(value) is str or type(value) is int:
        return value
    if type(value) is float:
        return {"f": value.hex()}
    if isinstance(value, bytes | memoryview):
        return {"b": bytes(value).hex()}
    raise MigrationError(f"unsupported stored value type {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def row_sha256(row: Iterable[object]) -> str:
    encoded = canonical_json([canonical_value(value) for value in row]).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sqlite_rows(
    connection: sqlite3.Connection, mapping: TableMapping, *, load_order: bool = False
) -> Iterator[tuple[object, ...]]:
    """Stream mapped columns in byte-wise key order (or foreign-key load order)."""
    columns = ", ".join(_quote(name) for name in mapping.column_names)
    order = [f"{_quote(name)} COLLATE BINARY" for name in mapping.primary_key]
    if load_order and mapping.spec.load_priority is not None:
        order.insert(0, mapping.spec.load_priority)
    # Identifiers come from the validated catalog mapping, never from input.
    cursor = connection.execute(
        f"SELECT {columns} FROM {_quote(mapping.name)} ORDER BY {', '.join(order)}"
    )
    try:
        while batch := cursor.fetchmany(_BATCH):
            yield from batch
    finally:
        cursor.close()


def postgres_rows(
    connection: psycopg.Connection, mapping: TableMapping, schema: str
) -> Iterator[tuple[object, ...]]:
    """Stream mapped columns through a server-side cursor in byte-wise key order."""
    order = [
        sql.SQL('{} COLLATE "C"').format(sql.Identifier(name))
        if mapping.target_type(name) == "text"
        else sql.Identifier(name)
        for name in mapping.primary_key
    ]
    query = sql.SQL("SELECT {} FROM {} ORDER BY {}").format(
        sql.SQL(", ").join(sql.Identifier(name) for name in mapping.column_names),
        sql.Identifier(schema, mapping.name),
        sql.SQL(", ").join(order),
    )
    with connection.cursor(name=f"seeon_reconcile_{mapping.name}") as cursor:
        cursor.itersize = _BATCH
        cursor.execute(query)
        yield from cursor


@dataclass(frozen=True, slots=True)
class TableSummary:
    rows: int
    pk_sha256: str
    content_sha256: str
    status: dict[str, dict[str, int]]

    def to_json(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "pk_sha256": self.pk_sha256,
            "content_sha256": self.content_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class TableResult:
    name: str
    reference: TableSummary
    candidate: TableSummary
    # Keys only in the reference, only in the candidate, and in both with other content.
    missing: int
    extra: int
    changed: int

    @property
    def passed(self) -> bool:
        return (
            self.reference == self.candidate
            and self.missing == 0
            and self.extra == 0
            and self.changed == 0
        )

    def to_json(self, labels: tuple[str, str] = ("source", "target")) -> dict[str, object]:
        reference, candidate = labels
        return {
            "table": self.name,
            "result": PASS if self.passed else FAIL,
            reference: self.reference.to_json(),
            candidate: self.candidate.to_json(),
            f"only_in_{reference}": self.missing,
            f"only_in_{candidate}": self.extra,
            "changed": self.changed,
        }


class _Side:
    """Order-checked running digests of one row stream."""

    __slots__ = ("content", "keys", "last", "mapping", "rows", "status")

    def __init__(self, mapping: TableMapping) -> None:
        self.mapping = mapping
        self.rows = 0
        self.keys = hashlib.sha256()
        self.content = hashlib.sha256()
        self.status: dict[str, Counter[str]] = {}
        self.last: tuple[object, ...] | None = None

    def add(self, row: tuple[object, ...]) -> tuple[tuple[object, ...], str]:
        key = tuple(row[index] for index in self.mapping.primary_key_indexes)
        if self.last is not None and not _less(self.last, key, self.mapping.name):
            raise MigrationError(f"{self.mapping.name} rows are not in strict key order")
        self.last = key
        digest = row_sha256(row)
        self.rows += 1
        key_json = canonical_json([canonical_value(value) for value in key])
        self.keys.update(key_json.encode("ascii") + b"\n")
        self.content.update(digest.encode("ascii") + b"\n")
        for name, index in zip(
            self.mapping.spec.status_columns, self.mapping.status_indexes, strict=True
        ):
            self.status.setdefault(name, Counter())[_status_label(row[index])] += 1
        return key, digest

    def summary(self) -> TableSummary:
        return TableSummary(
            rows=self.rows,
            pk_sha256=self.keys.hexdigest(),
            content_sha256=self.content.hexdigest(),
            status={
                name: dict(sorted(self.status.get(name, Counter()).items()))
                for name in self.mapping.spec.status_columns
            },
        )


def compare_rows(
    mapping: TableMapping,
    reference: Iterable[tuple[object, ...]],
    candidate: Iterable[tuple[object, ...]],
) -> TableResult:
    """Merge-join two key-ordered row streams without holding either in memory."""
    left = _Side(mapping)
    right = _Side(mapping)
    references = iter(reference)
    candidates = iter(candidate)
    a = _next(references, left)
    b = _next(candidates, right)
    missing = extra = changed = 0
    while a is not None or b is not None:
        if b is None or (a is not None and _less(a[0], b[0], mapping.name)):
            missing += 1
            a = _next(references, left)
        elif a is None or _less(b[0], a[0], mapping.name):
            extra += 1
            b = _next(candidates, right)
        else:
            changed += a[1] != b[1]
            a = _next(references, left)
            b = _next(candidates, right)
    return TableResult(
        name=mapping.name,
        reference=left.summary(),
        candidate=right.summary(),
        missing=missing,
        extra=extra,
        changed=changed,
    )


def compare_tables(
    snapshot: sqlite3.Connection,
    target: psycopg.Connection,
    mappings: Iterable[TableMapping],
    schema: str,
) -> tuple[TableResult, ...]:
    results = []
    for mapping in mappings:
        with (
            closing(sqlite_rows(snapshot, mapping)) as reference,
            closing(postgres_rows(target, mapping, schema)) as candidate,
        ):
            results.append(compare_rows(mapping, reference, candidate))
    return tuple(results)


def compare_sources(
    snapshot: sqlite3.Connection, live: sqlite3.Connection, mappings: Iterable[TableMapping]
) -> tuple[TableResult, ...]:
    results = []
    for mapping in mappings:
        with (
            closing(sqlite_rows(snapshot, mapping)) as reference,
            closing(sqlite_rows(live, mapping)) as candidate,
        ):
            results.append(compare_rows(mapping, reference, candidate))
    return tuple(results)


def fingerprint(results: Iterable[TableResult]) -> str:
    """Digest of the reference side: binds the snapshot content to the mapping used."""
    body = [
        [
            result.name,
            result.reference.rows,
            result.reference.pk_sha256,
            result.reference.content_sha256,
        ]
        for result in results
    ]
    return hashlib.sha256(canonical_json(body).encode("ascii")).hexdigest()


def source_identity_floors(
    connection: sqlite3.Connection, mappings: Iterable[TableMapping]
) -> dict[str, int]:
    """The next identity value must exceed every used id and every AUTOINCREMENT high-water."""
    identities = {
        mapping.name: mapping.identity_column
        for mapping in mappings
        if mapping.identity_column is not None
    }
    floors: dict[str, int] = {}
    for table, column in identities.items():
        (value,) = connection.execute(
            f"SELECT max({_quote(column)}) FROM {_quote(table)}"
        ).fetchone()
        if value is not None and type(value) is not int:
            raise MigrationError(f"{table}.{column} holds a non-integer identity")
        floors[table] = value or 0
    if "sqlite_sequence" in sqlite_table_names(connection):
        for table, value in connection.execute("SELECT name, seq FROM sqlite_sequence"):
            if table not in identities:
                raise MigrationError("sqlite_sequence tracks a table without an identity mapping")
            if type(value) is not int:
                raise MigrationError(f"sqlite_sequence for {table} is not an integer")
            floors[table] = max(floors[table], value)
    return floors


def target_identity(
    connection: psycopg.Connection, schema: str, table: str, column: str
) -> int | None:
    row = connection.execute(
        "SELECT pg_catalog.pg_sequence_last_value("
        "pg_catalog.pg_get_serial_sequence(%s, %s)::regclass)",
        (f"{schema}.{table}", column),
    ).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def ledger_rows(connection: psycopg.Connection, schema: str) -> list[tuple[object, ...]]:
    return [
        tuple(row)
        for row in connection.execute(
            sql.SQL(
                "SELECT version, name, checksum, source_schema_version, source_db_sha256, "
                "reconciliation_sha256 FROM {} ORDER BY version"
            ).format(sql.Identifier(schema, "schema_migrations"))
        ).fetchall()
    ]


def authority_state(connection: psycopg.Connection, schema: str) -> dict[str, object]:
    """Generation and switches only; the writer token never leaves the database."""
    rows = connection.execute(
        sql.SQL("SELECT generation, accepting, egress_enabled FROM {}").format(
            sql.Identifier(schema, "deployment_authority")
        )
    ).fetchall()
    if len(rows) != 1:
        raise MigrationError("deployment authority is not a singleton")
    generation, accepting, egress_enabled = rows[0]
    return {"generation": generation, "accepting": accepting, "egress_enabled": egress_enabled}


def seed_only_site(connection: psycopg.Connection, schema: str) -> bool:
    """Whether edge_site holds nothing beyond the activation seed's id, time and defaults.

    Any other value, including one equal to what a restored snapshot would hold,
    can only have been written after activation.
    """
    written = connection.execute(
        sql.SQL(
            "SELECT count(*) FROM {} AS site CROSS JOIN LATERAL jsonb_each("
            "to_jsonb(site) - 'id' - 'updated_at') AS kept(name, value) "
            "LEFT JOIN information_schema.columns AS col ON col.table_schema = %s "
            "AND col.table_name = 'edge_site' AND col.column_name = kept.name "
            "WHERE col.column_default IS DISTINCT FROM kept.value #>> '{{}}'"
        ).format(sql.Identifier(schema, "edge_site")),
        (schema,),
    ).fetchone()[0]
    return written == 0


def delivery_state(connection: psycopg.Connection, schema: str) -> dict[str, object]:
    rows = {
        table: int(
            connection.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(schema, table))
            ).fetchone()[0]
        )
        for table in DELIVERY_TABLES
    }
    states = connection.execute(
        sql.SQL("SELECT state, count(*) FROM {} GROUP BY state ORDER BY state").format(
            sql.Identifier(schema, "event_outbox")
        )
    ).fetchall()
    (active,) = connection.execute(
        sql.SQL(
            "SELECT count(*) FROM {} WHERE state = 'IN_FLIGHT' "
            "AND lease_until > pg_catalog.clock_timestamp()"
        ).format(sql.Identifier(schema, "event_outbox"))
    ).fetchone()
    return {
        "delivery_rows": rows,
        "outbox_states": {str(state): int(count) for state, count in states},
        "active_leases": int(active),
    }


def diagnostics_state(connection: psycopg.Connection, schema: str) -> dict[str, object]:
    """Live diagnostics are fresh, never imported: verify the layout and count rows only."""
    diagnostics = diagnostics_schema_name(schema)
    tables = postgres_table_names(connection, diagnostics)
    ledger: list[tuple[object, ...]] = []
    if "schema_migrations" in tables:
        ledger = [
            tuple(row)
            for row in connection.execute(
                sql.SQL("SELECT version, name, checksum FROM {} ORDER BY version").format(
                    sql.Identifier(diagnostics, "schema_migrations")
                )
            ).fetchall()
        ]
    expected = [(DIAGNOSTICS_SCHEMA_VERSION, DIAGNOSTICS_SCHEMA_NAME, diagnostics_checksum())]
    ledger_result = PASS if ledger == expected else FAIL
    tables_result = PASS if tables == DIAGNOSTICS_TARGET_TABLES else FAIL
    rows = (
        {
            table: int(
                connection.execute(
                    sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(diagnostics, table))
                ).fetchone()[0]
            )
            for table in sorted(DIAGNOSTICS_TABLES)
        }
        if tables_result == PASS
        else None
    )
    return {
        "schema": diagnostics,
        "mode": "live",
        "reconciled": False,
        "ledger": ledger_result,
        "tables": tables_result,
        "rows": rows,
        "result": PASS if ledger_result == tables_result == PASS else FAIL,
    }


def reconcile(
    database: PostgresDatabase,
    *,
    schema: str,
    snapshot_path: Path,
    source_path: Path | None = None,
    worker_state_dir: Path | None = None,
    expected_queue_sha256: str | None = None,
    after_transfer: bool = False,
    fence_receipt: Path | None = None,
) -> dict[str, object]:
    """Compare the snapshot with the target; counts and digests only.

    Before transfer the authority must be fenced. After transfer it must have moved
    past generation 1, and the only difference transfer itself may leave is the
    activation seed of an ``edge_site`` the snapshot never had. A source the fence
    stamped is compared with its receipt by bytes, since SQLite refuses to open it.
    """
    require_identifier(schema, "schema")
    if fence_receipt is not None and source_path is None:
        raise MigrationError("a fence receipt needs the source it fenced")
    source_sha256 = snapshot_sha256(snapshot_path)
    failures: list[str] = []
    with closing(open_snapshot(snapshot_path)) as snapshot:
        schema_version = verify_runtime_schema(snapshot)

        def inspect(connection: psycopg.Connection) -> dict[str, object]:
            mappings = build_mappings(snapshot, connection, schema)
            floors = source_identity_floors(snapshot, mappings)
            return {
                "mappings": mappings,
                "tables": compare_tables(snapshot, connection, mappings, schema),
                "ledger": ledger_rows(connection, schema),
                "authority": authority_state(connection, schema),
                "seed_only": seed_only_site(connection, schema),
                "pending": delivery_state(connection, schema),
                "identity": {
                    mapping.name: (
                        floors[mapping.name],
                        target_identity(connection, schema, mapping.name, mapping.identity_column),
                    )
                    for mapping in mappings
                    if mapping.identity_column is not None
                },
                "target_tail": _max_audit_id(connection, schema),
                "diagnostics": diagnostics_state(connection, schema),
            }

        target = database.read_snapshot(inspect)
        tables: tuple[TableResult, ...] = target["tables"]
        activation_seed = (
            after_transfer
            and target["seed_only"]
            and any(
                result.name == "edge_site"
                and (result.reference.rows, result.candidate.rows) == (0, 1)
                for result in tables
            )
        )
        failures.extend(
            f"table:{result.name}"
            for result in tables
            if not (result.passed or (activation_seed and result.name == "edge_site"))
        )
        reconciliation = fingerprint(tables)

        boundary: dict[str, object] = {
            "snapshot_audit_tail": snapshot.execute(
                "SELECT max(audit_id) FROM audit_events"
            ).fetchone()[0],
            "target_audit_tail": target["target_tail"],
            "live_source": None,
        }
        if boundary["snapshot_audit_tail"] != boundary["target_audit_tail"]:
            failures.append("boundary:audit_tail")
        if fence_receipt is not None:
            fence = read_fence_receipt(fence_receipt)
            section, reasons = inspect_fence(source_path, fence)
            if fence.snapshot_sha256 != source_sha256:
                reasons.append("sqlite:snapshot_mismatch")
            failures.extend(reasons)
            boundary["live_source"] = {
                "result": FAIL if reasons else PASS,
                "fence": section,
                "reasons": reasons,
            }
        elif source_path is not None:
            with open_fenced_source(source_path) as live:
                live_results = compare_sources(snapshot, live, target["mappings"])
            failures.extend(
                f"live_source:{result.name}" for result in live_results if not result.passed
            )
            boundary["live_source"] = {
                "result": PASS if all(result.passed for result in live_results) else FAIL,
                "tables": [result.to_json(("snapshot", "live")) for result in live_results],
            }
    if snapshot_sha256(snapshot_path) != source_sha256:
        raise MigrationError("snapshot changed during reconciliation")

    ledger = _ledger_section(target["ledger"], schema_version, source_sha256, reconciliation)
    failures.extend(f"ledger:{name}" for name in ledger["failures"])

    authority = target["authority"]
    if after_transfer:
        if authority["generation"] == 1:
            failures.append("authority:not_transferred")
    elif authority["accepting"] or authority["egress_enabled"]:
        failures.append("authority:not_fenced")

    diagnostics = target["diagnostics"]
    if diagnostics["result"] != PASS:
        failures.append("diagnostics:schema")

    pending = target["pending"]
    failures.extend(
        f"pending:{table}" for table, count in pending["delivery_rows"].items() if count
    )

    identity: dict[str, object] = {}
    for table, (floor, last_value) in target["identity"].items():
        passed = floor == 0 or (last_value is not None and last_value >= floor)
        identity[table] = {
            "source_floor": floor,
            "target_last_value": last_value,
            "result": PASS if passed else FAIL,
        }
        if not passed:
            failures.append(f"identity:{table}")

    queue: dict[str, object] | None = None
    if worker_state_dir is not None:
        digest = queue_digest(worker_state_dir)
        queue = digest.to_json()
        if expected_queue_sha256 is None:
            queue["result"] = "REPORTED"
        elif digest.sha256 == expected_queue_sha256:
            queue["result"] = PASS
        else:
            queue["result"] = FAIL
            failures.append("delivery_queue:sha256")

    return {
        "format": REPORT_FORMAT,
        "mode": "after_transfer" if after_transfer else "before_transfer",
        "result": FAIL if failures else PASS,
        "failures": failures,
        "snapshot": {"sha256": source_sha256, "schema_version": schema_version},
        "activation_seed": activation_seed,
        "tables": [result.to_json() for result in tables],
        "boundary": boundary,
        "ledger": {key: value for key, value in ledger.items() if key != "failures"},
        "authority": authority,
        "pending": pending,
        "identity": identity,
        "delivery_queue": queue,
        "diagnostics": diagnostics,
        "decisions": decision_table(),
        "reconciliation_sha256": reconciliation,
    }


def _ledger_section(
    rows: list[tuple[object, ...]], schema_version: int, source_sha256: str, reconciliation: str
) -> dict[str, object]:
    failures: list[str] = []
    if len(rows) != 1:
        return {"entries": len(rows), "result": FAIL, "failures": ["entries"]}
    version, name, checksum, stamped_version, stamped_sha256, stamped_reconciliation = rows[0]
    checks = {
        "version": version == SCHEMA_VERSION and name == SCHEMA_NAME,
        "checksum": checksum == schema_checksum(),
        "imported": stamped_sha256 is not None,
        "source_schema_version": stamped_version == schema_version,
        "source_db_sha256": stamped_sha256 == source_sha256,
        "reconciliation_sha256": stamped_reconciliation == reconciliation,
    }
    failures = [check for check, passed in checks.items() if not passed]
    return {
        "entries": 1,
        "checks": {check: PASS if passed else FAIL for check, passed in checks.items()},
        "result": FAIL if failures else PASS,
        "failures": failures,
    }


def _max_audit_id(connection: psycopg.Connection, schema: str) -> int | None:
    (value,) = connection.execute(
        sql.SQL("SELECT max(audit_id) FROM {}").format(sql.Identifier(schema, "audit_events"))
    ).fetchone()
    return value


def _status_label(value: object) -> str:
    # Raw only for short enum-like text; anything longer could be resident data.
    if value is None:
        return "<null>"
    text = value if isinstance(value, str) else canonical_json(canonical_value(value))
    if _STATUS_LABEL.fullmatch(text):
        return text
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _next(rows: Iterator[tuple[object, ...]], side: _Side) -> tuple[tuple[object, ...], str] | None:
    row = next(rows, None)
    return None if row is None else side.add(tuple(row))


def _less(left: tuple[object, ...], right: tuple[object, ...], table: str) -> bool:
    try:
        return left < right
    except TypeError as error:
        raise MigrationError(f"{table} primary key values are not comparable") from error


def _quote(identifier: str) -> str:
    return '"' + identifier + '"'


__all__ = [
    "FAIL",
    "PASS",
    "REPORT_FORMAT",
    "TableResult",
    "TableSummary",
    "authority_state",
    "canonical_json",
    "canonical_value",
    "compare_rows",
    "compare_sources",
    "compare_tables",
    "delivery_state",
    "diagnostics_state",
    "fingerprint",
    "ledger_rows",
    "postgres_rows",
    "reconcile",
    "row_sha256",
    "seed_only_site",
    "source_identity_floors",
    "sqlite_rows",
    "target_identity",
    "write_report",
]
