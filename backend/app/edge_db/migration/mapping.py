"""Table mapping derived from the live schema-19 SQLite and PostgreSQL catalogs."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from enum import StrEnum

import psycopg

from backend.app.edge_db.migration.errors import MigrationError

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class Disposition(StrEnum):
    MIGRATE = "migrate"
    LEDGER_NOT_COPIED = "ledger_not_copied"
    SEQUENCE_CARRIED = "sequence_carried"
    PLANNER_STATISTICS = "planner_statistics"
    TARGET_ONLY = "target_only"


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    status_columns: tuple[str, ...] = ()
    # SQLite ORDER BY prefix that satisfies self-referencing foreign keys on load.
    load_priority: str | None = None


# Load order satisfies every non-deferrable foreign key in the PostgreSQL DDL.
MIGRATED_TABLES: tuple[TableSpec, ...] = (
    TableSpec("credentials"),
    TableSpec("edge_site"),
    TableSpec(
        "locations",
        status_columns=("kind",),
        load_priority="CASE kind WHEN 'FLOOR' THEN 0 ELSE 1 END",
    ),
    TableSpec("cameras", status_columns=("mapping_state",)),
    TableSpec("policies", status_columns=("status",)),
    TableSpec("clips", status_columns=("local_state", "publish_state", "retention_state")),
    TableSpec("incidents", status_columns=("lifecycle_state", "provenance_state")),
    TableSpec("artifacts", status_columns=("state",)),
    TableSpec("audit_events", status_columns=("outcome",)),
    TableSpec("execution_provenance"),
    TableSpec("execution_segments", status_columns=("storage_state",)),
    TableSpec("execution_units", status_columns=("causal_state",)),
    TableSpec("execution_records", status_columns=("record_kind", "outcome")),
    TableSpec("execution_coverage", status_columns=("coverage_kind",)),
    TableSpec("execution_batches"),
)

# Every other SQLite table must appear here; an unlisted table refuses the export.
SOURCE_DECISIONS: dict[str, tuple[Disposition, str]] = {
    "schema_migrations": (
        Disposition.LEDGER_NOT_COPIED,
        "verified as schema 19 at export; PostgreSQL keeps its own ledger with the source stamp",
    ),
    "sqlite_sequence": (
        Disposition.SEQUENCE_CARRIED,
        "AUTOINCREMENT high-water marks become identity sequence floors",
    ),
    "sqlite_stat1": (Disposition.PLANNER_STATISTICS, "query-planner statistics, not records"),
    "sqlite_stat4": (Disposition.PLANNER_STATISTICS, "query-planner statistics, not records"),
}

# PostgreSQL tables with no SQLite source rows.
TARGET_DECISIONS: dict[str, str] = {
    "schema_migrations": "provision version row plus the import source stamp",
    "deployment_authority": "provisioned fenced at generation 1; transferred exactly once",
    "event_outbox": (
        "old runtime delivered synchronously; its pending set is the worker file queue, "
        "retained on its volume and replayed"
    ),
    "event_delivery_attempts": "starts empty; new sender history",
    "event_delivery_results": "starts empty; new sender history",
    "event_delivery_observations": "starts empty; new sender history",
}

EXPECTED_TARGET_TABLES = frozenset(spec.name for spec in MIGRATED_TABLES) | frozenset(
    TARGET_DECISIONS
)
DELIVERY_TABLES: tuple[str, ...] = (
    "event_outbox",
    "event_delivery_attempts",
    "event_delivery_results",
    "event_delivery_observations",
)
# Live execution writes go to a sibling schema; the product copies hold imported history.
DIAGNOSTICS_SUFFIX = "_diagnostics"
DIAGNOSTICS_TABLES = frozenset(
    {
        "execution_provenance",
        "execution_segments",
        "execution_units",
        "execution_records",
        "execution_coverage",
        "execution_batches",
    }
)
DIAGNOSTICS_TARGET_TABLES = DIAGNOSTICS_TABLES | frozenset({"schema_migrations"})

# PostgreSQL-only columns; each must carry a default so the load omits it.
PG_ONLY_COLUMNS: dict[str, frozenset[str]] = {"cameras": frozenset({"incarnation"})}

TYPE_MAP: dict[str, str] = {
    "INTEGER": "bigint",
    "INT": "bigint",
    "TEXT": "text",
    "BLOB": "bytea",
    "REAL": "double precision",
}

# Binary COPY type names for the mapped PostgreSQL types.
COPY_TYPES: dict[str, str] = {
    "bigint": "int8",
    "text": "text",
    "bytea": "bytea",
    "double precision": "float8",
}


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    source_type: str
    target_type: str


@dataclass(frozen=True, slots=True)
class TableMapping:
    spec: TableSpec
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    identity_column: str | None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def primary_key_indexes(self) -> tuple[int, ...]:
        names = self.column_names
        return tuple(names.index(name) for name in self.primary_key)

    @property
    def status_indexes(self) -> tuple[int, ...]:
        names = self.column_names
        return tuple(names.index(name) for name in self.spec.status_columns)

    def target_type(self, name: str) -> str:
        for column in self.columns:
            if column.name == name:
                return column.target_type
        raise MigrationError(f"{self.name}.{name} is not a mapped column")


def decision_table() -> list[dict[str, str]]:
    """The explicit per-table decisions, for reports and the runbook."""
    rows = [
        {"table": spec.name, "source": "sqlite", "decision": Disposition.MIGRATE.value}
        for spec in MIGRATED_TABLES
    ]
    rows.extend(
        {"table": name, "source": "sqlite", "decision": decision.value, "reason": reason}
        for name, (decision, reason) in SOURCE_DECISIONS.items()
    )
    rows.extend(
        {
            "table": name,
            "source": "postgresql",
            "decision": Disposition.TARGET_ONLY.value,
            "reason": reason,
        }
        for name, reason in TARGET_DECISIONS.items()
    )
    return rows


def require_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise MigrationError(f"{label} must be a lowercase SQL identifier of at most 63 bytes")
    return value


def diagnostics_schema_name(schema: str) -> str:
    """Return the live diagnostics schema the runtime derives from the product schema."""
    return require_identifier(
        require_identifier(schema, "schema") + DIAGNOSTICS_SUFFIX, "diagnostics schema"
    )


def sqlite_table_names(connection: sqlite3.Connection) -> frozenset[str]:
    rows = connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'").fetchall()
    return frozenset(str(row[0]) for row in rows)


def postgres_table_names(connection: psycopg.Connection, schema: str) -> frozenset[str]:
    rows = connection.execute(
        "SELECT c.relname FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = %s AND c.relkind IN ('r', 'p', 'v', 'm', 'f')",
        (schema,),
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def verify_source_tables(connection: sqlite3.Connection) -> None:
    """Refuse any SQLite table that has no recorded decision."""
    tables = sqlite_table_names(connection)
    migrated = {spec.name for spec in MIGRATED_TABLES}
    unmapped = sorted(tables - migrated - set(SOURCE_DECISIONS))
    if unmapped:
        raise MigrationError(f"source tables without a mapping decision: {', '.join(unmapped)}")
    missing = sorted(migrated - tables)
    if missing:
        raise MigrationError(f"source is missing mapped tables: {', '.join(missing)}")


def build_mappings(
    source: sqlite3.Connection, target: psycopg.Connection, schema: str
) -> tuple[TableMapping, ...]:
    """Derive column mappings from both catalogs; any drift refuses."""
    verify_source_tables(source)
    target_tables = postgres_table_names(target, schema)
    if target_tables != EXPECTED_TARGET_TABLES:
        unexpected = sorted(target_tables - EXPECTED_TARGET_TABLES)
        absent = sorted(EXPECTED_TARGET_TABLES - target_tables)
        raise MigrationError(
            "target schema tables differ from the provisioned set: "
            f"unexpected={','.join(unexpected) or '-'} missing={','.join(absent) or '-'}"
        )
    return tuple(_map_table(spec, source, target, schema) for spec in MIGRATED_TABLES)


def _map_table(
    spec: TableSpec, source: sqlite3.Connection, target: psycopg.Connection, schema: str
) -> TableMapping:
    source_columns: list[tuple[str, str]] = []
    source_key: list[tuple[int, str]] = []
    for name, declared, key_ordinal, hidden in source.execute(
        "SELECT name, type, pk, hidden FROM pragma_table_xinfo(?) ORDER BY cid", (spec.name,)
    ):
        if hidden:
            raise MigrationError(f"{spec.name}.{name} is a hidden or generated source column")
        require_identifier(str(name), f"source column {spec.name}")
        source_columns.append((str(name), str(declared).upper()))
        if key_ordinal:
            source_key.append((int(key_ordinal), str(name)))

    target_columns = {
        str(name): (str(type_name), bool(has_default), str(identity), str(generated))
        for name, type_name, has_default, identity, generated in target.execute(
            "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), a.atthasdef, "
            "a.attidentity, a.attgenerated FROM pg_catalog.pg_attribute a "
            "JOIN pg_catalog.pg_class c ON c.oid = a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped "
            "ORDER BY a.attnum",
            (schema, spec.name),
        )
    }
    target_key = tuple(
        str(row[0])
        for row in target.execute(
            "SELECT a.attname FROM pg_catalog.pg_index i "
            "JOIN pg_catalog.pg_class c ON c.oid = i.indrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "CROSS JOIN LATERAL generate_series(0, i.indnkeyatts - 1) AS k(position) "
            "JOIN pg_catalog.pg_attribute a "
            "ON a.attrelid = i.indrelid AND a.attnum = i.indkey[k.position] "
            "WHERE i.indisprimary AND n.nspname = %s AND c.relname = %s ORDER BY k.position",
            (schema, spec.name),
        )
    )

    columns: list[Column] = []
    identity_column: str | None = None
    for name, declared in source_columns:
        expected = TYPE_MAP.get(declared)
        if expected is None:
            raise MigrationError(f"{spec.name}.{name} has an unmapped source type")
        target_column = target_columns.get(name)
        if target_column is None:
            raise MigrationError(f"{spec.name}.{name} has no target column")
        type_name, _has_default, identity, generated = target_column
        if type_name != expected:
            raise MigrationError(f"{spec.name}.{name} target type differs from the source type")
        if generated:
            raise MigrationError(f"{spec.name}.{name} is a generated target column")
        if identity:
            if identity != "d" or identity_column is not None:
                raise MigrationError(f"{spec.name}.{name} identity must be BY DEFAULT and unique")
            identity_column = name
        columns.append(Column(name=name, source_type=declared, target_type=type_name))

    mapped = {name for name, _declared in source_columns}
    allowed_extra = PG_ONLY_COLUMNS.get(spec.name, frozenset())
    for name, (_type_name, has_default, _identity, _generated) in target_columns.items():
        if name in mapped:
            continue
        if name not in allowed_extra or not has_default:
            raise MigrationError(f"{spec.name}.{name} is an unmapped target column")

    primary_key = tuple(name for _ordinal, name in sorted(source_key))
    if not primary_key or primary_key != target_key:
        raise MigrationError(f"{spec.name} primary keys differ between source and target")
    if not set(spec.status_columns) <= mapped:
        raise MigrationError(f"{spec.name} status columns are not mapped")
    return TableMapping(
        spec=spec,
        columns=tuple(columns),
        primary_key=primary_key,
        identity_column=identity_column,
    )


__all__ = [
    "COPY_TYPES",
    "DELIVERY_TABLES",
    "DIAGNOSTICS_SUFFIX",
    "DIAGNOSTICS_TABLES",
    "DIAGNOSTICS_TARGET_TABLES",
    "EXPECTED_TARGET_TABLES",
    "MIGRATED_TABLES",
    "PG_ONLY_COLUMNS",
    "SOURCE_DECISIONS",
    "TARGET_DECISIONS",
    "TYPE_MAP",
    "Column",
    "Disposition",
    "TableMapping",
    "TableSpec",
    "build_mappings",
    "decision_table",
    "diagnostics_schema_name",
    "postgres_table_names",
    "require_identifier",
    "sqlite_table_names",
    "verify_source_tables",
]
