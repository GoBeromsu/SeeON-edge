"""Canonical machine-readable schema 18 structural contract."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from backend.app.edge_db.compact_schema import SCHEMA_18_STATEMENTS


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    type: str
    notnull: int
    default: str | None
    pk: int
    hidden: int


@dataclass(frozen=True, slots=True)
class ForeignKeySpec:
    id: int
    seq: int
    table: str
    from_col: str
    to_col: str | None
    on_update: str
    on_delete: str


@dataclass(frozen=True, slots=True)
class IndexColumnSpec:
    seqno: int
    name: str | None
    desc: int


@dataclass(frozen=True, slots=True)
class IndexSpec:
    name: str
    unique: int
    origin: str
    partial: int
    sql: str | None
    columns: tuple[IndexColumnSpec, ...]


@dataclass(frozen=True, slots=True)
class TriggerSpec:
    name: str
    sql: str


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    columns: tuple[ColumnSpec, ...]
    foreign_keys: tuple[ForeignKeySpec, ...]
    indexes: tuple[IndexSpec, ...]
    triggers: tuple[TriggerSpec, ...]
    check_sql: str


@dataclass(frozen=True, slots=True)
class Schema18Manifest:
    tables: tuple[TableSpec, ...]

    def diff(self, other: Schema18Manifest) -> tuple[str, ...]:
        if self == other:
            return ()
        left = {table.name: table for table in self.tables}
        right = {table.name: table for table in other.tables}
        names = tuple(sorted(set(left) | set(right)))
        deltas: list[str] = []
        for name in names:
            if name not in left:
                deltas.append(f"missing_table:{name}")
                continue
            if name not in right:
                deltas.append(f"extra_table:{name}")
                continue
            if left[name] != right[name]:
                deltas.append(f"altered_table:{name}")
        return tuple(deltas)


def read_schema18_manifest(connection: sqlite3.Connection) -> Schema18Manifest:
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return Schema18Manifest(tables=tuple(_read_table(connection, name) for name in tables))


def compile_schema18_manifest() -> Schema18Manifest:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in SCHEMA_18_STATEMENTS:
            connection.execute(statement)
        return read_schema18_manifest(connection)
    finally:
        connection.close()


def _normalized_sql(sql: str) -> str:
    return " ".join(sql.split())


def _read_table(connection: sqlite3.Connection, name: str) -> TableSpec:
    create_row = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    create_sql = "" if create_row is None or create_row[0] is None else str(create_row[0])
    columns = tuple(
        ColumnSpec(
            name=str(row[1]),
            type=str(row[2]),
            notnull=int(row[3]),
            default=None if row[4] is None else str(row[4]),
            pk=int(row[5]),
            hidden=int(row[6]),
        )
        for row in connection.execute(f'PRAGMA table_xinfo("{name}")')
    )
    foreign_keys = tuple(
        ForeignKeySpec(
            id=int(row[0]),
            seq=int(row[1]),
            table=str(row[2]),
            from_col=str(row[3]),
            to_col=None if row[4] is None else str(row[4]),
            on_update=str(row[5]),
            on_delete=str(row[6]),
        )
        for row in connection.execute(f'PRAGMA foreign_key_list("{name}")')
    )
    indexes = tuple(
        _read_index(connection, str(row[1]), int(row[2]), str(row[3]), int(row[4]))
        for row in connection.execute(f'PRAGMA index_list("{name}")')
    )
    triggers = tuple(
        TriggerSpec(name=str(row[0]), sql=str(row[1]))
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' "
            "AND tbl_name = ? ORDER BY name",
            (name,),
        )
    )
    return TableSpec(
        name=name,
        columns=columns,
        foreign_keys=foreign_keys,
        indexes=indexes,
        triggers=triggers,
        check_sql=_normalized_sql(create_sql),
    )


def _read_index(
    connection: sqlite3.Connection,
    name: str,
    unique: int,
    origin: str,
    partial: int,
) -> IndexSpec:
    sql_row = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type = 'index' AND name = ?",
        (name,),
    ).fetchone()
    columns = tuple(
        IndexColumnSpec(
            seqno=int(row[0]),
            name=None if row[2] is None else str(row[2]),
            desc=int(row[3]),
        )
        for row in connection.execute(f'PRAGMA index_xinfo("{name}")')
        if int(row[5]) == 1
    )
    return IndexSpec(
        name=name,
        unique=unique,
        origin=origin,
        partial=partial,
        sql=None if sql_row is None or sql_row[0] is None else str(sql_row[0]),
        columns=columns,
    )


__all__ = [
    "ColumnSpec",
    "ForeignKeySpec",
    "IndexColumnSpec",
    "IndexSpec",
    "Schema18Manifest",
    "TableSpec",
    "TriggerSpec",
    "compile_schema18_manifest",
    "read_schema18_manifest",
]
