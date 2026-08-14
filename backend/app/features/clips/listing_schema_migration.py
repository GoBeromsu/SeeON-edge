"""Strict initialization and draft migration for clip listing tables."""

from __future__ import annotations

import sqlite3
from enum import StrEnum
from typing import assert_never, final

from backend.app.features.clips.listing_schema import (
    CREATE_GENERATION_TABLE,
    CREATE_INDEX_STATEMENTS,
    CREATE_ROWS_TABLE,
    CREATE_STATEMENTS,
    CREATE_SUMMARY_TABLE,
    CREATE_THUMBNAILS_TABLE,
    KNOWN_DRAFT_ROWS_TABLE,
)

_DRAFT_BACKUP_TABLE = "clip_listing_rows_draft_268"
_COPY_RELEASED_COLUMNS = """INSERT INTO clip_listing_rows (
    generation, clip_id, manifest_path, manifest_mtime_ns, manifest_size_bytes,
    camera_id, event_ref, event_type, event_facet, started_at, duration_s, codec,
    media_path, video_available, video_error, finalized, size_bytes
) SELECT generation, clip_id, manifest_path, manifest_mtime_ns, manifest_size_bytes,
    camera_id, event_ref, event_type, event_facet, started_at, duration_s, codec,
    media_path, video_available, video_error, finalized, size_bytes
FROM clip_listing_rows_draft_268"""
_COPY_THUMBNAIL_COLUMNS = """INSERT INTO clip_listing_thumbnails (
    generation, clip_id, thumbnail_mtime_ns, thumbnail_size_bytes, thumbnail_available
) SELECT generation, clip_id, thumbnail_mtime_ns, thumbnail_size_bytes,
    thumbnail_available FROM clip_listing_rows_draft_268"""


class _RowsShape(StrEnum):
    ABSENT = "absent"
    RELEASED = "released"
    DRAFT = "draft"


@final
class ListingSchemaError(sqlite3.DatabaseError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def validate_listing_schema(connection: sqlite3.Connection) -> None:
    expected_tables = (
        ("clip_listing_generation", CREATE_GENERATION_TABLE),
        ("clip_listing_rows", CREATE_ROWS_TABLE),
        ("clip_listing_thumbnails", CREATE_THUMBNAILS_TABLE),
        ("clip_listing_summary", CREATE_SUMMARY_TABLE),
    )
    for table, expected_sql in expected_tables:
        sql = _table_sql(connection, table)
        if sql is None or _normalize(sql) != _normalize(expected_sql):
            raise ListingSchemaError(f"unsupported {table} schema")

    expected_indexes = {
        statement.split()[5]: _normalize(statement) for statement in CREATE_INDEX_STATEMENTS
    }
    index_rows = connection.execute(
        "SELECT name, sql FROM sqlite_schema WHERE type = 'index' "
        "AND tbl_name = 'clip_listing_rows' AND sql IS NOT NULL"
    ).fetchall()
    actual_indexes = {str(name): _normalize(str(sql)) for name, sql in index_rows}
    if actual_indexes != expected_indexes:
        raise ListingSchemaError("unsupported clip_listing_rows schema objects")

    generations = connection.execute(
        "SELECT id, active_generation, next_generation FROM clip_listing_generation"
    ).fetchall()
    if len(generations) != 1:
        raise ListingSchemaError("clip listing generation state is not initialized")
    identifier, active_generation, next_generation = generations[0]
    if (
        identifier != 1
        or not isinstance(active_generation, int)
        or not isinstance(next_generation, int)
        or active_generation < 0
        or next_generation <= active_generation
    ):
        raise ListingSchemaError("clip listing generation state is invalid")


def initialize_listing_schema(connection: sqlite3.Connection) -> None:
    rows_sql = _table_sql(connection, "clip_listing_rows")
    thumbnails_sql = _table_sql(connection, "clip_listing_thumbnails")
    shape = _classify_rows(rows_sql)
    match shape:
        case _RowsShape.ABSENT:
            if thumbnails_sql is not None:
                raise ListingSchemaError(
                    "unsupported clip_listing_rows schema: absent with side table"
                )
            _bootstrap(connection)
        case _RowsShape.RELEASED:
            _validate_thumbnail_table(connection, thumbnails_sql)
            _bootstrap(connection)
        case _RowsShape.DRAFT:
            if thumbnails_sql is not None:
                raise ListingSchemaError(
                    "unsupported clip_listing_rows schema: draft with side table"
                )
            _migrate_draft(connection)
        case unreachable:
            assert_never(unreachable)


def _classify_rows(sql: str | None) -> _RowsShape:
    if sql is None:
        return _RowsShape.ABSENT
    normalized = _normalize(sql)
    if normalized == _normalize(CREATE_ROWS_TABLE):
        return _RowsShape.RELEASED
    if normalized == _normalize(KNOWN_DRAFT_ROWS_TABLE):
        return _RowsShape.DRAFT
    raise ListingSchemaError("unsupported clip_listing_rows schema")


def _validate_thumbnail_table(connection: sqlite3.Connection, sql: str | None) -> None:
    if sql is None:
        return
    if _normalize(sql) != _normalize(CREATE_THUMBNAILS_TABLE):
        raise ListingSchemaError("unsupported clip_listing_thumbnails schema")
    secondary_indexes = connection.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'index' "
        "AND tbl_name = 'clip_listing_thumbnails' AND sql IS NOT NULL"
    ).fetchall()
    triggers = connection.execute(
        "SELECT name FROM sqlite_schema WHERE type = 'trigger' "
        "AND tbl_name = 'clip_listing_thumbnails'"
    ).fetchall()
    if secondary_indexes or triggers:
        raise ListingSchemaError("unsupported clip_listing_thumbnails schema objects")


def _table_sql(connection: sqlite3.Connection, table: str) -> str | None:
    row = connection.execute(
        "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if row is None:
        return None
    sql = row[0]
    if not isinstance(sql, str):
        raise ListingSchemaError(f"unsupported {table} schema metadata")
    return sql


def _normalize(sql: str) -> str:
    return "".join(sql.casefold().split()).replace("ifnotexists", "")


def _bootstrap(connection: sqlite3.Connection) -> None:
    for statement in CREATE_STATEMENTS:
        connection.execute(statement)


def _migrate_draft(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            f"ALTER TABLE clip_listing_rows RENAME TO {_DRAFT_BACKUP_TABLE}"
        )
        connection.execute(CREATE_ROWS_TABLE)
        connection.execute(CREATE_THUMBNAILS_TABLE)
        connection.execute(_COPY_RELEASED_COLUMNS)
        connection.execute(_COPY_THUMBNAIL_COLUMNS)
        connection.execute(f"DROP TABLE {_DRAFT_BACKUP_TABLE}")
        _bootstrap(connection)
        connection.execute("COMMIT")
    except sqlite3.Error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


__all__ = [
    "ListingSchemaError",
    "initialize_listing_schema",
    "validate_listing_schema",
]
