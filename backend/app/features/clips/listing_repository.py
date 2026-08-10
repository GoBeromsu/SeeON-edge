"""WAL-backed storage for immutable clip listing generations."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeAlias, final

from pydantic import TypeAdapter, ValidationError

from backend.app.features.clips.listing import ClipPage, EventTypeFacet
from backend.app.features.clips.listing_generation import (
    TOTAL_FACET,
    IndexedClip,
    PreparedGeneration,
    SqlParameter,
)
from backend.app.features.clips.listing_queries import (
    QueryPlans,
    build_page_statements,
)
from backend.app.features.clips.listing_schema import (
    ACTIVATE_GENERATION,
    ADVANCE_NEXT_GENERATION,
    DELETE_OLD_ROWS,
    DELETE_OLD_SUMMARIES,
    DELETE_OLD_THUMBNAILS,
    INSERT_CLIP,
    INSERT_SUMMARY,
    INSERT_THUMBNAIL,
    SELECT_ACTIVE_CLIPS,
    SELECT_ACTIVE_GENERATION,
    SELECT_NEXT_GENERATION,
)
from backend.app.features.clips.listing_schema_migration import initialize_listing_schema
from backend.app.features.clips.schemas import ClipListQuery
from backend.app.features.clips.store import ClipManifest
from backend.app.shared.sqlite_bootstrap import connect_catalog_store

SqlValue: TypeAlias = str | int | float | bytes | None
SqlRows: TypeAlias = list[tuple[SqlValue, ...]]
ActiveRow: TypeAlias = tuple[
    str,
    int,
    int,
    str,
    str,
    str,
    str | None,
    EventTypeFacet,
    str,
    float,
    str,
    str | None,
    int,
    str | None,
    int,
    int | None,
    int | None,
    int | None,
    int,
]
PageRow: TypeAlias = tuple[
    str,
    str,
    str,
    str | None,
    str,
    float,
    str,
    str | None,
    int,
    str | None,
    int,
    int | None,
    int,
]
_ACTIVE_ROWS = TypeAdapter(list[ActiveRow])
_PAGE_ROWS = TypeAdapter(list[PageRow])
_SUMMARY_ROWS = TypeAdapter(list[tuple[str, int]])
_GENERATION_ROWS = TypeAdapter(list[tuple[int]])
_PLAN_ROWS = TypeAdapter(list[tuple[int, int, int, str]])


class ListingRepositoryClosedError(RuntimeError):
    pass


@final
class ListingRepository:
    def __init__(self, path: Path, writer: sqlite3.Connection) -> None:
        self._path = path
        self._writer = writer
        self._closed = False

    @classmethod
    def open(cls, path: Path | str) -> ListingRepository:
        resolved = Path(path)
        writer = connect_catalog_store(resolved, ())
        try:
            initialize_listing_schema(writer)
        except sqlite3.Error:
            writer.close()
            raise
        return cls(resolved, writer)

    def active_clips(self) -> dict[str, IndexedClip]:
        self._ensure_open()
        raw: SqlRows = self._writer.execute(SELECT_ACTIVE_CLIPS).fetchall()
        clips = (_indexed_clip(row) for row in _ACTIVE_ROWS.validate_python(raw))
        return {clip.manifest_path: clip for clip in clips}

    def publish(self, prepared: PreparedGeneration) -> None:
        self._ensure_open()
        generation = self._reserve_generation()
        _ = self._writer.execute("BEGIN IMMEDIATE")
        try:
            _ = self._writer.executemany(
                INSERT_CLIP,
                (clip.base_sql_values(generation) for clip in prepared.clips),
            )
            _ = self._writer.executemany(
                INSERT_THUMBNAIL,
                (clip.thumbnail_sql_values(generation) for clip in prepared.clips),
            )
            _ = self._writer.executemany(
                INSERT_SUMMARY,
                (summary.sql_values(generation) for summary in prepared.summaries),
            )
            _ = self._writer.execute("COMMIT")
        except (sqlite3.Error, ValidationError):
            self.rollback()
            raise
        _ = self._writer.execute("BEGIN IMMEDIATE")
        try:
            _ = self._writer.execute(ACTIVATE_GENERATION, (generation,))
            _ = self._writer.execute("COMMIT")
        except sqlite3.Error:
            self.rollback()
            raise
        _ = self._writer.execute("BEGIN IMMEDIATE")
        try:
            _ = self._writer.execute(DELETE_OLD_THUMBNAILS, (generation,))
            _ = self._writer.execute(DELETE_OLD_ROWS, (generation,))
            _ = self._writer.execute(DELETE_OLD_SUMMARIES, (generation,))
            _ = self._writer.execute("COMMIT")
        except sqlite3.Error:
            self.rollback()
            raise

    def page(self, query: ClipListQuery) -> ClipPage:
        with self._reader() as connection:
            _ = connection.execute("BEGIN")
            try:
                generation = self._active_generation(connection)
                statements = build_page_statements(generation, query)
                raw_summaries: SqlRows = connection.execute(
                    statements.summary_sql, statements.summary_values
                ).fetchall()
                summary = dict(_SUMMARY_ROWS.validate_python(raw_summaries))
                total_key = query.event_type or TOTAL_FACET
                total = summary.get(total_key, 0)
                facets = {key: count for key, count in summary.items() if key != TOTAL_FACET}
                raw_rows: SqlRows = connection.execute(
                    statements.page_sql, statements.page_values
                ).fetchall()
                manifests = tuple(_manifest(row) for row in _PAGE_ROWS.validate_python(raw_rows))
                page = ClipPage(
                    manifests=manifests,
                    total=total,
                    has_more=query.offset + len(manifests) < total,
                    event_type_counts=facets,
                )
                _ = connection.execute("COMMIT")
            finally:
                if connection.in_transaction:
                    connection.rollback()
        return page

    def explain(self, query: ClipListQuery) -> QueryPlans:
        with self._reader() as connection:
            statements = build_page_statements(self._active_generation(connection), query)
            page = _plan(connection, statements.page_sql, statements.page_values)
            summary = _plan(connection, statements.summary_sql, statements.summary_values)
            return QueryPlans(page=page, summary=summary, summary_sql=statements.summary_sql)

    def rollback(self) -> None:
        if self._closed:
            return
        try:
            if self._writer.in_transaction:
                _ = self._writer.execute("ROLLBACK")
        except sqlite3.ProgrammingError:
            return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.close()

    def _reserve_generation(self) -> int:
        _ = self._writer.execute("BEGIN IMMEDIATE")
        try:
            raw: SqlRows = self._writer.execute(SELECT_NEXT_GENERATION).fetchall()
            generation = _GENERATION_ROWS.validate_python(raw)[0][0]
            _ = self._writer.execute(ADVANCE_NEXT_GENERATION)
            _ = self._writer.execute("COMMIT")
        except (sqlite3.Error, ValidationError):
            self.rollback()
            raise
        else:
            return generation

    @contextmanager
    def _reader(self) -> Generator[sqlite3.Connection, None, None]:
        self._ensure_open()
        connection = connect_catalog_store(self._path, ())
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _active_generation(connection: sqlite3.Connection) -> int:
        raw: SqlRows = connection.execute(SELECT_ACTIVE_GENERATION).fetchall()
        return _GENERATION_ROWS.validate_python(raw)[0][0]

    def _ensure_open(self) -> None:
        if self._closed:
            raise ListingRepositoryClosedError("clip listing repository is closed")


def _indexed_clip(row: ActiveRow) -> IndexedClip:
    return IndexedClip(
        *row[:12],
        bool(row[12]),
        row[13],
        bool(row[14]),
        row[15],
        row[16],
        row[17],
        bool(row[18]),
    )


def _manifest(row: PageRow) -> ClipManifest:
    return ClipManifest(
        *row[:8],
        bool(row[8]),
        row[9],
        bool(row[10]),
        row[11],
        bool(row[12]),
    )


def _plan(
    connection: sqlite3.Connection,
    sql: str,
    values: tuple[SqlParameter, ...],
) -> tuple[str, ...]:
    raw: SqlRows = connection.execute(f"EXPLAIN QUERY PLAN {sql}", values).fetchall()
    return tuple(row[3] for row in _PLAN_ROWS.validate_python(raw))


__all__ = ["ListingRepository", "ListingRepositoryClosedError"]
