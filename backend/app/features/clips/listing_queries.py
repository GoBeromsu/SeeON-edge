"""Bounded SQL construction for clip listing pages and summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from backend.app.features.clips.listing_generation import GLOBAL_CAMERA, SqlParameter
from backend.app.features.clips.schemas import ClipListQuery

SqlValues: TypeAlias = tuple[SqlParameter, ...]
SUMMARY_SQL = """SELECT event_facet, count FROM clip_listing_summary
    WHERE generation = ? AND camera_id = ? ORDER BY event_facet"""
_PAGE_COLUMNS = """SELECT rows.clip_id, rows.camera_id, rows.event_ref, rows.event_type,
    rows.started_at, rows.duration_s, rows.codec, rows.media_path, rows.video_available,
    rows.video_error, rows.finalized, rows.size_bytes,
    COALESCE(thumbnails.thumbnail_available, 0)
    FROM clip_listing_rows AS rows INDEXED BY {index_name}
    LEFT JOIN clip_listing_thumbnails AS thumbnails
      ON thumbnails.generation = rows.generation AND thumbnails.clip_id = rows.clip_id"""


@dataclass(frozen=True, slots=True)
class PageStatements:
    page_sql: str
    page_values: SqlValues
    summary_sql: str
    summary_values: SqlValues


@dataclass(frozen=True, slots=True)
class QueryPlans:
    page: tuple[str, ...]
    summary: tuple[str, ...]
    summary_sql: str


def build_page_statements(generation: int, query: ClipListQuery) -> PageStatements:
    clauses = ["rows.generation = ?"]
    values: list[SqlParameter] = [generation]
    index_name = "clip_listing_global_order_idx"
    if query.camera_id is not None:
        clauses.append("rows.camera_id = ?")
        values.append(query.camera_id)
        index_name = "clip_listing_camera_order_idx"
    if query.event_type is not None:
        clauses.append("rows.event_facet = ?")
        values.append(query.event_type)
        index_name = (
            "clip_listing_camera_facet_order_idx"
            if query.camera_id is not None
            else "clip_listing_global_facet_order_idx"
        )
    page_sql = (
        f"{_PAGE_COLUMNS.format(index_name=index_name)} "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY rows.started_at DESC, rows.clip_id DESC LIMIT ? OFFSET ?"
    )
    page_values = (*values, query.limit, query.offset)
    summary_camera = query.camera_id or GLOBAL_CAMERA
    return PageStatements(
        page_sql=page_sql,
        page_values=page_values,
        summary_sql=SUMMARY_SQL,
        summary_values=(generation, summary_camera),
    )


__all__ = ["PageStatements", "QueryPlans", "build_page_statements"]
