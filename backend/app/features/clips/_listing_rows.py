"""SQLite row adapters for immutable clip listing generations."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import TypeAdapter

from backend.app.features.clips.listing import EventTypeFacet
from backend.app.features.clips.listing_generation import IndexedClip
from backend.app.features.clips.store import ClipManifest

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
ACTIVE_ROWS = TypeAdapter(list[ActiveRow])
PAGE_ROWS = TypeAdapter(list[PageRow])


def indexed_clip_from_row(row: ActiveRow) -> IndexedClip:
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


def manifest_from_row(row: PageRow) -> ClipManifest:
    return ClipManifest(
        *row[:8],
        bool(row[8]),
        row[9],
        bool(row[10]),
        row[11],
        bool(row[12]),
    )
