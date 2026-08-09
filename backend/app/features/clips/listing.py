"""Deterministic filtering, facets, and pagination for clip manifests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from backend.app.features.clips.schemas import ClipEventType, ClipListQuery
from backend.app.features.clips.store import ClipManifest

EventTypeFacet: TypeAlias = ClipEventType
_EVENT_TYPE_FACETS: dict[str, ClipEventType] = {
    "fall": "fall",
    "bed-exit": "bed-exit",
}


@dataclass(frozen=True, slots=True)
class ClipPage:
    manifests: tuple[ClipManifest, ...]
    total: int
    has_more: bool
    event_type_counts: Mapping[str, int]


def event_type_facet(event_type: str | None) -> EventTypeFacet:
    return _EVENT_TYPE_FACETS.get(event_type or "", "other")


def effective_event_type(manifest: ClipManifest) -> EventTypeFacet:
    return event_type_facet(manifest.event_type or manifest.event_ref)


def select_clip_page(manifests: Iterable[ClipManifest], query: ClipListQuery) -> ClipPage:
    camera_manifests = [
        manifest
        for manifest in manifests
        if query.camera_id is None or manifest.camera_id == query.camera_id
    ]
    event_type_counts = dict(
        sorted(Counter(effective_event_type(manifest) for manifest in camera_manifests).items())
    )
    filtered = [
        manifest
        for manifest in camera_manifests
        if query.event_type is None or effective_event_type(manifest) == query.event_type
    ]
    ordered = sorted(
        filtered,
        key=lambda manifest: (manifest.started_at, manifest.clip_id),
        reverse=True,
    )
    total = len(ordered)
    if query.limit is None:
        page = ordered
    else:
        page = ordered[query.offset : query.offset + query.limit]
    return ClipPage(
        manifests=tuple(page),
        total=total,
        has_more=query.offset + len(page) < total,
        event_type_counts=event_type_counts,
    )
