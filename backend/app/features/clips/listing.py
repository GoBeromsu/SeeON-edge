"""Event-type facets shared by compact clip listing and receipt projection."""

from __future__ import annotations

from typing import TypeAlias

from backend.app.features.clips.schemas import ClipEventType
from backend.app.features.clips.store import ClipManifest

EventTypeFacet: TypeAlias = ClipEventType
_EVENT_TYPE_FACETS: dict[str, ClipEventType] = {
    "fall": "fall",
    "bed-exit": "bed-exit",
}


def event_type_facet(event_type: str | None) -> EventTypeFacet:
    return _EVENT_TYPE_FACETS.get(event_type or "", "other")


def effective_event_type(manifest: ClipManifest) -> EventTypeFacet:
    return event_type_facet(manifest.event_type or manifest.event_ref)


__all__ = ["EventTypeFacet", "effective_event_type", "event_type_facet"]
