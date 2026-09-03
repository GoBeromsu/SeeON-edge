from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TypedDict

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)


class BedRegionCacheCounterSnapshot(TypedDict):
    fresh: int
    cached: int
    expired: int
    reset: int
    scheduled_empty: int


@dataclass(slots=True)  # policy: MUTABLE_OK
class BedRegionCacheCounters:
    """Mutable telemetry counters owned by one camera's scene state."""

    fresh: int = 0
    cached: int = 0
    expired: int = 0
    reset: int = 0
    scheduled_empty: int = 0

    def snapshot(self) -> BedRegionCacheCounterSnapshot:
        return {
            "fresh": self.fresh,
            "cached": self.cached,
            "expired": self.expired,
            "reset": self.reset,
            "scheduled_empty": self.scheduled_empty,
        }


@dataclass(slots=True)  # policy: MUTABLE_OK
class SceneState:
    """Mutable per-camera owner of the latest observation and persisted bed polygon."""

    camera_id: str
    latest_observation: FrameObservation | None = None
    track_ids: tuple[int, ...] = field(default_factory=tuple)
    scheduled_empty_bed_cycles: int = 0
    bed_region_freshness: BedRegionCacheState = BedRegionCacheState.EMPTY
    bed_region_counters: BedRegionCacheCounters = field(default_factory=BedRegionCacheCounters)
    # Set once at camera-build time from `CameraRuntimeConfig.bed_zone_polygon`
    # and never mutated from a per-frame model result.
    persisted_bed_regions: tuple[BoundingBox, ...] = field(default_factory=tuple)
    # Source image size of ``persisted_bed_regions`` polygons. Poses arrive in
    # frame_width x frame_height; these are not guaranteed to match. None means
    # the stored polygon is already in frame space.
    bed_zone_image_width: int | None = None
    bed_zone_image_height: int | None = None

    def reset_bed_cache(self, _reason: str) -> None:
        """Retain the persisted polygon across a source lifecycle reset."""

    def reset_for_new_source(self, reason: str = "source_restart") -> None:
        self.reset_bed_cache(reason)
        self.latest_observation = None
        self.track_ids = ()

    def update(
        self,
        observation: FrameObservation,
        *,
        track_ids: tuple[int, ...] = (),
    ) -> FrameObservation:
        """Compatibility alias for an actual inference observation."""
        return self.observe(observation, track_ids=track_ids)

    def observe(
        self,
        observation: FrameObservation,
        *,
        track_ids: tuple[int, ...] = (),
    ) -> FrameObservation:
        """Store one inferred observation."""
        self.latest_observation = observation
        self.track_ids = track_ids
        return observation

    def coast(self) -> FrameObservation | None:
        """Return the last inferred scene without advancing it as empty."""
        return self.latest_observation

    def resolve_bed_regions(
        self,
        observation: FrameObservation,
        *,
        frame_index: int,
        bed_scheduled: bool,
        bed_interval: int,
    ) -> tuple[FrameObservation, BedRegionDebugSnapshot]:
        """Replace all per-frame bed output with the persisted polygon."""
        _ = frame_index, bed_scheduled, bed_interval
        if self.persisted_bed_regions:
            resolved = _replace_bed_boxes(observation, self.persisted_bed_regions)
            snapshot = BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH, empty_cycles=0)
            self.bed_region_freshness = snapshot.source
            self._mark_processed(resolved)
            return resolved, snapshot

        resolved = _replace_bed_boxes(observation, ())
        snapshot = BedRegionDebugSnapshot(source=BedRegionCacheState.EMPTY, empty_cycles=0)
        self.bed_region_freshness = snapshot.source
        self._mark_processed(resolved)
        return resolved, snapshot

    @property
    def bed_polygon_source(self) -> str:
        """Actual provenance of the polygon offered to bed decisions."""
        return "persisted" if self.persisted_bed_regions else "none"

    def _mark_processed(self, observation: FrameObservation) -> None:
        _ = self.update(observation)


def _replace_bed_boxes(
    observation: FrameObservation,
    bed_boxes: tuple[BoundingBox, ...],
) -> FrameObservation:
    return replace(observation, regions=(bed_boxes, observation.bed_exit_statuses))


__all__ = ["BedRegionCacheCounters", "SceneState"]
