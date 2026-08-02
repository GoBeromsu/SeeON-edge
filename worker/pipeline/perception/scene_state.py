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


@dataclass(slots=True)  # noqa: MUTABLE_OK
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


@dataclass(slots=True)  # noqa: MUTABLE_OK
class SceneState:
    """Mutable per-camera owner of the latest observation and bounded bed cache."""

    camera_id: str
    latest_observation: FrameObservation | None = None
    track_ids: tuple[int, ...] = field(default_factory=tuple)
    bed_regions: tuple[BoundingBox, ...] = field(default_factory=tuple)
    last_bed_frame_index: int | None = None
    last_processed_frame_index: int | None = None
    scheduled_empty_bed_cycles: int = 0
    bed_region_freshness: BedRegionCacheState = BedRegionCacheState.EMPTY
    bed_region_counters: BedRegionCacheCounters = field(default_factory=BedRegionCacheCounters)
    # An operator-recognized, persisted bed polygon (see the bed-zone
    # recognize endpoint) is authoritative whenever set: it never expires and
    # always wins over the live scheduled/cached segmentation cycle below.
    # Set once at camera-build time from `CameraRuntimeConfig.bed_zone_polygon`
    # -- never mutated per-frame.
    persisted_bed_regions: tuple[BoundingBox, ...] = field(default_factory=tuple)

    def reset_bed_cache(self, _reason: str) -> None:
        self.bed_regions = ()
        self.last_bed_frame_index = None
        self.scheduled_empty_bed_cycles = 0
        self.bed_region_freshness = BedRegionCacheState.EMPTY
        self.bed_region_counters.reset += 1

    def reset_for_new_source(self, reason: str = "source_restart") -> None:
        self.reset_bed_cache(reason)
        self.last_processed_frame_index = None
        self.latest_observation = None
        self.track_ids = ()

    def update(
        self,
        observation: FrameObservation,
        *,
        track_ids: tuple[int, ...] = (),
    ) -> FrameObservation:
        """Store one observation without discarding an existing non-empty bed cache."""
        self.latest_observation = observation
        self.track_ids = track_ids
        if observation.bed_boxes:
            self.bed_regions = observation.bed_boxes
        return observation

    def resolve_bed_regions(
        self,
        observation: FrameObservation,
        *,
        frame_index: int,
        bed_scheduled: bool,
        bed_interval: int,
    ) -> tuple[FrameObservation, BedRegionDebugSnapshot]:
        """Resolve fresh, cached, empty, or expired bed regions for one frame."""
        if self.persisted_bed_regions:
            resolved = _replace_bed_boxes(observation, self.persisted_bed_regions)
            snapshot = BedRegionDebugSnapshot(
                source=BedRegionCacheState.FRESH,
                age_frames=0,
                empty_cycles=0,
                reset_reason=None,
            )
            self._mark_processed(frame_index, resolved)
            return resolved, snapshot

        reset_reason = self._reset_if_discontinuous(frame_index)
        if bed_scheduled and observation.bed_boxes:
            self.bed_regions = observation.bed_boxes
            self.last_bed_frame_index = frame_index
            self.scheduled_empty_bed_cycles = 0
            self.bed_region_freshness = BedRegionCacheState.FRESH
            self.bed_region_counters.fresh += 1
            snapshot = BedRegionDebugSnapshot(
                source=BedRegionCacheState.FRESH,
                age_frames=0,
                empty_cycles=0,
                reset_reason=reset_reason,
            )
            self._mark_processed(frame_index, observation)
            return observation, snapshot

        if bed_scheduled:
            self.scheduled_empty_bed_cycles += 1
            self.bed_region_counters.scheduled_empty += 1
            if self.scheduled_empty_bed_cycles >= 2:
                empty_cycles = self.scheduled_empty_bed_cycles
                self._expire_bed_cache()
                resolved = _replace_bed_boxes(observation, ())
                snapshot = BedRegionDebugSnapshot(
                    source=BedRegionCacheState.EXPIRED,
                    age_frames=None,
                    empty_cycles=empty_cycles,
                    reset_reason=reset_reason,
                )
                self._mark_processed(frame_index, resolved)
                return resolved, snapshot

            self.bed_region_freshness = BedRegionCacheState.EMPTY
            snapshot = BedRegionDebugSnapshot(
                source=BedRegionCacheState.EMPTY,
                age_frames=self._age(frame_index),
                empty_cycles=self.scheduled_empty_bed_cycles,
                reset_reason=reset_reason,
            )
            self._mark_processed(frame_index, observation)
            return observation, snapshot

        cached = self._cached_boxes_if_fresh(frame_index, bed_interval)
        if cached:
            resolved = _replace_bed_boxes(observation, cached)
            self.bed_region_freshness = BedRegionCacheState.CACHED
            self.bed_region_counters.cached += 1
            snapshot = BedRegionDebugSnapshot(
                source=BedRegionCacheState.CACHED,
                age_frames=self._age(frame_index),
                empty_cycles=self.scheduled_empty_bed_cycles,
                reset_reason=reset_reason,
            )
            self._mark_processed(frame_index, resolved)
            return resolved, snapshot

        expired = bool(self.bed_regions) or self.last_bed_frame_index is not None
        source = BedRegionCacheState.EXPIRED if expired else BedRegionCacheState.EMPTY
        if expired:
            self._expire_bed_cache()
        else:
            self.bed_region_freshness = BedRegionCacheState.EMPTY
        resolved = _replace_bed_boxes(observation, ())
        snapshot = BedRegionDebugSnapshot(
            source=source,
            age_frames=None,
            empty_cycles=self.scheduled_empty_bed_cycles,
            reset_reason=reset_reason,
        )
        self._mark_processed(frame_index, resolved)
        return resolved, snapshot

    def _reset_if_discontinuous(self, frame_index: int) -> str | None:
        if self.last_processed_frame_index is None:
            return None
        if frame_index > self.last_processed_frame_index:
            return None
        self.reset_bed_cache("frame_index_discontinuity")
        return "frame_index_discontinuity"

    def _cached_boxes_if_fresh(
        self,
        frame_index: int,
        bed_interval: int,
    ) -> tuple[BoundingBox, ...]:
        if not self.bed_regions or self.last_bed_frame_index is None:
            return ()
        if self.scheduled_empty_bed_cycles >= 2:
            return ()
        age_frames = frame_index - self.last_bed_frame_index
        if age_frames < 0:
            return ()
        if age_frames <= 2 * max(1, bed_interval):
            return self.bed_regions
        return ()

    def _age(self, frame_index: int) -> int | None:
        if self.last_bed_frame_index is None:
            return None
        return max(0, frame_index - self.last_bed_frame_index)

    def _expire_bed_cache(self) -> None:
        self.bed_regions = ()
        self.last_bed_frame_index = None
        self.bed_region_freshness = BedRegionCacheState.EXPIRED
        self.bed_region_counters.expired += 1

    def _mark_processed(self, frame_index: int, observation: FrameObservation) -> None:
        self.last_processed_frame_index = frame_index
        _ = self.update(observation)


def _replace_bed_boxes(
    observation: FrameObservation,
    bed_boxes: tuple[BoundingBox, ...],
) -> FrameObservation:
    return replace(observation, regions=(bed_boxes, observation.bed_exit_statuses))


__all__ = ["BedRegionCacheCounters", "SceneState"]
