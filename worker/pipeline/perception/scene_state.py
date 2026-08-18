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
    # Only ever used as an existence sentinel (None vs not-None) since the
    # frame-index age comparison was deleted -- see resolve_bed_regions'
    # docstring. The actual frame number is no longer read anywhere.
    last_bed_frame_index: int | None = None
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
        """Store one inferred observation without discarding the bed cache."""
        self.latest_observation = observation
        self.track_ids = track_ids
        if observation.bed_boxes:
            self.bed_regions = observation.bed_boxes
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
        """Resolve fresh, cached, empty, or expired bed regions for one frame.

        ``scheduled_empty_bed_cycles >= 2`` is the sole invalidation
        mechanism: two consecutive scheduled segmentation cycles that find no
        bed is the only evidence this cache treats as "the bed is actually
        gone." There is deliberately no frame-index or wall-clock TTL on top
        of it.

        This repo's ingest lifecycle does not call ``reset_for_new_source()``
        on an RTSP reconnect (see #208), and a reconnected decode session
        typically restarts its frame counter near 0. An earlier version of
        this method additionally reset the cache whenever ``frame_index``
        stopped increasing (``_reset_if_discontinuous``) and gated the cache
        on ``frame_index - last_bed_frame_index`` staying within a bounded
        window (``_cached_boxes_if_fresh``'s age comparison). Both read
        ``frame_index`` as if it were a single monotonic counter for the
        camera's whole lifetime, when it is actually only monotonic *within
        one decode session*. Every reconnect made both of them misfire and
        wipe an otherwise-good cache -- which is indistinguishable, to
        everything downstream, from the bed genuinely being empty. Neither
        mechanism is reachable from bed_interval or persisted_bed_regions, so
        deleting both leaves this method's behavior unchanged across a
        reconnect and unchanged for every camera that isn't reconnecting.

        ``bed_interval`` is accepted for backward-compatible call
        compatibility (the scheduler already owns bed segmentation cadence
        via ``Scheduler.task_intervals["bed"]``) but is no longer read here.
        """
        if self.persisted_bed_regions:
            resolved = _replace_bed_boxes(observation, self.persisted_bed_regions)
            snapshot = BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH, empty_cycles=0)
            self._mark_processed(resolved)
            return resolved, snapshot

        if bed_scheduled and observation.bed_boxes:
            self.bed_regions = observation.bed_boxes
            self.last_bed_frame_index = frame_index
            self.scheduled_empty_bed_cycles = 0
            self.bed_region_freshness = BedRegionCacheState.FRESH
            self.bed_region_counters.fresh += 1
            snapshot = BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH, empty_cycles=0)
            self._mark_processed(observation)
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
                    empty_cycles=empty_cycles,
                )
                self._mark_processed(resolved)
                return resolved, snapshot

            self.bed_region_freshness = BedRegionCacheState.EMPTY
            snapshot = BedRegionDebugSnapshot(
                source=BedRegionCacheState.EMPTY,
                empty_cycles=self.scheduled_empty_bed_cycles,
            )
            self._mark_processed(observation)
            return observation, snapshot

        cached = self._cached_boxes_if_fresh()
        if cached:
            resolved = _replace_bed_boxes(observation, cached)
            self.bed_region_freshness = BedRegionCacheState.CACHED
            self.bed_region_counters.cached += 1
            snapshot = BedRegionDebugSnapshot(
                source=BedRegionCacheState.CACHED,
                empty_cycles=self.scheduled_empty_bed_cycles,
            )
            self._mark_processed(resolved)
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
            empty_cycles=self.scheduled_empty_bed_cycles,
        )
        self._mark_processed(resolved)
        return resolved, snapshot

    def _cached_boxes_if_fresh(self) -> tuple[BoundingBox, ...]:
        """Serve the last detected bed boxes as long as nothing has proven them gone.

        No frame-index or wall-clock age check: ``scheduled_empty_bed_cycles``
        is the only signal that invalidates this. See ``resolve_bed_regions``.
        """
        if not self.bed_regions or self.last_bed_frame_index is None:
            return ()
        if self.scheduled_empty_bed_cycles >= 2:
            return ()
        return self.bed_regions

    def _expire_bed_cache(self) -> None:
        self.bed_regions = ()
        self.last_bed_frame_index = None
        self.bed_region_freshness = BedRegionCacheState.EXPIRED
        self.bed_region_counters.expired += 1

    def _mark_processed(self, observation: FrameObservation) -> None:
        _ = self.update(observation)


def _replace_bed_boxes(
    observation: FrameObservation,
    bed_boxes: tuple[BoundingBox, ...],
) -> FrameObservation:
    return replace(observation, regions=(bed_boxes, observation.bed_exit_statuses))


__all__ = ["BedRegionCacheCounters", "SceneState"]
