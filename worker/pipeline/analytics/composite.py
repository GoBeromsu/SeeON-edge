from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    FrameObservation,
)
from worker.pipeline.analytics.merge import authoritative_boxes, merge_module_results
from worker.pipeline.analytics.models import NamedExtractor, ensure_unique_module_names
from worker.pipeline.bus import Scheduler
from worker.pipeline.perception import (
    GreedyIouTracker,
    SceneState,
    build_decision_input,
    build_frame_observation,
)
from worker.pipeline.perception.decision_input import bed_pose_features_for
from worker.pipeline.perception.scene_state import BedRegionCacheCounterSnapshot
from worker.types import CURRENT_TEMPORAL_PROFILE, DecisionInput, FramePacket, ModuleResult

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompositeResult:
    module_results: tuple[ModuleResult, ...]
    observation: FrameObservation
    decision_input: DecisionInput


class InferenceGuard(Protocol):
    """Structural view of ``InferenceWatchdog.guard()``.

    Pipeline code depends only on this narrow shape rather than importing
    ``worker.runtime.watchdog.InferenceWatchdog`` directly, so the pipeline
    layer never has to import from the runtime layer that composes it.
    """

    def guard(
        self,
        *,
        camera_id: str,
        task: str,
        frame_index: int | None = None,
        deadline_sec: float | None = None,
        model_artifact_digest: str | None = None,
    ) -> AbstractContextManager[int]: ...


class StageTimingRecorder(Protocol):
    """Structural view of ``WorkerDiagnostics.record_stage_timing()``.

    Kept narrow for the same layering reason as ``InferenceGuard``: the
    pipeline layer depends on this shape instead of importing
    ``worker.runtime.telemetry.runtime_diagnostics.WorkerDiagnostics``.
    """

    def record_stage_timing(self, camera_id: str, stage: str, elapsed_sec: float) -> None: ...


class BedRegionRecorder(Protocol):
    """Structural view of ``WorkerDiagnostics.record_bed_region()``.

    Kept narrow for the same layering reason as ``StageTimingRecorder``
    above (issue #207): the pipeline layer depends on this shape instead of
    importing ``worker.runtime.telemetry.runtime_diagnostics.WorkerDiagnostics``.
    """

    def record_bed_region(
        self,
        camera_id: str,
        freshness: BedRegionCacheState,
        counters: BedRegionCacheCounterSnapshot,
    ) -> None: ...


class CompositeExtractor:
    """Own one camera's analytics state while reusing shared named extractors."""

    def __init__(
        self,
        *,
        extractors: Sequence[NamedExtractor],
        scheduler: Scheduler,
        tracker: GreedyIouTracker,
        scene_state: SceneState,
        watchdog: InferenceGuard | None = None,
        stage_timing_recorder: StageTimingRecorder | None = None,
        bed_region_recorder: BedRegionRecorder | None = None,
    ) -> None:
        frozen_extractors = tuple(extractors)
        ensure_unique_module_names(tuple(extractor.module_name for extractor in frozen_extractors))
        self.extractors: tuple[NamedExtractor, ...] = frozen_extractors
        self.scheduler: Scheduler = scheduler
        self.tracker: GreedyIouTracker = tracker
        self.scene_state: SceneState = scene_state
        self._extractors_by_name: dict[str, NamedExtractor] = {
            extractor.module_name: extractor for extractor in frozen_extractors
        }
        self._watchdog = watchdog
        self._stage_timing_recorder = stage_timing_recorder
        self._bed_region_recorder = bed_region_recorder

    def _extract(self, extractor: NamedExtractor, packet: FramePacket) -> ModuleResult:
        """Run one module's forward pass, guarded against a hung driver.

        Without an injected watchdog this is a direct call -- tests and other
        non-production compositions that never pass ``watchdog=`` behave
        exactly as before. The already-computed ``elapsed_ms`` is then handed
        to an injected stage-timing recorder, when one is present, so per-stage
        latency is visible without recomputing it a second time.
        """
        if self._watchdog is None:
            result = extractor.extract(packet)
        else:
            with self._watchdog.guard(
                camera_id=self.scene_state.camera_id,
                task=extractor.module_name,
                frame_index=packet.frame.index,
            ):
                result = extractor.extract(packet)
        if self._stage_timing_recorder is not None:
            # Telemetry is reporting, never detection. Unguarded it sat directly
            # on the extraction path, so a raising recorder discarded the result
            # of a frame that may have carried a fall.
            try:
                self._stage_timing_recorder.record_stage_timing(
                    self.scene_state.camera_id, result.module_name, result.elapsed_ms / 1000.0
                )
            except Exception:  # noqa: BLE001 - telemetry never blocks detection
                _LOGGER.warning(
                    "stage-timing recorder failed for camera %s; detection continues",
                    self.scene_state.camera_id,
                    exc_info=True,
                )
        return result

    def process(
        self,
        packet: FramePacket,
        *,
        prefetched_results: Sequence[ModuleResult] = (),
    ) -> CompositeResult:
        """Merge coordinator-owned results with due per-camera extractors."""
        scheduled_names = self.scheduler.tasks_for_frame(packet.frame.index)
        prefetched = tuple(
            result for result in prefetched_results if result.module_name in scheduled_names
        )
        prefetched_names = {result.module_name for result in prefetched}
        module_results = prefetched + tuple(
            self._extract(extractor, packet)
            for name in scheduled_names
            if name not in prefetched_names
            and (extractor := self._extractors_by_name.get(name)) is not None
        )
        pose_observed = any(result.module_name == "pose" for result in module_results)
        merged = merge_module_results(module_results)
        if pose_observed:
            track_ids = self.tracker.observe(authoritative_boxes(merged))
            observation = build_frame_observation(
                detections=merged.detections,
                raw_boxes=merged.raw_boxes,
                poses=merged.poses,
                bed_boxes=merged.bed_boxes,
                track_ids=track_ids,
            )
            decision_input = build_decision_input(
                observation,
                frame_width=packet.width,
                frame_height=packet.height,
                live_track_ids=tuple(sorted(self.tracker.live_ids)),
                time_sec=packet.frame.time_sec,
                frame_index=packet.frame.index,
                scene_state=self.scene_state,
                bed_scheduled="bed" in scheduled_names,
                bed_interval=self.scheduler.task_intervals.get(
                    "bed", CURRENT_TEMPORAL_PROFILE.decision_interval_frames("bed")
                ),
            )
            final_observation = decision_input.observation
            _ = self.scene_state.observe(final_observation, track_ids=track_ids)
        else:
            self.tracker.coast()
            final_observation = self.scene_state.coast() or FrameObservation()
            decision_input = DecisionInput(
                observation=final_observation,
                frame_width=packet.width,
                frame_height=packet.height,
                live_track_ids=tuple(sorted(self.tracker.live_ids)),
                time_sec=packet.frame.time_sec,
                frame_index=packet.frame.index,
                bed_region=BedRegionDebugSnapshot(
                    source=self.scene_state.bed_region_freshness,
                    empty_cycles=self.scene_state.scheduled_empty_bed_cycles,
                ),
                bed_pose_features=bed_pose_features_for(
                    final_observation,
                    frame_width=packet.width,
                    frame_height=packet.height,
                    scene_state=self.scene_state,
                ),
            )
        if self._bed_region_recorder is not None:
            # `decision_input.bed_region.source` (a `BedRegionDebugSnapshot`)
            # is this frame's actual resolved state, not
            # `scene_state.bed_region_freshness` -- on the persisted-polygon
            # short-circuit (`resolve_bed_regions`, ~10/13 live cameras),
            # `resolve_bed_regions` returns FRESH every frame but never
            # touches `bed_region_freshness`/`bed_region_counters` at all, so
            # reading those directly would misreport those cameras as
            # permanently EMPTY (#207). `bed_region_counters` itself only
            # advances on the live-detection cache path; staying all-zero
            # forever on a persisted-polygon camera is correct and, paired
            # with a constant FRESH source, is itself informative --  it
            # marks the cache mechanism as never engaged rather than idle.
            #
            # Like `record_stage_timing` above, this only refreshes an
            # in-memory value the recorder holds -- it is not itself a log
            # call, so doing it every frame does not reproduce the
            # "per-frame logging across 13 cameras" outage the issue warns
            # against; the recorder decides its own emission cadence.
            try:
                self._bed_region_recorder.record_bed_region(
                    self.scene_state.camera_id,
                    decision_input.bed_region.source,
                    self.scene_state.bed_region_counters.snapshot(),
                )
            except Exception:  # noqa: BLE001 - telemetry never blocks detection
                _LOGGER.warning(
                    "bed-region recorder failed for camera %s; detection continues",
                    self.scene_state.camera_id,
                    exc_info=True,
                )
        return CompositeResult(
            module_results=module_results,
            observation=final_observation,
            decision_input=decision_input,
        )


__all__ = [
    "BedRegionRecorder",
    "CompositeExtractor",
    "CompositeResult",
    "InferenceGuard",
    "StageTimingRecorder",
]
