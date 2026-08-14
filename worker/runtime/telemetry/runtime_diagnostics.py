"""Worker-local diagnostics and frozen runtime-status wire projection."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from time import monotonic, time
from typing import final

from contracts.decode_diagnostics import DECODE_FALLBACK_REASONS, DecodeSelection
from contracts.encode_diagnostics import ENCODE_FALLBACK_REASONS, EncodeSelection
from contracts.observation import BedRegionCacheState
from worker.pipeline.perception.scene_state import BedRegionCacheCounterSnapshot
from worker.runtime.telemetry.local_metrics import (
    StageTimingAccumulator,
    bus_snapshot,
    log_snapshot,
)
from worker.runtime.telemetry.models import (
    BedExitScoringDiagnostics,
    BedRegionDiagnostics,
    BusMetricsSource,
    BusSubscriptionSnapshot,
    CameraDiagnosticsSnapshot,
    DeviceResidencyDiagnostics,
    EncoderLifecycleSnapshot,
    InvalidStageTimingError,
    RuntimeDiagnosticsSnapshot,
    StageTimingSnapshot,
    SubscriptionMetrics,
)
from worker.runtime.telemetry.status_store import StatusStore
from worker.runtime.telemetry.wire import (
    ClipRecorderStatus,
    RelayCameraPayload,
    RelayClipExportPayload,
    RelayClipRecorderPayload,
    RelayDecodePayload,
    RelayGpuPayload,
    RelayRuntimeStatusPayload,
    RelayWorkerPayload,
    camera_payload,
    facility_payload,
)

MEASURED_FPS_MAX_AGE_SEC = 10.0


@final
class WorkerDiagnostics:
    """Thread-safe local metrics and strict legacy relay projection."""

    def __init__(
        self,
        status_store: StatusStore | None = None,
        clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], float] = time,
    ) -> None:
        self._lock = threading.RLock()
        self._status_store = StatusStore() if status_store is None else status_store
        self._clock = clock
        self._wall_clock = wall_clock
        self._decode_by_camera: dict[str, DecodeSelection] = {}
        self._encode_by_camera: dict[str, EncodeSelection] = {}
        self._measured_fps_by_camera: dict[str, tuple[float, float | None]] = {}
        self._stage_timings: dict[str, dict[str, StageTimingAccumulator]] = {}
        self._buses: dict[str, tuple[BusMetricsSource, tuple[str, ...]]] = {}
        self._bed_region_by_camera: dict[str, BedRegionDiagnostics] = {}
        self._bed_exit_scoring_by_camera: dict[str, BedExitScoringDiagnostics] = {}
        self._device_residency_by_camera: dict[str, DeviceResidencyDiagnostics] = {}
        self._encoder = EncoderLifecycleSnapshot()
        self._clip_recorder = ClipRecorderStatus()
        self._clip_export = RelayClipExportPayload(enabled=False, version=0)
        self._gpu: RelayGpuPayload | None = None
        self._worker: RelayWorkerPayload | None = None

    def set_clip_recorder_status(self, status: ClipRecorderStatus) -> None:
        with self._lock:
            self._clip_recorder = status

    def set_clip_export_applied(self, *, enabled: bool, version: int) -> None:
        with self._lock:
            self._clip_export = RelayClipExportPayload(enabled=enabled, version=version)

    def set_gpu_status(self, status: RelayGpuPayload | None) -> None:
        with self._lock:
            self._gpu = None if status is None else status.copy()

    def set_worker_status(self, status: RelayWorkerPayload | None) -> None:
        with self._lock:
            self._worker = None if status is None else status.copy()

    def register_decode(self, camera_id: str, requested: str) -> None:
        self.update_decode(
            camera_id,
            DecodeSelection(
                requested=requested,
                selected=None,
                fallback_count=0,
                last_reason=None,
                updated_at_sec=self._wall_clock(),
            ),
        )

    def update_decode(self, camera_id: str, selection: DecodeSelection) -> None:
        with self._lock:
            self._decode_by_camera[camera_id] = selection

    def record_decode_open_failure(self, camera_id: str, reason: str) -> None:
        normalized = reason if reason in DECODE_FALLBACK_REASONS else "spawn_failed"
        with self._lock:
            previous = self._decode_by_camera.get(camera_id)
            if previous is None:
                return
            self._decode_by_camera[camera_id] = DecodeSelection(
                requested=previous.requested,
                selected=None,
                fallback_count=previous.fallback_count,
                last_reason=normalized,
                updated_at_sec=self._wall_clock(),
            )

    def decode_selection(self, camera_id: str) -> DecodeSelection | None:
        with self._lock:
            return self._decode_by_camera.get(camera_id)

    def decode_snapshot(self) -> Mapping[str, DecodeSelection]:
        with self._lock:
            return dict(self._decode_by_camera)

    def register_encode(self, camera_id: str, requested: str) -> None:
        self.update_encode(
            camera_id,
            EncodeSelection(
                requested=requested,
                selected=requested,
                fallback_count=0,
                last_reason=None,
                updated_at_sec=self._wall_clock(),
            ),
        )

    def update_encode(self, camera_id: str, selection: EncodeSelection) -> None:
        with self._lock:
            self._encode_by_camera[camera_id] = selection

    def record_encode_open_failure(self, camera_id: str, reason: str) -> None:
        """Record a camera's nvenc session-open failure and its libx264 demotion.

        Unlike `record_decode_open_failure` (which sets `selected=None` --
        decode has nothing safe to fall back to), #53 sanctions libx264 as
        encode's always-available fallback, so this records the demotion
        itself rather than a "no selection" state.
        """
        normalized = reason if reason in ENCODE_FALLBACK_REASONS else "session_open_failed"
        with self._lock:
            previous = self._encode_by_camera.get(camera_id)
            requested = previous.requested if previous is not None else "h264_nvenc"
            fallback_count = 1 + (previous.fallback_count if previous is not None else 0)
            self._encode_by_camera[camera_id] = EncodeSelection(
                requested=requested,
                selected="libx264",
                fallback_count=fallback_count,
                last_reason=normalized,
                updated_at_sec=self._wall_clock(),
            )

    def encode_selection(self, camera_id: str) -> EncodeSelection | None:
        with self._lock:
            return self._encode_by_camera.get(camera_id)

    def encode_snapshot(self) -> Mapping[str, EncodeSelection]:
        with self._lock:
            return dict(self._encode_by_camera)

    def record_bed_region(
        self,
        camera_id: str,
        freshness: BedRegionCacheState,
        counters: BedRegionCacheCounterSnapshot,
    ) -> None:
        """Refresh one camera's bed-region state (issue #207).

        Called once per processed frame from ``CompositeExtractor`` (see
        ``BedRegionRecorder`` in worker/pipeline/analytics/composite.py) --
        like ``record_stage_timing``, this only updates an in-memory value;
        it is not itself a log call, so per-frame frequency here does not
        reproduce the "per-frame logging across 13 cameras" outage the issue
        warns against. Actual emission is on `log_snapshot()`'s cadence.
        """
        with self._lock:
            self._bed_region_by_camera[camera_id] = BedRegionDiagnostics(
                freshness=freshness,
                counters=counters,
                updated_at_sec=self._wall_clock(),
            )

    def bed_region_selection(self, camera_id: str) -> BedRegionDiagnostics | None:
        with self._lock:
            return self._bed_region_by_camera.get(camera_id)

    def bed_region_snapshot(self) -> Mapping[str, BedRegionDiagnostics]:
        with self._lock:
            return dict(self._bed_region_by_camera)

    def record_bed_exit_scoring(
        self,
        camera_id: str,
        max_containment_observed: float,
        grace_positive_transitions: int,
        assignments_made: int,
    ) -> None:
        """Refresh one camera's cumulative bed_exit scoring signal (#238).

        Called from ``BedExitMonitor.update()`` (see ``BedExitScoringRecorder``
        in worker/domains/bed_exit/detector.py) once per processed frame --
        like ``record_bed_region``, this only overwrites an in-memory value;
        it is not itself a log call. Actual emission is on `log_snapshot()`'s
        cadence.
        """
        with self._lock:
            self._bed_exit_scoring_by_camera[camera_id] = BedExitScoringDiagnostics(
                max_containment_observed=max_containment_observed,
                grace_positive_transitions=grace_positive_transitions,
                assignments_made=assignments_made,
                updated_at_sec=self._wall_clock(),
            )

    def bed_exit_scoring_selection(self, camera_id: str) -> BedExitScoringDiagnostics | None:
        with self._lock:
            return self._bed_exit_scoring_by_camera.get(camera_id)

    def bed_exit_scoring_snapshot(self) -> Mapping[str, BedExitScoringDiagnostics]:
        with self._lock:
            return dict(self._bed_exit_scoring_by_camera)

    def record_device_residency(
        self, camera_id: str, diagnostics: DeviceResidencyDiagnostics
    ) -> None:
        """Refresh one camera's device-resident pipeline counters (Todo 17).

        Only ever called for a camera running the opt-in
        ``nvidia-device-experimental`` profile -- same overwrite-in-place,
        emission-on-``log_snapshot``-cadence convention as
        ``record_bed_region``/``record_bed_exit_scoring`` above.
        """
        with self._lock:
            self._device_residency_by_camera[camera_id] = diagnostics

    def device_residency_selection(self, camera_id: str) -> DeviceResidencyDiagnostics | None:
        with self._lock:
            return self._device_residency_by_camera.get(camera_id)

    def device_residency_snapshot(self) -> Mapping[str, DeviceResidencyDiagnostics]:
        with self._lock:
            return dict(self._device_residency_by_camera)

    def update_measured_fps(self, camera_id: str, measured_fps: float | None) -> None:
        with self._lock:
            self._measured_fps_by_camera[camera_id] = (self._clock(), measured_fps)

    def record_stage_timing(self, camera_id: str, stage: str, elapsed_sec: float) -> None:
        if elapsed_sec < 0:
            raise InvalidStageTimingError(elapsed_sec)
        with self._lock:
            camera_stages = self._stage_timings.setdefault(camera_id, {})
            camera_stages.setdefault(stage, StageTimingAccumulator()).add(elapsed_sec)

    def register_bus(
        self,
        camera_id: str,
        bus: BusMetricsSource,
        subscriptions: tuple[str, ...] = ("inference", "live", "evidence"),
    ) -> None:
        with self._lock:
            self._buses[camera_id] = (bus, subscriptions)

    def update_encoder_lifecycle(self, snapshot: EncoderLifecycleSnapshot) -> None:
        with self._lock:
            self._encoder = EncoderLifecycleSnapshot(
                process_starts=snapshot.process_starts,
                recreates=snapshot.recreates,
                failures=snapshot.failures,
                active_sessions=snapshot.active_sessions,
                finalized_segments=snapshot.finalized_segments,
                unavailable_cameras=tuple(sorted(snapshot.unavailable_cameras)),
            )

    def snapshot(self) -> RuntimeDiagnosticsSnapshot:
        statuses = {record.camera_id: record for record in self._status_store.snapshot().cameras}
        with self._lock:
            stage_timings = {
                camera_id: tuple(stages[stage].snapshot(stage) for stage in sorted(stages))
                for camera_id, stages in self._stage_timings.items()
            }
            buses = dict(self._buses)
            encoder = self._encoder
            encode_by_camera = dict(self._encode_by_camera)
            bed_region_by_camera = dict(self._bed_region_by_camera)
            bed_exit_scoring_by_camera = dict(self._bed_exit_scoring_by_camera)
            device_residency_by_camera = dict(self._device_residency_by_camera)
            camera_ids = (
                set(self._decode_by_camera)
                | set(stage_timings)
                | set(buses)
                | set(statuses)
                | set(encode_by_camera)
                | set(bed_region_by_camera)
                | set(bed_exit_scoring_by_camera)
                | set(device_residency_by_camera)
            )
        cameras = tuple(
            CameraDiagnosticsSnapshot(
                camera_id=camera_id,
                failure_category=(
                    None if camera_id not in statuses else statuses[camera_id].error_category
                ),
                stage_timings=stage_timings.get(camera_id, ()),
                bus=bus_snapshot(buses.get(camera_id)),
                encode=encode_by_camera.get(camera_id),
                bed_region=bed_region_by_camera.get(camera_id),
                bed_exit_scoring=bed_exit_scoring_by_camera.get(camera_id),
                device_residency=device_residency_by_camera.get(camera_id),
            )
            for camera_id in sorted(camera_ids)
        )
        return RuntimeDiagnosticsSnapshot(cameras=cameras, encoder=encoder)

    def log_snapshot(self) -> None:
        log_snapshot(self.snapshot())

    def to_payload(
        self,
        facility_id: str,
        generation: int | None,
        seq: int,
    ) -> RelayRuntimeStatusPayload:
        selections, measured_fps, clip_recorder, clip_export, gpu, worker = self._wire_inputs()
        cameras = [
            camera_payload(camera_id, selection, measured_fps.get(camera_id))
            for camera_id, selection in sorted(selections.items())
        ]
        return facility_payload(
            facility_id,
            generation,
            seq,
            cameras,
            clip_recorder,
            clip_export,
            gpu,
            worker,
        )

    def to_payloads(
        self,
        camera_facilities: Mapping[str, str],
        generation: int | None,
        seq: int,
    ) -> list[RelayRuntimeStatusPayload]:
        selections, measured_fps, clip_recorder, clip_export, gpu, worker = self._wire_inputs()
        cameras_by_facility: dict[str, list[RelayCameraPayload]] = {
            facility_id: [] for facility_id in set(camera_facilities.values())
        }
        for camera_id, selection in selections.items():
            facility_id = camera_facilities.get(camera_id)
            if facility_id is not None:
                cameras_by_facility[facility_id].append(
                    camera_payload(camera_id, selection, measured_fps.get(camera_id))
                )
        return [
            facility_payload(
                facility_id,
                generation,
                seq,
                cameras,
                clip_recorder,
                clip_export,
                gpu,
                worker,
            )
            for facility_id, cameras in sorted(cameras_by_facility.items())
        ]

    def _wire_inputs(
        self,
    ) -> tuple[
        dict[str, DecodeSelection],
        dict[str, float | None],
        ClipRecorderStatus,
        RelayClipExportPayload,
        RelayGpuPayload | None,
        RelayWorkerPayload | None,
    ]:
        with self._lock:
            selections = dict(self._decode_by_camera)
            measured_fps = {
                camera_id: measured
                for camera_id, (updated_at, measured) in self._measured_fps_by_camera.items()
                if self._clock() - updated_at <= MEASURED_FPS_MAX_AGE_SEC
            }
            clip_recorder = self._clip_recorder
            clip_export = self._clip_export.copy()
            gpu = None if self._gpu is None else self._gpu.copy()
            worker = None if self._worker is None else self._worker.copy()
        return selections, measured_fps, clip_recorder, clip_export, gpu, worker


__all__ = [
    "BedExitScoringDiagnostics",
    "BedRegionDiagnostics",
    "BusMetricsSource",
    "BusSubscriptionSnapshot",
    "CameraDiagnosticsSnapshot",
    "ClipRecorderStatus",
    "DeviceResidencyDiagnostics",
    "EncodeSelection",
    "EncoderLifecycleSnapshot",
    "RelayCameraPayload",
    "RelayClipExportPayload",
    "RelayClipRecorderPayload",
    "RelayDecodePayload",
    "RelayGpuPayload",
    "RelayRuntimeStatusPayload",
    "RelayWorkerPayload",
    "RuntimeDiagnosticsSnapshot",
    "StageTimingSnapshot",
    "SubscriptionMetrics",
    "WorkerDiagnostics",
]
