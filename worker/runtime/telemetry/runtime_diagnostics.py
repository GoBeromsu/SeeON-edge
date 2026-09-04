"""Worker-local diagnostics and frozen runtime-status wire projection."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from time import monotonic, time
from typing import final

from contracts.decode_diagnostics import DECODE_FALLBACK_REASONS, DecodeSelection
from contracts.encode_diagnostics import ENCODE_FALLBACK_REASONS, EncodeSelection
from contracts.observation import BedRegionCacheState
from worker.pipeline.inference_coordinator import CameraInferenceTelemetry
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
    DecodeBackendObservability,
    DeviceResidencyDiagnostics,
    EncoderLifecycleSnapshot,
    GeometryBatchHistogram,
    InferenceMetricsSource,
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
    RelayDetectionPayload,
    RelayGpuPayload,
    RelayRuntimeStatusPayload,
    RelayWorkerPayload,
    camera_payload,
    detection_payload,
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
        self._decode_backend_by_camera: dict[str, DecodeBackendObservability] = {}
        self._encode_by_camera: dict[str, EncodeSelection] = {}
        self._measured_fps_by_camera: dict[str, tuple[float, float | None]] = {}
        self._stage_timings: dict[str, dict[str, StageTimingAccumulator]] = {}
        self._buses: dict[str, tuple[BusMetricsSource, tuple[str, ...]]] = {}
        self._inference: InferenceMetricsSource | None = None
        self._bed_region_by_camera: dict[str, BedRegionDiagnostics] = {}
        self._bed_exit_scoring_by_camera: dict[str, BedExitScoringDiagnostics] = {}
        self._device_residency_by_camera: dict[str, DeviceResidencyDiagnostics] = {}
        self._decision_completed_by_camera: dict[str, int] = {}
        # Cameras whose detection results come from a producer other than the
        # host ``CapabilityInferenceCoordinator`` -- today the ``nvidia``
        # profile's ``NativePolicyPump``. Registered explicitly by the
        # composition root instead of being inferred from ``self._inference``
        # being ``None``: that fall-through silently reported every nvidia
        # camera as ``expected=False`` (rendered "detection disabled") no
        # matter what the producer was actually doing.
        self._native_detection_cameras: set[str] = set()
        # Real attempt count for the native producer. Without it the relay
        # payload had to synthesise admitted == completed, which pinned the
        # backend's recent_success_rate at 1.0 and hid every failed frame.
        self._native_attempts_by_camera: dict[str, int] = {}
        self._track_id_switches_by_camera: dict[str, int] = {}
        self._track_id_switches_absorbed_by_camera: dict[str, int] = {}
        self._replay_trace_write_failures_by_camera: dict[str, int] = {}
        self._bed_polygon_source_by_camera: dict[str, str] = {}
        self._resample_gap_rows_by_camera: dict[str, int] = {}
        self._fall_inference_device_by_camera: dict[str, str] = {}
        self._fall_unapplied_policy_threshold_by_camera: dict[str, float] = {}
        self._flow_recording_by_camera: dict[str, tuple[int, int, int]] = {}
        self._flow_nvenc_sessions_by_camera: dict[str, int] = {}
        self._flow_lifecycle_by_camera: dict[str, tuple[int, int]] = {}
        self._incident_managers: dict[str, object] = {}
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

    def record_resample_gap_rows(self, camera_id: str, count: int = 1) -> None:
        if count < 0:
            raise ValueError("resample gap count must be non-negative")
        with self._lock:
            self._resample_gap_rows_by_camera[camera_id] = (
                self._resample_gap_rows_by_camera.get(camera_id, 0) + count
            )

    def record_fall_inference_device(self, camera_id: str, device: str) -> None:
        if device != "cpu":
            raise ValueError("fall inference device must be cpu")
        with self._lock:
            self._fall_inference_device_by_camera[camera_id] = device

    def record_fall_unapplied_policy_threshold(self, camera_id: str, threshold: float) -> None:
        """Report an operator threshold that was received but is not applied."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("unapplied policy threshold must be a probability")
        with self._lock:
            self._fall_unapplied_policy_threshold_by_camera[camera_id] = threshold

    def record_flow_recording_counters(
        self, camera_id: str, *, extended: int, extension_raced: int, start_refused: int
    ) -> None:
        if min(extended, extension_raced, start_refused) < 0:
            raise ValueError("Flow recording counters must be non-negative")
        with self._lock:
            self._flow_recording_by_camera[camera_id] = (extended, extension_raced, start_refused)

    def record_flow_nvenc_sessions(self, camera_id: str, active: int) -> None:
        if active < 0:
            raise ValueError("Flow NVENC sessions must be non-negative")
        with self._lock:
            self._flow_nvenc_sessions_by_camera[camera_id] = active

    def record_flow_lifecycle_counters(
        self, camera_id: str, *, outages: int, recoveries: int
    ) -> None:
        if min(outages, recoveries) < 0:
            raise ValueError("Flow lifecycle counters must be non-negative")
        with self._lock:
            self._flow_lifecycle_by_camera[camera_id] = (outages, recoveries)

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

    def record_decode_backend(
        self,
        camera_id: str,
        *,
        requested_profile_decode: str,
        resolved_backend: str,
        actual_adapter_class: str,
    ) -> None:
        """Record local-only boot selection details for one camera.

        ``DecodeSelection`` remains the relay-compatible view. The profile
        token and concrete adapter class are intentionally retained only in
        the local runtime snapshot.
        """
        with self._lock:
            self._decode_backend_by_camera[camera_id] = DecodeBackendObservability(
                requested_profile_decode=requested_profile_decode,
                resolved_backend=resolved_backend,
                actual_adapter_class=actual_adapter_class,
            )

    def decode_backend_snapshot(self) -> Mapping[str, DecodeBackendObservability]:
        with self._lock:
            return dict(self._decode_backend_by_camera)

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
        """Refresh one camera's device-resident pipeline counters.

        Only ever called for a camera running the canonical ``nvidia``
        profile -- same overwrite-in-place, emission-on-``log_snapshot``-
        cadence convention as ``record_bed_region``/
        ``record_bed_exit_scoring`` above.
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

    def record_detection_completed(self, camera_id: str) -> None:
        with self._lock:
            self._decision_completed_by_camera[camera_id] = (
                self._decision_completed_by_camera.get(camera_id, 0) + 1
            )

    def record_native_detection_attempt(self, camera_id: str) -> None:
        """Count one perception frame the native producer took responsibility for."""
        with self._lock:
            self._native_attempts_by_camera[camera_id] = (
                self._native_attempts_by_camera.get(camera_id, 0) + 1
            )

    def record_track_id_switch(self, camera_id: str) -> None:
        with self._lock:
            self._track_id_switches_by_camera[camera_id] = (
                self._track_id_switches_by_camera.get(camera_id, 0) + 1
            )

    def record_track_id_switch_absorbed_total(self, camera_id: str, total: int) -> None:
        if isinstance(total, bool) or total < 0:
            raise ValueError("absorbed track id switch total must be non-negative")
        with self._lock:
            self._track_id_switches_absorbed_by_camera[camera_id] = total

    def record_replay_trace_write_failure(self, camera_id: str) -> None:
        with self._lock:
            self._replay_trace_write_failures_by_camera[camera_id] = (
                self._replay_trace_write_failures_by_camera.get(camera_id, 0) + 1
            )

    def record_bed_polygon_source(self, camera_id: str, source: str) -> None:
        if source not in {"persisted", "native-per-frame", "none"}:
            raise ValueError("invalid bed polygon source")
        with self._lock:
            self._bed_polygon_source_by_camera[camera_id] = source

    def register_incident_manager(self, camera_id: str, manager: object) -> None:
        """Expose the manager's cumulative cooldown counter in local snapshots."""
        with self._lock:
            self._incident_managers[camera_id] = manager

    def register_native_detection(self, camera_id: str) -> None:
        """Declare that a non-host producer owns this camera's detection.

        The composition root calls this when it activates a producer that does
        not populate the host inference telemetry source, so the relay payload
        reports the producer as present instead of falling through to
        ``expected=False``.
        """
        with self._lock:
            self._native_detection_cameras.add(camera_id)

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

    def register_inference(self, source: InferenceMetricsSource) -> None:
        with self._lock:
            self._inference = source

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
            inference = None if self._inference is None else self._inference.snapshot()
            encoder = self._encoder
            decode_backend_by_camera = dict(self._decode_backend_by_camera)
            encode_by_camera = dict(self._encode_by_camera)
            bed_region_by_camera = dict(self._bed_region_by_camera)
            bed_exit_scoring_by_camera = dict(self._bed_exit_scoring_by_camera)
            device_residency_by_camera = dict(self._device_residency_by_camera)
            decision_completed_by_camera = dict(self._decision_completed_by_camera)
            track_id_switches_by_camera = dict(self._track_id_switches_by_camera)
            track_id_switches_absorbed_by_camera = dict(self._track_id_switches_absorbed_by_camera)
            replay_trace_write_failures_by_camera = dict(
                self._replay_trace_write_failures_by_camera
            )
            bed_polygon_source_by_camera = dict(self._bed_polygon_source_by_camera)
            resample_gap_rows_by_camera = dict(self._resample_gap_rows_by_camera)
            fall_inference_device_by_camera = dict(self._fall_inference_device_by_camera)
            fall_unapplied_by_camera = dict(self._fall_unapplied_policy_threshold_by_camera)
            flow_recording_by_camera = dict(self._flow_recording_by_camera)
            flow_nvenc_sessions_by_camera = dict(self._flow_nvenc_sessions_by_camera)
            flow_lifecycle_by_camera = dict(self._flow_lifecycle_by_camera)
            incident_managers = dict(self._incident_managers)
            measured_fps_by_camera = dict(self._measured_fps_by_camera)
            camera_ids = (
                set(self._decode_by_camera)
                | set(decode_backend_by_camera)
                | set(stage_timings)
                | set(buses)
                | set(statuses)
                | set(encode_by_camera)
                | set(bed_region_by_camera)
                | set(bed_exit_scoring_by_camera)
                | set(device_residency_by_camera)
                | set(decision_completed_by_camera)
                | set(track_id_switches_by_camera)
                | set(track_id_switches_absorbed_by_camera)
                | set(replay_trace_write_failures_by_camera)
                | set(bed_polygon_source_by_camera)
                | set(resample_gap_rows_by_camera)
                | set(fall_inference_device_by_camera)
                | set(fall_unapplied_by_camera)
                | set(flow_recording_by_camera)
                | set(flow_nvenc_sessions_by_camera)
                | set(flow_lifecycle_by_camera)
                | set(incident_managers)
                | (set() if inference is None else set(inference.cameras))
            )
        cameras = tuple(
            CameraDiagnosticsSnapshot(
                camera_id=camera_id,
                failure_category=(
                    None if camera_id not in statuses else statuses[camera_id].error_category
                ),
                stage_timings=stage_timings.get(camera_id, ()),
                bus=bus_snapshot(buses.get(camera_id)),
                decode_backend=decode_backend_by_camera.get(camera_id),
                encode=encode_by_camera.get(camera_id),
                bed_region=bed_region_by_camera.get(camera_id),
                bed_exit_scoring=bed_exit_scoring_by_camera.get(camera_id),
                device_residency=device_residency_by_camera.get(camera_id),
                decision_completed=decision_completed_by_camera.get(camera_id, 0),
                inference=(None if inference is None else inference.cameras.get(camera_id)),
                batch_sizes=(() if inference is None else tuple(inference.batch_sizes.items())),
                geometry_batch_sizes=(
                    ()
                    if inference is None
                    else tuple(
                        GeometryBatchHistogram(geometry, tuple(sorted(sizes.items())))
                        for geometry, sizes in sorted(inference.geometry_batch_sizes.items())
                    )
                ),
                forward_p50_sec=(0.0 if inference is None else inference.forward_p50_sec),
                forward_p95_sec=(0.0 if inference is None else inference.forward_p95_sec),
                track_id_switch_total=track_id_switches_by_camera.get(camera_id, 0),
                track_id_switch_absorbed_total=track_id_switches_absorbed_by_camera.get(
                    camera_id, 0
                ),
                replay_trace_write_failures=replay_trace_write_failures_by_camera.get(camera_id, 0),
                incident_cooldown_suppressed_total=getattr(
                    incident_managers.get(camera_id), "cooldown_suppressed_total", 0
                ),
                bed_polygon_source=bed_polygon_source_by_camera.get(camera_id, "none"),
                inference_fps=measured_fps_by_camera.get(camera_id, (0.0, None))[1],
                camera_fps_unpinned=not _is_pinned_fps(
                    measured_fps_by_camera.get(camera_id, (0.0, None))[1]
                ),
                resample_gap_rows_total=resample_gap_rows_by_camera.get(camera_id, 0),
                fall_inference_device=fall_inference_device_by_camera.get(camera_id, "unknown"),
                fall_unapplied_policy_threshold=fall_unapplied_by_camera.get(camera_id),
                smart_record_extended_total=flow_recording_by_camera.get(camera_id, (0, 0, 0))[0],
                smart_record_extension_raced_total=flow_recording_by_camera.get(
                    camera_id, (0, 0, 0)
                )[1],
                smart_record_start_refused_total=flow_recording_by_camera.get(camera_id, (0, 0, 0))[
                    2
                ],
                nvenc_sessions_active=flow_nvenc_sessions_by_camera.get(camera_id, 0),
                flow_source_outages_total=flow_lifecycle_by_camera.get(camera_id, (0, 0))[0],
                flow_source_recoveries_total=flow_lifecycle_by_camera.get(camera_id, (0, 0))[1],
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
        (
            selections,
            measured_fps,
            detections,
            clip_recorder,
            clip_export,
            gpu,
            worker,
        ) = self._wire_inputs()
        cameras = [
            camera_payload(
                camera_id,
                selection,
                measured_fps.get(camera_id),
                detections[camera_id],
            )
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
        (
            selections,
            measured_fps,
            detections,
            clip_recorder,
            clip_export,
            gpu,
            worker,
        ) = self._wire_inputs()
        cameras_by_facility: dict[str, list[RelayCameraPayload]] = {
            facility_id: [] for facility_id in set(camera_facilities.values())
        }
        for camera_id, selection in selections.items():
            facility_id = camera_facilities.get(camera_id)
            if facility_id is not None:
                cameras_by_facility[facility_id].append(
                    camera_payload(
                        camera_id,
                        selection,
                        measured_fps.get(camera_id),
                        detections[camera_id],
                    )
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
        dict[str, RelayDetectionPayload],
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
            inference = None if self._inference is None else self._inference.snapshot()
            decision_completed_by_camera = dict(self._decision_completed_by_camera)
            native_detection_cameras = set(self._native_detection_cameras)
            native_attempts_by_camera = dict(self._native_attempts_by_camera)
            clip_recorder = self._clip_recorder
            clip_export = self._clip_export.copy()
            gpu = None if self._gpu is None else self._gpu.copy()
            worker = None if self._worker is None else self._worker.copy()
        inference_cameras = {} if inference is None else inference.cameras
        detections = {
            camera_id: _detection_for_camera(
                inference_cameras.get(camera_id),
                decision_completed_by_camera.get(camera_id, 0),
                native_producer=camera_id in native_detection_cameras,
                native_attempts=native_attempts_by_camera.get(camera_id, 0),
            )
            for camera_id in selections
        }
        return selections, measured_fps, detections, clip_recorder, clip_export, gpu, worker


def _is_pinned_fps(value: float | None) -> bool:
    return value is not None and 14.0 <= value <= 16.0


def _detection_for_camera(
    inference: CameraInferenceTelemetry | None,
    decision_completed: int,
    *,
    native_producer: bool,
    native_attempts: int,
) -> RelayDetectionPayload:
    """Report detection telemetry for whichever producer owns this camera.

    ``expected`` means "a detection producer is active for this camera", not
    "the host inference coordinator exists". The backend short-circuits to
    ``state="disabled"`` on ``expected=False`` before it looks at any counter
    (``backend/app/features/status/detection_health.py``), so inferring the
    answer from ``inference is None`` made every ``nvidia`` camera render as
    "detection disabled" whether or not its ``NativePolicyPump`` was working,
    and discarded the real ``decision_completed`` on the way out.

    The native producer owns decode and inference inside the DeepStream child,
    so it has no host-side admitted/succeeded/overwritten counts of its own.
    The wire contract nevertheless enforces the host pipeline's stage ordering
    (``decision_completed <= inference_succeeded <= inference_admitted``; see
    ``RelayDetectionStatus.counters_are_ordered`` in the backend relay router)
    and rejects the payload with HTTP 422 otherwise. For this producer the
    three counts are definitionally equal: the child only publishes a
    perception frame it has already inferred, and the pump completes a decision
    for every frame it accepts. Reporting them equal satisfies the invariant
    without inventing a number.
    """
    if inference is not None:
        return detection_payload(
            expected=True,
            inference_admitted=inference.admitted,
            inference_succeeded=inference.inferred,
            inference_overwritten=inference.overwritten,
            decision_completed=decision_completed,
        )
    if native_producer:
        # ``admitted`` is the real number of frames this producer took on;
        # ``succeeded`` collapses onto ``completed`` because a frame that
        # reaches a decision is by definition one the child already inferred.
        # Reporting a real ``admitted`` is what makes the backend's
        # recent_success_rate meaningful: it drops below 1.0 when frames fail
        # instead of being pinned at 1.0 by a synthesised count.
        return detection_payload(
            expected=True,
            inference_admitted=max(native_attempts, decision_completed),
            inference_succeeded=decision_completed,
            inference_overwritten=0,
            decision_completed=decision_completed,
        )
    return detection_payload(
        expected=False,
        inference_admitted=0,
        inference_succeeded=0,
        inference_overwritten=0,
        decision_completed=0,
    )


__all__ = [
    "BedExitScoringDiagnostics",
    "BedRegionDiagnostics",
    "BusMetricsSource",
    "BusSubscriptionSnapshot",
    "CameraDiagnosticsSnapshot",
    "ClipRecorderStatus",
    "DecodeBackendObservability",
    "DeviceResidencyDiagnostics",
    "EncodeSelection",
    "EncoderLifecycleSnapshot",
    "RelayCameraPayload",
    "RelayClipExportPayload",
    "RelayClipRecorderPayload",
    "RelayDecodePayload",
    "RelayDetectionPayload",
    "RelayGpuPayload",
    "RelayRuntimeStatusPayload",
    "RelayWorkerPayload",
    "RuntimeDiagnosticsSnapshot",
    "StageTimingSnapshot",
    "SubscriptionMetrics",
    "WorkerDiagnostics",
]
