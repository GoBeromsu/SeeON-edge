from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from types import MappingProxyType
from typing import Any, Final, Protocol, TypeAlias, cast, final, runtime_checkable

import worker.pipeline.ingest.lifecycle as ingest
import worker.runtime.bootstrap as bootstrap
import worker.runtime.telemetry.runtime_status_sender as runtime_status_sender_module
from contracts.decode_diagnostics import DecodeSelection
from contracts.observation import BoundingBox
from contracts.runner import BedRunnerResult, Image, RunnerProtocol
from shared.detection_policies import LATEST_POLICY_VERSIONS
from shared.edge_db import EDGE_DATABASE_PATH
from shared.edge_db.compatibility import CURRENT_SCHEMA_RANGE
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.evidence_http_transport import (
    bounded_request,
    classify_http_failure,
    encode_json,
)
from shared.events.relay_failure_log import RelayFailureLog
from shared.events.schemas import build_audit_envelope
from worker.adapters.decode.cpu_av.adapter import CpuAvAdapter
from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.cpu_av.probe import probe_opencv_ffmpeg_capability
from worker.adapters.decode.nvdec_cuvid.probe import probe_nvdec_cuvid_capability
from worker.adapters.decode.nvdec_device.capability import probe_device_resident_capability
from worker.adapters.decode.vaapi.probe import probe_vaapi_capability
from worker.adapters.device.cuda.probe import probe_cuda_capability, probe_nvenc_capability
from worker.adapters.device.mps.probe import probe_mps_capability
from worker.adapters.device.nvml.probe import probe_nvml_gpu_status
from worker.adapters.model import warmup_to_ready
from worker.adapters.model.errors import FatalAcceleratorError
from worker.adapters.model.fall_family_registry import (
    DEFAULT_FALL_MODEL_FAMILY_REGISTRY,
    UnknownFallModelTypeError,
)
from worker.domains import (
    AVAILABLE_OBSERVATION_CHANNELS,
    DETECTION_MODULE_REGISTRY,
    DOMAIN_REGISTRY,
    CameraModuleContext,
    CompiledDetectionModuleRegistry,
    DetectionModuleDefinition,
)
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall import FallModelProtocol
from worker.interfaces.decision import Decider
from worker.interfaces.output import EventSink
from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics import CompositeExtractor, NamedExtractor
from worker.pipeline.analytics.merge import result_merger_names
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.decision.event_identity import event_identity_path
from worker.pipeline.inference_coordinator import (
    CapabilityInferenceCoordinator,
    InferenceResultSlot,
)
from worker.pipeline.ingest.probe import RTSPProbeError, probe_first_frame
from worker.pipeline.ingest.registry import SourceRegistry
from worker.pipeline.output.event_sink import EventClipRecorder, EvidenceEventSink
from worker.pipeline.output.evidence.clip_config import DEFAULT_CLIP_STORE_DIR
from worker.pipeline.output.evidence.clip_frame_feeder import ClipFrameFeeder
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recorder_services import default_services
from worker.pipeline.output.evidence.clip_store_lock import (
    ClipStoreLock,
    ClipStoreLockedError,
)
from worker.pipeline.output.evidence.evidence_runtime import EvidenceExportRuntime
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import PacketRingLimits
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.output.live_view import LatestFrameStore, LiveViewSubscriber
from worker.pipeline.output.live_view_pump import LatestObservationStore, LiveViewPump
from worker.pipeline.output.mjpeg_server import (
    BedZoneNotFoundError,
    BedZonePayload,
    MjpegProbeError,
    MjpegProbePayload,
    MjpegServer,
    MjpegServerConfig,
    dev_mjpeg_config,
    start_optional_mjpeg_server,
)
from worker.pipeline.output.overlay import OverlayRenderer
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.pipeline.trace import BoundedTraceWriter, TraceCapture, TraceIdentity
from worker.runtime.clip_deletion_control import ClipDeletionControlService
from worker.runtime.config import (
    RELAY_HEARTBEAT_PATH,
    CameraRuntimeConfig,
    LiveClipExportPolicy,
    WorkerConfig,
)
from worker.runtime.derivative_runtime import (
    DerivativeControlService,
    DerivativeProductionRuntime,
)
from worker.runtime.faults.handler import FaultHandler
from worker.runtime.faults.record import make_fault_record
from worker.runtime.ingest_composition import (
    build_camera_source_registry,
    compose_camera_ingest_loop,
    resolve_decode_backend,
)
from worker.runtime.lease import GpuLease
from worker.runtime.model_composition import (
    ProvisionedSharedComponent,
    SharedComponentGraph,
    SharedComponentPool,
    SharedYoloExtractors,
    compose_shared_components,
)
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    DecodeProbe,
    EncodeProbe,
    VerifyResult,
    default_decode_probe,
    default_verifiers,
)
from worker.runtime.provenance import (
    AppliedDetectionWindow,
    AppliedRuntimeManifest,
    AppliedRuntimeManifestStore,
    RuntimeEnvironmentFacts,
    build_applied_camera_state,
    build_applied_runtime_manifest,
)
from worker.runtime.provenance.environment import collect_runtime_environment_facts
from worker.runtime.state_dir import resolve_state_dir
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RelayRuntimeStatusTransport,
    RuntimeStatusSender,
)
from worker.runtime.telemetry.wire import (
    ClipRecorderStatus,
    RelayGpuPayload,
    RelayWorkerPayload,
)
from worker.runtime.watchdog import InferenceWatchdog
from worker.types import BusinessEvent, DecisionInput, FramePacket

LOGGER: Final = logging.getLogger(__name__)
HEARTBEAT_TIMEOUT_SEC: Final = 0.5
# Matches edge/runtime/edge_worker.py's DETECTOR_VERSION -- same domain-detector
# generation, ported wholesale rather than re-derived per worker/AGENTS.md.
DETECTOR_VERSION: Final = "worker-domain-detectors-v1"


class EvidenceDeliveryError(RuntimeError):
    """Evidence delivery is enabled but cannot be brought up safely.

    ADR-0003: event delivery is always active. A relay misconfiguration or a
    clip store owned by another process is a real failure, not something to
    degrade past with a warning -- a worker
    that looks healthy while alerts pile up unsent is the exact failure mode
    that decision removes. Messages are sanitized: relay URLs and tokens stay
    on ``__cause__``.
    """


class _RunnableIngest(Protocol):
    @property
    def camera_id(self) -> str: ...

    def run(self) -> None: ...

    def stop(self) -> None: ...


CameraLoopFactory: TypeAlias = Callable[
    [CameraRuntimeConfig, BoundedFrameBus, ingest.IngestReporter], _RunnableIngest
]

PumpFactory: TypeAlias = Callable[
    [CameraRuntimeConfig, BoundedFrameBus, CompositeExtractor, EventAggregator, EventSink],
    _RunnableIngest,
]

EventSinkFactory: TypeAlias = Callable[[CameraRuntimeConfig], EventSink]

ClipRecorderFactory: TypeAlias = Callable[[CameraRuntimeConfig], EventClipRecorder]
EnvironmentFactsFactory: TypeAlias = Callable[[BootContext, str | None], RuntimeEnvironmentFacts]


def _processed_count(pump: _RunnableIngest) -> int:
    """Read a pump's frame-cap progress without widening `_RunnableIngest`.

    Defensive: test-injected pump fakes (e.g. `_NoOpPump`) need not expose
    `processed_count`; a missing attribute reads as 0 (never-complete).
    """
    return getattr(pump, "processed_count", 0)


def _required_extractor_names(domain_names: Sequence[str]) -> tuple[str, ...]:
    """Resolve normalized extractor requirements from compiled modules."""
    required: dict[str, None] = {}
    for definition in DETECTION_MODULE_REGISTRY.selected(domain_names):
        for binding in definition.shared_bindings:
            if binding.component_kind == "extractor" and binding.activation_flag is None:
                required.setdefault(binding.component_id, None)
    return tuple(required)


def _persisted_bed_regions(camera: CameraRuntimeConfig) -> tuple[BoundingBox, ...]:
    """Convert a pulled ``bed_zone_polygon`` into the one persisted bed region.

    Empty when the camera has no persisted polygon -- ``SceneState`` then
    falls back to its existing live-segmentation cache unchanged.
    """
    polygon = camera.bed_zone_polygon
    if not polygon:
        return ()
    xs = tuple(point[0] for point in polygon)
    ys = tuple(point[1] for point in polygon)
    return (
        BoundingBox(
            x1=min(xs),
            y1=min(ys),
            x2=max(xs),
            y2=max(ys),
            confidence=1.0,
            polygon=polygon,
        ),
    )


@runtime_checkable
class _Warmable(Protocol):
    def warmup(self) -> None: ...


def _debug_snapshots_provider(
    domain_deciders: Mapping[str, Decider],
    definitions: Mapping[str, DetectionModuleDefinition] | None = None,
) -> Callable[[int], tuple[Any, ...]]:
    """Build one camera's cross-domain debug-snapshot collector.

    Mirrors edge's per-frame `debug_snapshots` list (camera_worker.py:223-228):
    every domain that registered a `debug_snapshot_adapter` contributes its
    current snapshot, read from the live detector this closure captures.
    """

    def provider(frame_index: int) -> tuple[Any, ...]:
        snapshots: list[Any] = []
        for name, detector in domain_deciders.items():
            adapter = (
                DOMAIN_REGISTRY[name].debug_snapshot_adapter
                if definitions is None
                else definitions[name].debug_adapter
            )
            if adapter is None:
                continue
            snapshot = adapter(detector, frame_index)
            if snapshot is not None:
                snapshots.append(snapshot)
        return tuple(snapshots)

    return provider


@dataclass(frozen=True, slots=True)
class CameraRuntimeContext:
    bus: BoundedFrameBus
    tracker: GreedyIouTracker
    scene_state: SceneState
    scheduler: Scheduler
    analytics: CompositeExtractor
    decision: EventAggregator
    heartbeat: HeartbeatReporter
    ingest_loop: _RunnableIngest
    pump: _RunnableIngest
    clip_frame_feeder: ClipFrameFeeder | None = None
    inference_results: InferenceResultSlot | None = None
    live_view_pump: LiveViewPump | None = None


@dataclass(frozen=True, slots=True)
class CameraDetectionPlan:
    tracker: GreedyIouTracker
    schedule: Mapping[str, int]
    detection_windows: Mapping[str, DetectionWindow | None]
    decision: EventAggregator
    domain_audit: Mapping[str, Mapping[str, object]]
    domain_deciders: Mapping[str, Decider]
    definitions: Mapping[str, DetectionModuleDefinition]


@final
class HeartbeatReporter:
    """Translate a camera READY transition into one bounded relay heartbeat."""

    def __init__(self, worker: WorkerConfig, camera: CameraRuntimeConfig) -> None:
        self._worker, self._camera = worker, camera
        self._last_attempt: float | None = None
        self.failure_count = 0
        self._failure_log = RelayFailureLog(
            LOGGER, channel=f"heartbeat camera_id={camera.camera_id}", method="POST"
        )

    def mark_starting(self, camera_id: str) -> None:
        del camera_id

    def mark_ready(self, camera_id: str) -> None:
        now = monotonic()
        worker, camera = self._worker, self._camera
        if self._last_attempt is not None and (
            now < self._last_attempt + camera.heartbeat_interval_sec
        ):
            return
        self._last_attempt = now
        payload = {
            "camera_id": camera_id,
            "facility_id": camera.facility_id,
            "config_version": worker.version,
        }
        headers = {
            "Content-Type": "application/json",
            "X-Edge-Relay-Token": worker.relay.token.get_secret_value(),
        }
        try:
            result = bounded_request(
                worker.relay_heartbeat_url,
                "POST",
                headers,
                encode_json(payload),
                HEARTBEAT_TIMEOUT_SEC,
            )
        except FatalAcceleratorError:
            raise
        except Exception as exc:  # noqa: BLE001 - relay I/O is a non-fatal camera boundary
            self._record_failure(
                DeliveryFailure(
                    DeliveryDisposition.RETRY,
                    "UNEXPECTED",
                    transport_error=f"{type(exc).__name__}: {exc}",
                )
            )
            return
        if isinstance(result, DeliveryFailure):
            self._record_failure(result)
            return
        status, headers_out, _body = result
        if not 200 <= status < 300:
            self._record_failure(classify_http_failure(status, headers_out))
            return
        self._record_success()

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        del camera_id, category

    def emit(self, event: ingest.IngestEvent) -> None:
        del event

    def _record_failure(self, failure: DeliveryFailure) -> None:
        self.failure_count += 1
        self._failure_log.record_failure(failure, path=RELAY_HEARTBEAT_PATH)

    def _record_success(self) -> None:
        self._failure_log.record_success(path=RELAY_HEARTBEAT_PATH)


@final
class _FaultAwareLoop:
    def __init__(
        self,
        loop: _RunnableIngest,
        handler: FaultHandler,
        profile: str,
        *,
        stage: str = "camera_ingest",
    ) -> None:
        self._loop, self._handler = loop, handler
        self._profile = profile
        self._stage = stage

    @property
    def camera_id(self) -> str:
        return self._loop.camera_id

    def run(self) -> None:
        try:
            self._loop.run()
        except FatalAcceleratorError as exc:
            record = make_fault_record(
                exc,
                profile=self._profile,
                task=exc.task or "inference",
                stage=self._stage,
                camera_id=exc.camera_id or self.camera_id,
            )
            self._handler.handle(exc, record)

    def stop(self) -> None:
        self._loop.stop()


@final
class _NullClipRecorder:
    """Interim ``EventClipRecorder`` used when recording is unavailable."""

    def on_event(
        self,
        trigger_packet: FramePacket,
        event: BusinessEvent,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        del trigger_packet, event, allow_new_clip
        return None


@final
@dataclass(frozen=True, slots=True)
class _CameraClipRecorderView:
    """Per-camera ``EventClipRecorder`` view over one shared :class:`ClipRecorder`.

    ``ClipRecorder`` is a single actor (one encoder, one queue, one thread) that
    already keys all mutable per-clip state by ``camera_id`` internally, so
    clip state is never cross-contaminated between cameras even though the
    underlying encoder resource is shared -- that sharing is the existing
    design, not something introduced here. This view exists so composition
    hands each camera a distinct ``EventClipRecorder`` object (never the same
    reference), matching the DI seam style used elsewhere in this module.
    """

    recorder: ClipRecorder
    camera_id: str

    def on_event(
        self,
        trigger_packet: FramePacket,
        event: BusinessEvent,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        if trigger_packet.camera_id != self.camera_id:
            raise ValueError("trigger packet camera does not match recorder view")
        return self.recorder.on_event(
            trigger_packet,
            event,
            allow_new_clip=allow_new_clip,
        )


@final
@dataclass(frozen=True, slots=True)
class _WindowGatedDecider:
    """Common per-domain detection-window gate (issue #24).

    Wraps another domain's :class:`Decider` so ``update()`` is skipped
    entirely -- returning a no-decision (``()``) -- whenever ``clock()`` falls
    outside ``window``. The wrapped decider's internal state is never touched
    while gated, which is safe for domains whose state can simply freeze
    outside their window (e.g. "fall"). ``bed_exit`` is deliberately never
    wrapped by this gate: see :meth:`WorkerRuntime._build_decider`.
    """

    decider: Decider
    window: DetectionWindow
    clock: Callable[[], datetime]
    last_trace_snapshots: object = ()

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        if not self.window.contains(self.clock()):
            # Local import keeps this shared composition-root file's change
            # confined to the window-gate hunk owned by this remediation lane.
            from worker.types.trace import DecisionTraceSnapshot

            object.__setattr__(
                self,
                "last_trace_snapshots",
                (
                    DecisionTraceSnapshot(
                        reason="outside-detection-window",
                        previous_state="not-evaluated",
                        current_state="not-evaluated",
                        triggered=False,
                        track_id=None,
                        bed_id=None,
                        missing_values={"decision_state": "outside-detection-window"},
                    ),
                ),
            )
            return ()
        events = self.decider.update(input_value)
        object.__setattr__(
            self,
            "last_trace_snapshots",
            getattr(self.decider, "last_trace_snapshots", ()),
        )
        return events


def _evidence_outbox_path(state_dir: Path) -> Path:
    return (
        EDGE_DATABASE_PATH
        if state_dir == resolve_state_dir()
        else state_dir / "worker-state.sqlite3"
    )


def _derivative_schema_available(database_path: Path) -> bool:
    if not database_path.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='derivative_jobs'"
                ).fetchone()
                is not None
            )
    except sqlite3.Error:
        return False


def _verify_opencv_decode() -> VerifyResult:
    capability = probe_opencv_ffmpeg_capability()
    return VerifyResult(capability.available, "cpu", "decode", capability.reason)


def _verify_nvdec_decode() -> VerifyResult:
    capability = probe_nvdec_cuvid_capability()
    return VerifyResult(capability.available, "cuda", "decode", capability.reason)


def _verify_vaapi_decode() -> VerifyResult:
    capability = probe_vaapi_capability()
    return VerifyResult(capability.available, "igpu", "decode", capability.reason)


_PRODUCTION_DECODE_PROBES: Final[Mapping[str, Callable[[], VerifyResult]]] = MappingProxyType(
    {
        "opencv": _verify_opencv_decode,
        "nvdec": _verify_nvdec_decode,
        "vaapi": _verify_vaapi_decode,
    }
)


def production_decode_probe(decode: str) -> VerifyResult:
    """The real ``DecodeProbe`` :class:`WorkerRuntime` injects by default.

    Wraps the adapter-level, hardware-touching capability checks --
    ``probe_opencv_ffmpeg_capability`` (``worker.adapters.decode.cpu_av.probe``,
    checks ``cv2`` imports in this process) and ``probe_nvdec_cuvid_capability``
    (``worker.adapters.decode.nvdec_cuvid.probe``, checks the resolvable
    ``ffmpeg`` build supports ``cuda``/``cuvid``/``nvdec`` hwaccel or a
    ``*_cuvid`` decoder) -- into the ``VerifyResult`` shape
    :func:`~worker.runtime.profile.registry.default_decode_probe` expects for
    the ``decode_capability`` bootstrap stage. Both adapter probes are ports
    of ``default_decode_probe`` in ``edge/runtime/profile/registry.py:151-163``
    (pre-migration reference, read-only) -- this function is the equivalent of
    that reference function's dispatch-by-``decode``-name role, adapted to the
    injected-probes-map pattern already established by
    :func:`~worker.runtime.profile.registry.default_decode_probe`.

    This composition lives here, in the composition root, rather than in
    ``worker.runtime.profile`` (policy only, no hardware access per
    ``worker/runtime/AGENTS.md``) or in ``worker.adapters`` (forbidden from
    importing ``worker.runtime``'s ``VerifyResult`` per
    ``worker/adapters/AGENTS.md``) -- it is the one place allowed to depend on
    both.  Both probes fail closed (``available=False``) on any error,
    missing binary, or unsupported build, so an environment without real
    OpenCV/FFMPEG decode support never gets a false positive.
    """
    return default_decode_probe(decode, _PRODUCTION_DECODE_PROBES)


def production_encode_probe() -> VerifyResult:
    capability = probe_nvenc_capability()
    return VerifyResult(capability.available, "cuda", "encode", capability.reason)


def _production_cuda_source() -> CudaProbe:
    capability = probe_cuda_capability()
    return CudaProbe(
        available=capability.available,
        reason=capability.reason,
        device_count=capability.device_count,
        arch_list=capability.arch_list,
    )


def _production_mps_source() -> bool:
    return probe_mps_capability().available


def _production_device_resident_source() -> VerifyResult:
    """Todo 17's own concrete-stage verifier for `nvidia-device-experimental`.

    Wraps ``probe_device_resident_capability``
    (``worker.adapters.decode.nvdec_device.capability``): a strictly
    narrower, distinct check than ``_production_cuda_source`` above --
    it additionally requires NVML device identity, real
    ``torch.cuda.Stream``/``torch.cuda.Event`` construction, and DLPack
    support before this experimental profile's boot gate can pass. A host
    that only satisfies the plain-CUDA check (e.g. `nvidia-host-bridge`
    already boots there) still fails this one closed with the probe's own
    truthful reason -- see ``DeviceResidentCapability``.
    """
    capability = probe_device_resident_capability()
    return VerifyResult(
        capability.available, "nvidia-device-experimental", "device", capability.reason
    )


def production_boot_dependencies() -> bootstrap.BootDependencies:
    """The real ``BootDependencies`` :class:`WorkerRuntime` injects by default.

    Wraps the adapter-level, hardware-touching ``probe_cuda_capability``
    (``worker.adapters.device.cuda.probe``, checks ``torch`` imports and
    ``torch.cuda.is_available()``) into the ``CudaProbeSource`` shape
    ``worker.runtime.profile.registry.default_verifiers`` expects for the
    ``profile_device`` bootstrap stage's ``cuda`` verifier.

    Unlike ``decode_probe`` (a bare callable field, defaulted with
    ``decode_probe or production_decode_probe``), ``boot_dependencies`` is a
    ``BootDependencies`` *value* -- so the production default has to be
    constructed here, not merely referenced, and ``WorkerRuntime.__init__``
    calls this function rather than assigning it. Before this wiring existed,
    ``WorkerRuntime`` threaded ``boot_dependencies=None`` straight through to
    ``bootstrap.profile_device_stage``, which falls back to
    ``BootDependencies(default_verifiers())`` -- fail-closed with "CUDA
    capability probe is not configured" on every profile that needs a real
    ``cuda`` verify. This is the real default that fixes that.

    ``mps_source`` wraps the adapter-level ``probe_mps_capability``
    (``worker.adapters.device.mps.probe``, checks ``torch`` imports,
    ``torch.backends.mps.is_built()``, and ``torch.backends.mps.is_available()``)
    into the ``MpsProbeSource`` shape ``default_verifiers`` expects for the
    ``mps`` verifier -- a bare ``Callable[[], bool]``
    (``worker.runtime.profile.registry.MpsProbeSource``), unlike
    ``CudaProbeSource`` which carries a richer result dataclass through; only
    the ``available`` flag crosses that boundary, so ``_verify_mps`` reports a
    generic "MPS is available"/"MPS is unavailable" reason rather than the
    probe's detailed diagnostic string. Before this wiring existed, ``mps``
    kept failing closed with "MPS capability probe is not configured" on
    every host, including Apple Silicon where ``torch.backends.mps`` reports
    available. This is the real default that fixes that. The ``cpu`` profile
    is unaffected either way -- ``_verify_cpu`` never consults a source.
    """
    return bootstrap.BootDependencies(
        default_verifiers(
            cuda_source=_production_cuda_source,
            mps_source=_production_mps_source,
            device_resident_source=_production_device_resident_source,
        )
    )


def _production_gpu_status() -> RelayGpuPayload:
    """The real GPU-telemetry producer `WorkerRuntime.__init__` calls once at boot.

    Issue #132: `RelayGpuPayload` (`worker/runtime/telemetry/wire.py:56-64`) and
    `WorkerDiagnostics.set_gpu_status` (`worker/runtime/telemetry/runtime_diagnostics.py`)
    have existed since the relay wire schema was defined, but nothing in
    production ever called `set_gpu_status` -- the same failure class as #124
    (`update_decode` never called in production, fixed in 583d02e). This
    function is the fix's composition point: it combines two adapter-level,
    hardware-touching probes into the wire payload's shape.

    `nvml_available`/`driver_version`/`device_name` come from
    `probe_nvml_gpu_status` (`worker.adapters.device.nvml.probe`, checks
    `pynvml` imports and `nvmlInit`/`nvmlDeviceGetCount`). `cuda_context_ok`
    reuses `probe_cuda_capability` (`worker.adapters.device.cuda.probe`,
    already probed above for the `profile_device` bootstrap stage) rather than
    re-deriving CUDA-context health from NVML data -- NVML enumerating a
    device is a necessary but not sufficient signal that this process's torch
    build can actually construct a `device="cuda"` model (see
    `probe_cuda_capability`'s own docstring on the broken-wheel failure mode),
    so this deliberately answers "is NVML available" and "is CUDA usable" as
    two independent questions, exactly as the wire schema's two separate
    boolean fields imply.

    Both probes fail closed and never raise, so an environment with neither
    NVML nor CUDA (this repo's macOS dev/CI machines) reports
    `nvml_available=False`/`cuda_context_ok=False` with a clear `nvml_error`
    rather than breaking boot.
    """
    nvml_status = probe_nvml_gpu_status()
    cuda_capability = probe_cuda_capability()
    return RelayGpuPayload(
        nvml_available=nvml_status.nvml_available,
        cuda_context_ok=cuda_capability.available,
        driver_version=nvml_status.driver_version,
        device_name=nvml_status.device_name,
        captured_at_sec=time.time(),
        nvml_error=None if nvml_status.nvml_available else nvml_status.reason,
    )


@final
class WorkerRuntime:
    """Own process-wide models and camera-local mutable pipeline state."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        loop_factory: CameraLoopFactory | None = None,
        serving_client: ServingClient,
        env: Mapping[str, str] | None = None,
        acquire_lease: bootstrap.LeaseAcquirer | None = None,
        decode_probe: DecodeProbe | None = None,
        encode_probe: EncodeProbe | None = None,
        boot_dependencies: bootstrap.BootDependencies | None = None,
        hard_exit: Callable[[int], None] = os._exit,  # noqa: SLF001
        restart_check: Callable[[], bool] | None = None,
        clip_export_policy: LiveClipExportPolicy | None = None,
        pump_factory: PumpFactory | None = None,
        event_sink_factory: EventSinkFactory | None = None,
        clip_recorder_factory: ClipRecorderFactory | None = None,
        max_frames_per_camera: int | None = None,
        state_dir: Path | None = None,
        clip_store_dir: Path | None = None,
        module_registry: CompiledDetectionModuleRegistry | None = None,
        restart_generation: int = 0,
        build_revision: str | None = None,
        environment_facts_factory: EnvironmentFactsFactory = collect_runtime_environment_facts,
    ) -> None:
        self.config = config
        self._module_registry = module_registry or DETECTION_MODULE_REGISTRY
        self._module_versions = config.domains.selected_versions(self._module_registry)
        self._restart_generation = restart_generation
        self._build_revision = build_revision
        self._environment_facts_factory = environment_facts_factory
        self._boot_instance_id = f"worker:{uuid.uuid4()}"
        self._runtime_manifest: AppliedRuntimeManifest | None = None
        self._trace_writer: BoundedTraceWriter | None = None
        self._camera_trace_captures: dict[str, TraceCapture] = {}
        self._clip_store_dir = (
            Path(DEFAULT_CLIP_STORE_DIR) if clip_store_dir is None else clip_store_dir
        )
        # Issue #191: a relay pull that carried no domains signal at all used
        # to silently resolve to zero active domains (no fall/bed_exit
        # detection scheduled, no error). Logging the resolved set -- and
        # whether anything actually overrode the registry -- at startup
        # makes that state visible instead of only discoverable by noticing
        # detections never arrive. ``config.enabled_domains`` (the registry
        # overlaid by ``config.domains.resolved_overrides()``) never returns
        # an undefined/None state anymore, so there is no separate fallback
        # branch here: an empty override map *is* "registry default".
        resolved_domain_names = tuple(self._module_versions)
        domain_source = (
            "config override" if self.config.domains.resolved_overrides() else "registry default"
        )
        LOGGER.info(
            "resolved active detection domains (%s): %s",
            domain_source,
            ", ".join(resolved_domain_names) or "(none)",
        )
        if not resolved_domain_names:
            LOGGER.warning("resolved active detection domains is empty; no detection will run")
        self._env = os.environ if env is None else env
        self._serving = serving_client
        self._state_dir = state_dir if state_dir is not None else resolve_state_dir()
        LOGGER.info("worker state directory resolved to %s", self._state_dir)
        self._loop_factory = loop_factory or self._default_loop_factory
        self._pump_factory = pump_factory or self._default_pump_factory
        self._sink_factory = event_sink_factory or self._default_event_sink
        self._clip_recorder_factory = clip_recorder_factory or self._default_clip_recorder
        self._acquire = acquire_lease or (lambda: GpuLease.acquire(self._state_dir))
        self._decode_probe = decode_probe or production_decode_probe
        self._encode_probe = encode_probe or production_encode_probe
        self._boot_dependencies = boot_dependencies or production_boot_dependencies()
        self._hard_exit = hard_exit
        self._restart_check = restart_check
        self._clip_export_policy = clip_export_policy or LiveClipExportPolicy(
            config.clip_export_enabled,
            config.clip_export_version,
        )
        self._max_frames_per_camera = max_frames_per_camera
        self._context = bootstrap.BootstrapContext()
        self._boot: BootContext | None = None
        self._supervisor: ingest.IngestSupervisor | None = None
        self.shared_yolo: SharedYoloExtractors | None = None
        self.fall_model: FallModelProtocol | None = None
        self._shared_component_pool = SharedComponentPool()
        self._shared_graph: SharedComponentGraph | None = None
        self._warmed_component_ids: frozenset[str] = frozenset()
        self.fault_handler: FaultHandler | None = None
        self.watchdog: InferenceWatchdog | None = None
        self.inference_coordinator: CapabilityInferenceCoordinator | None = None
        self.cameras: tuple[CameraRuntimeContext, ...] = ()
        self._clip_recorder: ClipRecorder | None = None
        self._packet_repository: PacketRingRepository | None = None
        self._evidence_export_runtime: EvidenceExportRuntime | None = None
        self._derivative_runtime: DerivativeProductionRuntime | None = None
        self._derivative_control: DerivativeControlService | None = None
        self._clip_deletion_control: ClipDeletionControlService | None = None
        self._runtime_status_sender: RuntimeStatusSender | None = None
        self._clip_frame_feeders: tuple[ClipFrameFeeder, ...] = ()
        self._clip_frame_feeder_threads: tuple[threading.Thread, ...] = ()
        self._camera_source_registry: SourceRegistry | None = None
        # GAP #1/#2 wiring (todo 20): one shared diagnostics sink, overlay
        # renderer, and snapshot store per process -- same "one shared actor"
        # pattern as `_clip_recorder`/`_compose_evidence_export` -- plus the
        # per-camera evidence attacher `_default_pump_factory` reads.
        self.diagnostics = WorkerDiagnostics()
        # #132: `set_gpu_status` existed with zero production callers --
        # `runtime.device` in `/status` stayed permanently empty. Probing and
        # recording once here (not per-camera, not periodically refreshed --
        # see `_production_gpu_status`'s docstring for the follow-up note)
        # mirrors `_boot_dependencies`' own eager probe call two lines above.
        self.diagnostics.set_gpu_status(_production_gpu_status())
        # Explicit non-default mode: this renderer also feeds
        # `AlertEvidenceAttacher` alert snapshots (fall + bed_exit), where
        # `OverlayRenderer`'s new default `mode="none"` would silently render
        # blank evidence images. `"bedexit"` still draws person boxes/skeleton
        # (useful for fall review too) while keeping bed context for bed_exit
        # alerts.
        self._overlay_renderer = OverlayRenderer(mode="bedexit")
        self._snapshot_store = SnapshotStore(self._resolved_clip_store_dir())
        self._camera_evidence_attachers: dict[str, AlertEvidenceAttacher] = {}
        # #15 (resolved): `mjpeg_server.py` had been ported without a call
        # site, so `:8090` never opened and the dashboard's camera view stayed
        # dead even though `compose.edge.yaml` enables the switch and the
        # backend proxies `/api/v1/streams/{id}` there. The dev MJPEG server
        # IS the sanctioned viewer and is really started by
        # `_start_live_view_server` below (real bind, covered by
        # tests/test_worker_live_view_composition.py). The switch is resolved
        # once here so the per-camera live-view pumps built during `_activate`
        # can be handed the tap.
        self._mjpeg_config = self._resolve_mjpeg_config()
        self._live_frames = LatestFrameStore()
        # Written by each camera's pipeline pump right after `analytics.process`
        # (a dict write, never a model call) and read by that camera's
        # `LiveViewPump`: the preview overlays the LATEST cached observation
        # instead of waiting for the current frame's pose forward.
        self._live_observations = LatestObservationStore()
        # #40: the live view gets its own per-camera pose-overlay renderer
        # (LiveViewSubscriber's default, keyed off `self._live_frames`) rather
        # than sharing `self._overlay_renderer` -- that instance stays a
        # fixed, no-toggle renderer for the evidence attacher below, so a
        # runtime pose toggle for one camera's live view can never leak into
        # another camera or into audit snapshots.
        self._live_view: LiveViewSubscriber | None = (
            LiveViewSubscriber(self._live_frames) if self._mjpeg_config.enabled else None
        )
        self._mjpeg_server: MjpegServer | None = None
        self._live_view_pumps: tuple[LiveViewPump, ...] = ()
        self._live_view_pump_threads: tuple[threading.Thread, ...] = ()
        self._camera_debug_snapshots: dict[str, Callable[[int], tuple[Any, ...]]] = {}
        self._camera_inference_results: dict[str, InferenceResultSlot] = {}

    def _resolve_mjpeg_config(self) -> MjpegServerConfig:
        """Settle the live view's two switches into one answer.

        Both exist on purpose and neither may be silently ignored: the YAML
        ``dev_mjpeg`` block is how an operator pins host/port in a config file,
        and ``ML_WORKER_DEV_MJPEG*`` is how the shipped ``compose.edge.yaml``
        turns it on for the deployed worker. An explicit ``dev_mjpeg.enabled``
        in the config wins outright (it is the more specific statement); with
        the config silent, the environment decides.

        The relay token doubles as the probe token, matching edge, so the
        backend's probe origin authenticates against the same secret it
        already holds.
        """
        configured = self.config.dev_mjpeg
        source = (
            MjpegServerConfig(enabled=True, host=configured.host, port=configured.port)
            if configured.enabled
            else dev_mjpeg_config(self._env)
        )
        return MjpegServerConfig(
            enabled=source.enabled,
            host=source.host,
            port=source.port,
            # `_authorized_probe` does a plain string comparison, so the
            # SecretStr has to be unwrapped here or every probe would 403.
            probe_token=self.config.relay.token.get_secret_value(),
        )

    def run(self) -> None:
        stages = bootstrap.named_stages(
            self._context,
            self._env,
            initializers={"models": self._initialize_models},
            warmups={"models": lambda _models: self._warm_models()},
            activate=self._activate,
            decode_probe=self._decode_probe,
            encode_probe=self._encode_probe,
            deps=self._boot_dependencies,
            acquire=self._acquire,
        )
        try:
            _ = bootstrap.bootstrap_or_exit(stages, context=self._context)
            # `bootstrap_or_exit` exits the process on any boot failure, so
            # reaching this line already means boot succeeded -- there is no
            # deferred/partial failure state left to report here.
            self.diagnostics.set_worker_status(
                RelayWorkerPayload(
                    alive=True,
                    pid=os.getpid(),
                    started_at_sec=time.time(),
                    profile_boot_error=None,
                )
            )
            self._start_export_sender()
            self._start_derivative_runtime()
            self._start_runtime_status_sender()
            self._start_clip_frame_feeders()
            self._start_live_view_pumps()
            self._start_live_view_server()
            if self._supervisor is not None:
                # 판정 기준은 **설정된 로스터**(`config.cameras`)이지 활성화에
                # 성공한 카메라(`self.cameras`)가 아니다. 카메라가 설정돼
                # 있는데 전부 활성화에 실패한 경우(예: 도메인이 미등록
                # extractor를 요구해 fail-closed되는 이슈 #47 경로)에는
                # 예전처럼 즉시 반환해서 프로세스가 끝나야 한다 -- 그걸
                # 무한 대기로 만들면 재시작 정책이 걸려 있어도 되살아나지
                # 못하고 그냥 매달린다.
                if self.config.cameras:
                    self._supervisor.join()
                else:
                    # Issue #150: zero configured cameras is a valid boot
                    # state -- every gate above already passed. `join()` has
                    # no ingest threads to wait on and would return here
                    # immediately, exiting the process right after boot and
                    # taking the probe/MJPEG server down with it before an
                    # installer ever gets to validate a camera's RTSP URL.
                    # `wait_until_stopped()` blocks on the supervisor's own
                    # stop signal instead, which an external SIGTERM/SIGINT
                    # (`WorkerRuntime.stop()`) still sets, and which the
                    # restart watcher (already started unconditionally in
                    # `_activate`, camera count aside) sets itself the moment
                    # it observes a fresh config pull -- i.e. once the first
                    # camera is registered, this worker exits clean and the
                    # container's `restart: unless-stopped` brings it back up
                    # to pull the new roster and start the real pipeline.
                    self._supervisor.wait_until_stopped()
        finally:
            self.stop()

    def stop(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop()
        if self.watchdog is not None:
            self.watchdog.stop()
        if self._evidence_export_runtime is not None:
            self._evidence_export_runtime.stop_sender()
        if self._runtime_status_sender is not None:
            self._runtime_status_sender.stop()
        if self._mjpeg_server is not None:
            self._mjpeg_server.stop()
            self._mjpeg_server = None
        for live_pump in self._live_view_pumps:
            live_pump.stop()
        for live_thread in self._live_view_pump_threads:
            live_thread.join(timeout=5.0)
        self._live_view_pump_threads = ()
        if self._derivative_runtime is not None:
            self._derivative_runtime.stop()
            self._derivative_runtime = None
            self._derivative_control = None
        self._clip_deletion_control = None
        for feeder in self._clip_frame_feeders:
            feeder.stop()
        for thread in self._clip_frame_feeder_threads:
            thread.join(timeout=5.0)
        if self._clip_recorder is not None:
            self._clip_recorder.stop()
        if self._packet_repository is not None:
            self._packet_repository.close()
            self._packet_repository = None
        if self._trace_writer is not None:
            self._trace_writer.stop()
            self._trace_writer = None
        self._context.release_lease()

    def _start_live_view_server(self) -> None:
        """Open the operator MJPEG port once the cameras behind it exist.

        Runs after ``bootstrap_or_exit`` so every camera is already registered
        in ``_live_frames`` -- a request for an unknown camera is a 404, and
        binding earlier would serve those for the whole boot window.

        ``start_optional_mjpeg_server`` returns ``None`` both when the feature
        is off and when the bind fails. That is the intended asymmetry: a
        cosmetic view losing its port must not take fall detection down with
        it, so the failure is logged and the worker keeps running.
        """
        if (
            self._live_view is None
            and self._derivative_control is None
            and self._clip_deletion_control is None
        ):
            LOGGER.info("dev_mjpeg disabled; live view server not started")
            return
        server_config = self._mjpeg_config
        if (
            self._derivative_control is not None or self._clip_deletion_control is not None
        ) and not server_config.enabled:
            server_config = replace(server_config, enabled=True)
        self._mjpeg_server = start_optional_mjpeg_server(
            self._live_frames,
            server_config,
            probe=self._rtsp_probe,
            bed_zone_recognizer=self._bed_zone_recognizer,
            derivative_control=self._derivative_control,
            clip_deletion_control=self._clip_deletion_control,
        )
        if self._mjpeg_server is None:
            LOGGER.warning(
                "live view enabled but its server could not bind: host=%s port=%d",
                self._mjpeg_config.host,
                self._mjpeg_config.port,
                extra={
                    "host": self._mjpeg_config.host,
                    "port": self._mjpeg_config.port,
                },
            )
        else:
            surface = "live view" if self._live_view is not None else "derivative control"
            LOGGER.info(
                "%s server bound: host=%s port=%d",
                surface,
                self._mjpeg_config.host,
                self._mjpeg_server.port,
                extra={
                    "host": self._mjpeg_config.host,
                    "port": self._mjpeg_config.port,
                },
            )

    def _rtsp_probe(self, rtsp_url: str) -> MjpegProbePayload:
        """Give the dev MJPEG server's ``POST /probe`` a real decode attempt.

        ``start_optional_mjpeg_server`` falls back to its ``_unavailable_probe``
        default (always ``MjpegProbeError("decode")``) when no ``probe=`` is
        passed in -- this method is that callable, wired in below in
        ``_start_live_view_server``. It reuses camera ingest's own decode path
        (``CpuAvAdapter`` + ``probe_first_frame``) rather than a bespoke probe
        implementation, and the same open/read timeouts real ingest uses
        (``self.config.runtime``) rather than inventing separate probe-only
        ones, so a registration probe fails for the same reasons the camera
        itself would have.

        Destination policy is re-applied here (static + every A/AAAA answer)
        so a forged internal ``/probe`` call cannot bypass API admission
        (SSRF / DNS rebinding). The decoder opens the pinned IP URL.
        Facility LAN and local fixture QA opt in via
        ``ML_RTSP_ALLOW_PRIVATE_DESTINATIONS`` /
        ``ML_RTSP_ALLOW_LOCAL_DESTINATIONS``.
        """
        from shared.rtsp_url_policy import assert_rtsp_endpoint_allowed

        try:
            endpoint = assert_rtsp_endpoint_allowed(rtsp_url)
        except ValueError as exc:
            raise MjpegProbeError("unsupported") from exc
        # Open the pinned IP URL so the decoder cannot re-resolve past policy.
        pinned_url = endpoint.pinned_url
        config = CpuAvConfig(
            camera_id="probe",
            url=pinned_url,
            open_timeout_ms=self.config.runtime.open_timeout_ms,
            read_timeout_ms=self.config.runtime.read_timeout_ms,
        )
        try:
            result = probe_first_frame(
                pinned_url,
                decoder=CpuAvAdapter(),
                config=config,
                requested_backend="cpu_av",
                selected_backend="cpu_av",
            )
        except RTSPProbeError as exc:
            raise MjpegProbeError(exc.error_class) from exc
        return result.as_dict()

    def _bed_zone_recognizer(self, image: Image) -> BedZonePayload:
        """Run one on-demand bed-segmentation pass for the recognize endpoint.

        Reuses the shared bed-seg runner directly (``self.shared_yolo.bed.runner``)
        rather than going through ``NamedExtractor.extract``/a ``FramePacket`` --
        this is a single HTTP-thread call, not a per-camera pump frame. The
        shared YOLO runner instances are already invoked concurrently by every
        camera's pump thread with no explicit lock (``NamedExtractor.extract``
        has none), so this call reuses that same no-additional-locking
        precedent instead of introducing a new one. Only reachable once
        ``_start_live_view_server`` runs, which is after ``bootstrap_or_exit``
        has set ``self.shared_yolo``.
        """
        if self.shared_yolo is None:
            raise RuntimeError("bed-zone recognizer called before models were initialized")
        runner = self.shared_yolo.bed.runner
        # Mirrors `worker.pipeline.analytics.models._runner_call`'s exact
        # is-callable-or-`.run` resolution instead of reaching into
        # `NamedExtractor`'s private `_call` field.
        call = runner if callable(runner) else runner.run
        result = call(image)
        if not isinstance(result, BedRunnerResult):
            raise BedZoneNotFoundError("bed runner returned an unexpected result")
        height, width = int(image.shape[0]), int(image.shape[1])
        best_box: Sequence[float | Sequence[Sequence[int]]] | None = None
        best_score = -1.0
        for box in result.boxes:
            if not isinstance(box[4], (int, float)):
                continue
            score = float(box[4])
            if score > best_score:
                best_score = score
                best_box = box
        if best_box is None:
            raise BedZoneNotFoundError("no bed detected in the current frame")
        coordinates = cast("Sequence[float]", best_box[:5])
        polygon_field = best_box[5] if len(best_box) > 5 else ()
        polygon = (
            [
                [int(point[0]), int(point[1])]
                for point in polygon_field
                if isinstance(point, Sequence)
            ]
            if isinstance(polygon_field, Sequence)
            else []
        )
        if not polygon:
            x1, y1, x2, y2 = (
                int(coordinates[0]),
                int(coordinates[1]),
                int(coordinates[2]),
                int(coordinates[3]),
            )
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return {"polygon": polygon, "image_width": width, "image_height": height}

    def _start_export_sender(self) -> None:
        """Start delivering staged evidence to the relay.

        Camera activation (and the clip recorder it composes) has already run
        by the time ``bootstrap_or_exit`` returns, so this only needs to flip
        the sender's background thread on.

        ``self._evidence_export_runtime is None`` means camera activation did
        not compose delivery. A runtime that *does* exist and then fails to
        start its sender is not
        an optional boundary. ADR-0003: swallowing it leaves staged alerts
        accumulating in the local outbox while the worker reports healthy --
        the same silent-degrade this decision removes. Fail closed instead.
        """
        if self._evidence_export_runtime is None:
            return
        try:
            self._evidence_export_runtime.start_sender()
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed sanitized error
            raise EvidenceDeliveryError(
                "evidence export sender failed to start; staged alerts would not reach the relay"
            ) from exc

    def _start_derivative_runtime(self) -> None:
        database_path = _evidence_outbox_path(self._state_dir)
        if not _derivative_schema_available(database_path):
            return
        runtime = DerivativeProductionRuntime(
            database_path,
            self._resolved_clip_store_dir(),
        )
        runtime.start()
        self._derivative_runtime = runtime
        self._derivative_control = DerivativeControlService(runtime)

    def _start_runtime_status_sender(self) -> None:
        """Start periodic runtime-status relay delivery (default 5s cadence).

        A separate channel from :class:`HeartbeatReporter` -- that is a
        READY-gated liveness ping on ``camera.heartbeat_interval_sec`` (default
        30s) to ``POST /api/v1/relay/heartbeat``; this publishes the process's
        ``WorkerDiagnostics`` telemetry (decode selection, measured fps,
        clip-recorder/gpu/worker status) to ``POST /api/v1/relay/runtime-status``
        on its own timer, independent of any camera's readiness. Never fatal to
        camera activation.

        ``facility_by_camera`` is built from every configured camera (never a
        filtered subset), so no camera's telemetry silently drops out of the
        payload for lacking a mapping entry.

        ``request`` is looked up from ``runtime_status_sender_module`` here
        (rather than relying on ``RelayRuntimeStatusTransport``'s default
        parameter, which binds ``bounded_request`` once at class-definition
        time) so tests can substitute the HTTP transport by monkeypatching
        ``runtime_status_sender_module.bounded_request``.
        """
        facility_by_camera: Mapping[str, str] = MappingProxyType(
            {camera.camera_id: camera.facility_id for camera in self.config.cameras}
        )
        transport = RelayRuntimeStatusTransport(
            self.config.relay.url,
            self.config.relay.token.get_secret_value(),
            request=runtime_status_sender_module.bounded_request,
        )
        sender = RuntimeStatusSender(
            self.diagnostics,
            facility_by_camera,
            transport,
            before_publish=self._refresh_runtime_status_telemetry,
        )
        try:
            sender.start()
        except Exception:  # noqa: BLE001 - runtime-status delivery is a non-fatal camera boundary
            LOGGER.warning("runtime status sender failed to start", exc_info=True)
            return
        self._runtime_status_sender = sender

    def _start_clip_frame_feeders(self) -> None:
        """Start each camera's clip-frame-feeder thread (production gap A).

        Deliberately independent of ``IngestSupervisor`` rather than folded
        into its managed loop set: a feeder drains ``bus.evidence`` for as
        long as the process runs, the same "background concern, not the
        work" shape as the export sender and runtime-status sender above.
        Folding it into the supervisor's loops would make
        ``self._supervisor.join()`` -- and therefore ``run()`` -- block
        forever whenever clip recording is active, including in bounded
        ``--max-frames-per-camera`` runs and any test composing a real
        ``ClipRecorder`` without that cap: the completion/restart watchers
        only observe ingest and pump loops, and a feeder never raises
        ``FatalAcceleratorError`` to unblock itself either. Started here
        (after camera activation, so ``self._clip_frame_feeders`` is already
        settled) and reaped by name in ``stop()``, exactly like the export
        sender's own background thread.
        """
        threads = []
        for feeder in self._clip_frame_feeders:
            thread = threading.Thread(
                target=feeder.run,
                name=f"clip-frame-feeder-{feeder.camera_id}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        self._clip_frame_feeder_threads = tuple(threads)

    def _start_live_view_pumps(self) -> None:
        """Run each camera's ``bus.live`` consumer on its own thread.

        Deliberately outside ``IngestSupervisor`` for the same reason as the
        clip frame feeders: these loops never complete, so folding them into
        the supervisor would make ``join()`` -- and any bounded
        ``--max-frames-per-camera`` run -- hang. Reaped by name in ``stop()``.
        """
        threads = []
        for live_pump in self._live_view_pumps:
            thread = threading.Thread(
                target=live_pump.run,
                name=f"live-view-pump-{live_pump.camera_id}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        self._live_view_pump_threads = tuple(threads)
        if threads:
            LOGGER.info(
                "live view pumps started: cameras=%d",
                len(threads),
                extra={"cameras": len(threads)},
            )

    def _initialize_models(self, boot: BootContext) -> SharedComponentGraph:
        self._boot = boot
        self.fault_handler = FaultHandler(
            boot.profile.name, hard_exit=self._hard_exit, state_dir=self._state_dir
        )
        self.watchdog = InferenceWatchdog(self.fault_handler, profile=boot.profile.name)
        flags = {"person-box-source": self.config.models.box_source == "person"}
        graph = compose_shared_components(
            self._module_registry,
            module_versions=self._module_versions,
            serving_client=self._serving,
            runtime=boot.runtime_profile.effective_inference_backend,
            device=boot.device,
            flags=flags,
            pool=self._shared_component_pool,
            provisioners={
                "fall-model-family-registry": lambda _binding, device: ProvisionedSharedComponent(
                    self._create_fall_model(device)
                ),
            },
        )
        self._shared_graph = graph
        fall_component = graph.components.get("fall-classifier")
        self.fall_model = (
            cast("FallModelProtocol", fall_component)
            if isinstance(fall_component, FallModelProtocol)
            else None
        )
        pose = graph.components.get("pose")
        bed = graph.components.get("bed")
        person = graph.components.get("person")
        if isinstance(pose, NamedExtractor) and isinstance(bed, NamedExtractor):
            self.shared_yolo = SharedYoloExtractors(
                pose=pose,
                person=person if isinstance(person, NamedExtractor) else None,
                bed=bed,
            )
        return graph

    def _create_fall_model(self, device: str) -> FallModelProtocol:
        """Construct the fall model, fail closed if none or unknown is configured.

        Fall model selection has no implicit fallback: an operator who omits
        ``models.fall`` gets a refused boot, not a silent switch to a
        different model with different performance characteristics. "Which
        model ran that night" must never be answered by an unconfigured
        default (same fail-closed principle as ``ML_WORKER_PROFILE`` and the
        decode policy's unknown-value ``RuntimeError``).

        Issue #65: which model *family* runs is config/metadata-driven via
        ``DEFAULT_FALL_MODEL_FAMILY_REGISTRY``, keyed by ``configured.type``.
        This method never grows a new branch for a new family -- onboarding
        one means registering a factory, not editing this call site. An
        unregistered ``type`` value is just as fail-closed as an absent
        config: it refuses to boot with a diagnostic error naming every
        registered family, never a silent fallback to "lstm".
        """
        configured = self.config.models.fall
        if configured is None:
            raise RuntimeError("fall model must be explicitly configured; refusing to boot")
        try:
            return DEFAULT_FALL_MODEL_FAMILY_REGISTRY.create(configured.type, configured, device)
        except UnknownFallModelTypeError as exc:
            raise RuntimeError(str(exc)) from exc

    def _warm_models(self) -> tuple[str, ...]:
        if self._shared_graph is None or self._boot is None:
            raise RuntimeError("models cannot warm before initialization")
        required_bindings = self._module_registry.shared_bindings(
            self._module_versions,
            flags={"person-box-source": self.config.models.box_source == "person"},
        )
        for binding in required_bindings:
            component = self._shared_graph.components[binding.component_id]
            model = component.runner if isinstance(component, NamedExtractor) else component
            if binding.warmup_required:
                self._warm_one(cast("RunnerProtocol | FallModelProtocol", model), self._boot.device)
        self._warmed_component_ids = frozenset(
            binding.component_id for binding in required_bindings
        )
        return tuple(sorted(self._warmed_component_ids))

    def _warm_one(self, model: RunnerProtocol | FallModelProtocol, device: str) -> None:
        if not isinstance(model, _Warmable):
            raise TypeError("configured model does not expose warmup")
        _ = warmup_to_ready(model, device=device)

    def _activate(self, boot: BootContext) -> tuple[bootstrap.CameraStageOutcome, ...]:
        graph, handler, watchdog = self._shared_graph, self.fault_handler, self.watchdog
        if graph is None or handler is None or watchdog is None:
            raise RuntimeError("camera activation requires initialized shared state")
        # Structural graph failures are global boot failures. Build every complete
        # camera plan before entering the camera-local degradation boundary.
        plans = {
            camera.camera_id: self._preflight_camera_graph(camera) for camera in self.config.cameras
        }
        self._apply_runtime_manifest(boot, plans)
        self._compose_evidence_export(boot)
        contexts: list[CameraRuntimeContext] = []
        outcomes: list[bootstrap.CameraStageOutcome] = []
        for camera in self.config.cameras:
            built: list[CameraRuntimeContext] = []
            outcomes.append(
                bootstrap.run_camera_stage(
                    camera.camera_id,
                    lambda camera=camera, built=built: built.append(
                        self._build_camera(
                            camera,
                            self.shared_yolo,
                            plans[camera.camera_id],
                        )
                    ),
                )
            )
            contexts.extend(built)
        self.cameras = tuple(contexts)
        coordinator = self._compose_inference_coordinator(graph, watchdog, contexts)
        self.inference_coordinator = coordinator
        loops = tuple(
            _FaultAwareLoop(item.ingest_loop, handler, boot.profile.name) for item in contexts
        ) + tuple(
            _FaultAwareLoop(item.pump, handler, boot.profile.name, stage="camera_pipeline_pump")
            for item in contexts
        )
        if coordinator is not None:
            loops += (
                _FaultAwareLoop(
                    coordinator,
                    handler,
                    boot.profile.name,
                    stage="capability_inference_coordinator",
                ),
            )
        for loop in loops:
            handler.register_loop(loop)
        # Feeders run independently of the supervisor (see
        # `_start_clip_frame_feeders`'s docstring) but still register with the
        # fault handler so a fatal error elsewhere stops them too, instead of
        # leaving a feeder thread draining a bus nobody's producing to anymore.
        self._clip_frame_feeders = tuple(
            item.clip_frame_feeder for item in contexts if item.clip_frame_feeder is not None
        )
        for feeder in self._clip_frame_feeders:
            handler.register_loop(feeder)
        # Same shape as the feeders above: a cosmetic tap runs beside the
        # pipeline, never inside `IngestSupervisor`'s completion accounting
        # (a live pump never finishes, so joining it would hang bounded runs).
        self._live_view_pumps = tuple(
            item.live_view_pump for item in contexts if item.live_view_pump is not None
        )
        for live_pump in self._live_view_pumps:
            handler.register_loop(live_pump)
        completion_check = (
            None if self._max_frames_per_camera is None else self._max_frames_completion_check
        )
        self._supervisor = ingest.IngestSupervisor(
            loops, restart_check=self._restart_check, completion_check=completion_check
        )
        watchdog.start()
        self._supervisor.start()
        return tuple(outcomes)

    def _compose_inference_coordinator(
        self,
        graph: SharedComponentGraph,
        watchdog: InferenceWatchdog,
        contexts: Sequence[CameraRuntimeContext],
    ) -> CapabilityInferenceCoordinator | None:
        client = graph.batch_serving_client
        pose = graph.components.get("pose")
        if client is None or not isinstance(pose, NamedExtractor) or not contexts:
            return None
        boot = self._boot
        if boot is None:
            raise RuntimeError("inference coordinator requires initialized boot context")
        coordinator = CapabilityInferenceCoordinator(
            client,
            watchdog,
            stage_timing_recorder=self.diagnostics,
            pose_output_adapter=pose.output_adapter,
            pose_device=boot.device,
        )
        for context in contexts:
            results = context.inference_results
            if results is None:
                raise RuntimeError("batched pose camera has no result handoff")
            coordinator.register(context.scene_state.camera_id, context.bus.inference, results)
        self.diagnostics.register_inference(coordinator)
        return coordinator

    def _apply_runtime_manifest(
        self,
        boot: BootContext,
        plans: Mapping[str, CameraDetectionPlan],
    ) -> None:
        database_path = _evidence_outbox_path(self._state_dir)
        if database_path.name != "edge.sqlite3":
            return
        graph = self._shared_graph
        if graph is None:
            raise RuntimeError("runtime provenance requires initialized components")
        cameras = tuple(
            build_applied_camera_state(
                camera_id=camera.camera_id,
                effective_decode_backend=resolve_decode_backend(boot.decode, camera.decode_backend),
                ingest_target_fps=camera.fps,
                module_qualified_ids=tuple(
                    definition.qualified_id
                    for definition in plans[camera.camera_id].definitions.values()
                ),
                schedule=plans[camera.camera_id].schedule,
                detection_windows={
                    module_id: (
                        None
                        if window is None
                        else AppliedDetectionWindow(window.start, window.end, window.tz)
                    )
                    for module_id, window in plans[camera.camera_id].detection_windows.items()
                },
                policies=MappingProxyType(
                    {
                        definition.module_id: self.config.detection_policies.resolve(
                            camera.camera_id,
                            definition.module_id,
                            definition.version,
                        )
                        for definition in plans[camera.camera_id].definitions.values()
                    }
                ),
                bed_zone_polygon=camera.bed_zone_polygon,
                bed_zone_image_width=camera.bed_zone_image_width,
                bed_zone_image_height=camera.bed_zone_image_height,
            )
            for camera in self.config.cameras
        )
        manifest = build_applied_runtime_manifest(
            boot=boot,
            module_registry=self._module_registry,
            module_versions=self._module_versions,
            component_identities=graph.identities,
            cameras=cameras,
            config_version=self.config.version,
            restart_generation=self._restart_generation,
            detector_version=DETECTOR_VERSION,
            environment=self._environment_facts_factory(boot, self._build_revision),
            edge_database_schema_version=CURRENT_SCHEMA_RANGE.maximum,
        )
        applied_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        AppliedRuntimeManifestStore(database_path).persist(
            manifest,
            boot_instance_id=self._boot_instance_id,
            applied_at=applied_at,
        )
        self._runtime_manifest = manifest
        trace_writer = BoundedTraceWriter(database_path)
        trace_writer.start()
        self._trace_writer = trace_writer

    def _build_camera(
        self,
        camera: CameraRuntimeConfig,
        yolo: SharedYoloExtractors | None,
        plan: CameraDetectionPlan | None = None,
    ) -> CameraRuntimeContext:
        resolved_plan = plan or self._preflight_camera_graph(camera)
        graph = self._shared_graph
        if graph is None and yolo is None:
            raise RuntimeError("camera activation requires initialized shared components")
        # The evidence tap is a FIFO that only ``ClipFrameFeeder`` drains. With
        # clip recording off there is no feeder, so a full-size tap would retain
        # frames nothing will ever read. Size it from the same flag that decides
        # whether a recorder exists at all.
        bus = (
            BoundedFrameBus() if self.config.clip.enabled else BoundedFrameBus(evidence_capacity=1)
        )
        self.diagnostics.register_bus(camera.camera_id, bus)
        # #(runtime.cameras): `register_decode` is the only thing that seeds
        # a camera into the runtime-status payload's `cameras` list (see
        # `WorkerDiagnostics.to_payload`/`to_payloads`, which build it solely
        # from `_decode_by_camera`); without this call every camera is
        # invisible to `/status`'s `runtime.cameras` forever, even though the
        # sender keeps posting 200s. `selected` starts unset here and is
        # refined by `update_decode`/`record_decode_open_failure` once actual
        # decode selection runs.
        self.diagnostics.register_decode(camera.camera_id, camera.decode_backend or "auto")
        tracker = resolved_plan.tracker
        persisted_bed_regions = _persisted_bed_regions(camera)
        if graph is not None:
            extractors = (
                tuple(item for item in graph.extractors if item.module_name != "pose")
                if graph.batch_serving_client is not None
                else graph.extractors
            )
        elif yolo is not None:
            extractors = yolo.extractors
        else:
            raise RuntimeError("camera activation has no extractor graph")
        scene = SceneState(
            camera.camera_id,
            persisted_bed_regions=persisted_bed_regions,
        )
        scheduler = Scheduler(dict(resolved_plan.schedule))
        analytics = CompositeExtractor(
            extractors=extractors,
            scheduler=scheduler,
            tracker=tracker,
            scene_state=scene,
            watchdog=self.watchdog,
            stage_timing_recorder=self.diagnostics,
            bed_region_recorder=self.diagnostics,
        )
        decision = resolved_plan.decision
        domain_audit = resolved_plan.domain_audit
        domain_deciders = resolved_plan.domain_deciders
        if self._trace_writer is not None:
            self._camera_trace_captures[camera.camera_id] = self._build_trace_capture(
                camera.camera_id,
                resolved_plan,
            )
        # One collector per camera, shared by the two consumers that need the
        # same per-frame snapshots: the alert overlay burned into evidence, and
        # the operator live view. Building it twice would read the same live
        # deciders through two closures for no reason.
        debug_snapshots = _debug_snapshots_provider(domain_deciders, resolved_plan.definitions)
        self._camera_debug_snapshots[camera.camera_id] = debug_snapshots
        self._camera_evidence_attachers[camera.camera_id] = AlertEvidenceAttacher(
            domain_audit=domain_audit,
            overlay_renderer=self._overlay_renderer,
            debug_snapshots_provider=debug_snapshots,
            runtime_manifest_sha256=(
                None if self._runtime_manifest is None else self._runtime_manifest.sha256
            ),
        )
        if self._live_view is not None:
            # Register before the server binds so a camera is never a 404 on a
            # live view that is meant to carry it.
            self._live_frames.register_camera(camera.camera_id)
        heartbeat = HeartbeatReporter(self.config, camera)
        loop = self._loop_factory(camera, bus, heartbeat)
        sink = self._sink_factory(camera)
        inference_results = (
            InferenceResultSlot()
            if graph is not None and graph.batch_serving_client is not None
            else None
        )
        if inference_results is not None:
            self._camera_inference_results[camera.camera_id] = inference_results
        pump = self._pump_factory(camera, bus, analytics, decision, sink)
        clip_frame_feeder = self._build_clip_frame_feeder(camera.camera_id, bus)
        live_view_pump = self._build_live_view_pump(camera.camera_id, bus)
        return CameraRuntimeContext(
            bus,
            tracker,
            scene,
            scheduler,
            analytics,
            decision,
            heartbeat,
            loop,
            pump,
            clip_frame_feeder,
            inference_results,
            live_view_pump,
        )

    def _build_live_view_pump(
        self, camera_id: str, bus: BoundedFrameBus
    ) -> LiveViewPump | None:
        """Give ``bus.live`` its consumer (todo 10).

        Ingest has always fanned every decoded frame into the latest-only
        ``live`` lane and nothing drained it; preview was published from the
        pipeline pump *after* pose, so it inherited every inference stall.
        One pump per camera drains that lane directly and overlays the newest
        cached observation. ``None`` when the live view is off -- draining a
        lane whose only consumer is a disabled viewer would be pure waste,
        and the lane's latest-only eviction keeps it bounded either way.
        """
        if self._live_view is None:
            return None
        return LiveViewPump(camera_id, bus.live, self._live_view, self._live_observations)

    def _build_clip_frame_feeder(
        self, camera_id: str, bus: BoundedFrameBus
    ) -> ClipFrameFeeder | None:
        """Feed ``bus.evidence`` into the shared clip recorder (production gap A).

        ``self._clip_recorder`` is resolved once by ``_compose_evidence_export``
        before any camera is built (see ``_activate``), so it is already
        settled by the time this runs. ``None`` when clip recording never
        started -- no point draining a subscription nobody reads.
        """
        if self._clip_recorder is None:
            return None
        return ClipFrameFeeder(camera_id, bus.evidence, self._clip_recorder)

    def _default_pump_factory(
        self,
        camera: CameraRuntimeConfig,
        bus: BoundedFrameBus,
        analytics: CompositeExtractor,
        decision: EventAggregator,
        sink: EventSink,
    ) -> CameraPipelinePump:
        results = self._camera_inference_results.get(camera.camera_id)
        if results is None:
            raise RuntimeError("camera pipeline requires the batched pose coordinator")
        return CameraPipelinePump(
            camera.camera_id,
            results,
            analytics,
            decision,
            sink,
            evidence_attacher=self._camera_evidence_attachers.get(camera.camera_id),
            diagnostics=self.diagnostics,
            max_frames=self._max_frames_per_camera,
            observation_recorder=(
                None if self._live_view is None else self._live_observations
            ),
            debug_snapshots_provider=self._camera_debug_snapshots.get(camera.camera_id),
            trace_capture=self._camera_trace_captures.get(camera.camera_id),
            trace_writer=self._trace_writer,
        )

    def _build_trace_capture(
        self,
        camera_id: str,
        plan: CameraDetectionPlan,
    ) -> TraceCapture:
        manifest = self._runtime_manifest
        graph = self._shared_graph
        if manifest is None or graph is None:
            raise RuntimeError("trace capture requires applied runtime provenance")
        shared_digests = {
            identity.component_id: identity.artifact_digest for identity in graph.identities
        }
        identities: list[TraceIdentity] = []
        for module_id, definition in plan.definitions.items():
            decider = plan.domain_deciders[module_id]
            policy = self.config.detection_policies.resolve(
                camera_id,
                module_id,
                definition.version,
            )
            component_ids = tuple(
                f"{binding.component_id}.sha256."
                + shared_digests.get(
                    binding.component_id,
                    hashlib.sha256(
                        f"{definition.qualified_id}:{binding.component_id}".encode()
                    ).hexdigest(),
                )
                for binding in definition.component_bindings
            )

            def snapshots(
                definition: DetectionModuleDefinition = definition,
                decider: Decider = decider,
            ) -> object:
                adapter = definition.trace_adapter
                return {} if adapter is None else adapter(decider)

            identities.append(
                TraceIdentity(
                    module_qualified_id=definition.qualified_id,
                    component_qualified_ids=component_ids,
                    policy_qualified_id=definition.policy_schema.qualified_id,
                    effective_policy_id=policy.effective_policy_id,
                    runtime_manifest_sha256=manifest.sha256,
                    snapshot_provider=snapshots,
                )
            )
        return TraceCapture(tuple(identities))

    def _max_frames_completion_check(self) -> bool:
        """True once every camera's pump has processed its frame cap.

        Mirrors edge's `_done` (edge/runtime/edge_worker_supervisor.py): "all
        cameras reached the cap", not "any". Read defensively via
        `_processed_count` so test-injected pump fakes without a
        `processed_count` attribute (e.g. `_NoOpPump`) are simply treated as
        never-complete rather than breaking unrelated tests.
        """
        cap = self._max_frames_per_camera
        if cap is None or not self.cameras:
            return False
        return all(_processed_count(context.pump) >= cap for context in self.cameras)

    def _default_event_sink(self, camera: CameraRuntimeConfig) -> EventSink:
        stager = DurableEvidenceStager(
            database_path=_evidence_outbox_path(self._state_dir),
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            resident_id=camera.resident_id,
            config_version=self.config.version,
            clock=time.time,
            runtime_manifest_sha256=(
                None if self._runtime_manifest is None else self._runtime_manifest.sha256
            ),
        )
        return EvidenceEventSink(
            stager=stager,
            recorder=self._clip_recorder_factory(camera),
            snapshot_store=self._snapshot_store,
        )

    def _compose_evidence_export(self, boot: BootContext) -> None:
        """Compose evidence delivery, and clip recording only when enabled.

        These are two different capabilities that used to share one method.

        *Evidence delivery* (``EvidenceExportRuntime``: outbox reconciliation
        and the relay export sender) is always composed. Alerts stage durably
        and ship regardless of whether this worker records video.

        *Clip recording* (``ClipRecorder`` plus the per-camera views and frame
        feeders) is opt-in through ``config.clip.enabled``, default off.

        The lifecycle invariant both branches must preserve:
        ``initialize_under_lock()`` runs **exactly once, while the
        ``ClipStoreLock`` is held**, before ``start_sender()``.

        * recorder on -- ``ClipRecorder.start()`` acquires the lock, runs the
          startup hook, then sweeps/rotates/admits in that order. That ordering
          is a contract (``tests/test_evidence_export_startup.py``), so the
          hook stays exactly where it is.
        * recorder off -- nothing else would take the lock, so delivery
          composition acquires it itself, initializes once, and releases. The
          sender then starts against an initialized runtime instead of raising
          ``evidence runtime must initialize before sender start``.
        """
        evidence_runtime = self._compose_evidence_delivery()
        self._evidence_export_runtime = evidence_runtime
        if not self.config.clip.enabled:
            self._initialize_delivery_without_recorder(evidence_runtime)
            return
        self._compose_clip_recording(boot, evidence_runtime)

    def _resolved_clip_store_dir(self) -> Path:
        """The clip store root this worker records into.

        The constructor's default ``/var/lib/clip-store`` is the fixed baked
        production volume. Tests and alternate runtime compositions may inject
        a portable root through that constructor seam without reviving the
        retired ``CLIP_STORE_DIR`` environment authority. A backend-selected
        ``clip.store_subdir`` (see ``ClipRecordingConfig``, pulled from ml-api's
        persisted clip-storage-location choice) is appended underneath it so
        an operator's dashboard selection actually changes where clips land.
        ``pull_models.py`` already validates the subdir (relative, no ``..``
        traversal) before it ever reaches ``WorkerConfig``, but a path used for
        filesystem construction is re-checked here too rather than trusted at
        a distance -- ``Path(base) / value`` silently discards ``base`` and
        becomes absolute if ``value`` starts with ``/``.
        """
        base = self._clip_store_dir
        subdir = self.config.clip.store_subdir
        if not subdir:
            return base
        candidate = PurePosixPath(subdir)
        if candidate.is_absolute() or ".." in candidate.parts:
            return base
        return base / subdir

    def _compose_evidence_delivery(self) -> EvidenceExportRuntime:
        """Always build event delivery; the live policy only gates clip claims."""
        probe_camera_id = self.config.cameras[0].camera_id if self.config.cameras else "worker"
        try:
            return EvidenceExportRuntime.from_config(
                store_dir=self._resolved_clip_store_dir(),
                relay_url=self.config.relay.url,
                relay_token=self.config.relay.token.get_secret_value(),
                probe_camera_id=probe_camera_id,
                clip_export_enabled=self._clip_export_policy.enabled,
                database_path=_evidence_outbox_path(self._state_dir),
            )
        except ValueError as exc:
            raise EvidenceDeliveryError(
                "evidence delivery is misconfigured: relay URL, relay token, "
                "and a probe identity are required"
            ) from exc

    def _initialize_delivery_without_recorder(
        self,
        evidence_runtime: EvidenceExportRuntime | None,
    ) -> None:
        """Hold the clip-store lock just long enough to initialize the runtime.

        With clip recording disabled there is no ``ClipRecorder`` to own the
        lock, but reconciliation still has to run under it exactly once before
        the sender starts.
        """
        if evidence_runtime is None:
            return
        store_dir = self._resolved_clip_store_dir()
        try:
            with ClipStoreLock.acquire(store_dir):
                evidence_runtime.initialize_under_lock()
        except ClipStoreLockedError as exc:
            # Another process already owns this clip store. Continuing would run
            # two workers against one outbox, so this is fatal, not degradable.
            raise EvidenceDeliveryError(
                "clip store is locked by another process; refusing to start "
                "evidence delivery against a store this worker does not own"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - any init failure is fatal
            # Matches the startup-hook path: every failure initializing delivery
            # is fatal and typed. Narrowing this to OSError let other exception
            # types escape as raw, unsanitized errors instead.
            raise EvidenceDeliveryError(
                "evidence delivery failed to initialize under the clip-store lock"
            ) from exc

    def _compose_clip_recording(
        self,
        boot: BootContext,
        evidence_runtime: EvidenceExportRuntime | None,
    ) -> None:
        """Build the one shared clip recorder for this process.

        ``ClipRecorder`` is one shared actor/encoder for the whole process (the
        existing design; see ``_CameraClipRecorderView``), so it is built once
        here, before any per-camera sink needs it. An *encoder* failure, or any
        other recorder-side start failure, degrades to ``_NullClipRecorder``
        (events still stage and ship, just without a bound clip), matching the
        branch ``EvidenceEventSink`` already exercises when recording is
        unavailable. Clip-store lock acquisition is not in that category --
        see below.

        Delivery initialization is *not* part of that optional boundary. It
        happens inside ``recorder.start()`` via the startup hook, so its failure
        would otherwise be swallowed by the same broad catch and leave the
        runtime uninitialized -- ``start_sender()`` then refuses and alerts
        strand in the local outbox while the worker looks healthy. The hook
        therefore re-raises as ``EvidenceDeliveryError``, which this method lets
        through.
        """
        clip_config = ClipRecorderConfig(store_dir=self._resolved_clip_store_dir())
        hook_ran = False

        def _startup_hook() -> None:
            # Tracks whether the recorder actually reached the hook, so a later
            # failure inside start() cannot make us initialize a second time.
            nonlocal hook_ran
            hook_ran = True
            if evidence_runtime is None:
                return
            try:
                evidence_runtime.initialize_under_lock()
            except ClipStoreLockedError as exc:
                raise EvidenceDeliveryError(
                    "clip store is locked by another process; refusing to start "
                    "evidence delivery against a store this worker does not own"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - any init failure is fatal
                # Every failure inside delivery initialization is fatal, not an
                # optional clip boundary. Narrowing this would let a new
                # exception type slip back into the clip-only broad catch below
                # and strand alerts in the local outbox again.
                raise EvidenceDeliveryError(
                    "evidence delivery failed to initialize under the clip-store lock"
                ) from exc

        packet_repository = PacketRingRepository(
            tuple(camera.camera_id for camera in self.config.cameras),
            per_camera_limits=PacketRingLimits(
                clip_config.packet_ring_max_packets,
                clip_config.packet_ring_max_bytes_per_camera,
                clip_config.pre_event_seconds
                + clip_config.post_event_seconds
                + clip_config.finalize_grace_seconds,
            ),
            global_max_bytes=clip_config.packet_ring_global_max_bytes,
        )
        self._packet_repository = packet_repository
        recorder = ClipRecorder(
            clip_config,
            services=default_services(clip_config, packet_repository),
            is_clip_held=None if evidence_runtime is None else evidence_runtime.is_clip_held,
            begin_clip_purge=(
                None if evidence_runtime is None else evidence_runtime.begin_clip_purge
            ),
            complete_clip_purge=(
                None if evidence_runtime is None else evidence_runtime.complete_clip_purge
            ),
            fail_clip_purge=(
                None if evidence_runtime is None else evidence_runtime.fail_clip_purge
            ),
            operator_delete_preflight=(
                None
                if evidence_runtime is None
                else getattr(evidence_runtime, "operator_delete_preflight", None)
            ),
            startup_hook=_startup_hook,
            on_clip_finalized=(
                None if evidence_runtime is None else evidence_runtime.notify_clip_finalized
            ),
        )
        try:
            recorder.start()
        except EvidenceDeliveryError:
            # Delivery initialization is fatal, not an optional clip boundary.
            raise
        except Exception:  # noqa: BLE001 - clip recording is a non-fatal camera boundary
            packet_repository.close()
            self._packet_repository = None
            LOGGER.warning("clip recorder failed to start; clips disabled", exc_info=True)
            # Fail-visible: clips are always-on by default, so a start failure
            # must surface through runtime diagnostics (`/status`) rather than
            # silently degrading -- the worker keeps running without clips,
            # but that degraded state is now observable.
            self.diagnostics.set_clip_recorder_status(ClipRecorderStatus(available=False))
            # `initialize_under_lock()` must run exactly once before the sender
            # starts. If start() failed *before* the hook, nothing initialized
            # the runtime and delivery would refuse to start, so initialize here.
            # If start() failed *after* the hook, the runtime is already
            # initialized and re-running it would violate the exactly-once
            # invariant.
            if not hook_ran:
                self._initialize_delivery_without_recorder(evidence_runtime)
            return
        self._clip_recorder = recorder
        self._clip_deletion_control = ClipDeletionControlService(
            delete_clip=recorder.delete_clip,
            retention_state=(
                (lambda _clip_id: None)
                if evidence_runtime is None
                else evidence_runtime.clip_retention_state
            ),
            complete_pending_purge=(
                None if evidence_runtime is None else evidence_runtime.complete_clip_purge
            ),
        )
        self.diagnostics.set_clip_recorder_status(ClipRecorderStatus(available=True))

    def _refresh_runtime_status_telemetry(self) -> None:
        enabled, version = self._clip_export_policy.snapshot()
        self.diagnostics.set_clip_export_applied(enabled=enabled, version=version)
        self._refresh_clip_recorder_telemetry()

    def _refresh_clip_recorder_telemetry(self) -> None:
        """Push the clip recorder's live counters into diagnostics.

        ``set_clip_recorder_status`` (above) is only ever called at recorder
        start/failure, so without this the counters it seeds
        (``dropped_frames``, ``finalized_clips``, ``video_unavailable_clips``,
        ...) stay frozen at their startup values for the rest of the
        process -- ``ClipRecorderStats`` (``clip_actor.py``) keeps
        incrementing them the whole time, but nothing ever re-reads them into
        telemetry, so ``/status`` never reflects clip failures (#165). Wired
        as ``RuntimeStatusSender``'s ``before_publish`` hook so it runs on
        every periodic tick, right before that tick's payload is built.
        """
        recorder = self._clip_recorder
        if recorder is None:
            return
        stats = recorder.stats
        self.diagnostics.set_clip_recorder_status(
            ClipRecorderStatus(
                available=True,
                dropped_frames=stats.dropped_frames,
                dropped_events=stats.dropped_events,
                failed_writes=stats.failed_writes,
                finalized_clips=stats.finalized_clips,
                video_unavailable_clips=stats.video_unavailable_clips,
                active_clips=stats.active_clips,
                encoder=stats.encoder,
            )
        )

    def _default_clip_recorder(self, camera: CameraRuntimeConfig) -> EventClipRecorder:
        if self._clip_recorder is None:
            return _NullClipRecorder()
        return _CameraClipRecorderView(self._clip_recorder, camera.camera_id)

    def _default_loop_factory(
        self,
        camera: CameraRuntimeConfig,
        bus: BoundedFrameBus,
        reporter: ingest.IngestReporter,
    ) -> _RunnableIngest:
        """Compose the real per-camera ingest loop from the boot-resolved profile.

        ``self._boot`` is set by ``_initialize_models`` before camera activation
        (and therefore before this ever runs), so the decode token is always the
        one the profile verified at boot -- never a second, parallel resolution.
        """
        if self._boot is None:
            raise RuntimeError("camera ingest composition requires a resolved boot profile")
        # `resolve_decode_backend` raises on an incompatible override before
        # any adapter is built (fail-fast, ADR-0002); computing it here first
        # means a rejected override never reaches `_record_decode_selection`,
        # so diagnostics never claims a resolution that didn't actually happen.
        resolved_backend = resolve_decode_backend(self._boot.decode, camera.decode_backend)
        loop = compose_camera_ingest_loop(
            camera,
            bus,
            reporter,
            decode=self._boot.decode,
            registry=self._ingest_source_registry(),
            runtime=self.config.runtime,
            packet_sink=self._packet_repository,
        )
        self._record_decode_selection(camera, resolved_backend)
        self.diagnostics.record_decode_backend(
            camera.camera_id,
            requested_profile_decode=self._boot.decode,
            resolved_backend=resolved_backend,
            actual_adapter_class=type(loop.decode_adapter).__name__,
        )
        return loop

    def _record_decode_selection(self, camera: CameraRuntimeConfig, resolved_backend: str) -> None:
        """Surface the effective decode backend to runtime-status diagnostics.

        `register_decode` (called from `_build_camera`) only ever records the
        *requested* backend with `selected=None`; nothing previously carried
        the value `resolve_decode_backend` actually computed for this camera
        into `WorkerDiagnostics`, so `runtime.cameras[*].decode.selected`
        stayed permanently null even though the real decode adapter was built
        from exactly this token. Layer `selected` onto whatever `requested`/
        `fallback_count`/`last_reason` are already on record rather than
        resetting them, since `register_decode`/`record_decode_open_failure`
        may have already run for this camera.
        """
        previous = self.diagnostics.decode_selection(camera.camera_id)
        fallback_requested = camera.decode_backend or "auto"
        self.diagnostics.update_decode(
            camera.camera_id,
            DecodeSelection(
                requested=previous.requested if previous is not None else fallback_requested,
                selected=resolved_backend,
                fallback_count=previous.fallback_count if previous is not None else 0,
                last_reason=previous.last_reason if previous is not None else None,
                updated_at_sec=time.time(),
            ),
        )

    def _ingest_source_registry(self) -> SourceRegistry:
        if self._camera_source_registry is None:
            self._camera_source_registry = build_camera_source_registry(self.config.cameras)
        return self._camera_source_registry

    def _active_domain_names(self) -> tuple[str, ...]:
        return tuple(self._module_versions)

    def _preflight_camera_graph(
        self,
        camera: CameraRuntimeConfig,
        tracker: GreedyIouTracker | None = None,
    ) -> CameraDetectionPlan:
        graph = self._shared_graph
        if graph is None:
            raise RuntimeError("detection graph preflight requires initialized components")
        persisted_bed_regions = _persisted_bed_regions(camera)
        flags = {
            "person-box-source": self.config.models.box_source == "person",
            "persisted-bed-region": bool(persisted_bed_regions),
        }
        activation = self._module_registry.activation(
            module_versions=self._module_versions,
            available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
            available_component_ids=graph.components,
            warmed_component_ids=self._warmed_component_ids,
            output_adapter_ids=result_merger_names(),
            camera_frame_stride=camera.frame_stride,
            flags=flags,
        )
        camera_components: Mapping[str, object] = MappingProxyType(
            {} if tracker is None else {"person-tracker": tracker}
        )
        domain_deciders: dict[str, Decider] = {}
        domain_audit: dict[str, Mapping[str, object]] = {}
        definitions: dict[str, DetectionModuleDefinition] = {}
        detection_windows: dict[str, DetectionWindow | None] = {}
        for definition in activation.definitions:
            window = self._resolved_window(definition.module_id)
            detection_windows[definition.module_id] = window
            context = CameraModuleContext(
                camera_id=camera.camera_id,
                facility_id=camera.facility_id,
                shared_components=graph.components,
                camera_components=camera_components,
                detection_window=window,
                clock=lambda: datetime.now(UTC),
                diagnostics=self.diagnostics,
                policy=(
                    self.config.detection_policies.resolve(
                        camera.camera_id,
                        definition.module_id,
                        definition.version,
                    )
                    if definition.module_id in LATEST_POLICY_VERSIONS
                    else None
                ),
            )
            camera_module = definition.create_camera_module(context)
            camera_components = camera_module.camera_components
            decider = camera_module.decider
            if definition.window_mode == "external" and window is not None:
                decider = _WindowGatedDecider(decider, window, clock=lambda: datetime.now(UTC))
            domain_deciders[definition.module_id] = decider
            definitions[definition.module_id] = definition
            if definition.audit_adapter is not None:
                audit_context = replace(context, camera_components=camera_components)
                snapshot = definition.audit_adapter(audit_context)
                domain_audit[definition.module_id] = build_audit_envelope(
                    model_version=snapshot.model_version,
                    detector_version=DETECTOR_VERSION,
                    operating_threshold=snapshot.operating_threshold,
                )
        resolved_tracker = camera_components.get("person-tracker")
        if not isinstance(resolved_tracker, GreedyIouTracker):
            resolved_tracker = tracker or GreedyIouTracker()
        incidents = IncidentManager(
            identity_path=event_identity_path(camera.camera_id, self._state_dir)
        )
        aggregator = EventAggregator(deciders=tuple(domain_deciders.values()), incidents=incidents)
        return CameraDetectionPlan(
            tracker=resolved_tracker,
            schedule=activation.schedule,
            detection_windows=MappingProxyType(detection_windows),
            decision=aggregator,
            domain_audit=MappingProxyType(domain_audit),
            domain_deciders=MappingProxyType(domain_deciders),
            definitions=MappingProxyType(definitions),
        )

    def _build_decision_stage(
        self, camera: CameraRuntimeConfig, tracker: GreedyIouTracker
    ) -> tuple[EventAggregator, Mapping[str, Mapping[str, object]], Mapping[str, Decider]]:
        plan = self._preflight_camera_graph(camera, tracker)
        return plan.decision, plan.domain_audit, plan.domain_deciders

    def _build_decider(
        self,
        name: str,
        camera: CameraRuntimeConfig,
        fall_model: FallModelProtocol,
        tracker: GreedyIouTracker | None = None,
    ) -> Decider:
        """Compile one registered module without domain-name dispatch."""
        definition = self._module_registry.get(name, self._module_versions.get(name))
        window = self._resolved_window(name)
        context = CameraModuleContext(
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            shared_components=MappingProxyType({"fall-classifier": fall_model}),
            camera_components=MappingProxyType({"person-tracker": tracker or GreedyIouTracker()}),
            detection_window=window,
            clock=lambda: datetime.now(UTC),
            diagnostics=self.diagnostics,
            policy=self.config.detection_policies.resolve(
                camera.camera_id,
                definition.module_id,
                definition.version,
            ),
        )
        decider = definition.create_camera_module(context).decider
        if definition.window_mode == "external" and window is not None:
            return _WindowGatedDecider(decider, window, clock=lambda: datetime.now(UTC))
        return decider

    def _resolved_window(self, name: str) -> DetectionWindow | None:
        configured = self.config.domains.resolved_detection_window(name)
        if configured is None:
            return None
        return DetectionWindow(start=configured.start, end=configured.end, tz=configured.tz)


__all__ = [
    "CameraRuntimeContext",
    "HeartbeatReporter",
    "WorkerRuntime",
    "production_boot_dependencies",
    "production_decode_probe",
]
