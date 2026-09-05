from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from time import monotonic
from types import MappingProxyType
from typing import Any, Final, Protocol, final, runtime_checkable

import worker.runtime.telemetry.runtime_status_sender as runtime_status_sender_module
from contracts.observation import BoundingBox
from contracts.runner import BedRunnerResult, Image, RunnerProtocol
from shared.detection_policies import LATEST_POLICY_VERSIONS
from shared.events.delivery_queue import DeliveryQueue
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.evidence_http_transport import (
    bounded_request,
    classify_http_failure,
    encode_json,
)
from shared.events.relay_failure_log import RelayFailureLog
from shared.events.schemas import build_audit_envelope
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION
from worker.adapters.device.cuda.probe import probe_cuda_capability
from worker.adapters.device.mps.probe import probe_mps_capability
from worker.adapters.device.nvml.probe import probe_nvml_gpu_status
from worker.adapters.model import warmup_to_ready
from worker.adapters.model.errors import FatalAcceleratorError
from worker.adapters.model.fall_family_registry import (
    DEFAULT_FALL_MODEL_FAMILY_REGISTRY,
    UnknownFallModelTypeError,
)
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.domains import (
    AVAILABLE_OBSERVATION_CHANNELS,
    DETECTION_MODULE_REGISTRY,
    DOMAIN_REGISTRY,
    CameraModuleContext,
    CompiledDetectionModuleRegistry,
    DetectionModuleDefinition,
    SharedComponentIdentity,
)
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall import FallV2DomainDecider
from worker.domains.tracker import GreedyIouTracker
from worker.interfaces.decision import Decider
from worker.interfaces.fall_model import FallV2ModelProtocol
from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics.merge import result_merger_names
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.decision.event_identity import event_identity_path
from worker.pipeline.output.evidence.clip_config import DEFAULT_CLIP_STORE_DIR
from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator
from worker.pipeline.output.evidence.clip_publication import ClipPublisher
from worker.pipeline.output.evidence.clip_store_lock import (
    ClipStoreLock,
    ClipStoreLockedError,
)
from worker.pipeline.output.evidence.evidence_runtime import EvidenceExportRuntime
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublisher
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.live_view_api import BedZoneRecognizeResponse
from worker.pipeline.output.mjpeg_server import (
    BedZoneNotFoundError,
    MjpegServer,
    MjpegServerConfig,
    dev_mjpeg_config,
    start_optional_mjpeg_server,
)
from worker.pipeline.perception import SceneState
from worker.pipeline.trace.replay_trace_writer import ReplayTraceWriter
from worker.runtime import bootstrap
from worker.runtime.config import (
    RELAY_HEARTBEAT_PATH,
    CameraRuntimeConfig,
    LiveClipExportPolicy,
    WorkerConfig,
    replay_trace_directory_from_environment,
)
from worker.runtime.faults.handler import FaultHandler
from worker.runtime.faults.record import make_fault_record
from worker.runtime.flow.cold_start import FlowWarmupTimeout, verify_flow_boot_inputs
from worker.runtime.flow.evidence import FlowEvidenceBinding
from worker.runtime.flow.lifecycle_supervisor import FlowLifecycleSupervisor
from worker.runtime.flow.media_plane import FlowMediaPlane, FlowMediaPlaneConfig
from worker.runtime.flow.policy_pump import (
    NativePolicyContext,
    NativePolicyPump,
)
from worker.runtime.lease import GpuLease
from worker.runtime.model_composition import (
    SharedComponentGraph,
    SharedYoloExtractors,
)
from worker.runtime.nvidia_bed_zone_recognizer import (
    DEFAULT_BED_ZONE_RECOGNITION_TIMEOUT_S,
    NvidiaBedZoneRecognizer,
)
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    VerifyResult,
    default_verifiers,
)
from worker.runtime.provenance import (
    AppliedDetectionWindow,
    AppliedRuntimeManifest,
    build_applied_camera_state,
    build_applied_runtime_manifest,
)
from worker.runtime.provenance.environment import (
    RuntimeEnvironmentFacts,
    collect_runtime_environment_facts,
)
from worker.runtime.provenance.model_bundle import ModelBundleProof, admit_model_bundle
from worker.runtime.provenance.store import AppliedRuntimeManifestStore
from worker.runtime.state_dir import resolve_state_dir
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RelayRuntimeStatusTransport,
    RuntimeStatusSender,
)
from worker.runtime.telemetry.wire import (
    RelayGpuPayload,
    RelayWorkerPayload,
)
from worker.runtime.watchdog import InferenceWatchdog
from worker.types import (
    CURRENT_TEMPORAL_PROFILE,
    BusinessEvent,
    DecisionInput,
    TemporalProfile,
)

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


def _persisted_bed_regions(camera: CameraRuntimeConfig) -> tuple[BoundingBox, ...]:
    """Convert a pulled ``bed_zone_polygon`` into the one persisted bed region.

    Empty when the camera has no persisted polygon. Runtime bed-exit decisions
    use only persisted regions; segmentation is on-demand recognition only.
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
class _NativeEngineComponent:
    artifact_digest: str
    preprocessing_identity: str


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

    def _record_failure(self, failure: DeliveryFailure) -> None:
        self.failure_count += 1
        self._failure_log.record_failure(failure, path=RELAY_HEARTBEAT_PATH)

    def _record_success(self) -> None:
        self._failure_log.record_success(path=RELAY_HEARTBEAT_PATH)


@dataclass(frozen=True, slots=True)
@final
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


def _absorbed_track_id_switch_total(decision: EventAggregator) -> int:
    for decider in decision.deciders:
        fall_decider = decider.decider if isinstance(decider, _WindowGatedDecider) else decider
        if isinstance(fall_decider, FallV2DomainDecider):
            return fall_decider.track_id_switch_absorbed_total
    raise RuntimeError("native policy decision lacks a fall V2 absorbed-switch counter")


def _delivery_queue_dir(state_dir: Path) -> Path:
    """Directory backing the publish-once delivery queue for this slot.

    The queue directory is its own capacity authority: count and byte totals are
    reconstructed by scanning it under the queue's cross-process lock, so there
    is no second persisted ledger to diverge from it after a crash.
    """
    return state_dir / "delivery-queue"


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


def production_boot_dependencies() -> bootstrap.BootDependencies:
    """The real ``BootDependencies`` :class:`WorkerRuntime` injects by default.

    Wraps the adapter-level, hardware-touching ``probe_cuda_capability``
    (``worker.adapters.device.cuda.probe``, checks ``torch`` imports and
    ``torch.cuda.is_available()``) into the ``CudaProbeSource`` shape
    ``worker.runtime.profile.registry.default_verifiers`` expects for the
    ``profile_device`` bootstrap stage's ``cuda`` verifier.

    ``BootDependencies`` is a value, so the production default is constructed
    here and injected into ``WorkerRuntime`` rather than being left to a
    bootstrap fallback. That keeps missing capability wiring fail-closed.

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


def _production_device_resident_source() -> VerifyResult:
    """Device residency for `flow`, established from NVML rather than decode.

    The retired `nvidia` profile proved residency by opening an NVDEC device.
    Under `flow` the SDK owns decode inside its own process graph, so the
    parent's evidence is NVML naming a driver and a device - the same source
    the boot telemetry and provenance already use. Failing to see one is a
    refusal to start, not a warning: ADR-0002 keeps required GPU infrastructure
    fail-fast.
    """
    status = probe_nvml_gpu_status()
    if not status.nvml_available:
        return VerifyResult(False, "flow", "device", status.reason)
    device = status.device_name or "an unnamed device"
    driver = status.driver_version or "an unreported driver"
    return VerifyResult(True, "flow", "device", f"NVML reports {device} on driver {driver}")


def _production_gpu_status(*, probe_python_cuda: bool = True) -> RelayGpuPayload:
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
    cuda_context_ok = probe_cuda_capability().available if probe_python_cuda else False
    return RelayGpuPayload(
        nvml_available=nvml_status.nvml_available,
        cuda_context_ok=cuda_context_ok,
        driver_version=nvml_status.driver_version,
        device_name=nvml_status.device_name,
        captured_at_sec=time.time(),
        nvml_error=None if nvml_status.nvml_available else nvml_status.reason,
    )


@final
class NativeHeartbeatLoop:
    """Periodic per-camera liveness for the Flow policy plane.

    Flow source readiness is reported by its lifecycle supervisor. This loop
    additionally reports liveness only while each policy pump advances, so a
    stalled policy path cannot be pinned online by a timer.

    Liveness here is observed, not assumed: a camera is reported ready only
    when its pump's ``processed_count`` advanced since the previous tick, so a
    stalled camera stops heartbeating instead of being pinned online by a
    timer.

    This deliberately runs off the decision path. The policy pump is the sole
    consumer of a capacity-one metadata slot; blocking I/O inside it stalls
    fall detection and overwrites frames.
    """

    def __init__(
        self,
        worker: WorkerConfig,
        cameras: Sequence[CameraRuntimeConfig],
        pumps: Sequence[NativePolicyPump],
        *,
        tick_sec: float = 5.0,
    ) -> None:
        by_id = {camera.camera_id: camera for camera in cameras}
        self._pumps = tuple(pump for pump in pumps if pump.camera_id in by_id)
        self._reporters = {
            pump.camera_id: HeartbeatReporter(worker, by_id[pump.camera_id]) for pump in self._pumps
        }
        # The reporter rate-limits to camera.heartbeat_interval_sec on its own,
        # so ticking faster only shortens how long a newly live camera waits to
        # appear; it does not increase relay traffic.
        self._tick_sec = tick_sec
        # Seeded from the live counter, not from a sentinel below zero: a
        # camera that has processed nothing must not be reported live by the
        # very first tick.
        self._seen: dict[str, int] = {pump.camera_id: pump.processed_count for pump in self._pumps}
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self._tick_sec):
            for pump in self._pumps:
                camera_id = pump.camera_id
                processed = pump.processed_count
                advanced = processed > self._seen[camera_id]
                self._seen[camera_id] = processed
                if not advanced:
                    continue
                try:
                    self._reporters[camera_id].mark_ready(camera_id)
                except FatalAcceleratorError:
                    raise
                except Exception:  # noqa: BLE001 - relay I/O is a non-fatal boundary
                    LOGGER.warning(
                        "native heartbeat failed: camera_id=%s", camera_id, exc_info=True
                    )

    def stop(self) -> None:
        self._stop.set()


@final
class WorkerRuntime:
    """Own process-wide models and camera-local mutable pipeline state."""

    #: Flow is the worker's sole production media plane.
    _flow_media_plane: FlowMediaPlane | None = None
    _flow_lifecycle_supervisor: FlowLifecycleSupervisor | None = None

    def __init__(
        self,
        config: WorkerConfig,
        *,
        serving_client: ServingClient,
        env: Mapping[str, str] | None = None,
        acquire_lease: bootstrap.LeaseAcquirer | None = None,
        boot_dependencies: bootstrap.BootDependencies | None = None,
        hard_exit: Callable[[int], None] = os._exit,  # noqa: SLF001
        restart_check: Callable[[], bool] | None = None,
        clip_export_policy: LiveClipExportPolicy | None = None,
        max_frames_per_camera: int | None = None,
        state_dir: Path | None = None,
        clip_store_dir: Path | None = None,
        module_registry: CompiledDetectionModuleRegistry | None = None,
        restart_generation: int = 0,
        build_revision: str | None = None,
        environment_facts_factory: Callable[
            [BootContext, str | None], RuntimeEnvironmentFacts
        ] = collect_runtime_environment_facts,
        temporal_profile: TemporalProfile = CURRENT_TEMPORAL_PROFILE,
        flow_media_plane: FlowMediaPlane | None = None,
    ) -> None:
        self.config = config
        self.temporal_profile = temporal_profile
        self._module_registry = module_registry or DETECTION_MODULE_REGISTRY
        self._module_versions = config.domains.selected_versions(self._module_registry)
        self._restart_generation = restart_generation
        self._build_revision = build_revision
        self._environment_facts_factory = environment_facts_factory
        self._worker_boot_uuid = uuid.uuid4()
        self._boot_instance_id = f"worker:{self._worker_boot_uuid}"
        self._runtime_manifest: AppliedRuntimeManifest | None = None
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
        self._acquire = acquire_lease or (lambda: GpuLease.acquire(self._state_dir))
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
        self.shared_yolo: SharedYoloExtractors | None = None
        self.fall_model: FallV2ModelProtocol | None = None
        self._shared_graph: SharedComponentGraph | None = None
        self._warmed_component_ids: frozenset[str] = frozenset()
        self.fault_handler: FaultHandler | None = None
        self.watchdog: InferenceWatchdog | None = None
        self._evidence_export_runtime: EvidenceExportRuntime | None = None
        self._runtime_status_sender: RuntimeStatusSender | None = None
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
        self.diagnostics.set_gpu_status(_production_gpu_status(probe_python_cuda=False))
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
        self._mjpeg_server: MjpegServer | None = None
        self._flow_media_plane: FlowMediaPlane | None = flow_media_plane
        self._flow_lifecycle_supervisor: FlowLifecycleSupervisor | None = None
        self._native_policy_pumps: tuple[NativePolicyPump, ...] = ()
        self._policy_pump_threads: tuple[threading.Thread, ...] = ()
        self._selected_bundle_admission: ModelBundleProof | None = None

    def _stop_flow_media_plane(self) -> None:
        """Stop the Flow and let it go, without touching its roster.

        Removing its sources on the way out bought nothing - the plane and its
        slot are discarded here, and a roster change requires a restart anyway -
        while driving the SDK's per-stream teardown, which core-dumped the
        process on a 13-camera shutdown after the Flow failed to stop in time.
        """
        if self._flow_media_plane is None:
            return
        self._live_frames.set_demand_listener(None)
        self._flow_media_plane.stop()
        self._flow_media_plane = None

    def _resolved_clip_store_dir(self) -> Path:
        base = self._clip_store_dir
        subdir = self.config.clip.store_subdir
        if not subdir:
            return base
        candidate = PurePosixPath(subdir)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError("clip store subdirectory must be relative and traversal-free")
        return base / subdir

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
        def decode_probe(_decode: str) -> VerifyResult:
            return VerifyResult(True, "flow", "decode", "Flow media plane")

        def encode_probe() -> VerifyResult:
            return VerifyResult(True, "flow", "encode", "Flow media plane")

        stages = bootstrap.named_stages(
            self._context,
            self._env,
            initializers={"models": self._initialize_models},
            warmups={"models": lambda _models: self._warm_models()},
            activate=self._activate,
            decode_probe=decode_probe,
            encode_probe=encode_probe,
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
            self._start_runtime_status_sender()
            self._start_live_view_server()
            while not self._max_frames_completion_check() and (
                not self._restart_check or not self._restart_check()
            ):
                time.sleep(1)
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_flow_media_plane()
        for pump in self._native_policy_pumps:
            pump.stop()
        for thread in self._policy_pump_threads:
            thread.join(timeout=5.0)
        self._policy_pump_threads = ()
        if self.watchdog is not None:
            self.watchdog.stop()
        if self._evidence_export_runtime is not None:
            self._evidence_export_runtime.stop_sender()
        if self._runtime_status_sender is not None:
            self._runtime_status_sender.stop()
        if self._mjpeg_server is not None:
            self._mjpeg_server.stop()
            self._mjpeg_server = None
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
        if not self._mjpeg_config.enabled:
            LOGGER.info("dev_mjpeg disabled; live view server not started")
            return
        self._mjpeg_server = start_optional_mjpeg_server(
            self._live_frames,
            self._mjpeg_config,
            bed_zone_recognizer=(
                NvidiaBedZoneRecognizer(
                    self._serving,
                    timeout_s=DEFAULT_BED_ZONE_RECOGNITION_TIMEOUT_S,
                )
                if self._boot is not None and self._boot.profile.name == "flow"
                else self._bed_zone_recognizer
            ),
            replay_fall_model=self.fall_model,
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
            LOGGER.info(
                "live view server bound: host=%s port=%d",
                self._mjpeg_config.host,
                self._mjpeg_server.port,
                extra={
                    "host": self._mjpeg_config.host,
                    "port": self._mjpeg_config.port,
                },
            )

    def _bed_zone_recognizer(self, image: Image) -> BedZoneRecognizeResponse:
        """Run one on-demand bed-segmentation pass for the recognition endpoint."""
        if self.shared_yolo is None:
            raise RuntimeError("bed-zone recognizer called before models were initialized")
        runner = self.shared_yolo.bed.runner
        call = runner if callable(runner) else runner.run
        result = call(image)
        if not isinstance(result, BedRunnerResult):
            raise BedZoneNotFoundError("bed runner returned an unexpected result")
        height, width = int(image.shape[0]), int(image.shape[1])
        best_box: Sequence[float | Sequence[Sequence[int]]] | None = None
        best_score = -1.0
        for box in result.boxes:
            if isinstance(box[4], (int, float)) and float(box[4]) > best_score:
                best_score = float(box[4])
                best_box = box
        if best_box is None:
            raise BedZoneNotFoundError("no bed detected in the current frame")
        coordinates = best_box[:5]
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
        return BedZoneRecognizeResponse(
            polygon=tuple((point[0], point[1]) for point in polygon),
            image_width=width,
            image_height=height,
        )

    def _max_frames_completion_check(self) -> bool:
        cap = self._max_frames_per_camera
        return (
            cap is not None
            and bool(self._native_policy_pumps)
            and all(pump.processed_count >= cap for pump in self._native_policy_pumps)
        )

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
            delivery_queue=DeliveryQueue(_delivery_queue_dir(self._state_dir)),
        )
        try:
            sender.start()
        except Exception:  # noqa: BLE001 - runtime-status delivery is a non-fatal camera boundary
            LOGGER.warning("runtime status sender failed to start", exc_info=True)
            return
        self._runtime_status_sender = sender

    def _initialize_models(self, boot: BootContext) -> SharedComponentGraph:
        self._boot = boot
        self._admit_selected_fall_bundle()
        self.fault_handler = FaultHandler(
            boot.profile.name, hard_exit=self._hard_exit, state_dir=self._state_dir
        )
        self.watchdog = InferenceWatchdog(self.fault_handler, profile=boot.profile.name)
        return self._initialize_flow_media_plane(boot)

    def _admit_selected_fall_bundle(self) -> None:
        """Verify the selected active bundle before any model warmup or camera starts."""
        selected = self.config.models.selected
        if selected is None:
            return
        if self.config.models.box_source != "pose":
            raise RuntimeError("selected fall bundle requires box_source=pose")
        if (
            selected.desired.selection is None
            or selected.desired.selection.input_observation_schema != "pose-bbox56.v1"
        ):
            raise RuntimeError(
                "selected fall bundle requires input_observation_schema=pose-bbox56.v1"
            )
        self._selected_bundle_admission = admit_model_bundle(selected.models_root, selected.desired)

    def _initialize_flow_media_plane(self, boot: BootContext) -> SharedComponentGraph:
        """Compose Flow only from explicitly provisioned DeepStream artifacts."""
        # The boot gate already ran verify_flow_boot_inputs for this profile;
        # re-verify here so the plane is never constructed against inputs that
        # changed since the gate, and keep the returned identity for the manifest.
        self._flow_engine_identity = verify_flow_boot_inputs(
            self._env, deployed_batch=len(self.config.cameras)
        )
        if self._flow_media_plane is None:
            self._flow_media_plane = FlowMediaPlane(
                FlowMediaPlaneConfig(
                    infer_config_path=self._env["ML_WORKER_FLOW_INFER_CONFIG"],
                    tracker_config_path=self._env["ML_WORKER_FLOW_TRACKER_CONFIG"],
                    tracker_library_path=self._env["ML_WORKER_FLOW_TRACKER_LIBRARY"],
                    record_dir=Path(self._env["ML_WORKER_FLOW_RECORD_DIR"]),
                    record_cache_seconds=int(self._env["ML_WORKER_FLOW_RECORD_CACHE_SECONDS"]),
                    frame_width=int(self._env["ML_WORKER_FLOW_FRAME_WIDTH"]),
                    frame_height=int(self._env["ML_WORKER_FLOW_FRAME_HEIGHT"]),
                    snapshot_branch_enabled=True,
                    source_silence_timeout_sec=float(
                        self._env.get(
                            "ML_WORKER_FLOW_SOURCE_SILENCE_TIMEOUT_SEC",
                            str(FlowLifecycleSupervisor.DEFAULT_SILENCE_TIMEOUT_SEC),
                        )
                    ),
                ),
                worker_boot_id=str(self._worker_boot_uuid),
            )
        self._flow_media_plane.bind_live_frames(self._live_frames)
        # The plane is not started here: a pyservicemaker Flow fixes its sources
        # when it is built, so _activate_flow registers the roster first, then
        # starts it, then waits for the first accepted frame (the real-batch
        # warmup). Roster changes go through the worker restart directive.
        return self._initialize_flow_policy_graph(boot)

    def _initialize_flow_policy_graph(self, boot: BootContext) -> SharedComponentGraph:
        """Build the CPU policy graph for the Flow media plane."""
        fall_model = self._create_fall_model("cpu", require_onnxruntime=True)
        selected = self.config.models.selected
        selection = None if selected is None else selected.desired.selection
        flags = {"person-box-source": self.config.models.box_source == "person"}
        bindings = self._module_registry.shared_bindings(self._module_versions, flags=flags)
        components: dict[str, object] = {}
        identities: list[SharedComponentIdentity] = []
        for binding in bindings:
            if binding.component_id == "fall-classifier":
                # A pulled selection names its publication digest; the packaged
                # default names the published weights member of its own bundle
                # manifest (the lineage the ONNX export derives from).
                digest = (
                    self._packaged_fall_member_digest()
                    if selection is None
                    else selection.model_publication.bundle_sha256
                )
                preprocessing = (
                    binding.preprocessing_identity
                    if selection is None
                    else selection.input_observation_schema
                )
                components[binding.component_id] = fall_model
                identities.append(
                    SharedComponentIdentity(
                        binding.component_id, digest, "cpu-policy", "cpu", preprocessing
                    )
                )
                continue
            # The media plane owns every perception component. Their applied
            # identity is the published weights each derives from (the ONNX
            # export and the TensorRT engine are build products of that
            # artifact, verified separately by the engine identity file at
            # boot), so the runtime manifest names their published lineage.
            digest = binding.artifact_digest
            preprocessing = binding.preprocessing_identity
            if not digest or not preprocessing:
                raise RuntimeError(f"flow component {binding.component_id!r} has no identity")
            components[binding.component_id] = _NativeEngineComponent(digest, preprocessing)
            identities.append(
                SharedComponentIdentity(
                    binding.component_id,
                    digest,
                    "deepstream-flow" if binding.component_id == "pose" else "onnxruntime-cpu",
                    boot.device if binding.component_id == "pose" else "cpu",
                    preprocessing,
                )
            )
        graph = SharedComponentGraph(MappingProxyType(components), (), tuple(identities), None)
        self._shared_graph = graph
        self.fall_model = fall_model
        self._warmed_component_ids = frozenset(graph.components)
        return graph

    def _packaged_fall_member_digest(self) -> str:
        from worker.adapters.model.pose_bbox56_bundle_support import member_digest, read_json

        configured = self.config.models.fall
        if configured is None:
            raise RuntimeError("fall model must be explicitly configured; refusing to boot")
        # The published weights are the bundle's identity on every profile;
        # under flow the ONNX member is an export of them, pinned to the same
        # bundle manifest and verified by the ORT runner before it loads.
        manifest = read_json(configured.artifact_dir / "bundle-manifest.json")
        return member_digest(manifest, "model.pt")

    def _create_fall_model(
        self, device: str, *, require_onnxruntime: bool = False
    ) -> FallV2ModelProtocol:
        """Construct the selected bundle or the packaged fall model.

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
        selected = self.config.models.selected
        if selected is not None:
            selection = selected.desired.selection
            if selection is None:
                raise RuntimeError("selected fall bundle has no selection contract")
            if require_onnxruntime:
                artifact_dir = selected.models_root / "bundles" / selected.desired.bundle_sha256
                model_onnx = artifact_dir / "model.onnx"
                if not model_onnx.is_file():
                    raise RuntimeError(
                        f"flow profile requires ONNX fall bundle model.onnx: {model_onnx}"
                    )
                if selection.runtime_format != "onnxruntime":
                    raise RuntimeError(
                        "flow profile refuses a Torch fall bundle; the selected runtime_format "
                        "must be onnxruntime (export model.onnx with worker.tools.export_fall_onnx)"
                    )
                return OrtPoseBbox56Runner.from_artifact_dir(artifact_dir, device="cpu")
            try:
                return DEFAULT_FALL_MODEL_FAMILY_REGISTRY.create_bundle(
                    selection.runtime_format,
                    selected.models_root / "bundles" / selected.desired.bundle_sha256,
                    device,
                )
            except UnknownFallModelTypeError as exc:
                raise RuntimeError(str(exc)) from exc

        configured = self.config.models.fall
        if configured is None:
            raise RuntimeError("fall model must be explicitly configured; refusing to boot")
        if require_onnxruntime and configured.framework != "onnxruntime":
            raise RuntimeError(
                "flow profile requires the ONNX Runtime fall model; the packaged config "
                f"resolved framework={configured.framework!r} (export model.onnx with "
                "worker.tools.export_fall_onnx and boot with ML_WORKER_PROFILE=flow)"
            )
        try:
            return DEFAULT_FALL_MODEL_FAMILY_REGISTRY.create(configured.type, configured, device)
        except UnknownFallModelTypeError as exc:
            raise RuntimeError(str(exc)) from exc

    def _warm_models(self) -> tuple[str, ...]:
        if self._shared_graph is None or self._boot is None:
            raise RuntimeError("models cannot warm before initialization")
        if self.fall_model is None or self._flow_media_plane is None:
            raise RuntimeError("flow media plane is not initialized")
        self._warm_one(self.fall_model, "cpu")
        return tuple(sorted(self._warmed_component_ids))

    def _warm_one(self, model: RunnerProtocol | FallV2ModelProtocol, device: str) -> None:
        if not isinstance(model, _Warmable):
            raise TypeError("configured model does not expose warmup")
        _ = warmup_to_ready(model, device=device)

    def _activate(self, boot: BootContext) -> tuple[bootstrap.CameraStageOutcome, ...]:
        handler = self.fault_handler
        if handler is None:
            raise RuntimeError("camera activation requires initialized fault handler")
        return self._activate_flow(boot, handler)

    def _activate_flow(
        self,
        boot: BootContext,
        handler: FaultHandler,
    ) -> tuple[bootstrap.CameraStageOutcome, ...]:
        """Activate Flow sources and the existing image-free policy pumps."""
        media_plane = self._flow_media_plane
        if media_plane is None:
            raise RuntimeError("flow media plane is not initialized")
        self._compose_evidence_export()
        plans = {
            camera.camera_id: self._preflight_camera_graph(camera) for camera in self.config.cameras
        }
        self._apply_runtime_manifest(boot, plans)
        pumps: list[NativePolicyPump] = []
        sealed_bindings: list[FlowEvidenceBinding] = []
        outcomes = tuple(
            bootstrap.run_camera_stage(
                camera.camera_id,
                partial(self._build_flow_camera, camera, pumps, sealed_bindings),
            )
            for camera in self.config.cameras
        )
        self._replay_sealed_clips(sealed_bindings)
        # Every roster source is registered; build and run the Flow now, then
        # require one accepted metadata frame before any pump or readiness
        # exists. This is the real-batch warmup: engines are verified, never
        # built (ADR-0002), so the first frame proves the whole chain.
        media_plane.start()
        self._await_flow_first_frame(pumps)
        self._native_policy_pumps = tuple(pumps)
        for pump in pumps:
            handler.register_loop(pump)
        cameras = {camera.camera_id: camera for camera in self.config.cameras}
        reporters = {
            camera_id: HeartbeatReporter(self.config, camera)
            for camera_id, camera in cameras.items()
        }

        def on_fatal(error: str) -> None:
            LOGGER.error("flow media plane fatal: error=%s", error)
            exc = FatalAcceleratorError(error, task="flow_media_plane")
            handler.handle(
                exc,
                make_fault_record(
                    exc,
                    profile="flow",
                    task="flow_media_plane",
                    stage="flow_media_plane",
                ),
            )

        def on_unready(camera_id: str) -> None:
            LOGGER.warning("flow source outage: camera_id=%s category=metadata_silence", camera_id)

        self._flow_lifecycle_supervisor = FlowLifecycleSupervisor(
            media_plane,
            cameras,
            on_ready=lambda camera_id: reporters[camera_id].mark_ready(camera_id),
            on_unready=on_unready,
            on_fatal=on_fatal,
            silence_timeout_sec=media_plane.config.source_silence_timeout_sec,
        )
        heartbeat = NativeHeartbeatLoop(self.config, self.config.cameras, pumps)
        handler.register_loop(heartbeat)
        threading.Thread(target=heartbeat.run, name="flow-heartbeat", daemon=True).start()
        self._policy_pump_threads = tuple(
            threading.Thread(
                target=pump.run,
                name=f"flow-policy-{pump.camera_id}",
                daemon=True,
            )
            for pump in pumps
        )
        for thread in self._policy_pump_threads:
            thread.start()
        return outcomes

    def _compose_evidence_export(self) -> None:
        """Build and initialize the durable evidence exporter before activation."""
        probe_camera_id = self.config.cameras[0].camera_id if self.config.cameras else "worker"
        try:
            runtime = EvidenceExportRuntime.from_config(
                store_dir=self._resolved_clip_store_dir(),
                queue_directory=_delivery_queue_dir(self._state_dir),
                relay_url=self.config.relay.url,
                relay_token=self.config.relay.token.get_secret_value(),
                probe_camera_id=probe_camera_id,
                clip_export_enabled=self._clip_export_policy.enabled,
                flow_sealed_sidecar_directory=self._state_dir / "flow-sealed",
            )
            with ClipStoreLock.acquire(self._resolved_clip_store_dir()):
                runtime.initialize_under_lock()
        except ClipStoreLockedError as exc:
            raise EvidenceDeliveryError(
                "clip store is locked by another process; refusing evidence delivery"
            ) from exc
        except ValueError as exc:
            raise EvidenceDeliveryError(
                "evidence delivery is misconfigured: relay URL, relay token, "
                "and a probe identity are required"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - delivery startup is required
            raise EvidenceDeliveryError(
                "evidence delivery failed to initialize under the clip-store lock"
            ) from exc
        self._evidence_export_runtime = runtime

    def _start_export_sender(self) -> None:
        """Start the initialized evidence sender after Flow activation succeeds."""
        if self._evidence_export_runtime is None:
            raise EvidenceDeliveryError("evidence delivery was not composed")
        try:
            self._evidence_export_runtime.start_sender()
        except Exception as exc:  # noqa: BLE001 - delivery startup is required
            raise EvidenceDeliveryError("evidence export sender failed to start") from exc

    @staticmethod
    def _replay_sealed_clips(bindings: Sequence[FlowEvidenceBinding]) -> int:
        """Republish clips a previous boot sealed but could not publish.

        Returns the number that failed again. A sidecar exists precisely because
        a publication already failed once; if it fails again the media and the
        sidecar are still on disk for the next attempt. Refusing to activate
        cameras over one stale clip would trade every camera for one recording.
        """
        failures = 0
        for binding in bindings:
            try:
                binding.replay_sealed()
            except Exception:  # noqa: BLE001 - a stale clip must never brick the boot
                failures += 1
                LOGGER.exception(
                    "replaying a sealed clip failed for camera_id=%s; the media and its "
                    "sidecar are retained and cameras continue to activate",
                    binding.camera_id,
                )
        return failures

    def _await_flow_first_frame(
        self, pumps: list[NativePolicyPump], *, timeout_sec: float = 30.0
    ) -> None:
        """Block until one registered source publishes an accepted frame."""
        media_plane = self._flow_media_plane
        if media_plane is None:
            raise RuntimeError("flow media plane is not initialized")
        cameras = {camera.camera_id: camera for camera in self.config.cameras}
        pending = [
            (pump.camera_id, binding)
            for pump, binding in (
                (pump, media_plane.metadata.expected_binding(pump.camera_id)) for pump in pumps
            )
            if binding is not None
        ]
        if not pending:
            raise FlowWarmupTimeout("Flow warmup has no registered source to wait for")
        deadline = time.monotonic() + timeout_sec
        tokens = {camera_id: media_plane.metadata.subscribe(b) for camera_id, b in pending}
        ready: set[str] = set()
        while time.monotonic() < deadline and len(ready) < len(tokens):
            for camera_id, token in tokens.items():
                if camera_id in ready:
                    continue
                try:
                    _ = media_plane.metadata.wait_accepted(token, timeout_sec=1.0)
                except TimeoutError:
                    continue
                # A camera is READY only once its own accepted frame proves the
                # whole chain for it; announcing readiness before the plane
                # produced anything would advertise a camera that cannot alert.
                ready.add(camera_id)
                HeartbeatReporter(self.config, cameras[camera_id]).mark_ready(camera_id)
        if not ready:
            raise FlowWarmupTimeout(
                "Flow warmup did not receive an accepted metadata frame from any source"
            )
        unproven = sorted(set(tokens) - ready)
        if unproven:
            # The plane's per-source reconnect keeps trying; these cameras are
            # simply not READY yet, and the operator must be able to see which.
            LOGGER.warning(
                "flow warmup: %d of %d cameras published a frame; still waiting on %s",
                len(ready),
                len(tokens),
                ", ".join(unproven),
            )

    def _build_flow_camera(
        self,
        camera: CameraRuntimeConfig,
        pumps: list[NativePolicyPump],
        sealed_bindings: list[FlowEvidenceBinding],
    ) -> None:
        from shared.rtsp_url_policy import assert_rtsp_endpoint_allowed

        if camera.decode_backend not in {None, "auto", "nvdec"}:
            raise RuntimeError("flow cameras cannot override decode to a host backend")
        media_plane = self._flow_media_plane
        if media_plane is None:
            raise RuntimeError("flow media plane is not initialized")
        endpoint = assert_rtsp_endpoint_allowed(camera.inference_rtsp_url)
        # Decode is the SDK's now; the host decode-selection record went with the
        # host pipeline. The diagnostic still names what the roster asked for.
        self.diagnostics.register_decode(camera.camera_id, camera.decode_backend or "auto")
        self._live_frames.register_camera(camera.camera_id)
        binding = media_plane.add_source(camera.camera_id, endpoint.pinned_url)
        plan = self._preflight_camera_graph(
            camera,
            episode_source_identity=(
                str(self._worker_boot_uuid),
                str(binding.stream_epoch),
                binding.source_generation,
            ),
        )
        scene = SceneState(
            camera.camera_id,
            persisted_bed_regions=_persisted_bed_regions(camera),
            bed_zone_image_width=camera.bed_zone_image_width,
            bed_zone_image_height=camera.bed_zone_image_height,
        )
        attacher = AlertEvidenceAttacher(
            domain_audit=plan.domain_audit,
            snapshot_renderer=None,
            debug_snapshots_provider=_debug_snapshots_provider(
                plan.domain_deciders, plan.definitions
            ),
            runtime_manifest_sha256=(
                None if self._runtime_manifest is None else self._runtime_manifest.sha256
            ),
        )
        self._camera_evidence_attachers[camera.camera_id] = attacher
        stager = DurableEvidenceStager(
            queue_directory=_delivery_queue_dir(self._state_dir),
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            resident_id=camera.resident_id,
            config_version=self.config.version,
            clock=time.time,
            runtime_manifest_sha256=(
                None if self._runtime_manifest is None else self._runtime_manifest.sha256
            ),
        )
        sealed_binding: list[FlowEvidenceBinding] = []
        actor = media_plane.smart_recorder(
            camera.camera_id,
            sink=lambda sealed: sealed_binding[0].on_sealed(sealed),
        )
        sink = FlowEvidenceBinding(
            actor=actor,
            stager=stager,
            publisher=FlowClipPublisher(
                ClipIdAllocator(self._resolved_clip_store_dir()),
                ClipPublisher(
                    self._resolved_clip_store_dir(),
                    delivery_queue_directory=_delivery_queue_dir(self._state_dir),
                ),
            ),
            sidecars=FlowSealedSidecars(self._state_dir / "flow-sealed"),
            camera_id=camera.camera_id,
        )
        sealed_binding.append(sink)
        sealed_bindings.append(sink)
        pump = NativePolicyPump(
            binding,
            NativePolicyContext(
                media_plane.metadata,
                media_plane,
                scene,
                plan.decision,
                sink,
                attacher,
                self.diagnostics,
                plan.schedule.get("bed", self.temporal_profile.decision_interval_frames("bed")),
                replay_trace=(
                    None
                    if (trace_directory := replay_trace_directory_from_environment()) is None
                    else ReplayTraceWriter(trace_directory, camera.camera_id)
                ),
                night_window_active=_night_window_active(plan.detection_windows.get("bed_exit")),
                recreate_decision=lambda rebuilt: (
                    self._preflight_camera_graph(
                        camera,
                        episode_source_identity=(
                            str(self._worker_boot_uuid),
                            str(rebuilt.stream_epoch),
                            rebuilt.source_generation,
                        ),
                        incidents=plan.decision.incidents,
                    ).decision
                ),
                track_id_switch_absorbed_total=_absorbed_track_id_switch_total,
            ),
        )
        pumps.append(pump)
        self.diagnostics.register_native_detection(camera.camera_id)
        # Readiness is announced by the warmup once this camera's own accepted
        # frame arrives, not here: the plane has not even started yet.

    def _apply_runtime_manifest(
        self,
        boot: BootContext,
        plans: Mapping[str, CameraDetectionPlan],
    ) -> None:
        """Apply provenance after shared identities and camera plans settle.

        The schema value identifies the release this worker targets, not a
        database observed at runtime; workers deliberately do not open one.
        Provenance is auxiliary, so failures must never prevent detection.
        """
        graph = self._shared_graph
        if graph is None:
            raise RuntimeError("runtime provenance requires initialized components")
        try:
            cameras = tuple(
                build_applied_camera_state(
                    camera_id=camera.camera_id,
                    # The profile's declared decode, not the profile's name: the
                    # manifest records which decoder actually ran, and under flow
                    # that is the SDK's NVDEC. Reporting "flow" here made every
                    # boot fail the manifest's vocabulary check.
                    effective_decode_backend=self._boot.profile.effective_decode_backend,
                    ingest_target_fps=self.temporal_profile.target_fps,
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
            self._runtime_manifest = build_applied_runtime_manifest(
                boot=boot,
                module_registry=self._module_registry,
                module_versions=self._module_versions,
                component_identities=graph.identities,
                cameras=cameras,
                config_version=self.config.version,
                restart_generation=self._restart_generation,
                detector_version=DETECTOR_VERSION,
                environment=self._environment_facts_factory(boot, self._build_revision),
                edge_database_schema_version=EDGE_DATABASE_SCHEMA_VERSION,
            )
            AppliedRuntimeManifestStore(self._state_dir / "runtime-manifest").persist(
                self._runtime_manifest,
                boot_instance_id=self._boot_instance_id,
                applied_at=datetime.now(UTC).isoformat(),
            )
        except Exception:  # noqa: BLE001 - provenance must not stop camera activation
            self._runtime_manifest = None
            if self._selected_bundle_admission is not None:
                raise
            LOGGER.warning(
                "runtime provenance could not be applied; continuing without it",
                exc_info=True,
            )

    def _refresh_runtime_status_telemetry(self) -> None:
        enabled, version = self._clip_export_policy.snapshot()
        self.diagnostics.set_clip_export_applied(enabled=enabled, version=version)
        self._refresh_flow_recording_telemetry()

    def _refresh_flow_recording_telemetry(self) -> None:
        """Publish Flow Smart Record and NVENC counters on every status tick."""
        media_plane = self._flow_media_plane
        if media_plane is None:
            return
        lifecycle = self._flow_lifecycle_supervisor
        if lifecycle is None:
            raise RuntimeError("flow lifecycle supervisor is not initialized")
        lifecycle.tick()
        status = media_plane.status()
        for source in status.sources:
            extended, extension_raced, start_refused = media_plane.recorder_counters(
                source.camera_id
            )
            self.diagnostics.record_flow_recording_counters(
                source.camera_id,
                extended=extended,
                extension_raced=extension_raced,
                start_refused=start_refused,
            )
            self.diagnostics.record_flow_nvenc_sessions(
                source.camera_id, status.nvenc_sessions_active
            )
            counters = lifecycle.counters(source.camera_id)
            self.diagnostics.record_flow_lifecycle_counters(
                source.camera_id,
                outages=counters.outages,
                recoveries=counters.recoveries,
            )
            LOGGER.info(
                "flow lifecycle: camera_id=%s outages=%d recoveries=%d",
                source.camera_id,
                counters.outages,
                counters.recoveries,
            )

    def _active_domain_names(self) -> tuple[str, ...]:
        return tuple(self._module_versions)

    def _preflight_camera_graph(
        self,
        camera: CameraRuntimeConfig,
        tracker: GreedyIouTracker | None = None,
        episode_source_identity: tuple[str, str, int] | None = None,
        incidents: IncidentManager | None = None,
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
            temporal_profile=self.temporal_profile,
        )
        if episode_source_identity is None:
            episode_source_identity = (str(self._worker_boot_uuid), "0", 0)
        camera_component_values: dict[str, object] = {
            "episode-identity": episode_source_identity,
        }
        if tracker is not None:
            camera_component_values["person-tracker"] = tracker
        camera_components: Mapping[str, object] = MappingProxyType(camera_component_values)
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
                envelope = build_audit_envelope(
                    model_version=snapshot.model_version,
                    detector_version=DETECTOR_VERSION,
                    operating_threshold=snapshot.operating_threshold,
                )
                if snapshot.threshold_source is not None:
                    envelope["threshold_source"] = snapshot.threshold_source
                if snapshot.receipt_threshold is not None:
                    envelope["receipt_threshold"] = snapshot.receipt_threshold
                if snapshot.unapplied_policy_threshold is not None:
                    # The alert envelope is a frozen wire contract, so an
                    # operator threshold that P1a does not apply is reported on
                    # the runtime status surface instead of silently dropped.
                    self.diagnostics.record_fall_unapplied_policy_threshold(
                        camera.camera_id, snapshot.unapplied_policy_threshold
                    )
                domain_audit[definition.module_id] = envelope
        resolved_tracker = camera_components.get("person-tracker")
        if not isinstance(resolved_tracker, GreedyIouTracker):
            resolved_tracker = tracker or GreedyIouTracker()
        if incidents is None:
            incidents = IncidentManager(
                identity_path=event_identity_path(camera.camera_id, self._state_dir)
            )
        aggregator = EventAggregator(deciders=tuple(domain_deciders.values()), incidents=incidents)
        self.diagnostics.register_incident_manager(camera.camera_id, incidents)
        if _is_confirmed_cpu_fall_runner(self.fall_model):
            self.diagnostics.record_fall_inference_device(camera.camera_id, "cpu")
        return CameraDetectionPlan(
            tracker=resolved_tracker,
            schedule=activation.schedule,
            detection_windows=MappingProxyType(detection_windows),
            decision=aggregator,
            domain_audit=MappingProxyType(domain_audit),
            domain_deciders=MappingProxyType(domain_deciders),
            definitions=MappingProxyType(definitions),
        )

    def _build_decider(
        self,
        name: str,
        camera: CameraRuntimeConfig,
        fall_model: FallV2ModelProtocol,
        tracker: GreedyIouTracker | None = None,
    ) -> Decider:
        """Compile one registered module without domain-name dispatch."""
        definition = self._module_registry.get(name, self._module_versions.get(name))
        window = self._resolved_window(name)
        context = CameraModuleContext(
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            shared_components=MappingProxyType({"fall-classifier": fall_model}),
            camera_components=MappingProxyType(
                {
                    "person-tracker": tracker or GreedyIouTracker(),
                    "episode-identity": (str(self._worker_boot_uuid), "0", 0),
                }
            ),
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


def _night_window_active(window: DetectionWindow | None) -> Callable[[], bool]:
    return (lambda: False) if window is None else lambda: window.contains(datetime.now(UTC))


def _is_confirmed_cpu_fall_runner(model: FallV2ModelProtocol | None) -> bool:
    """Report CPU policy inference only from a runner that declares CPU placement."""
    if model is None:
        return False
    device = getattr(model, "device", None)
    return str(device) == "cpu"


__all__ = [
    "HeartbeatReporter",
    "WorkerRuntime",
    "production_boot_dependencies",
]
