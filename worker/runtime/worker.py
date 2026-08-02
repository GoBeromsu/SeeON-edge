from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from types import MappingProxyType
from typing import Any, Final, Protocol, TypeAlias, final, runtime_checkable

import worker.pipeline.ingest.lifecycle as ingest
import worker.runtime.bootstrap as bootstrap
import worker.runtime.telemetry.runtime_status_sender as runtime_status_sender_module
from contracts.runner import RunnerProtocol
from shared.events.evidence_export_contract import DeliveryFailure
from shared.events.evidence_http_transport import bounded_request, encode_json
from shared.events.schemas import build_audit_envelope
from worker.adapters.decode.cpu_av.probe import probe_opencv_ffmpeg_capability
from worker.adapters.decode.nvdec_cuvid.probe import probe_nvdec_cuvid_capability
from worker.adapters.device.cuda.probe import probe_cuda_capability
from worker.adapters.device.mps.probe import probe_mps_capability
from worker.adapters.model import LstmFallRunner, warmup_to_ready
from worker.adapters.model.errors import FatalAcceleratorError
from worker.domains import (
    DOMAIN_REGISTRY,
    AuditContext,
    BedExitDomainDependencies,
    FallDomainDependencies,
    enabled_domains,
)
from worker.domains.bed_exit import BedExitConfig, NightWindow
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall import FallModelProtocol
from worker.interfaces.decision import Decider
from worker.interfaces.output import EventSink
from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.decision.event_identity import event_identity_path
from worker.pipeline.ingest.registry import SourceRegistry
from worker.pipeline.output.event_sink import EventClipRecorder, EvidenceEventSink
from worker.pipeline.output.evidence.clip_frame_feeder import ClipFrameFeeder
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recorder_services import default_services
from worker.pipeline.output.evidence.clip_store_lock import (
    ClipStoreLock,
    ClipStoreLockedError,
)
from worker.pipeline.output.evidence.evidence_runtime import (
    OUTBOX_PATH_ENV,
    EvidenceExportRuntime,
    export_enabled,
)
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.output.live_view import LatestFrameStore, LiveViewSubscriber
from worker.pipeline.output.mjpeg_server import (
    MjpegServer,
    MjpegServerConfig,
    dev_mjpeg_config,
    start_optional_mjpeg_server,
)
from worker.pipeline.output.overlay import OverlayRenderer
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.faults.handler import FaultHandler
from worker.runtime.faults.record import make_fault_record
from worker.runtime.ingest_composition import (
    build_camera_source_registry,
    compose_camera_ingest_loop,
)
from worker.runtime.lease import GpuLease, resolve_state_dir
from worker.runtime.model_composition import SharedYoloExtractors, compose_yolo_extractors
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    DecodeProbe,
    VerifyResult,
    default_decode_probe,
    default_verifiers,
)
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RelayRuntimeStatusTransport,
    RuntimeStatusSender,
)
from worker.runtime.watchdog import InferenceWatchdog
from worker.types import BusinessEvent, DecisionInput

LOGGER: Final = logging.getLogger(__name__)
HEARTBEAT_TIMEOUT_SEC: Final = 0.5
# Matches edge/runtime/edge_worker.py's DETECTOR_VERSION -- same domain-detector
# generation, ported wholesale rather than re-derived per worker/AGENTS.md.
DETECTOR_VERSION: Final = "worker-domain-detectors-v1"


class EvidenceDeliveryError(RuntimeError):
    """Evidence delivery is enabled but cannot be brought up safely.

    ADR-0003: the export env gate is the explicit opt-out. Once export is
    switched on, a misconfiguration or a clip store owned by another process is
    a real failure, not something to degrade past with a warning -- a worker
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


def _processed_count(pump: _RunnableIngest) -> int:
    """Read a pump's frame-cap progress without widening `_RunnableIngest`.

    Defensive: test-injected pump fakes (e.g. `_NoOpPump`) need not expose
    `processed_count`; a missing attribute reads as 0 (never-complete).
    """
    return getattr(pump, "processed_count", 0)


@runtime_checkable
class _Warmable(Protocol):
    def warmup(self) -> None: ...


def _debug_snapshots_provider(
    domain_deciders: Mapping[str, Decider],
) -> Callable[[int], tuple[Any, ...]]:
    """Build one camera's cross-domain debug-snapshot collector.

    Mirrors edge's per-frame `debug_snapshots` list (camera_worker.py:223-228):
    every domain that registered a `debug_snapshot_adapter` contributes its
    current snapshot, read from the live detector this closure captures.
    """

    def provider(frame_index: int) -> tuple[Any, ...]:
        snapshots: list[Any] = []
        for name, detector in domain_deciders.items():
            adapter = DOMAIN_REGISTRY[name].debug_snapshot_adapter
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
    clip_frame_feeder: _RunnableIngest | None = None


@final
class HeartbeatReporter:
    """Translate a camera READY transition into one bounded relay heartbeat."""

    def __init__(self, worker: WorkerConfig, camera: CameraRuntimeConfig) -> None:
        self._worker, self._camera = worker, camera
        self._last_attempt: float | None = None
        self.failure_count = 0

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
        except Exception:  # noqa: BLE001 - relay I/O is a non-fatal camera boundary
            self._record_failure()
            return
        if isinstance(result, DeliveryFailure) or not 200 <= result[0] < 300:
            self._record_failure()

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        del camera_id, category

    def emit(self, event: ingest.IngestEvent) -> None:
        del event

    def _record_failure(self) -> None:
        self.failure_count += 1
        LOGGER.warning("relay heartbeat failed", extra={"camera_id": self._camera.camera_id})


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
    """Interim ``EventClipRecorder``: real clip binding lands with the clip-encoder
    composition; until then, events still stage durably without a bound clip
    (the same branch :class:`EvidenceEventSink` already exercises when
    recording is unavailable).
    """

    def on_event(
        self,
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        del camera_id, event_ref, event_type, allow_new_clip
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
        camera_id: str,
        event_ref: str,
        event_type: str | None = None,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        return self.recorder.on_event(
            camera_id, event_ref, event_type, allow_new_clip=allow_new_clip
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

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        if not self.window.contains(self.clock()):
            return ()
        return self.decider.update(input_value)


def _evidence_outbox_path() -> Path:
    configured = os.environ.get(OUTBOX_PATH_ENV, "").strip()
    if configured:
        return Path(configured)
    return resolve_state_dir() / "evidence-outbox.sqlite3"


def _verify_opencv_decode() -> VerifyResult:
    capability = probe_opencv_ffmpeg_capability()
    return VerifyResult(capability.available, "cpu", "decode", capability.reason)


def _verify_nvdec_decode() -> VerifyResult:
    capability = probe_nvdec_cuvid_capability()
    return VerifyResult(capability.available, "cuda", "decode", capability.reason)


_PRODUCTION_DECODE_PROBES: Final[Mapping[str, Callable[[], VerifyResult]]] = MappingProxyType(
    {
        "opencv": _verify_opencv_decode,
        "nvdec": _verify_nvdec_decode,
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
        default_verifiers(cuda_source=_production_cuda_source, mps_source=_production_mps_source)
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
        boot_dependencies: bootstrap.BootDependencies | None = None,
        hard_exit: Callable[[int], None] = os._exit,  # noqa: SLF001
        restart_check: Callable[[], bool] | None = None,
        pump_factory: PumpFactory | None = None,
        event_sink_factory: EventSinkFactory | None = None,
        clip_recorder_factory: ClipRecorderFactory | None = None,
        max_frames_per_camera: int | None = None,
    ) -> None:
        self.config = config
        self._env = os.environ if env is None else env
        self._serving = serving_client
        self._loop_factory = loop_factory or self._default_loop_factory
        self._pump_factory = pump_factory or self._default_pump_factory
        self._sink_factory = event_sink_factory or self._default_event_sink
        self._clip_recorder_factory = clip_recorder_factory or self._default_clip_recorder
        self._acquire = acquire_lease or GpuLease.acquire
        self._decode_probe = decode_probe or production_decode_probe
        self._boot_dependencies = boot_dependencies or production_boot_dependencies()
        self._hard_exit = hard_exit
        self._restart_check = restart_check
        self._max_frames_per_camera = max_frames_per_camera
        self._context = bootstrap.BootstrapContext()
        self._boot: BootContext | None = None
        self._supervisor: ingest.IngestSupervisor | None = None
        self.shared_yolo: SharedYoloExtractors | None = None
        self.fall_model: FallModelProtocol | None = None
        self.fault_handler: FaultHandler | None = None
        self.watchdog: InferenceWatchdog | None = None
        self.cameras: tuple[CameraRuntimeContext, ...] = ()
        self._clip_recorder: ClipRecorder | None = None
        self._evidence_export_runtime: EvidenceExportRuntime | None = None
        self._runtime_status_sender: RuntimeStatusSender | None = None
        self._clip_frame_feeders: tuple[ClipFrameFeeder, ...] = ()
        self._clip_frame_feeder_threads: tuple[threading.Thread, ...] = ()
        self._camera_source_registry: SourceRegistry | None = None
        # GAP #1/#2 wiring (todo 20): one shared diagnostics sink, overlay
        # renderer, and snapshot store per process -- same "one shared actor"
        # pattern as `_clip_recorder`/`_compose_evidence_export` -- plus the
        # per-camera evidence attacher `_default_pump_factory` reads.
        self.diagnostics = WorkerDiagnostics()
        self._overlay_renderer = OverlayRenderer()
        self._snapshot_store = SnapshotStore()
        self._camera_evidence_attachers: dict[str, AlertEvidenceAttacher] = {}
        # #15: `mjpeg_server.py` was ported without a call site, so `:8090`
        # never opened and the dashboard's camera view stayed dead even though
        # `compose.edge.yaml` enables the switch and the backend proxies
        # `/api/v1/streams/{id}` there. Resolve the switch once, here, so the
        # per-camera pumps built during `_activate` can be handed the tap.
        self._mjpeg_config = self._resolve_mjpeg_config()
        self._live_frames = LatestFrameStore()
        self._live_view: LiveViewSubscriber | None = (
            LiveViewSubscriber(self._live_frames, renderer=self._overlay_renderer)
            if self._mjpeg_config.enabled
            else None
        )
        self._mjpeg_server: MjpegServer | None = None
        self._camera_debug_snapshots: dict[str, Callable[[int], tuple[Any, ...]]] = {}

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
            deps=self._boot_dependencies,
            acquire=self._acquire,
        )
        try:
            _ = bootstrap.bootstrap_or_exit(stages, context=self._context)
            self._start_export_sender()
            self._start_runtime_status_sender()
            self._start_clip_frame_feeders()
            self._start_live_view_server()
            if self._supervisor is not None:
                self._supervisor.join()
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
        for feeder in self._clip_frame_feeders:
            feeder.stop()
        for thread in self._clip_frame_feeder_threads:
            thread.join(timeout=5.0)
        if self._clip_recorder is not None:
            self._clip_recorder.stop()
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
        if self._live_view is None:
            return
        self._mjpeg_server = start_optional_mjpeg_server(
            self._live_frames, self._mjpeg_config
        )
        if self._mjpeg_server is None:
            LOGGER.warning(
                "live view enabled but its server could not bind",
                extra={
                    "host": self._mjpeg_config.host,
                    "port": self._mjpeg_config.port,
                },
            )

    def _start_export_sender(self) -> None:
        """Start delivering staged evidence to the relay.

        Camera activation (and the clip recorder it composes) has already run
        by the time ``bootstrap_or_exit`` returns, so this only needs to flip
        the sender's background thread on.

        ``self._evidence_export_runtime is None`` is the explicit opt-out: the
        export env gate was off, so there is nothing to deliver and returning
        is correct.

        A runtime that *does* exist and then fails to start its sender is not
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
                "evidence export sender failed to start; staged alerts would "
                "not reach the relay"
            ) from exc

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
        sender = RuntimeStatusSender(self.diagnostics, facility_by_camera, transport)
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

    def _initialize_models(
        self, boot: BootContext
    ) -> tuple[SharedYoloExtractors, FallModelProtocol]:
        self._boot = boot
        self.fault_handler = FaultHandler(boot.profile.name, hard_exit=self._hard_exit)
        self.watchdog = InferenceWatchdog(self.fault_handler, profile=boot.profile.name)
        self.shared_yolo = compose_yolo_extractors(self._serving, device=boot.device)
        self.fall_model = self._create_fall_model(boot.device)
        return self.shared_yolo, self.fall_model

    def _create_fall_model(self, device: str) -> FallModelProtocol:
        """Construct the fall model, fail closed if none is configured.

        Fall model selection has no implicit fallback: an operator who omits
        ``models.fall`` gets a refused boot, not a silent switch to a
        different model with different performance characteristics. "Which
        model ran that night" must never be answered by an unconfigured
        default (same fail-closed principle as ``ML_WORKER_PROFILE`` and the
        decode policy's unknown-value ``RuntimeError``).
        """
        configured = self.config.models.fall
        if configured is None:
            raise RuntimeError(
                "fall model must be explicitly configured; refusing to boot"
            )
        return LstmFallRunner.from_artifact_dir(
            configured.artifact_dir,
            device=device,
            expected_schema_version=configured.schema_version,
            expected_preprocessing_identity=configured.preprocessing_identity,
        )

    def _warm_models(self) -> tuple[str, ...]:
        if self.shared_yolo is None or self.fall_model is None or self._boot is None:
            raise RuntimeError("models cannot warm before initialization")
        for extractor in self.shared_yolo.extractors:
            self._warm_one(extractor.runner, self._boot.device)
        self._warm_one(self.fall_model, self._boot.device)
        return ("pose", "person", "bed", "fall")

    def _warm_one(self, model: RunnerProtocol | FallModelProtocol, device: str) -> None:
        if not isinstance(model, _Warmable):
            raise TypeError("configured model does not expose warmup")
        _ = warmup_to_ready(model, device=device)

    def _activate(self, boot: BootContext) -> tuple[bootstrap.CameraStageOutcome, ...]:
        yolo, handler, watchdog = self.shared_yolo, self.fault_handler, self.watchdog
        if yolo is None or handler is None or watchdog is None:
            raise RuntimeError("camera activation requires initialized shared state")
        self._compose_evidence_export(boot)
        contexts: list[CameraRuntimeContext] = []
        outcomes: list[bootstrap.CameraStageOutcome] = []
        for camera in self.config.cameras:
            built: list[CameraRuntimeContext] = []
            outcomes.append(
                bootstrap.run_camera_stage(
                    camera.camera_id,
                    lambda camera=camera, built=built: built.append(
                        self._build_camera(camera, yolo)
                    ),
                )
            )
            contexts.extend(built)
        self.cameras = tuple(contexts)
        loops = tuple(
            _FaultAwareLoop(item.ingest_loop, handler, boot.profile.name) for item in contexts
        ) + tuple(
            _FaultAwareLoop(item.pump, handler, boot.profile.name, stage="camera_pipeline_pump")
            for item in contexts
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
        completion_check = (
            None if self._max_frames_per_camera is None else self._max_frames_completion_check
        )
        self._supervisor = ingest.IngestSupervisor(
            loops, restart_check=self._restart_check, completion_check=completion_check
        )
        watchdog.start()
        self._supervisor.start()
        return tuple(outcomes)

    def _build_camera(
        self, camera: CameraRuntimeConfig, yolo: SharedYoloExtractors
    ) -> CameraRuntimeContext:
        if self.fall_model is None:
            raise RuntimeError("camera activation requires an initialized fall model")
        # The evidence tap is a FIFO that only ``ClipFrameFeeder`` drains. With
        # clip recording off there is no feeder, so a full-size tap would retain
        # frames nothing will ever read. Size it from the same flag that decides
        # whether a recorder exists at all.
        bus = (
            BoundedFrameBus()
            if self.config.clip.enabled
            else BoundedFrameBus(evidence_capacity=1)
        )
        self.diagnostics.register_bus(camera.camera_id, bus)
        tracker = GreedyIouTracker()
        intervals = {"pose": camera.frame_stride, "person": camera.frame_stride, "bed": 30}
        scene, scheduler = SceneState(camera.camera_id), Scheduler(intervals)
        analytics = CompositeExtractor(
            extractors=yolo.extractors,
            scheduler=scheduler,
            tracker=tracker,
            scene_state=scene,
            watchdog=self.watchdog,
            stage_timing_recorder=self.diagnostics,
        )
        decision, domain_audit, domain_deciders = self._build_decision_stage(
            camera, self.fall_model
        )
        # One collector per camera, shared by the two consumers that need the
        # same per-frame snapshots: the alert overlay burned into evidence, and
        # the operator live view. Building it twice would read the same live
        # deciders through two closures for no reason.
        debug_snapshots = _debug_snapshots_provider(domain_deciders)
        self._camera_debug_snapshots[camera.camera_id] = debug_snapshots
        self._camera_evidence_attachers[camera.camera_id] = AlertEvidenceAttacher(
            domain_audit=domain_audit,
            overlay_renderer=self._overlay_renderer,
            debug_snapshots_provider=debug_snapshots,
        )
        if self._live_view is not None:
            # Register before the server binds so a camera is never a 404 on a
            # live view that is meant to carry it.
            self._live_frames.register_camera(camera.camera_id)
        heartbeat = HeartbeatReporter(self.config, camera)
        loop = self._loop_factory(camera, bus, heartbeat)
        sink = self._sink_factory(camera)
        pump = self._pump_factory(camera, bus, analytics, decision, sink)
        clip_frame_feeder = self._build_clip_frame_feeder(camera.camera_id, bus)
        return CameraRuntimeContext(
            bus, tracker, scene, scheduler, analytics, decision, heartbeat, loop, pump,
            clip_frame_feeder,
        )

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
        return CameraPipelinePump(
            camera.camera_id,
            bus.inference,
            analytics,
            decision,
            sink,
            evidence_attacher=self._camera_evidence_attachers.get(camera.camera_id),
            diagnostics=self.diagnostics,
            max_frames=self._max_frames_per_camera,
            live_view=self._live_view,
            debug_snapshots_provider=self._camera_debug_snapshots.get(camera.camera_id),
        )

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
            database_path=_evidence_outbox_path(),
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            resident_id=camera.resident_id,
            config_version=self.config.version,
            clock=time.time,
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

    def _compose_evidence_delivery(self) -> EvidenceExportRuntime | None:
        """Build the export runtime. Never gated on clip recording.

        The export gate is checked *before* ``ClipRecorderConfig()`` is built.
        That config reads clip-only environment variables and raises on
        malformed values, so building it unconditionally would let a stray clip
        setting fail a worker that has both clip recording and export switched
        off -- a failure with no relation to anything that worker uses.
        """
        if not export_enabled():
            return None
        clip_config = ClipRecorderConfig()
        try:
            return EvidenceExportRuntime.from_environment(
                store_dir=clip_config.store_dir,
                relay_url=self.config.relay.url,
                relay_token=self.config.relay.token.get_secret_value(),
                probe_camera_id=self.config.cameras[0].camera_id,
                database_path=_evidence_outbox_path(),
            )
        except ValueError as exc:
            # ADR-0003: the env gate (ML_WORKER_EVENT_CLIP_EXPORT_ENABLED) is the
            # explicit opt-out and returns None on its own. Reaching this branch
            # means the operator switched export ON and left it misconfigured, so
            # degrading to "no delivery" would hide a configuration error behind a
            # worker that looks healthy. Fail closed. The message is sanitized --
            # the relay URL and token stay on __cause__, never in this text.
            raise EvidenceDeliveryError(
                "evidence export is enabled but misconfigured: relay URL, relay "
                "token, and probe camera id are all required"
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
        clip_config = ClipRecorderConfig()
        try:
            with ClipStoreLock.acquire(clip_config.store_dir):
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
        clip_config = ClipRecorderConfig()
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

        recorder = ClipRecorder(
            clip_config,
            services=default_services(clip_config, boot.encode),
            is_clip_held=None if evidence_runtime is None else evidence_runtime.is_clip_held,
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
            LOGGER.warning("clip recorder failed to start; clips disabled", exc_info=True)
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
        return compose_camera_ingest_loop(
            camera,
            bus,
            reporter,
            decode=self._boot.decode,
            registry=self._ingest_source_registry(),
        )

    def _ingest_source_registry(self) -> SourceRegistry:
        if self._camera_source_registry is None:
            self._camera_source_registry = build_camera_source_registry(self.config.cameras)
        return self._camera_source_registry

    def _build_decision_stage(
        self, camera: CameraRuntimeConfig, fall_model: FallModelProtocol
    ) -> tuple[EventAggregator, Mapping[str, Mapping[str, object]], Mapping[str, Decider]]:
        domain_names = self.config.enabled_domains
        if domain_names is None:
            domain_names = enabled_domains()
        deciders = tuple(self._build_decider(name, camera, fall_model) for name in domain_names)
        domain_deciders = dict(zip(domain_names, deciders, strict=True))
        domain_audit = {name: self._build_domain_audit(name, fall_model) for name in domain_names}
        incidents = IncidentManager(identity_path=event_identity_path(camera.camera_id))
        aggregator = EventAggregator(deciders=deciders, incidents=incidents)
        return aggregator, domain_audit, domain_deciders

    def _build_domain_audit(self, name: str, fall_model: FallModelProtocol) -> Mapping[str, object]:
        """Precompute one domain's static audit envelope (GAP #1, todo 20).

        `DomainRegistration.audit_metadata_provider` only takes an
        `AuditContext` (worker/domains/registry.py's `_audit_snapshot` is a
        pure passthrough), so this is safe to resolve once per camera build
        rather than per event -- mirrors edge's `_attach_alert_metadata`
        (edge/runtime/camera_worker.py:289-336) using `build_audit_envelope`.
        """
        registration = DOMAIN_REGISTRY[name]
        if registration.audit_metadata_provider is None:
            return {}
        snapshot = registration.audit_metadata_provider(
            self._domain_audit_context(name, fall_model)
        )
        return build_audit_envelope(
            model_version=snapshot.model_version,
            detector_version=DETECTOR_VERSION,
            operating_threshold=snapshot.operating_threshold,
        )

    def _domain_audit_context(self, name: str, fall_model: FallModelProtocol) -> AuditContext:
        """Resolve model identity for the audit trail.

        Only "fall" has an ML model; `FallModelProtocol` has no `model_version`
        field, so `LstmFallRunner.manifest.artifact_digest` is the closest
        existing identity concept (read defensively -- a serving-client model
        may not expose `.manifest` at all). "bed_exit" is a geometric monitor
        with no model and no probability-threshold analog worth misrepresenting
        as one, so it gets an empty context (still yields a `clock_source`
        envelope from `build_audit_envelope`).
        """
        if name != "fall":
            return AuditContext(model_version=None, operating_threshold=None)
        manifest = getattr(fall_model, "manifest", None)
        model_version = None if manifest is None else getattr(manifest, "artifact_digest", None)
        return AuditContext(
            model_version=model_version,
            operating_threshold=fall_model.operating_threshold,
        )

    def _build_decider(
        self, name: str, camera: CameraRuntimeConfig, fall_model: FallModelProtocol
    ) -> Decider:
        registration = DOMAIN_REGISTRY[name]
        if name == "fall":
            dependencies: object = FallDomainDependencies(
                model=fall_model,
                camera_id=camera.camera_id,
                facility_id=camera.facility_id,
            )
        elif name == "bed_exit":
            dependencies = BedExitDomainDependencies(
                config=self._bed_exit_config(camera),
                clock=lambda: datetime.now(UTC),
            )
        else:
            raise RuntimeError(f"unsupported domain in registry: {name}")
        decider = registration.factory(dependencies)
        # bed_exit gates its own window internally (BedExitMonitor.update()
        # keeps tracking per-frame containment/latch state regardless of the
        # window and only gates final event *emission*); wrapping it here as
        # well would freeze that internal state while the window is closed,
        # which would delay/alter its emitted events at window-open
        # boundaries. So bed_exit keeps its existing wiring and is never
        # wrapped by the common gate below -- every other domain is.
        if name == "bed_exit":
            return decider
        window = self._resolved_window(name)
        if window is None:
            return decider
        return _WindowGatedDecider(decider, window, clock=lambda: datetime.now(UTC))

    def _resolved_window(self, name: str) -> DetectionWindow | None:
        configured = self.config.domains.resolved_detection_window(name)
        if configured is None:
            return None
        return DetectionWindow(start=configured.start, end=configured.end, tz=configured.tz)

    def _bed_exit_config(self, camera: CameraRuntimeConfig) -> BedExitConfig:
        configured = self.config.domains.resolved_detection_window("bed_exit")
        night_window = (
            None
            if configured is None
            else NightWindow(start=configured.start, end=configured.end, tz=configured.tz)
        )
        return BedExitConfig(
            camera_id=camera.camera_id,
            facility_id=camera.facility_id,
            night_window=night_window,
        )


__all__ = [
    "CameraRuntimeContext",
    "HeartbeatReporter",
    "WorkerRuntime",
    "production_boot_dependencies",
    "production_decode_probe",
]
