from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast, final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.observation import BoundingBox, FrameObservation
from contracts.runner import Image, RunnerResult
from shared.events.evidence_http_transport import HttpResult
from worker.adapters.model.errors import FatalAcceleratorError
from worker.domains import DETECTION_MODULE_REGISTRY
from worker.domains.bed_exit import BedExitMonitor
from worker.domains.fall import FallEventLatch
from worker.domains.module_definition import ComponentBinding, ScheduleRule
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.faults.handler import FaultHandler
from worker.runtime.faults.record import FirstFaultRecord
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime
from worker.types import BusinessEvent, FramePacket


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class _FakeRunner:
    def __init__(self, task: str) -> None:
        binding = _compiled_binding_for_task(task)
        self.task = task
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.artifact_digest = binding.artifact_digest
        self.preprocessing_identity = binding.preprocessing_identity
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("composition tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


def _compiled_binding_for_task(task: str) -> ComponentBinding:
    component_id = "fall-classifier" if task == "fall" else task
    return next(
        binding
        for definition in DETECTION_MODULE_REGISTRY.definitions
        for binding in definition.shared_bindings
        if binding.component_id == component_id
    )


@final
class _FakeServingClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, _FakeRunner]] = []

    def create(
        self,
        task: str,
        **_options: str | int | float | bool | None,
    ) -> _FakeRunner:
        runner = _FakeRunner(task)
        self.created.append((task, runner))
        return runner


@final
class _FakeLoop:
    def __init__(
        self,
        camera_id: str,
        reporter: IngestReporter,
        *,
        warmed: Callable[[], bool],
        fatal: bool,
    ) -> None:
        self.camera_id = camera_id
        self.reporter = reporter
        self._warmed = warmed
        self._fatal = fatal
        self.started_after_warmup = False
        self.continued_after_ready = False
        self.stop_count = 0

    def run(self) -> None:
        if self.stop_count:
            return
        self.started_after_warmup = self._warmed()
        self.reporter.mark_starting(self.camera_id)
        if self._fatal:
            raise FatalAcceleratorError(
                "injected accelerator failure",
                camera_id=self.camera_id,
                task="pose",
            )
        self.reporter.mark_ready(self.camera_id)
        self.continued_after_ready = True

    def stop(self) -> None:
        self.stop_count += 1


@final
class _NoOpPump:
    """Fake pump loop: composition tests assert on ingest wiring, not pump
    throughput, so this returns immediately instead of polling an empty bus
    forever (the real ``CameraPipelinePump`` blocks until ``stop()``)."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.stop_count = 0

    def run(self) -> None:
        return None

    def stop(self) -> None:
        self.stop_count += 1


def _pump_factory(
    camera: CameraRuntimeConfig,
    _bus: BoundedFrameBus,
    _analytics: object,
    _decision: object,
    _sink: object,
) -> _NoOpPump:
    return _NoOpPump(camera.camera_id)


@final
class _LoopFactory:
    def __init__(self, serving: _FakeServingClient, *, fatal_camera_id: str | None = None) -> None:
        self._serving = serving
        self._fatal_camera_id = fatal_camera_id
        self.loops: list[_FakeLoop] = []

    def __call__(
        self,
        camera: CameraRuntimeConfig,
        _bus: BoundedFrameBus,
        reporter: IngestReporter,
    ) -> _FakeLoop:
        loop = _FakeLoop(
            camera.camera_id,
            reporter,
            warmed=lambda: all(runner.warmup_count == 1 for _, runner in self._serving.created),
            fatal=camera.camera_id == self._fatal_camera_id,
        )
        self.loops.append(loop)
        return loop


def _stub_heartbeat_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def request(
        _url: str,
        _method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        return 204, {}, b""

    monkeypatch.setattr(worker_module, "bounded_request", request)


@pytest.fixture(autouse=True)
def _fall_model_via_serving_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """These composition tests predate explicit fall-model configuration and
    assert the fall runner comes from the injected ``_FakeServingClient``
    (e.g. ``serving.created`` below), not a real LSTM artifact on disk.
    ``_create_fall_model`` no longer falls back to the serving client in
    production (fail-closed boot, see ``WorkerRuntime._create_fall_model``),
    so pin the old behavior here, scoped to this test module only."""

    def _fall_via_serving(self: WorkerRuntime, _device: str) -> object:
        return self._serving.create("fall")  # noqa: SLF001

    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", _fall_via_serving)


def test_two_cameras_isolate_mutable_state_and_share_yolo_extractors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    runtime = _runtime(_config("camera-a", "camera-b"), serving, loops, tmp_path)

    runtime.run()

    first, second = runtime.cameras
    assert first.bus is not second.bus
    assert first.tracker is not second.tracker
    assert first.scene_state is not second.scene_state
    assert first.scheduler is not second.scheduler
    assert first.analytics.extractors == second.analytics.extractors
    assert all(
        left is right
        for left, right in zip(
            first.analytics.extractors,
            second.analytics.extractors,
            strict=True,
        )
    )
    # box_source defaults to "pose" (issue #44): person is never provisioned.
    assert [task for task, _runner in serving.created] == ["pose", "bed", "fall"]
    assert all(loop.started_after_warmup for loop in loops.loops)


def test_four_cameras_isolate_decision_state_and_share_the_fall_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    camera_ids = ("camera-a", "camera-b", "camera-c", "camera-d")
    runtime = _runtime(_config(*camera_ids), serving, loops, tmp_path)

    runtime.run()

    assert len(runtime.cameras) == len(camera_ids)
    fall_deciders: list[FallEventLatch] = []
    bed_exit_deciders: list[BedExitMonitor] = []
    for camera_id, camera in zip(camera_ids, runtime.cameras, strict=True):
        fall, bed_exit = camera.decision.deciders
        assert isinstance(fall, FallEventLatch)
        assert isinstance(bed_exit, BedExitMonitor)
        assert fall.camera_id == camera_id
        # The fall model itself stays the one shared bundle built once per task per process.
        assert fall.classifier.model is runtime.fall_model
        fall_deciders.append(fall)
        bed_exit_deciders.append(bed_exit)

    # Per-camera decision state (deciders, aggregators, cooldown/idempotency tracking) is
    # never shared across cameras, even as the camera count grows.
    assert len(set(map(id, fall_deciders))) == len(camera_ids)
    assert len(set(map(id, bed_exit_deciders))) == len(camera_ids)
    assert len({id(camera.decision) for camera in runtime.cameras}) == len(camera_ids)
    assert len({id(camera.decision.incidents) for camera in runtime.cameras}) == len(camera_ids)


def test_ready_camera_posts_heartbeat_with_canonical_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[tuple[str, str, bytes | None]] = []

    def bounded_request(
        url: str,
        method: str,
        _headers: dict[str, str],
        data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        requests.append((url, method, data))
        return 204, {}, b""

    monkeypatch.setattr(worker_module, "bounded_request", bounded_request)
    serving = _FakeServingClient()
    runtime = _runtime(_config("camera-a"), serving, _LoopFactory(serving), tmp_path)

    runtime.run()

    assert requests == [
        (
            "http://relay.test/api/v1/relay/heartbeat",
            "POST",
            b'{"camera_id":"camera-a","facility_id":"facility-a","config_version":7}',
        )
    ]


def test_heartbeat_exception_is_nonfatal_and_camera_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def failing_request(
        _url: str,
        _method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        raise RuntimeError("relay unavailable")

    monkeypatch.setattr(worker_module, "bounded_request", failing_request)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    exit_codes: list[int] = []
    runtime = _runtime(
        _config("camera-a"), serving, loops, tmp_path, hard_exit=exit_codes.append
    )

    runtime.run()

    assert loops.loops[0].continued_after_ready is True
    assert runtime.cameras[0].heartbeat.failure_count == 1
    assert exit_codes == []


def test_fatal_accelerator_fault_stops_every_camera_and_exits_four(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    received: list[FatalAcceleratorError] = []
    real_handle = FaultHandler.handle

    def record_handle(
        handler: FaultHandler, exc: FatalAcceleratorError, record: FirstFaultRecord
    ) -> None:
        received.append(exc)
        real_handle(handler, exc, record)

    monkeypatch.setattr(FaultHandler, "handle", record_handle)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving, fatal_camera_id="camera-a")
    exit_codes: list[int] = []
    runtime = _runtime(
        _config("camera-a", "camera-b"),
        serving,
        loops,
        tmp_path,
        hard_exit=exit_codes.append,
    )

    runtime.run()

    assert len(received) == 1
    assert all(loop.stop_count >= 1 for loop in loops.loops)
    assert exit_codes == [4]


def test_worker_runtime_init_logs_resolved_state_directory(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``WorkerRuntime`` must log its resolved state directory at
    construction (issue #35's fixed resolver has no env override to inspect
    externally, so the log line is the operator-visible record of where it
    landed)."""
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)

    with caplog.at_level("INFO"):
        _runtime(_config("camera-a"), serving, loops, tmp_path)

    assert any(
        record.getMessage() == f"worker state directory resolved to {tmp_path}"
        for record in caplog.records
    )


def test_camera_with_persisted_bed_zone_polygon_seeds_scene_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A camera pulled with a persisted ``bed_zone_polygon`` (see the
    bed-zone recognize endpoint) must have its ``SceneState`` seeded with the
    equivalent ``BoundingBox`` at camera-build time, so bed-exit treats it as
    the authoritative bed region from frame one -- and a camera without one
    is unaffected."""
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    config = WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://example.test/camera-a",
                    "heartbeat_interval_sec": 30.0,
                    "bed_zone_polygon": [[1, 2], [9, 2], [9, 8], [1, 8]],
                    "bed_zone_image_width": 640,
                    "bed_zone_image_height": 480,
                },
                {
                    "camera_id": "camera-b",
                    "facility_id": "facility-b",
                    "rtsp_url": "rtsp://example.test/camera-b",
                    "heartbeat_interval_sec": 30.0,
                },
            ],
        }
    )
    runtime = _runtime(config, serving, loops, tmp_path)

    runtime.run()

    with_polygon, without_polygon = runtime.cameras
    assert with_polygon.scene_state.persisted_bed_regions == (
        BoundingBox(
            x1=1,
            y1=2,
            x2=9,
            y2=8,
            confidence=1.0,
            polygon=((1, 2), (9, 2), (9, 8), (1, 8)),
        ),
    )
    assert without_polygon.scene_state.persisted_bed_regions == ()
    # Issue #41: a persisted polygon is authoritative and never expires, so
    # scheduling the live bed-seg extractor on top of it would only pay its
    # cost for a result nothing ever reads.
    assert "bed" not in with_polygon.scheduler.task_intervals
    assert "bed" in without_polygon.scheduler.task_intervals


def test_bed_exit_disabled_excludes_bed_from_intervals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #47: the extraction schedule is derived from the union of active
    domains' declared ``requires``, not a fixed dict. With bed_exit disabled,
    only "fall" is active and its sole requirement is "pose" -- "bed" must
    never appear in the built schedule."""
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    config = WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "domains": {"enabled": ["fall"]},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://example.test/camera-a",
                    "heartbeat_interval_sec": 30.0,
                }
            ],
        }
    )
    runtime = _runtime(config, serving, loops, tmp_path)

    runtime.run()

    intervals = runtime.cameras[0].scheduler.task_intervals
    assert set(intervals) == {"pose"}
    assert "bed" not in intervals


def _registry_with_ghost_bed_exit() -> object:
    registry = DETECTION_MODULE_REGISTRY
    bed_exit = registry.get("bed_exit")
    ghost_bed_exit = replace(
        bed_exit,
        component_bindings=bed_exit.component_bindings
        + (
            ComponentBinding(
                component_id="ghost",
                component_kind="extractor",
                model_family="ghost-family",
                provisioner="test",
                artifact_digest="a" * 64,
                preprocessing_identity="ghost-v1",
                output_adapter="ghost",
                warmup_required=True,
            ),
        ),
        schedule_rules=bed_exit.schedule_rules + (ScheduleRule("ghost", "camera-frame-stride"),),
    )
    definitions = tuple(
        ghost_bed_exit if definition.module_id == "bed_exit" else definition
        for definition in registry.definitions
    )
    return replace(
        registry,
        definitions=definitions,
        by_id=MappingProxyType({**registry.by_id, "bed_exit": ghost_bed_exit}),
    )


def test_domain_requiring_an_unregistered_extractor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #47: a domain that declares a ``requires`` name with no matching
    provisioned extractor refuses to boot that camera -- fail-closed,
    mirroring ``_build_decider``'s existing unsupported-domain RuntimeError.
    A per-camera failure degrades only that camera (``run_camera_stage``), so
    this asserts the camera never activates rather than that ``run()``
    raises.
    """
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    monkeypatch.setattr(worker_module, "DETECTION_MODULE_REGISTRY", _registry_with_ghost_bed_exit())
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)

    with caplog.at_level("CRITICAL"), pytest.raises(SystemExit):
        runtime.run()

    assert runtime.cameras == ()
    assert any(
        "detection module requires unavailable component(s): ghost" in record.getMessage()
        for record in caplog.records
    )


def test_configured_cameras_that_all_fail_to_activate_do_not_hang_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """이슈 #150 회귀 방지: `run()`의 대기 분기는 **설정된 로스터**로 갈린다.

    0대 부팅을 허용하면서 `run()`이 `join()` 대신 `wait_until_stopped()`를
    타는 경로가 생겼는데, 그 판정을 활성화에 성공한 카메라(`self.cameras`)로
    하면 "카메라가 설정돼 있는데 전부 활성화에 실패한" 경우까지 무한 대기에
    빠진다 -- 실제로 `test_domain_requiring_an_unregistered_extractor_fails_closed`
    가 영원히 멈췄다. 그 경우는 예전처럼 즉시 반환해야 재시작 정책이 걸린
    환경에서 프로세스가 되살아난다.

    (0대 로스터가 대기 상태로 계속 떠 있는 쪽은
    `tests/test_worker_startup_config_resolution.py`가 고정한다.)
    """
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    monkeypatch.setattr(worker_module, "DETECTION_MODULE_REGISTRY", _registry_with_ghost_bed_exit())
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)

    with pytest.raises(SystemExit):
        runtime.run()

    assert runtime.config.cameras != ()
    assert runtime.cameras == ()


def test_default_box_source_schedules_pose_and_bed_without_person(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #44: box_source defaults to "pose", so with both domains active
    (fall + bed_exit, the default) the schedule is exactly their requires
    union -- person is never added."""
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)

    runtime.run()

    intervals = runtime.cameras[0].scheduler.task_intervals
    assert set(intervals) == {"pose", "bed"}
    assert "person" not in [task for task, _runner in serving.created]


def test_box_source_person_schedules_and_provisions_person(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #44: box_source="person" additionally schedules and provisions
    the person extractor, on top of the domain-declared pose/bed set."""
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    loops = _LoopFactory(serving)
    config = WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "models": {"box_source": "person"},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://example.test/camera-a",
                    "heartbeat_interval_sec": 30.0,
                }
            ],
        }
    )
    runtime = _runtime(config, serving, loops, tmp_path)

    runtime.run()

    intervals = runtime.cameras[0].scheduler.task_intervals
    assert set(intervals) == {"pose", "bed", "person"}
    assert "person" in [task for task, _runner in serving.created]


def _runtime(
    config: WorkerConfig,
    serving: _FakeServingClient,
    loops: _LoopFactory,
    state_dir: Path,
    *,
    hard_exit: Callable[[int], None] = lambda _code: None,
) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=serving,
        loop_factory=loops,
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=hard_exit,
        state_dir=state_dir,
        clip_store_dir=state_dir / "clip-store",
        build_revision="a" * 40,
    )


def _config(*camera_ids: str) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": f"facility-{camera_id.removeprefix('camera-')}",
                    "rtsp_url": f"rtsp://example.test/{camera_id}",
                    "heartbeat_interval_sec": 30.0,
                }
                for camera_id in camera_ids
            ],
        }
    )


def test_absent_trace_provenance_degrades_capture_not_the_camera(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Analysis-trace capture is auxiliary and must never fail a camera.

    A trace-publishing slice raised when runtime provenance was not yet applied.
    Because that raise happened inside the per-camera stage, `run_camera_stage`
    degraded the camera, and every camera went down together: no heartbeat, no
    runners, no detection. An optional QA capability must not be able to take
    fall detection offline, so the camera must still come up and heartbeat with
    capture simply absent.
    """
    requests: list[tuple[str, str, bytes | None]] = []

    def bounded_request(
        url: str,
        method: str,
        _headers: dict[str, str],
        data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        requests.append((url, method, data))
        return 204, {}, b""

    monkeypatch.setattr(worker_module, "bounded_request", bounded_request)
    serving = _FakeServingClient()
    runtime = _runtime(_config("camera-a"), serving, _LoopFactory(serving), tmp_path)

    def fail_manifest(**_kwargs: object) -> object:
        raise RuntimeError("injected manifest composition failure")

    monkeypatch.setattr(worker_module, "build_applied_runtime_manifest", fail_manifest)

    runtime.run()

    assert runtime._runtime_manifest is None  # noqa: SLF001 - composition outcome
    assert runtime.cameras
    assert runtime._camera_trace_captures == {}  # noqa: SLF001 - tracing degraded only
    assert requests == [
        (
            "http://relay.test/api/v1/relay/heartbeat",
            "POST",
            b'{"camera_id":"camera-a","facility_id":"facility-a","config_version":7}',
        )
    ], "the camera did not come up, so an auxiliary capability took detection down"


def test_runtime_manifest_is_applied_to_emitted_event_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    runtime = _runtime(_config("camera-a"), serving, _LoopFactory(serving), tmp_path)

    runtime.run()

    manifest = runtime._runtime_manifest  # noqa: SLF001 - composition outcome
    assert manifest is not None
    attacher = replace(
        runtime._camera_evidence_attachers["camera-a"],  # noqa: SLF001
        overlay_renderer=None,
    )
    emitted = attacher.attach(
        BusinessEvent("fall", "fall.detected", "event-1", "camera-a", "facility-a", 1.0, 0.9),
        cast(FramePacket, None),  # no renderer reads the packet in this composition test
        cast(
            FrameObservation, None
        ),  # no renderer reads the observation in this composition test
    )
    assert emitted.audit is not None
    assert emitted.audit["runtime_manifest_sha256"] == manifest.sha256


def test_applied_manifest_makes_the_trace_producer_live(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A composed manifest must actually switch the analysis-trace producer on.

    `_runtime_manifest` was initialised to None and never assigned, so
    `_build_trace_capture` always returned None, capture and writer were never
    composed, and `AnalysisTraceSender.send` was unreachable on a live frame.
    The tables it feeds therefore stayed empty in production and the replay
    command refused forever -- a consumer wired to a producer that never ran.
    """
    _stub_heartbeat_transport(monkeypatch)
    serving = _FakeServingClient()
    runtime = _runtime(_config("camera-a"), serving, _LoopFactory(serving), tmp_path)

    runtime.run()

    assert runtime._runtime_manifest is not None, (  # noqa: SLF001
        "no manifest was applied, so trace capture degrades and the producer is inert"
    )
    # The writer is created at the top of run() and cleared on shutdown, so
    # after run() returns only the per-camera captures remain observable. Their
    # presence is the necessary condition: a camera with no capture never
    # reaches the publisher at all.
    assert runtime._camera_trace_captures.get("camera-a") is not None, (  # noqa: SLF001
        "the camera has no trace capture, so its frames never reach the publisher"
    )


def test_a_captured_frame_actually_reaches_the_trace_publisher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Composition is necessary but not sufficient; the frame must arrive.

    Two slices in this effort shipped a consumer whose producer never ran. The
    composed capture proves the wiring exists; this proves a frame submitted
    through the real writer reaches the publisher callable that posts to the
    backend relay route.
    """
    published: list[tuple[object, ...]] = []

    def _capture_publisher(frames: tuple[object, ...], truncation: object) -> None:
        published.append(frames)

    from worker.pipeline.trace import AnalysisTrace, OptionalNumber, TraceFrame
    from worker.pipeline.trace.writer import BoundedTraceWriter

    writer = BoundedTraceWriter(tmp_path / "runtime-analysis", publisher=_capture_publisher)
    writer.start()
    try:
        frame = TraceFrame(
            AnalysisTrace(
                trace_id="analysis-1",
                frame_key=("boot-a", "camera-a", 1, 1),
                pts=OptionalNumber(1.0),
                source_time=OptionalNumber(1.0),
                frame_width=4,
                frame_height=4,
                bed_region_provenance="fresh",
                persons=(),
                beds=(),
                components=(),
            ),
            (),
        )
        assert writer.submit(frame, require_persisted=True)
    finally:
        writer.stop()

    assert published, (
        "no frame reached the publisher, so runtime_analysis_* stays empty in "
        "production and the replay command has nothing to recover"
    )
