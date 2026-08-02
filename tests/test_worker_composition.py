from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.runner import Image, RunnerResult
from shared.events.evidence_http_transport import HttpResult
from worker.adapters.model.errors import FatalAcceleratorError
from worker.domains.bed_exit import BedExitMonitor
from worker.domains.fall import FallEventLatch
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.faults.handler import FaultHandler
from worker.runtime.faults.record import FirstFaultRecord
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class _FakeRunner:
    def __init__(self, task: str) -> None:
        self.task = task
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("composition tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


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
    assert [task for task, _runner in serving.created] == ["pose", "person", "bed", "fall"]
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
