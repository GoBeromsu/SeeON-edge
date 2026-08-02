"""Composition of the ``--max-frames-per-camera`` bounded-run cap.

Proves ``WorkerRuntime`` wires the configured cap into the supervisor's
completion watcher such that ``run()`` returns (the worker exits 0) only once
every camera's pump has reached the cap -- not just the first one to reach
it -- and that leaving the cap unset wires no completion watcher at all,
preserving today's run-forever default.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.runner import Image, RunnerResult
from shared.events.evidence_http_transport import HttpResult
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
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
    """Mirrors tests/test_worker_evidence_export_composition.py's fake:
    `_initialize_models` always composes real pose/person/bed/fall runners
    through `serving_client.create(...)`, even in composition tests that
    otherwise never touch inference, so this must return something usable
    rather than raise.
    """

    def __init__(self, task: str) -> None:
        self.task = task
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("max-frames composition tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _FakeServingClient:
    def create(self, task: str, **_options: object) -> _FakeRunner:
        return _FakeRunner(task)


@final
class _InstantLoop:
    """Ingest loop fake that reports ready and returns immediately."""

    def __init__(self, camera_id: str, reporter: IngestReporter) -> None:
        self.camera_id = camera_id
        self._reporter = reporter
        self.stop_count = 0

    def run(self) -> None:
        if self.stop_count:
            return
        self._reporter.mark_starting(self.camera_id)
        self._reporter.mark_ready(self.camera_id)

    def stop(self) -> None:
        self.stop_count += 1


@final
class _InstantLoopFactory:
    def __call__(
        self, camera: CameraRuntimeConfig, _bus: object, reporter: IngestReporter
    ) -> _InstantLoop:
        return _InstantLoop(camera.camera_id, reporter)


@final
class _CountingPump:
    """Self-terminating fake standing in for the real ``CameraPipelinePump``:
    increments ``processed_count`` up to ``cap`` (optionally paced by
    ``delay_sec``) so the supervisor's completion watcher has real per-camera
    progress to observe, without needing a real analytics/decision stack.
    """

    def __init__(self, camera_id: str, cap: int, *, delay_sec: float = 0.0) -> None:
        self.camera_id = camera_id
        self._cap = cap
        self._delay_sec = delay_sec
        self.processed_count = 0
        self.stop_count = 0

    def run(self) -> None:
        while self.processed_count < self._cap:
            if self._delay_sec:
                time.sleep(self._delay_sec)
            self.processed_count += 1

    def stop(self) -> None:
        self.stop_count += 1


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


def _runtime(
    config: WorkerConfig,
    pump_factory: Callable[..., object],
    state_dir: Path,
    *,
    max_frames_per_camera: int | None,
) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=_FakeServingClient(),
        loop_factory=_InstantLoopFactory(),
        pump_factory=pump_factory,  # type: ignore[arg-type]
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=lambda _code: None,
        max_frames_per_camera=max_frames_per_camera,
    )


@pytest.fixture(autouse=True)
def _fall_model_via_serving_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """These composition tests predate explicit fall-model configuration and
    rely on the fall runner coming from the injected ``_FakeServingClient``,
    not a real LSTM artifact on disk. ``_create_fall_model`` no longer falls
    back to the serving client in production (fail-closed boot, see
    ``WorkerRuntime._create_fall_model``), so pin the old behavior here,
    scoped to this test module only."""

    def _fall_via_serving(self: WorkerRuntime, _device: str) -> object:
        return self._serving.create("fall")  # noqa: SLF001

    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", _fall_via_serving)


def test_run_returns_only_once_every_camera_pump_reaches_the_configured_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    pumps: dict[str, _CountingPump] = {}

    def pump_factory(
        camera: CameraRuntimeConfig, _bus: object, _analytics: object, _decision: object,
        _sink: object,
    ) -> _CountingPump:
        # camera-slow lags behind camera-fast: if the supervisor stopped as
        # soon as any one camera reached the cap, camera-slow would be caught
        # short of it.
        delay = 0.0 if camera.camera_id == "camera-fast" else 0.02
        pump = _CountingPump(camera.camera_id, cap=4, delay_sec=delay)
        pumps[camera.camera_id] = pump
        return pump

    runtime = _runtime(
        _config("camera-fast", "camera-slow"), pump_factory, tmp_path, max_frames_per_camera=4
    )

    runtime.run()

    assert pumps["camera-fast"].processed_count == 4
    assert pumps["camera-slow"].processed_count == 4
    assert pumps["camera-fast"].stop_count >= 1
    assert pumps["camera-slow"].stop_count >= 1


def test_omitting_max_frames_per_camera_wires_no_completion_watcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_heartbeat_transport(monkeypatch)

    def pump_factory(
        camera: CameraRuntimeConfig, _bus: object, _analytics: object, _decision: object,
        _sink: object,
    ) -> _CountingPump:
        # This fake self-terminates on its own (cap=1) purely so run() returns
        # promptly in this test; that is independent of -- and must not be
        # mistaken for -- the supervisor's own completion-watcher wiring,
        # which is what this test actually asserts on below.
        return _CountingPump(camera.camera_id, cap=1)

    runtime = _runtime(_config("camera-a"), pump_factory, tmp_path, max_frames_per_camera=None)

    runtime.run()

    assert runtime._supervisor is not None  # noqa: SLF001
    assert runtime._supervisor._completion_check is None  # noqa: SLF001


def test_max_frames_per_camera_wires_the_runtimes_own_completion_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_heartbeat_transport(monkeypatch)

    def pump_factory(
        camera: CameraRuntimeConfig, _bus: object, _analytics: object, _decision: object,
        _sink: object,
    ) -> _CountingPump:
        return _CountingPump(camera.camera_id, cap=2)

    runtime = _runtime(_config("camera-a"), pump_factory, tmp_path, max_frames_per_camera=2)

    runtime.run()

    assert runtime._supervisor is not None  # noqa: SLF001
    # Bound-method attribute access mints a fresh wrapper object each time
    # (`a.method is a.method` is False in general), so equality -- which bound
    # methods define via `__self__`/`__func__` -- is the correct check here.
    assert runtime._supervisor._completion_check == runtime._max_frames_completion_check  # noqa: SLF001


def test_max_frames_completion_check_requires_every_camera_not_just_one(
    tmp_path: Path,
) -> None:
    """Direct unit-level lock on ``_max_frames_completion_check``'s own
    boundary logic: it must use ``all(...)`` over every camera's pump, not
    ``any(...)``, mirroring edge's ``_done`` (all cameras reached the cap).
    Exercised straight against the method with hand-placed fake pumps --
    bypassing the full boot sequence entirely -- to pin this specific
    boundary independently of the slower end-to-end composition test above
    (which proves the same thing indirectly, by asserting both cameras'
    final `processed_count`).
    """

    def pump_factory(
        camera: CameraRuntimeConfig, _bus: object, _analytics: object, _decision: object,
        _sink: object,
    ) -> _CountingPump:
        return _CountingPump(camera.camera_id, cap=2)

    runtime = _runtime(
        _config("camera-a", "camera-b"), pump_factory, tmp_path, max_frames_per_camera=2
    )

    pump_a = SimpleNamespace(processed_count=2)  # at cap
    pump_b = SimpleNamespace(processed_count=1)  # short of cap
    runtime.cameras = (
        SimpleNamespace(pump=pump_a),
        SimpleNamespace(pump=pump_b),
    )

    # Given: only one of two cameras has reached the cap.
    assert runtime._max_frames_completion_check() is False  # noqa: SLF001

    # When: the second camera also reaches the cap.
    pump_b.processed_count = 2

    # Then: only now does the completion check report done.
    assert runtime._max_frames_completion_check() is True  # noqa: SLF001
