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
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.model_composition import PRODUCTION_EXTRACTOR_NAMES
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime

ServingOption = str | int | float | bool | None


def test_production_extractor_names_intentionally_includes_person() -> None:
    # No config toggle exists to omit "person": it is unconditionally
    # provisioned alongside "pose" and "bed" for every profile.
    assert PRODUCTION_EXTRACTOR_NAMES == ("pose", "person", "bed")


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

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("runner-sharing tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> float:
        return 0.0

    def warmup(self) -> None:
        return None


@final
class _RecordingServingClient:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, dict[str, ServingOption]]] = []

    def create(self, task: str, **options: ServingOption) -> _FakeRunner:
        self.create_calls.append((task, dict(options)))
        return _FakeRunner(task)


@final
class _FakeLoop:
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
class _NoOpPump:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id

    def run(self) -> None:
        return None

    def stop(self) -> None:
        return None


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
    def __init__(self) -> None:
        self.loops: list[_FakeLoop] = []

    def __call__(
        self,
        camera: CameraRuntimeConfig,
        _bus: BoundedFrameBus,
        reporter: IngestReporter,
    ) -> _FakeLoop:
        loop = _FakeLoop(camera.camera_id, reporter)
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


def _config(*camera_ids: str) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://127.0.0.1:8000", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": "facility-1",
                    "rtsp_url": f"rtsp://camera-{index}.local/trackID=2",
                    "heartbeat_interval_sec": 30.0,
                }
                for index, camera_id in enumerate(camera_ids, start=1)
            ],
        }
    )


def _runtime(
    config: WorkerConfig, serving: _RecordingServingClient, loops: _LoopFactory, state_dir: Path
) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=serving,
        loop_factory=loops,
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
    )


@pytest.fixture(autouse=True)
def _fall_model_via_serving_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """This composition test predates explicit fall-model configuration and
    asserts the fall runner comes from the injected serving client (see
    ``serving.create_calls`` below), not a real LSTM artifact on disk.
    ``_create_fall_model`` no longer falls back to the serving client in
    production (fail-closed boot, see ``WorkerRuntime._create_fall_model``),
    so pin the old behavior here, scoped to this test module only."""

    def _fall_via_serving(self: WorkerRuntime, _device: str) -> object:
        return self._serving.create("fall")  # noqa: SLF001

    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", _fall_via_serving)


def test_pose_person_bed_and_fall_runners_are_created_once_and_shared_across_four_cameras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    serving = _RecordingServingClient()
    loops = _LoopFactory()
    camera_ids = ("camera-1", "camera-2", "camera-3", "camera-4")
    runtime = _runtime(_config(*camera_ids), serving, loops, tmp_path)

    runtime.run()

    # Each task is created exactly once regardless of camera count; "device"
    # is forwarded to the YOLO extractors, the fall model takes no kwargs.
    # box_source defaults to "pose" (issue #44): person is never provisioned.
    assert serving.create_calls == [
        ("pose", {"device": "cpu"}),
        ("bed", {"device": "cpu"}),
        ("fall", {}),
    ]

    assert len(runtime.cameras) == len(camera_ids)
    shared_extractors = runtime.cameras[0].analytics.extractors
    for camera in runtime.cameras[1:]:
        assert camera.analytics.extractors == shared_extractors
        assert all(
            left.runner is right.runner
            for left, right in zip(camera.analytics.extractors, shared_extractors, strict=True)
        )
        fall, _bed_exit = camera.decision.deciders
        assert fall.classifier.model is runtime.fall_model
