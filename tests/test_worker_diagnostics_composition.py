"""``_build_camera`` wires each camera's bus and analytics into diagnostics
(issue #45).

Before this wiring, ``WorkerDiagnostics.register_bus`` and
``record_stage_timing`` were only ever exercised directly against a bare
``WorkerDiagnostics`` instance in unit tests -- never from the real
composition root. This asserts on the two seams ``_build_camera`` must use so
frame-drop counters and stage timings are actually populated at runtime:
``self.diagnostics.register_bus(camera_id, bus)`` right after the camera's
bus is built, and ``stage_timing_recorder=self.diagnostics`` passed into the
``CompositeExtractor`` it constructs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.runner import Image
from worker.domains.module_definition import ComponentBinding
from worker.pipeline.bus import BoundedFrameBus
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY, VerifyResult
from worker.runtime.worker import WorkerRuntime


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

    def __call__(self, _image: Image) -> object:
        raise AssertionError("composition tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> float:
        return 0.0

    def warmup(self) -> None:
        return None


def _compiled_binding_for_task(task: str) -> ComponentBinding:
    component_id = "fall-classifier" if task == "fall" else task
    return next(
        binding
        for definition in worker_module.DETECTION_MODULE_REGISTRY.definitions
        for binding in definition.shared_bindings
        if binding.component_id == component_id
    )


@final
class _FakeServingClient:
    def create(self, task: str, **_options: object) -> _FakeRunner:
        return _FakeRunner(task)


def _config(camera_id: str = "camera-a") -> WorkerConfig:
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
            ],
        }
    )


def _runtime(config: WorkerConfig, state_dir: Path) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=_FakeServingClient(),
        loop_factory=lambda *_a, **_k: SimpleNamespace(),
        pump_factory=lambda *_a, **_k: SimpleNamespace(),
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=lambda _code: None,
        state_dir=state_dir,
        clip_store_dir=state_dir / "clip-store",
    )


@pytest.fixture(autouse=True)
def _fall_model_via_serving_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fall_via_serving(self: WorkerRuntime, _device: str) -> object:
        return self._serving.create("fall")  # noqa: SLF001

    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", _fall_via_serving)


def _build_camera_through_preflight(runtime: WorkerRuntime) -> None:
    """Build through the same global graph and camera preflight as activation."""
    profile = PROFILE_REGISTRY["cpu"]
    boot = BootContext(profile, profile.device, profile.decode, profile.encode)
    _ = runtime._initialize_models(boot)  # noqa: SLF001
    _ = runtime._warm_models()  # noqa: SLF001
    camera = runtime.config.cameras[0]
    plan = runtime._preflight_camera_graph(camera)  # noqa: SLF001
    _ = runtime._build_camera(camera, runtime.shared_yolo, plan)  # noqa: SLF001


def test_build_camera_registers_the_bus_with_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[BoundedFrameBus] = []
    real_bus = worker_module.BoundedFrameBus

    def _capturing_bus(**kwargs: object) -> BoundedFrameBus:
        bus = real_bus(**kwargs)  # type: ignore[arg-type]
        captured.append(bus)
        return bus

    monkeypatch.setattr(worker_module, "BoundedFrameBus", _capturing_bus)

    runtime = _runtime(_config("camera-a"), tmp_path)
    _build_camera_through_preflight(runtime)

    assert captured, "composition never constructed a frame bus"
    (camera,) = runtime.diagnostics.snapshot().cameras
    assert camera.camera_id == "camera-a"
    subscription_names = {entry.name for entry in camera.bus}
    assert subscription_names == {"inference", "live", "evidence"}


def test_build_camera_wires_the_composite_extractor_to_record_stage_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_kwargs: list[dict[str, object]] = []
    real_composite_extractor = worker_module.CompositeExtractor

    def _capturing_composite_extractor(**kwargs: object) -> object:
        captured_kwargs.append(kwargs)
        return real_composite_extractor(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(worker_module, "CompositeExtractor", _capturing_composite_extractor)

    runtime = _runtime(_config("camera-a"), tmp_path)
    _build_camera_through_preflight(runtime)

    assert captured_kwargs, "composition never constructed a CompositeExtractor"
    assert captured_kwargs[0]["stage_timing_recorder"] is runtime.diagnostics
