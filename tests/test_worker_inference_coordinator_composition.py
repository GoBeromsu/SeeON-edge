from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, final

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, bed_result, pose_result
from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import ModelRegistry
from worker.domains.module_definition import ComponentBinding
from worker.pipeline.analytics import NamedExtractor
from worker.runtime.config import WorkerConfig
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.worker import DETECTION_MODULE_REGISTRY, WorkerRuntime
from worker.types import FallModelInput, FramePacket


def _binding(component_id: str) -> ComponentBinding:
    return next(
        binding
        for definition in DETECTION_MODULE_REGISTRY.definitions
        for binding in definition.shared_bindings
        if binding.component_id == component_id
    )


@final
class _RecordingRunner:
    def __init__(self, task: str, device: str, forwards: list[tuple[int, str]]) -> None:
        binding = _binding(task)
        self.task = task
        self.device = device
        self.artifact_digest = binding.artifact_digest
        self.preprocessing_identity = binding.preprocessing_identity
        self._forwards = forwards

    def __call__(self, _image: Image) -> RunnerResult:
        return bed_result(()) if self.task == "bed" else pose_result((), ())

    def run_batch(self, images: Sequence[Image]) -> tuple[RunnerResult, ...]:
        self._forwards.append((id(self), self.device))
        return tuple(pose_result((), ()) for _image in images)

    def warmup(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class _FallModel:
    metadata = _FallMetadata()
    operating_threshold = 0.5

    def __init__(self) -> None:
        binding = _binding("fall-classifier")
        self.artifact_digest = binding.artifact_digest
        self.preprocessing_identity = binding.preprocessing_identity

    def predict(self, _features: FallModelInput) -> float:
        return 0.0

    def warmup(self) -> None:
        return None


def _config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "relay": {"url": "http://relay.test", "token": "token"},
            "clip": {"enabled": False},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://camera-a.test/stream",
                }
            ],
        }
    )


def _packet() -> FramePacket:
    return FramePacket(
        "camera-a",
        Frame(index=1, time_sec=1.0, image=np.zeros((2, 2, 3), dtype=np.uint8)),
        1.0,
        1,
        2,
        2,
        0.0,
    )


def test_composed_coordinator_removes_pose_and_reuses_the_cuda_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[_RecordingRunner] = []
    forwards: list[tuple[int, str]] = []
    registry = ModelRegistry()

    def factory(task: str):
        def create(*, device: str = "cpu") -> _RecordingRunner:
            runner = _RecordingRunner(task, device, forwards)
            created.append(runner)
            return runner

        return create

    registry.register("pose", factory("pose"))
    registry.register("bed", factory("bed"))
    serving = InProcessServingClient(registry)
    runtime = WorkerRuntime(
        _config(),
        serving_client=serving,
        loop_factory=lambda *_args: SimpleNamespace(run=lambda: None, stop=lambda: None),
        state_dir=tmp_path,
        clip_store_dir=tmp_path / "clips",
    )
    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", lambda *_args: _FallModel())
    profile = PROFILE_REGISTRY["nvidia-host-bridge"]
    boot = BootContext(profile, profile.device, profile.decode, profile.encode)
    graph = runtime._initialize_models(boot)  # noqa: SLF001
    runtime._warmed_component_ids = frozenset(graph.components)  # noqa: SLF001
    camera = runtime.config.cameras[0]
    plan = runtime._preflight_camera_graph(camera)  # noqa: SLF001
    context = runtime._build_camera(camera, runtime.shared_yolo, plan)  # noqa: SLF001
    assert runtime.watchdog is not None
    coordinator = runtime._compose_inference_coordinator(  # noqa: SLF001
        graph, runtime.watchdog, (context,)
    )
    assert coordinator is not None

    try:
        assert [extractor.module_name for extractor in context.analytics.extractors] == ["bed"]
        assert context.inference_results is not None
        assert context.pump._results is context.inference_results  # type: ignore[attr-defined]  # noqa: SLF001
        assert coordinator._lanes[0].results is context.inference_results  # noqa: SLF001

        pooled_pose = graph.components["pose"]
        assert isinstance(pooled_pose, NamedExtractor)
        context.bus.publish(_packet())
        assert coordinator.run_cycle() == 1
        delivered = context.inference_results.take(timeout_sec=0)
        assert delivered is not None
        delivered.packet.release()

        pose_runners = [runner for runner in created if runner.task == "pose"]
        assert [(runner.device, runner is pooled_pose.runner) for runner in pose_runners] == [
            ("cuda", True)
        ]
        assert forwards == [(id(pooled_pose.runner), "cuda")]
    finally:
        coordinator.stop()
        context.pump.stop()
        context.bus.close()
