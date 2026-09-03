"""Worker-config composition coverage for the V2 fall and bed-exit domains."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from contracts.runner import Image, RunnerResult
from shared.events.evidence_http_transport import HttpResult
from worker.domains.bed_exit import BedExitMonitor
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallV2Probabilities
from worker.domains.registry import DETECTION_MODULE_REGISTRY
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.pipeline.perception.pts_resample import CADENCE_NS
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime
from worker.types import BusinessEvent, DecisionInput

ServingOption = str | int | float | bool | None


def _compiled_identity(task: str) -> tuple[str, str]:
    component_id = "fall-classifier" if task == "fall" else task
    module_id = "bed_exit" if component_id in {"person", "bed"} else "fall"
    binding = next(
        binding
        for binding in DETECTION_MODULE_REGISTRY.get(module_id).shared_bindings
        if binding.component_id == component_id
    )
    assert binding.artifact_digest is not None
    assert binding.preprocessing_identity is not None
    return binding.artifact_digest, binding.preprocessing_identity


@final
class _FakeRunner:
    """V2 bundle seam used by the real registry-composed decider."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.artifact_digest, self.preprocessing_identity = _compiled_identity(task)
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("composition tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> FallV2Probabilities:
        return FallV2Probabilities(0.0, 0.99, 0.1)

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _FakeServingClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, _FakeRunner]] = []

    def create(self, task: str, **_options: ServingOption) -> _FakeRunner:
        runner = _FakeRunner(task)
        self.created.append((task, runner))
        return runner


@final
class _NoOpLoop:
    def __init__(self, camera_id: str, reporter: IngestReporter) -> None:
        self.camera_id = camera_id
        self.reporter = reporter

    def run(self) -> None:
        self.reporter.mark_starting(self.camera_id)
        self.reporter.mark_ready(self.camera_id)

    def stop(self) -> None:
        return None


@final
class _NoOpPump:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id

    def run(self) -> None:
        return None

    def stop(self) -> None:
        return None


def _loop_factory(
    camera: CameraRuntimeConfig, _bus: BoundedFrameBus, reporter: IngestReporter
) -> _NoOpLoop:
    return _NoOpLoop(camera.camera_id, reporter)


def _pump_factory(
    camera: CameraRuntimeConfig,
    _bus: BoundedFrameBus,
    _analytics: object,
    _decision: object,
    _sink: object,
) -> _NoOpPump:
    return _NoOpPump(camera.camera_id)


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


def _config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "domains": {"enabled": ["fall", "bed_exit"]},
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://example.test/camera-1",
                }
            ],
        }
    )


@pytest.fixture(autouse=True)
def _fall_model_via_serving_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the V2 bundle runner through the serving seam."""

    def _fall_via_serving(self: WorkerRuntime, _device: str) -> object:
        return self._serving.create("fall")  # noqa: SLF001

    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", _fall_via_serving)


def _build_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> WorkerRuntime:
    _stub_heartbeat_transport(monkeypatch)
    return WorkerRuntime(
        _config(),
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=_FakeServingClient(),
        loop_factory=_loop_factory,
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        state_dir=tmp_path,
        clip_store_dir=tmp_path / "clip-store",
    )


def test_worker_activation_uses_the_injected_clip_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    runtime.run()

    clip_store_dir = tmp_path / "clip-store"
    evidence_runtime = runtime._evidence_export_runtime  # noqa: SLF001
    assert evidence_runtime is not None
    assert evidence_runtime.store_dir == clip_store_dir
    assert runtime._snapshot_store.store_dir == clip_store_dir  # noqa: SLF001
    assert clip_store_dir.is_dir()


def _fall_frame(time_sec: float, frame_index: int) -> DecisionInput:
    # A single live track with 17 COCO keypoints. V2 requires the production
    # 30x56 window and emits every fifth 15fps frame.
    pose = tuple((index, index + 1, 0.8) for index in range(17))
    return DecisionInput(
        observation=FrameObservation(
            detections=((BoundingBox(0, 0, 50, 80, 0.9),), ()),
            track_ids=(1,),
            poses=(pose,),
        ),
        frame_width=100,
        frame_height=100,
        live_track_ids=(1,),
        time_sec=time_sec,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def _box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1, y1, x2, y2, confidence)


def _bed_exit_frame(
    person: BoundingBox, bed: BoundingBox, time_sec: float, frame_index: int
) -> DecisionInput:
    # No track_ids/live_track_ids: BedExitMonitor falls back to positional
    # person ids, and the fall classifier ignores frames with no keypoints
    # -- so this cannot spuriously perturb the already-latched fall state.
    return DecisionInput(
        observation=FrameObservation(detections=((person,), ()), regions=((bed,), ())),
        frame_width=200,
        frame_height=200,
        live_track_ids=(),
        time_sec=time_sec,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH),
    )


def test_full_worker_config_constructs_real_deciders_and_emits_a_fall_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    runtime.run()

    assert len(runtime.cameras) == 1
    camera = runtime.cameras[0]
    fall, bed_exit = camera.decision.deciders
    assert isinstance(fall, FallV2DomainDecider)
    assert isinstance(bed_exit, BedExitMonitor)
    assert isinstance(fall.policy, FallPolicyDeciderV2)
    # The classifier's model is the exact fall runner WorkerRuntime built
    # from this WorkerConfig via the registry -- real construction, not a
    # hand-assembled decider.
    assert fall.classifier.model is runtime.fall_model

    emitted: list[BusinessEvent] = []
    for frame_index in range(60):
        emitted.extend(
            camera.decision.update(
                _fall_frame(frame_index * CADENCE_NS / 1_000_000_000, frame_index)
            )
        )

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, BusinessEvent)
    assert event.domain == "fall"
    assert event.event_type == "fall"
    assert event.camera_id == "camera-1"
    assert event.facility_id == "facility-1"
    assert event.time_sec > 0.0


def test_full_worker_config_constructs_real_deciders_and_emits_a_bed_exit_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.run()
    camera = runtime.cameras[0]
    bed = _box(0, 0, 100, 100)
    inside = _box(10, 10, 90, 90)
    outside = _box(150, 150, 190, 190)

    # hold_frames=2 (unconfigured BedExitConfig default): two contained
    # frames are required before the person is assigned to the bed.
    admitted: tuple[BusinessEvent, ...] = ()
    for frame_index in range(2):
        admitted = camera.decision.update(
            _bed_exit_frame(inside, bed, float(frame_index), frame_index)
        )
        assert admitted == ()

    # grace_frames=3 (same default): an event fires only once grace_frames
    # is exceeded, i.e. on the 4th consecutive uncontained frame.
    for offset in range(1, 5):
        frame_index = 2 + offset - 1
        admitted = camera.decision.update(
            _bed_exit_frame(outside, bed, float(frame_index), frame_index)
        )
        if offset < 4:
            assert admitted == ()

    assert len(admitted) == 1
    event = admitted[0]
    assert isinstance(event, BusinessEvent)
    assert event.domain == "bed_exit"
    assert event.event_type == "bed-exit"
    assert event.camera_id == "camera-1"
    assert event.facility_id == "facility-1"
    assert event.person_id == 0
    assert event.bed_id == 0


def test_preflight_reuses_the_boot_scoped_incident_manager_on_reconnect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A source rebuild recreates the V2 deciders on the new epoch identity but
    keeps the camera's IncidentManager: cooldown is boot-scoped, exactly as the
    replay engine models boot segments versus reconnect epochs."""
    runtime = _build_runtime(monkeypatch, tmp_path)
    runtime.run()
    camera = runtime.config.cameras[0]

    first = runtime._preflight_camera_graph(  # noqa: SLF001
        camera, episode_source_identity=("boot", "1", 0)
    )
    rebuilt = runtime._preflight_camera_graph(  # noqa: SLF001
        camera, episode_source_identity=("boot", "2", 1), incidents=first.decision.incidents
    )

    assert rebuilt.decision.incidents is first.decision.incidents
    first_fall, _ = first.decision.deciders
    rebuilt_fall, _ = rebuilt.decision.deciders
    assert isinstance(first_fall, FallV2DomainDecider)
    assert isinstance(rebuilt_fall, FallV2DomainDecider)
    assert rebuilt_fall is not first_fall
    assert (rebuilt_fall.policy.stream_epoch, rebuilt_fall.policy.source_generation) == ("2", 1)
    assert (first_fall.policy.stream_epoch, first_fall.policy.source_generation) == ("1", 0)
