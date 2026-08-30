"""Worker-side successor of the per-camera fall/decision-state characterization.

Original edge coverage (edge/runtime/edge_worker.py, edge/runtime/camera_worker.py)
and its worker-side disposition:

- ``test_fall_classifier_state_is_per_camera`` (distinct tracker/detector/
  classifier instances per camera, one shared fall model across all cameras,
  one shared pose/bed runner per process) -> superseded 1:1 by
  ``tests/test_worker_composition.py::test_four_cameras_isolate_decision_state_and_share_the_fall_model``
  (worker/runtime/worker.py:197-227), which asserts across four real cameras
  that ``fall.classifier.model is runtime.fall_model`` while every
  ``FallEventLatch``/``BedExitMonitor``/``EventAggregator``/``IncidentManager``
  instance is distinct per camera, and by
  ``test_two_cameras_isolate_mutable_state_and_share_yolo_extractors``
  (worker/runtime/worker.py:168-195), which asserts the shared pose/person/bed
  runner identity across cameras (``["pose", "person", "bed", "fall"]`` created
  exactly once). Not duplicated here.

Two guarantees from that same cluster had no worker-side home and are ported
below at the composition-root level:

- Each camera gets its own ``DurableEvidenceStager`` (worker's replacement for
  edge's evidence-sink wiring) rather than a shared one -- worker's analog of
  ``test_supervisor_wires_the_camera_specific_durable_stager``.
- One tracker-assigned identity flows unchanged into every registered
  decider from a single frame -- worker's analog of
  ``test_camera_worker_stamps_one_shared_tracker_id_for_fall_and_bed_exit``.
  This decomposes cleanly along worker's architecture into two already-proven
  halves plus one still-missing link: (a) ``CompositeExtractor``+
  ``GreedyIouTracker`` produce exactly one ``track_ids`` tuple that feeds one
  ``DecisionInput`` per frame (already covered by
  ``tests/test_worker_composite_extractor.py``'s
  ``result.observation.track_ids`` assertions); (b) the missing link, ported
  below, is that ``EventAggregator.update`` (worker/pipeline/decision/event_aggregator.py)
  hands that *same* ``DecisionInput`` object to every registered decider in one
  generator expression, so nothing downstream of the tracker can see a
  different identity than any other decider does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, final

import pytest

from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from contracts.runner import Image, RunnerResult
from worker.domains.fall import FallPolicyDeciderV2, FallV2Probabilities
from worker.domains.registry import DETECTION_MODULE_REGISTRY
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime
from worker.types import BusinessEvent, DecisionInput


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


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
    def __init__(self, task: str) -> None:
        self.task = task
        self.metadata = _FallMetadata()
        self.artifact_digest, self.preprocessing_identity = _compiled_identity(task)
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("composition tests must not run model inference")

    def predict(self, _features: object) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _FakeServingClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, _FakeRunner]] = []

    def create(self, task: str, **_options: str | int | float | bool | None) -> _FakeRunner:
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
                }
                for camera_id in camera_ids
            ],
        }
    )


@pytest.fixture(autouse=True)
def _fall_model_via_serving_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """This composition test predates explicit fall-model configuration and
    relies on the fall runner coming from the injected ``_FakeServingClient``,
    not a real LSTM artifact on disk. ``_create_fall_model`` no longer falls
    back to the serving client in production (fail-closed boot, see
    ``WorkerRuntime._create_fall_model``), so pin the old behavior here,
    scoped to this test module only."""

    def _fall_via_serving(self: WorkerRuntime, _device: str) -> object:
        return self._serving.create("fall")  # noqa: SLF001

    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", _fall_via_serving)


def test_worker_wires_a_distinct_durable_stager_per_camera(tmp_path: Path) -> None:
    serving = _FakeServingClient()
    captured: dict[str, object] = {}

    def _pump_factory(
        camera: CameraRuntimeConfig,
        _bus: BoundedFrameBus,
        _analytics: object,
        _decision: object,
        sink: object,
    ) -> _NoOpPump:
        captured[camera.camera_id] = sink
        return _NoOpPump(camera.camera_id)

    runtime = WorkerRuntime(
        _config("camera-a", "camera-b"),
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=serving,
        loop_factory=_loop_factory,
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        state_dir=tmp_path,
        clip_store_dir=tmp_path / "clip-store",
    )

    runtime.run()

    assert set(captured) == {"camera-a", "camera-b"}
    sink_a, sink_b = captured["camera-a"], captured["camera-b"]
    assert isinstance(sink_a, EvidenceEventSink)
    assert isinstance(sink_b, EvidenceEventSink)
    assert sink_a is not sink_b
    assert sink_a.stager is not sink_b.stager
    assert isinstance(sink_a.stager, DurableEvidenceStager)
    assert isinstance(sink_b.stager, DurableEvidenceStager)
    assert sink_a.stager.camera_id == "camera-a"
    assert sink_b.stager.camera_id == "camera-b"
    assert sink_a.stager.facility_id == "facility-a"
    # Both cameras publish through the process-wide delivery queue.
    assert sink_a.stager.queue_directory == sink_b.stager.queue_directory


@dataclass(slots=True)
class _RecordingDecider:
    seen: list[DecisionInput] = field(default_factory=list)

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        self.seen.append(input_value)
        return ()


def test_event_aggregator_shares_one_decision_input_across_every_registered_decider(
    tmp_path: Path,
) -> None:
    fall_recorder = _RecordingDecider()
    bed_exit_recorder = _RecordingDecider()
    aggregator = EventAggregator(
        deciders=(fall_recorder, bed_exit_recorder),
        incidents=IncidentManager(identity_path=tmp_path / "identities.jsonl"),
    )
    frame = DecisionInput(
        observation=FrameObservation(track_ids=(42,)),
        frame_width=10,
        frame_height=10,
        live_track_ids=(42,),
        time_sec=1.0,
        frame_index=0,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )

    aggregator.update(frame)

    assert fall_recorder.seen == [frame]
    assert bed_exit_recorder.seen == [frame]
    # Not just equal -- the identical object, so no decider can observe a
    # tracker identity the others do not.
    assert fall_recorder.seen[0] is frame
    assert bed_exit_recorder.seen[0] is frame
    assert fall_recorder.seen[0].live_track_ids == (42,)
    assert bed_exit_recorder.seen[0].observation.track_ids == (42,)


def test_v2_fall_policy_state_is_camera_local() -> None:
    camera_a = FallPolicyDeciderV2(camera_id="camera-a", facility_id="facility")
    camera_b = FallPolicyDeciderV2(camera_id="camera-b", facility_id="facility")
    score = FallV2Probabilities(0.0, 0.7, 0.0)

    for frame in range(2):
        assert camera_a.update({1: score}, (1,), frame_index=frame, time_sec=float(frame)) == ()
    event = camera_a.update({1: score}, (1,), frame_index=2, time_sec=2.0)[0]

    assert event.camera_id == "camera-a"
    assert camera_b.generation_for(1) is None
