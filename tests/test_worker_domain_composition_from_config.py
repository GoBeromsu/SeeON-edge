"""Closes gap 11c (.omo/research/todo30c-triage.md): "full WorkerConfig ->
constructed detectors / emitted event" composition coverage lost when
test_ml_worker_yaml_runtime.py was rewritten off edge.* imports.

The lost tests (``test_worker_yaml_lstm_runtime_emits_fall_event``,
``test_domain_detectors_inject_bed_exit_night_window``,
``test_domain_detectors_leave_fall_without_time_gate``) drove
``edge_worker._build_supervisor``/``edge_worker._domain_detectors`` end to
end: full YAML config -> real detector construction -> a bounded run ->
one emitted fall/bed-exit event. CameraWorker-cluster-part-1
(tests/test_worker_composition.py, commit 50a5078) already characterizes
*registry-level* composition (a locally-built ``DomainRegistration`` composes
generically) and *shared-model* construction
(``test_four_cameras_isolate_decision_state_and_share_the_fall_model``
constructs real ``FallEventLatch``/``BedExitMonitor`` instances via
``WorkerRuntime._build_decider`` from a full ``WorkerConfig``), but nothing
worker-side then drives those real, config-constructed deciders with a
``DecisionInput`` and asserts an actual emitted ``BusinessEvent`` -- the
"...and emits an event" half of the original guarantee. That is what this
file adds.

Artifact handling: ``worker/runtime/worker.py._create_fall_model`` only
loads real LSTM weights from disk when ``config.models.fall`` is explicitly
set (``LstmFallRunner.from_artifact_dir(...)``); when it is left unset it now
refuses to boot (fail-closed: no implicit fallback model). This composition
test predates that change and still wants to exercise real decider wiring
without a real LSTM artifact on disk, so the autouse
``_fall_model_via_serving_client`` fixture below monkeypatches
``_create_fall_model`` back to ``self._serving.create("fall")`` -- the same
DI seam ``compose_yolo_extractors`` uses for pose/person/bed -- scoped to
this test module only. Real LSTM weight loading is therefore avoidable
entirely: this file drives a full, valid
``WorkerConfig`` (``domains.enabled = ["fall", "bed_exit"]``, one camera)
through the real ``WorkerRuntime.run()`` -> ``_build_decider`` path with a
``_FakeServingClient`` (same pattern as ``test_worker_composition.py``),
which still produces genuine ``FallEventLatch``/``BedExitMonitor``
instances wired to a real ``FallWindowClassifier`` and a real
``EventAggregator``/``IncidentManager`` -- only the underlying neural-net
call is faked, exactly as the existing composition suite already does. No
torch weights, no artifact fixtures needed; per the task's "do not force it"
guidance, forcing real weight loading here would only exercise
``LstmFallRunner``/``ModelLoadError`` paths that belong to
``worker/adapters/model`` unit tests, not composition-root wiring -- out of
scope for this gap.

Both events are covered ("if cheap" per the assignment): the fall event
needs only 2 synthetic frames (the fall model's window is 2, satisfied by
the fake runner's ``_FallMetadata``); the bed-exit event costs 6 frames
under ``_bed_exit_config``'s real (unconfigured) defaults
(``hold_frames=2``, ``grace_frames=3`` -- worker/domains/bed_exit/schema.py)
-- both driven through the *same* real ``camera.decision`` aggregator that
``WorkerRuntime`` built, proving the full chain: WorkerConfig -> real typed
deciders via the registry -> EventAggregator -> IncidentManager -> emitted
BusinessEvent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import pytest

import worker.runtime.worker as worker_module
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from shared.events.evidence_http_transport import HttpResult
from worker.domains.bed_exit import BedExitMonitor
from worker.domains.fall import FallEventLatch
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.pipeline.output.evidence.evidence_runtime import OUTBOX_PATH_ENV
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


@final
class _FakeRunner:
    """Same DI seam every composition test uses; ``predict`` is fixed high
    for the "fall" task only, so a real ``FallWindowClassifier`` (wired to
    this runner by the real ``WorkerRuntime``) reports a fall as soon as its
    2-frame window fills -- no real model weights involved.
    """

    def __init__(self, task: str) -> None:
        self.task = task
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def __call__(self, _image: object) -> object:
        raise AssertionError("composition tests must not run model inference")

    def predict(self, _features: object) -> float:
        return 0.99 if self.task == "fall" else 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _FakeServingClient:
    def __init__(self) -> None:
        self.created: list[tuple[str, _FakeRunner]] = []

    def create(self, task: str, **_options: object) -> _FakeRunner:
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
    """See module docstring: pins the pre-fail-closed fallback behavior for
    this test module only."""

    def _fall_via_serving(self: WorkerRuntime, _device: str) -> object:
        return self._serving.create("fall")  # noqa: SLF001

    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", _fall_via_serving)


def _build_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> WorkerRuntime:
    monkeypatch.setenv("ML_WORKER_STATE_DIR", str(tmp_path))
    monkeypatch.setenv(OUTBOX_PATH_ENV, str(tmp_path / "outbox.sqlite3"))
    _stub_heartbeat_transport(monkeypatch)
    return WorkerRuntime(
        _config(),
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=_FakeServingClient(),
        loop_factory=_loop_factory,
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
    )


def _fall_frame(time_sec: float, frame_index: int) -> DecisionInput:
    # A single live track with 17 COCO keypoints, enough to fill the fake
    # fall model's 2-frame window; bed_region is EMPTY since these frames
    # carry no bed/person detections for the bed-exit decider to act on.
    pose = tuple((index, index + 1, 0.8) for index in range(17))
    return DecisionInput(
        observation=FrameObservation(track_ids=(1,), poses=(pose,)),
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
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH, 0),
    )


def test_full_worker_config_constructs_real_deciders_and_emits_a_fall_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = _build_runtime(monkeypatch, tmp_path)

    runtime.run()

    assert len(runtime.cameras) == 1
    camera = runtime.cameras[0]
    fall, bed_exit = camera.decision.deciders
    assert isinstance(fall, FallEventLatch)
    assert isinstance(bed_exit, BedExitMonitor)
    # The classifier's model is the exact fall runner WorkerRuntime built
    # from this WorkerConfig via the registry -- real construction, not a
    # hand-assembled decider.
    assert fall.classifier.model is runtime.fall_model

    first = camera.decision.update(_fall_frame(0.0, 0))
    assert first == ()

    second = camera.decision.update(_fall_frame(1.0, 1))

    assert len(second) == 1
    event = second[0]
    assert isinstance(event, BusinessEvent)
    assert event.domain == "fall"
    assert event.event_type == "fall"
    assert event.camera_id == "camera-1"
    assert event.facility_id == "facility-1"
    assert event.time_sec == 1.0


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
