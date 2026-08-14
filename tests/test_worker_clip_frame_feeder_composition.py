"""Composition of production gap A: ``bus.evidence`` -> ``ClipFrameFeeder`` ->
the shared ``ClipRecorder.on_frame``.

``tests/test_worker_clip_frame_feeder.py`` already proves the feeder itself
drains a subscription correctly in isolation. This file proves the
composition root actually wires that feeder to each camera's *real* bus and
starts it, so frames a production ingest loop publishes genuinely reach the
shared recorder -- not just that a feeder object gets constructed and left
unused.

Also proves the feeder threads ``WorkerRuntime.run()`` starts
(``_start_clip_frame_feeders``) are deliberately independent of
``IngestSupervisor`` -- folding them into its managed loop set would block
``run()`` forever, since a feeder never terminates on its own and the
supervisor's ``join()`` waits for every managed loop to finish -- and that
``WorkerRuntime.stop()`` still reaps them cleanly (no zombie thread survives
``run()`` returning).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.frame import Frame
from contracts.runner import Image, RunnerResult
from shared.events.evidence_http_transport import HttpResult
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime
from worker.types import FramePacket


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((2, 3, 3), seq, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 3, 2, 0.25)


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
    def create(self, task: str, **_options: object) -> _FakeRunner:
        return _FakeRunner(task)


@final
class _NoOpPump:
    """Composition tests assert on wiring, not pump throughput."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.stop_count = 0

    def run(self) -> None:
        return None

    def stop(self) -> None:
        self.stop_count += 1


def _pump_factory(
    camera: CameraRuntimeConfig,
    _bus: object,
    _analytics: object,
    _decision: object,
    _sink: object,
) -> _NoOpPump:
    return _NoOpPump(camera.camera_id)


def _make_recording_clip_recorder_class(created: list[object]) -> type:
    """Composition always resolves a real encoder-backed ``ClipRecorder``
    (see ``test_worker_evidence_export_composition.py``'s
    ``test_per_camera_clip_recorder_views_are_distinct_objects_over_one_shared_recorder``),
    but proving frames reach ``on_frame`` doesn't need a real ffmpeg pipeline
    -- only that the composed feeder actually calls it. Monkeypatching
    ``worker_module.ClipRecorder`` (a bare-name module-global lookup at the
    call site inside ``_compose_evidence_export``, so this substitution takes
    effect) swaps in this lightweight recording fake instead.
    """

    @final
    class _RecordingClipRecorder:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self.frames: list[tuple[str, Frame]] = []
            self._lock = threading.Lock()
            startup_hook = kwargs.get("startup_hook")
            self._startup_hook = startup_hook if callable(startup_hook) else lambda: None
            created.append(self)

        def start(self) -> None:
            self._startup_hook()

        def stop(self, *, timeout: float = 5.0) -> None:
            del timeout

        def on_frame(self, camera_id: str, frame: Frame) -> bool:
            with self._lock:
                self.frames.append((camera_id, frame))
            return True

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

    return _RecordingClipRecorder


@final
class _FramePublishingLoop:
    """Ingest loop fake that publishes real packets onto the camera's *real*
    bus (the same one the composed feeder drains), then blocks until the
    feeder has actually consumed every one of them before returning.

    Without this wait, ``IngestSupervisor.join()`` (which only waits on
    ingest/pump loops, not the independently-started feeder threads) could
    return -- triggering ``stop()`` and tearing everything down -- before the
    feeder's background thread ever got a chance to drain the bus, making the
    delivery assertion flaky.
    """

    def __init__(
        self,
        camera_id: str,
        bus: BoundedFrameBus,
        reporter: IngestReporter,
        packets: tuple[FramePacket, ...],
        created: list[object],
    ) -> None:
        self.camera_id = camera_id
        self._bus = bus
        self._reporter = reporter
        self._packets = packets
        self._created = created
        self.stop_count = 0

    def run(self) -> None:
        if self.stop_count:
            return
        self._reporter.mark_starting(self.camera_id)
        for packet in self._packets:
            self._bus.publish(packet)
        assert self._wait_for_drain(), "clip frame feeder never drained the published frames"
        self._reporter.mark_ready(self.camera_id)

    def _wait_for_drain(self) -> bool:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._created and len(self._created[0].frames) >= len(self._packets):
                return True
            time.sleep(0.01)
        return False

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


def _config(*camera_ids: str, clip_enabled: bool = True) -> WorkerConfig:
    """Config for these tests.

    Clip recording is opt-in and off by default (``ClipRecordingConfig``).
    These tests are *about* the clip frame feeders, so they enable it
    explicitly rather than relying on a default that no longer exists.
    """
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "clip": {"enabled": clip_enabled},
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
    loop_factory: Callable[..., object],
    state_dir: Path,
) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=_FakeServingClient(),
        loop_factory=loop_factory,  # type: ignore[arg-type]
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=lambda _code: None,
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


def test_frames_published_on_the_bus_actually_reach_the_clip_recorders_on_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    created: list[object] = []
    monkeypatch.setattr(worker_module, "ClipRecorder", _make_recording_clip_recorder_class(created))
    packets = tuple(_packet("camera-a", seq) for seq in range(1, 6))

    def loop_factory(
        camera: CameraRuntimeConfig, bus: BoundedFrameBus, reporter: IngestReporter
    ) -> _FramePublishingLoop:
        return _FramePublishingLoop(camera.camera_id, bus, reporter, packets, created)

    runtime = _runtime(_config("camera-a"), loop_factory, tmp_path)
    runtime.run()

    assert created, "composition never constructed a ClipRecorder"
    recorder = created[0]
    assert [frame.index for _camera_id, frame in recorder.frames] == [1, 2, 3, 4, 5]  # type: ignore[attr-defined]
    assert all(camera_id == "camera-a" for camera_id, _frame in recorder.frames)  # type: ignore[attr-defined]


def test_worker_stop_joins_every_clip_frame_feeder_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    created: list[object] = []
    monkeypatch.setattr(worker_module, "ClipRecorder", _make_recording_clip_recorder_class(created))
    packets_by_camera = {
        "camera-a": (_packet("camera-a", 1),),
        "camera-b": (_packet("camera-b", 1),),
    }

    def loop_factory(
        camera: CameraRuntimeConfig, bus: BoundedFrameBus, reporter: IngestReporter
    ) -> _FramePublishingLoop:
        return _FramePublishingLoop(
            camera.camera_id, bus, reporter, packets_by_camera[camera.camera_id], created
        )

    runtime = _runtime(_config("camera-a", "camera-b"), loop_factory, tmp_path)
    runtime.run()

    assert len(runtime._clip_frame_feeders) == 2  # noqa: SLF001
    assert len(runtime._clip_frame_feeder_threads) == 2  # noqa: SLF001
    names = {thread.name for thread in runtime._clip_frame_feeder_threads}  # noqa: SLF001
    assert names == {"clip-frame-feeder-camera-a", "clip-frame-feeder-camera-b"}
    # run()'s finally block calls stop(), which joins every feeder thread: by
    # the time run() returns, none may still be alive.
    assert all(not thread.is_alive() for thread in runtime._clip_frame_feeder_threads)  # noqa: SLF001


def test_no_clip_recorder_means_no_feeder_threads_are_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the (real, encoder-backed) ClipRecorder fails to start -- the
    existing degrade-gracefully branch in ``_compose_evidence_export`` --
    gap A's feeders must not spin up threads that would drain a bus into a
    recorder that was never created."""
    _stub_heartbeat_transport(monkeypatch)

    @final
    class _AlwaysFailsToStartClipRecorder:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("simulated encoder start failure")

    monkeypatch.setattr(worker_module, "ClipRecorder", _AlwaysFailsToStartClipRecorder)

    @final
    class _InstantLoop:
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

    def loop_factory(
        camera: CameraRuntimeConfig, _bus: object, reporter: IngestReporter
    ) -> _InstantLoop:
        return _InstantLoop(camera.camera_id, reporter)

    runtime = _runtime(_config("camera-a"), loop_factory, tmp_path)
    runtime.run()

    assert runtime._clip_recorder is None  # noqa: SLF001
    assert runtime._clip_frame_feeders == ()  # noqa: SLF001
    assert runtime._clip_frame_feeder_threads == ()  # noqa: SLF001
