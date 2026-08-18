"""Composition of the operator live view: the MJPEG server actually running.

``worker/pipeline/output/mjpeg_server.py`` and ``live_view.py`` were ported
from edge with their own unit tests, but nothing in ``worker/runtime/`` ever
constructed them -- the port never opened, so the dashboard's camera view was
dead even though ``compose.edge.yaml`` sets ``ML_WORKER_DEV_MJPEG`` and the
backend proxies ``/api/v1/streams/{id}`` to ``:8090``
(https://github.com/SeniorAILab/eldercare-fall-ml-v2/issues/15).

The pieces' own behaviour is already covered elsewhere:
``tests/test_worker_mjpeg_server.py`` (HTTP surface),
``tests/test_runtime_latest_frame.py`` (store semantics), and
``tests/test_worker_overlay_debug_snapshot.py`` (rendering). None of those
fail when the feature is unreachable, which is exactly how it regressed. This
file asserts the wiring instead: the switch is read, cameras are registered,
a real socket is listening, ``stop()`` closes it, and the per-frame tap
reaches the store the server serves from.
"""

from __future__ import annotations

import http.client
import socket
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from shared.events.evidence_http_transport import HttpResult
from worker.domains.module_definition import ComponentBinding
from worker.pipeline.analytics import CompositeExtractor
from worker.pipeline.bus import BoundedFrameBus, Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import CoordinatedInference, InferenceResultSlot
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.pipeline.ingest.probe import RTSPProbeError, RTSPProbeResult
from worker.pipeline.output.live_view import LatestFrameStore, LiveViewSubscriber
from worker.pipeline.output.live_view_pump import LatestObservationStore, LiveViewPump
from worker.pipeline.output.mjpeg_server import MjpegProbeError
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.worker import WorkerRuntime
from worker.types import BusinessEvent, DecisionInput, FramePacket, ModuleResult


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

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("live view composition tests must not run model inference")

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

    def infer_batch(
        self, task: str, frames: Sequence[FramePacket], **_options: object
    ) -> tuple[RunnerResult, ...]:
        assert task == "pose"
        return tuple(pose_result((), ()) for _frame in frames)


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


@final
class _InstantLoopFactory:
    def __call__(
        self, camera: CameraRuntimeConfig, _bus: object, reporter: IngestReporter
    ) -> _InstantLoop:
        return _InstantLoop(camera.camera_id, reporter)


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


def _config(*camera_ids: str, dev_mjpeg: dict[str, object] | None = None) -> WorkerConfig:
    payload: dict[str, object] = {
        "version": 7,
        "relay": {"url": "http://relay.test", "token": "relay-token"},
        "cameras": [
            {
                "camera_id": camera_id,
                "facility_id": "facility-a",
                "rtsp_url": f"rtsp://example.test/{camera_id}",
                "heartbeat_interval_sec": 30.0,
            }
            for camera_id in camera_ids
        ],
    }
    if dev_mjpeg is not None:
        payload["dev_mjpeg"] = dev_mjpeg
    return WorkerConfig.model_validate(payload)


def _runtime(config: WorkerConfig, state_dir: Path, env: dict[str, str]) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu", **env},
        serving_client=_FakeServingClient(),
        loop_factory=_InstantLoopFactory(),
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=lambda _code: None,
        clip_store_dir=state_dir / "clip-store",
        build_revision="1" * 40,
    )


def _wait_for(predicate: Callable[[], bool], *, timeout_sec: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@contextmanager
def _running(runtime: WorkerRuntime, ready: Callable[[], bool]) -> Iterator[None]:
    """Run a worker for the length of a ``with`` block, then stop it cleanly.

    The real ``CameraPipelinePump`` never returns on its own here (no cap, and
    an ingest loop that publishes nothing), so ``run()`` has to be driven from
    a thread and released by ``stop()`` -- which is also what proves ``stop()``
    reaps the live view server rather than leaking its socket.
    """
    thread = threading.Thread(target=runtime.run, daemon=True)
    thread.start()
    try:
        assert _wait_for(ready), "worker never reached the expected state"
        yield
    finally:
        runtime.stop()
        thread.join(timeout=10)
    assert not thread.is_alive(), "worker thread outlived stop()"


def _get(port: int, path: str, *, token: str | None = "relay-token") -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        headers = {} if token is None else {"X-Edge-Relay-Token": token}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _post_probe(port: int, token: str | None) -> int:
    # Admission-only URL: DNS fails closed as unsupported without opening a
    # decoder. Keep the client timeout tight so a hung handler fails the test
    # instead of waiting on real RTSP.
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["X-Edge-Relay-Token"] = token
        connection.request(
            "POST",
            "/probe",
            # Loopback is rejected by admission instantly (no DNS, no decoder).
            body=b'{"rtsp_url": "rtsp://127.0.0.1/camera"}',
            headers=headers,
        )
        return connection.getresponse().status
    finally:
        connection.close()


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


def test_enabled_worker_binds_the_live_view_port_and_serves_its_cameras(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The regression this file exists for: a real listening socket.

    Port 0 so the kernel picks a free one -- pinning 8090 would make the test
    fail on a developer machine that happens to be running a worker.
    """
    _stub_heartbeat_transport(monkeypatch)
    runtime = _runtime(
        _config("camera-a"),
        tmp_path,
        {"ML_WORKER_DEV_MJPEG": "true", "ML_WORKER_DEV_MJPEG_PORT": "0"},
    )
    port = 0
    caplog.set_level("INFO", logger="worker.runtime.worker")
    with _running(runtime, lambda: runtime._mjpeg_server is not None):  # noqa: SLF001
        server = runtime._mjpeg_server  # noqa: SLF001
        assert server is not None
        port = server.port
        assert port > 0

        # Auth gates first (security finding #3 + probe SecretStr unwrap).
        # `relay.token` is a `SecretStr`; `_authorized_probe` compares raw
        # strings -- forwarding the wrapper instead of its value would 403.
        assert _post_probe(port, "wrong-token") == 403
        assert _post_probe(port, None) == 403
        assert _post_probe(port, "relay-token") != 403
        assert _get(port, "/snapshot/camera-a", token=None)[0] == 403
        assert _get(port, "/snapshot/camera-a", token="wrong-token")[0] == 403

        # Registered during camera activation: a configured camera is "known"
        # (503, awaiting its first frame) rather than unknown (404).
        assert _get(port, "/snapshot/camera-a")[0] == 503
        assert _get(port, "/snapshot/camera-absent")[0] == 404

        # The server serves out of the very store the runtime composed, so a
        # frame published through that store becomes retrievable over HTTP.
        runtime._live_frames.publish_jpeg(  # noqa: SLF001
            "camera-a", b"jpeg-bytes", frame_index=1
        )
        status, body = _get(port, "/snapshot/camera-a")
        assert status == 200
        assert body == b"jpeg-bytes"
        assert b"relay-token" not in body

    # stop() must release the port, not leak a bound socket for the process's
    # remaining lifetime.
    assert runtime._mjpeg_server is None  # noqa: SLF001
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", port)) != 0

    # Issue #113: a successful bind must also be logged -- the disabled path
    # gained a matching "not started" line, so the enabled+bound path must
    # not be the one left silent.
    assert any("live view server bound" in record.message for record in caplog.records)


def test_disabled_live_view_still_opens_production_derivative_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Disabling visual streaming does not disable derivative control.

    The shared HTTP listener remains reachable for authenticated STILL/VIDEO
    requests, while no camera is registered in the visual frame store.
    """
    _stub_heartbeat_transport(monkeypatch)
    runtime = _runtime(_config("camera-a"), tmp_path, {})

    assert runtime._live_view is None  # noqa: SLF001
    assert runtime._mjpeg_config.enabled is False  # noqa: SLF001

    with caplog.at_level("INFO", logger="worker.runtime.worker"):
        with _running(runtime, lambda: runtime._mjpeg_server is not None):  # noqa: SLF001
            server = runtime._mjpeg_server  # noqa: SLF001
            assert server is not None
            port = server.port
            assert runtime._live_frames.is_known("camera-a") is False  # noqa: SLF001
            with socket.create_connection(("127.0.0.1", port), timeout=2.0):
                pass

    assert runtime._mjpeg_server is None  # noqa: SLF001
    assert any("derivative control server bound" in record.message for record in caplog.records)


def test_either_switch_alone_enables_the_live_view(tmp_path: Path) -> None:
    """Both switches are live, and the config's host/port win when it is the
    one that enabled the feature.

    ``compose.edge.yaml`` turns the view on through the environment while a
    YAML-configured operator turns it on through ``dev_mjpeg``. Honouring only
    one would silently ignore the other.
    """
    env_only = _runtime(
        _config("camera-a"),
        tmp_path,
        {
            "ML_WORKER_DEV_MJPEG": "yes",
            "ML_WORKER_DEV_MJPEG_HOST": "0.0.0.0",
            "ML_WORKER_DEV_MJPEG_PORT": "9111",
        },
    )
    assert env_only._mjpeg_config.enabled is True  # noqa: SLF001
    assert env_only._mjpeg_config.host == "0.0.0.0"  # noqa: SLF001
    assert env_only._mjpeg_config.port == 9111  # noqa: SLF001
    assert env_only._live_view is not None  # noqa: SLF001

    config_only = _runtime(
        _config("camera-a", dev_mjpeg={"enabled": True, "host": "127.0.0.1", "port": 9222}),
        tmp_path,
        {},
    )
    assert config_only._mjpeg_config.enabled is True  # noqa: SLF001
    assert config_only._mjpeg_config.port == 9222  # noqa: SLF001
    assert config_only._live_view is not None  # noqa: SLF001

    # The relay token doubles as the probe token so the backend's probe origin
    # authenticates with the secret it already holds.
    assert config_only._mjpeg_config.probe_token == "relay-token"  # noqa: SLF001


def test_enabled_worker_wires_a_real_probe_not_the_unavailable_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression for the bug this file's sibling method exists to fix:
    ``_start_live_view_server`` used to call ``start_optional_mjpeg_server``
    without ``probe=``, so every registration probe fell back to
    ``_unavailable_probe`` and always failed with ``error_class=decode``. The
    started server's ``probe`` must be the runtime's own ``_rtsp_probe``.
    """
    _stub_heartbeat_transport(monkeypatch)
    runtime = _runtime(
        _config("camera-a"),
        tmp_path,
        {"ML_WORKER_DEV_MJPEG": "true", "ML_WORKER_DEV_MJPEG_PORT": "0"},
    )
    with _running(runtime, lambda: runtime._mjpeg_server is not None):  # noqa: SLF001
        server = runtime._mjpeg_server  # noqa: SLF001
        assert server is not None
        assert server.probe == runtime._rtsp_probe  # noqa: SLF001


def test_rtsp_probe_translates_probe_error_into_mjpeg_probe_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_rtsp_probe`` must carry the decoder's error class through as the
    HTTP handler's ``MjpegProbeError``, not swallow or relabel it. Faking
    ``probe_first_frame`` keeps this a plain-CI unit test -- no real RTSP."""
    runtime = _runtime(_config("camera-a"), tmp_path, {})

    def _raise_timeout(*_args: object, **_kwargs: object) -> RTSPProbeResult:
        raise RTSPProbeError("timeout", "timed out waiting for a frame", "rtsp://masked")

    monkeypatch.setattr(worker_module, "probe_first_frame", _raise_timeout)

    with pytest.raises(MjpegProbeError) as excinfo:
        # Public literal IP clears admission DNS pinning so the injected
        # probe_first_frame path is what raises.
        runtime._rtsp_probe("rtsp://8.8.8.8/camera-a")  # noqa: SLF001
    assert excinfo.value.error_class == "timeout"


def test_rtsp_probe_returns_the_probe_result_payload_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On success ``_rtsp_probe`` returns ``RTSPProbeResult.as_dict()``
    untouched -- the HTTP handler's ``_sanitize_probe_payload`` is what trims
    it down for the wire, not this method."""
    runtime = _runtime(_config("camera-a"), tmp_path, {})
    result = RTSPProbeResult(
        masked_url="rtsp://masked",
        requested_backend="cpu_av",
        backend="cpu_av",
        width=640,
        height=480,
        channels=3,
    )

    def _succeed(*_args: object, **_kwargs: object) -> RTSPProbeResult:
        return result

    monkeypatch.setattr(worker_module, "probe_first_frame", _succeed)

    payload = runtime._rtsp_probe("rtsp://8.8.8.8/camera-a")  # noqa: SLF001
    assert payload == result.as_dict()


def test_enabled_worker_gives_every_camera_a_live_bus_consumer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live lane must have a real consumer, and the pump must feed it.

    Composing a server and a store while nothing drains ``bus.live`` would
    leave the port open and permanently frameless -- a different shape of the
    same "ported but unwired" bug. The pipeline pump no longer publishes the
    preview itself (that put it behind pose); it records the observation the
    live pump overlays.
    """
    _stub_heartbeat_transport(monkeypatch)
    runtime = _runtime(
        _config("camera-a", "camera-b"),
        tmp_path,
        {"ML_WORKER_DEV_MJPEG": "true", "ML_WORKER_DEV_MJPEG_PORT": "0"},
    )
    # Wait on the live pump threads, not on camera count: activation registers
    # the cameras, but `run()` starts these threads a few statements later.
    with _running(
        runtime,
        lambda: len(runtime._live_view_pump_threads) == 2,  # noqa: SLF001
    ):
        assert {pump.camera_id for pump in runtime._live_view_pumps} == {  # noqa: SLF001
            "camera-a",
            "camera-b",
        }
        assert {thread.name for thread in runtime._live_view_pump_threads} == {  # noqa: SLF001
            "live-view-pump-camera-a",
            "live-view-pump-camera-b",
        }
        assert all(
            thread.is_alive()
            for thread in runtime._live_view_pump_threads  # noqa: SLF001
        )
        for context in runtime.cameras:
            pump = context.pump
            assert isinstance(pump, CameraPipelinePump)
            assert context.live_view_pump is not None
            assert (
                pump._observation_recorder  # noqa: SLF001
                is runtime._live_observations  # noqa: SLF001
            )
            # Each camera gets its own collector, not a shared or missing one.
            assert (
                pump._debug_snapshots_provider  # noqa: SLF001
                is runtime._camera_debug_snapshots[pump.camera_id]  # noqa: SLF001
            )
    assert not any(
        thread.is_alive() for thread in runtime._live_view_pump_threads  # noqa: SLF001
    )


def test_disabled_live_view_composes_no_live_bus_consumer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the viewer off there is nothing to drain the lane for."""
    _stub_heartbeat_transport(monkeypatch)
    runtime = _runtime(_config("camera-a"), tmp_path, {})

    with _running(runtime, lambda: len(runtime.cameras) == 1):
        assert runtime._live_view is None  # noqa: SLF001
        assert runtime._live_view_pumps == ()  # noqa: SLF001
        context = runtime.cameras[0]
        assert context.live_view_pump is None
        pump = context.pump
        assert isinstance(pump, CameraPipelinePump)
        assert pump._observation_recorder is None  # noqa: SLF001


# --- the tap itself, against real collaborators -------------------------------


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((4, 4, 3), seq, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 4, 4, 0.25)


def _blank_analytics(camera_id: str) -> CompositeExtractor:
    return CompositeExtractor(
        extractors=(),
        scheduler=Scheduler(task_intervals={}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id=camera_id),
    )


@final
class _NoDecider:
    def update(self, _input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        return ()


@final
class _NullSink:
    def emit(self, _event: BusinessEvent) -> None:
        return None


def _pump(
    observation_recorder: object | None,
    bus: BoundedFrameBus,
    *,
    debug_snapshots_provider: Callable[[int], tuple[object, ...]] | None = None,
) -> CameraPipelinePump:
    packet = bus.inference.take(timeout_sec=0)
    assert packet is not None
    results = InferenceResultSlot()
    results.publish(
        CoordinatedInference(
            packet, ModuleResult("pose", pose_result((), ()), 0.0, "pose")
        )
    )
    return CameraPipelinePump(
        "camera-a",
        results,
        _blank_analytics("camera-a"),
        EventAggregator(deciders=(_NoDecider(),), incidents=IncidentManager()),
        _NullSink(),
        poll_timeout_sec=0.02,
        max_frames=1,
        observation_recorder=observation_recorder,  # type: ignore[arg-type]
        debug_snapshots_provider=debug_snapshots_provider,  # type: ignore[arg-type]
    )


def test_pipeline_pump_and_live_pump_together_fill_the_live_view_store() -> None:
    """Real pipeline pump, real ``LiveViewPump``, real store -- end to end.

    The pipeline pump only caches its observation; the live pump is what takes
    the frame off ``bus.live`` and encodes it.
    """
    store = LatestFrameStore()
    # Viewer gating (#48): publish() is a no-op with zero viewers, so this
    # regression check for "still works with viewers" needs one connected.
    store.mark_viewer_connected("camera-a")
    observations = LatestObservationStore()
    bus = BoundedFrameBus()
    bus.publish(_packet("camera-a", 3))

    _pump(observations, bus).run()
    live_pump = LiveViewPump("camera-a", bus.live, LiveViewSubscriber(store), observations)
    assert live_pump.run_once()

    assert observations.latest("camera-a") is not None
    latest = store.get_latest("camera-a")
    assert latest is not None
    assert latest.frame_index == 3
    assert latest.jpeg.startswith(b"\xff\xd8")  # a real JPEG, actually encoded
    assert latest.overlay_stale is False
    assert bus.metrics("live").taken == 1


def test_pump_passes_the_frame_index_to_the_debug_snapshot_collector() -> None:
    """``_debug_snapshots_provider`` is keyed by frame index, not zero-arg."""
    seen: list[int] = []

    @final
    class _RecordingRecorder:
        def record(
            self,
            _camera_id: str,
            _observation: object,
            debug_snapshots: tuple[object, ...] = (),
            *,
            frame_index: int,
        ) -> None:
            assert debug_snapshots == ("snapshot-7",)
            assert frame_index == 7

    bus = BoundedFrameBus()
    bus.publish(_packet("camera-a", 7))

    def provider(frame_index: int) -> tuple[object, ...]:
        seen.append(frame_index)
        return ("snapshot-7",)

    _pump(_RecordingRecorder(), bus, debug_snapshots_provider=provider).run()

    assert seen == [7]


def test_a_failing_observation_recorder_never_stops_detection() -> None:
    """A cosmetic view is a tap, not a stage.

    An exploding recorder must not be counted as a pipeline failure, and must
    not prevent the frame from being processed.
    """

    @final
    class _ExplodingRecorder:
        def record(
            self,
            _camera_id: str,
            _observation: object,
            _debug_snapshots: tuple[object, ...] = (),
            *,
            frame_index: int,
        ) -> None:
            raise RuntimeError("observation cache exploded")

    bus = BoundedFrameBus()
    bus.publish(_packet("camera-a", 1))
    pump = _pump(_ExplodingRecorder(), bus)

    pump.run()

    assert pump.processed_count == 1
    assert pump.failure_count == 0
