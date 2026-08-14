"""Composition of production gap B: ``_start_runtime_status_sender`` actually
instantiating and starting a real ``RuntimeStatusSender`` /
``RelayRuntimeStatusTransport`` pair against the configured relay.

``RuntimeStatusSender``'s own periodic-delivery, retry, and backoff behavior
is already unit-tested directly in ``tests/test_runtime_status_sender.py``
(e.g. with a fast ``publish_interval_sec`` injected straight into the class).
This file does not re-prove that -- it proves the *wiring*: that
``WorkerRuntime.run()`` actually constructs and starts the real classes (not
just that they exist unused), with a *complete* ``facility_by_camera``
mapping (one entry per configured camera, never a filtered subset), that a
real delivery reaches ``POST /api/v1/relay/runtime-status`` -- a separate
channel from ``HeartbeatReporter``'s READY-gated 30s-default liveness ping to
``/api/v1/relay/heartbeat`` -- and that ``WorkerRuntime.stop()`` cleanly reaps
its background thread.

Must monkeypatch ``runtime_status_sender_module.bounded_request``, not
``worker_module`` or ``shared.events.evidence_export_client``:
``RelayRuntimeStatusTransport``'s ``request`` parameter default binds
``bounded_request`` once at class-definition time, so the composition root
passes ``request=runtime_status_sender_module.bounded_request`` explicitly (a
fresh attribute lookup at call time) specifically so this monkeypatch target
takes effect instead of silently hitting the real network.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.telemetry.runtime_status_sender as runtime_status_sender_module
import worker.runtime.worker as worker_module
from contracts.runner import Image, RunnerResult
from shared.events.evidence_http_transport import HttpResult
from worker.domains.module_definition import ComponentBinding
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderStats
from worker.runtime.config import CameraRuntimeConfig, LiveClipExportPolicy, WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.wire import ClipRecorderStatus
from worker.runtime.worker import WorkerRuntime

_TEST_BUILD_REVISION = "1" * 40


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
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("composition tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


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


@final
class _NoOpPump:
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
class _DeliveryWaitingLoop:
    """Ingest loop fake that blocks camera-a until the status sender has
    delivered every configured facility's payload.

    Without this, ``IngestSupervisor.join()`` could return -- triggering
    ``stop()`` -- before the sender's independent background thread ever got
    to post, making the delivery assertions flaky.
    """

    def __init__(
        self, camera_id: str, reporter: IngestReporter, delivered: threading.Event
    ) -> None:
        self.camera_id = camera_id
        self._reporter = reporter
        self._delivered = delivered
        self.stop_count = 0

    def run(self) -> None:
        if self.stop_count:
            return
        self._reporter.mark_starting(self.camera_id)
        assert self._delivered.wait(timeout=5.0), "runtime status sender never delivered"
        self._reporter.mark_ready(self.camera_id)

    def stop(self) -> None:
        self.stop_count += 1


@final
class _DeliveryWaitingLoopFactory:
    def __init__(self, delivered: threading.Event) -> None:
        self._delivered = delivered

    def __call__(
        self, camera: CameraRuntimeConfig, _bus: object, reporter: IngestReporter
    ) -> _DeliveryWaitingLoop | _InstantLoop:
        if camera.camera_id == "camera-a":
            return _DeliveryWaitingLoop(camera.camera_id, reporter, self._delivered)
        return _InstantLoop(camera.camera_id, reporter)


def _stub_heartbeat_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches the *heartbeat* transport (`worker_module.bounded_request`) --
    a deliberately different module attribute than the runtime-status
    transport under test, so a passing test here proves the two channels are
    actually independent rather than accidentally sharing one fake."""

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


def _config(*camera_facility_pairs: tuple[str, str]) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": facility_id,
                    "rtsp_url": f"rtsp://example.test/{camera_id}",
                    "heartbeat_interval_sec": 30.0,
                }
                for camera_id, facility_id in camera_facility_pairs
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
        build_revision=_TEST_BUILD_REVISION,
        serving_client=_FakeServingClient(),
        loop_factory=loop_factory,  # type: ignore[arg-type]
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=lambda _code: None,
        clip_store_dir=state_dir / "clip-store",
    )


def _fake_accepted_response() -> HttpResult:
    return 200, {}, json.dumps({"accepted": True, "generation": 1}).encode()


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


def test_runtime_status_sender_delivers_a_complete_facility_mapping_to_the_relay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    delivered = threading.Event()
    posts: list[tuple[str, dict[str, str], bytes | None]] = []

    def fake_bounded_request(
        url: str,
        method: str,
        headers: dict[str, str],
        data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        assert method == "POST"
        posts.append((url, headers, data))
        if len(posts) >= 2:  # both facility-a and facility-b payloads sent
            delivered.set()
        return _fake_accepted_response()

    monkeypatch.setattr(runtime_status_sender_module, "bounded_request", fake_bounded_request)

    loops = _DeliveryWaitingLoopFactory(delivered)
    runtime = _runtime(
        _config(("camera-a", "facility-a"), ("camera-b", "facility-b")), loops, tmp_path
    )

    runtime.run()

    assert runtime._runtime_status_sender is not None  # noqa: SLF001
    assert delivered.is_set()
    assert len(posts) >= 2
    urls = {url for url, _headers, _data in posts}
    assert urls == {"http://relay.test/api/v1/relay/runtime-status"}
    for _url, headers, _data in posts:
        assert headers["Authorization"] == "Bearer relay-token"
    facility_ids = {json.loads(data or b"{}")["facility_id"] for _url, _headers, data in posts}
    # Complete mapping: both configured cameras' facilities show up, none
    # silently dropped for lacking a `facility_by_camera` entry.
    assert facility_ids == {"facility-a", "facility-b"}


def test_worker_stop_joins_the_runtime_status_sender_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_heartbeat_transport(monkeypatch)

    delivered = threading.Event()
    sender_threads: list[threading.Thread] = []

    def fake_bounded_request(
        _url: str,
        method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        assert method == "POST"
        sender_threads.append(threading.current_thread())
        delivered.set()
        return _fake_accepted_response()

    monkeypatch.setattr(runtime_status_sender_module, "bounded_request", fake_bounded_request)

    loops = _DeliveryWaitingLoopFactory(delivered)
    runtime = _runtime(_config(("camera-a", "facility-a")), loops, tmp_path)

    runtime.run()

    assert sender_threads, "runtime status sender never delivered a payload"
    sender_thread = sender_threads[0]
    assert sender_thread.name == "runtime-status-sender"
    # run()'s finally block calls stop(), which stops the sender: by the time
    # run() returns, its background thread must already be joined and dead --
    # no zombie thread survives.
    assert not sender_thread.is_alive()


def test_runtime_status_payload_reports_registered_cameras_and_worker_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test: ``register_decode``/``set_worker_status`` used to be
    defined on ``WorkerDiagnostics`` but never called anywhere in
    ``WorkerRuntime`` -- every runtime-status POST kept succeeding with 200,
    but carried an empty ``cameras`` list and no ``worker`` field forever, so
    ``/status``'s ``runtime.cameras``/``runtime.worker`` stayed ``{}``/``null``
    no matter how long the worker ran."""
    _stub_heartbeat_transport(monkeypatch)

    delivered = threading.Event()
    posts: list[bytes | None] = []

    def fake_bounded_request(
        _url: str,
        method: str,
        _headers: dict[str, str],
        data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        assert method == "POST"
        posts.append(data)
        delivered.set()
        return _fake_accepted_response()

    monkeypatch.setattr(runtime_status_sender_module, "bounded_request", fake_bounded_request)

    loops = _DeliveryWaitingLoopFactory(delivered)
    runtime = _runtime(_config(("camera-a", "facility-a")), loops, tmp_path)

    runtime.run()

    assert posts, "runtime status sender never delivered a payload"
    payload = json.loads(posts[0] or b"{}")
    assert [camera["camera_id"] for camera in payload["cameras"]] == ["camera-a"]
    assert payload["cameras"][0]["decode"]["requested"] == "auto"
    assert payload["cameras"][0]["decode"]["selected"] is None
    assert payload["worker"]["alive"] is True
    assert isinstance(payload["worker"]["pid"], int)
    assert isinstance(payload["worker"]["started_at_sec"], float)
    assert payload["worker"]["profile_boot_error"] is None


def test_runtime_status_tick_reads_the_latest_applied_clip_export_policy() -> None:
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime.diagnostics = WorkerDiagnostics()
    runtime._clip_recorder = None  # noqa: SLF001
    runtime._clip_export_policy = LiveClipExportPolicy(False, 0)  # noqa: SLF001

    runtime._clip_export_policy.apply(enabled=True, version=4)  # noqa: SLF001
    runtime._refresh_runtime_status_telemetry()  # noqa: SLF001

    payload = runtime.diagnostics.to_payload("facility-a", None, 0)
    assert payload["clip_export"] == {"enabled": True, "version": 4}


def test_refresh_clip_recorder_telemetry_reads_live_stats_into_diagnostics() -> None:
    """Regression: ``set_clip_recorder_status`` is only ever called at
    recorder start/failure (``worker.py``), so without
    ``_refresh_clip_recorder_telemetry`` -- wired as ``RuntimeStatusSender``'s
    ``before_publish`` hook -- every runtime-status payload's
    ``clip_recorder`` counters stay frozen at their startup values forever,
    even though ``ClipRecorderStats`` (``clip_actor.py``) keeps incrementing
    the whole time (#165). This drives the method directly (isolated via
    ``WorkerRuntime.__new__``, the same seam
    ``tests/test_worker_real_warmup_no_stub.py`` uses for
    ``_warm_models``) against a live ``ClipRecorderStats``, not a value
    frozen at construction.
    """
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime.diagnostics = WorkerDiagnostics()

    class _StubRecorder:
        def __init__(self) -> None:
            self.stats = ClipRecorderStats()

    recorder = _StubRecorder()
    recorder.stats.dropped_frames = 4
    recorder.stats.finalized_clips = 9
    recorder.stats.video_unavailable_clips = 7
    recorder.stats.encoder = "libx264"
    runtime._clip_recorder = recorder  # type: ignore[assignment]  # noqa: SLF001

    runtime._refresh_clip_recorder_telemetry()  # noqa: SLF001

    payload = runtime.diagnostics.to_payload("facility-a", None, 0)
    assert payload["clip_recorder"] == {
        "available": True,
        "dropped_frames": 4,
        "dropped_events": 0,
        "failed_writes": 0,
        "finalized_clips": 9,
        "video_unavailable_clips": 7,
        "active_clips": 0,
        "encoder": "libx264",
    }

    # A later tick (more frames/clips processed) reflects the *new* live
    # values, not what was read the first time.
    recorder.stats.dropped_frames = 11
    recorder.stats.finalized_clips = 10

    runtime._refresh_clip_recorder_telemetry()  # noqa: SLF001

    payload = runtime.diagnostics.to_payload("facility-a", None, 0)
    assert payload["clip_recorder"]["dropped_frames"] == 11
    assert payload["clip_recorder"]["finalized_clips"] == 10


def test_refresh_clip_recorder_telemetry_is_a_noop_when_the_recorder_never_started() -> None:
    runtime = WorkerRuntime.__new__(WorkerRuntime)
    runtime.diagnostics = WorkerDiagnostics()
    runtime.diagnostics.set_clip_recorder_status(ClipRecorderStatus(available=False))
    runtime._clip_recorder = None  # noqa: SLF001

    runtime._refresh_clip_recorder_telemetry()  # noqa: SLF001

    payload = runtime.diagnostics.to_payload("facility-a", None, 0)
    assert payload["clip_recorder"]["available"] is False
