"""Composition of the evidence egress path: outbox -> relay HTTP, and the
real per-camera clip recorder wiring that replaces ``_NullClipRecorder``.

Covers the properties the composition root relies on but does not itself
prove: an event already staged in the durable outbox actually reaches the
relay's ``/api/v1/relay/alerts`` endpoint once ``WorkerRuntime.run()`` starts
the export sender, ``stop()`` really reaps that sender's background thread
instead of leaking it, and per-camera clip recorder composition hands each
camera a distinct object while sharing the one real encoder/actor underneath
(the existing ``ClipRecorder`` design, not a new policy).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import shared.events.evidence_export_client as evidence_export_client_module
import worker.runtime.worker as worker_module
from contracts.frame import Frame
from contracts.runner import Image, RunnerResult
from shared.events.evidence_http_transport import HttpResult
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import IngestReporter
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox
from worker.pipeline.output.evidence.evidence_outbox_types import EdgeEventId, StagedEvent
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
    def __init__(self) -> None:
        self.created: list[tuple[str, _FakeRunner]] = []

    def create(
        self,
        task: str,
        **_options: str | int | float | bool | None,
    ) -> _FakeRunner:
        runner = _FakeRunner(task)
        self.created.append((task, runner))
        return runner


@final
class _FakeClipRecorder:
    """Stands in for the real ``ClipRecorder``: ``_build_clip_frame_feeder``
    composition wiring only needs an object identity to bind, not real
    encode behavior."""

    def on_frame(self, camera_id: str, frame: Frame) -> bool:
        return True


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
class _DeliveryWaitingLoop:
    """Ingest loop fake that blocks camera-a until export delivery is observed.

    Without this, ``IngestSupervisor.join()`` (and therefore ``stop()``,
    which reaps the export sender thread) could race ahead of the sender's
    first delivery attempt, making the delivery assertion flaky.
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
        assert self._delivered.wait(timeout=5.0), "evidence export never delivered the event"
        self._reporter.mark_ready(self.camera_id)

    def stop(self) -> None:
        self.stop_count += 1


@final
class _DeliveryWaitingLoopFactory:
    def __init__(self, delivered: threading.Event) -> None:
        self._delivered = delivered

    def __call__(
        self, camera: CameraRuntimeConfig, _bus: object, reporter: IngestReporter
    ) -> _DeliveryWaitingLoop:
        return _DeliveryWaitingLoop(camera.camera_id, reporter, self._delivered)


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


def _runtime(
    config: WorkerConfig,
    serving: _FakeServingClient,
    loop_factory: Callable[..., object],
    state_dir: Path,
    *,
    hard_exit: Callable[[int], None] = lambda _code: None,
    clip_recorder_factory: Callable[[CameraRuntimeConfig], object] | None = None,
) -> WorkerRuntime:
    return WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=serving,
        loop_factory=loop_factory,
        pump_factory=_pump_factory,
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=hard_exit,
        clip_recorder_factory=clip_recorder_factory,
    )


def _config(*camera_ids: str, clip_enabled: bool = True) -> WorkerConfig:
    """Config for these tests.

    Clip recording is opt-in and off by default (``ClipRecordingConfig``).
    Tests that assert on the shared ``ClipRecorder`` enable it explicitly;
    default-off behaviour has its own dedicated tests below.
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


_EDGE_EVENT_ID = EdgeEventId("11111111-1111-4111-8111-111111111111")


def _stage_one_admitted_event(outbox_path: Path) -> str:
    payload_json = json.dumps(
        {
            "edge_event_id": _EDGE_EVENT_ID,
            "event_type": "fall_detected",
            "probability": 0.9,
            "detected_at": "2026-08-01T00:00:00Z",
            "camera_id": "camera-a",
            "facility_id": "facility-a",
            "evidence": {},
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    with EvidenceOutbox.open(outbox_path) as outbox:
        outbox.stage(
            StagedEvent(
                edge_event_id=_EDGE_EVENT_ID,
                detected_at="2026-08-01T00:00:00Z",
                payload_json=payload_json,
                queued_at=1.0,
            )
        )
        # Mirrors EvidenceEventSink.emit(): stage() alone leaves the row in
        # the non-claimable STAGED state; complete() (here: no clip bound)
        # transitions it to READY, the only state EvidenceSender.claim() polls.
        outbox.mark_ready(_EDGE_EVENT_ID)
    return payload_json


def _enable_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("ML_WORKER_EVENT_CLIP_EXPORT_ENABLED", "1")
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    outbox_path = tmp_path / "evidence-outbox.sqlite3"
    monkeypatch.setenv("ML_WORKER_EVIDENCE_OUTBOX_PATH", str(outbox_path))
    return outbox_path


def _fake_capabilities_response() -> HttpResult:
    return 200, {}, json.dumps({"event_idempotency": 1, "clip_export": 0}).encode()


def _fake_event_accepted_response() -> HttpResult:
    return (
        200,
        {},
        json.dumps(
            {"status": "accepted", "edge_event_id": _EDGE_EVENT_ID, "event_id": "backend-1"}
        ).encode(),
    )


def test_admitted_event_staged_in_the_outbox_is_delivered_to_the_relay_alerts_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Wired lifecycle: outbox row -> EvidenceSender -> RelayEvidenceClient ->
    a real HTTP-transport-level fake, asserting the actual POST URL/payload."""
    _stub_heartbeat_transport(monkeypatch)
    outbox_path = _enable_export(monkeypatch, tmp_path)
    _stage_one_admitted_event(outbox_path)

    posts: list[tuple[str, bytes | None]] = []
    delivered = threading.Event()

    def fake_bounded_request(
        url: str,
        method: str,
        _headers: dict[str, str],
        data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        if method == "GET" and "/api/v1/relay/capabilities" in url:
            return _fake_capabilities_response()
        if method == "POST" and url.endswith("/api/v1/relay/alerts"):
            posts.append((url, data))
            delivered.set()
            return _fake_event_accepted_response()
        raise AssertionError(f"unexpected relay request: {method} {url}")

    monkeypatch.setattr(evidence_export_client_module, "bounded_request", fake_bounded_request)

    serving = _FakeServingClient()
    loops = _DeliveryWaitingLoopFactory(delivered)
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)

    runtime.run()

    assert [url for url, _data in posts] == ["http://relay.test/api/v1/relay/alerts"]
    posted = json.loads(posts[0][1] or b"{}")
    assert posted["edge_event_id"] == _EDGE_EVENT_ID
    assert posted["camera_id"] == "camera-a"
    assert posted["facility_id"] == "facility-a"


def test_worker_stop_joins_the_evidence_export_sender_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_heartbeat_transport(monkeypatch)
    outbox_path = _enable_export(monkeypatch, tmp_path)
    _stage_one_admitted_event(outbox_path)

    delivered = threading.Event()
    sender_threads: list[threading.Thread] = []

    def fake_bounded_request(
        url: str,
        method: str,
        _headers: dict[str, str],
        _data: bytes | None,
        _timeout: float,
        _on_response: Callable[[int], None] | None = None,
    ) -> HttpResult:
        if method == "GET" and "/api/v1/relay/capabilities" in url:
            return _fake_capabilities_response()
        if method == "POST" and url.endswith("/api/v1/relay/alerts"):
            sender_threads.append(threading.current_thread())
            delivered.set()
            return _fake_event_accepted_response()
        raise AssertionError(f"unexpected relay request: {method} {url}")

    monkeypatch.setattr(evidence_export_client_module, "bounded_request", fake_bounded_request)

    serving = _FakeServingClient()
    loops = _DeliveryWaitingLoopFactory(delivered)
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)

    runtime.run()

    assert len(sender_threads) == 1
    sender_thread = sender_threads[0]
    assert sender_thread.name == "evidence-sender"
    # run()'s finally block calls stop(), which calls stop_sender(): by the
    # time run() returns, the sender thread must already be joined and dead.
    assert not sender_thread.is_alive()


def test_per_camera_clip_recorder_views_are_distinct_objects_over_one_shared_recorder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every profile in PROFILE_REGISTRY resolves a real encoder (h264_nvenc or
    libx264); there is no "no encode support" case in this codebase, so this
    exercises the always-real-recorder composition path. ClipRecorder itself
    is one shared actor/encoder (the existing design), but composition must
    still hand each camera a distinct EventClipRecorder object."""
    _stub_heartbeat_transport(monkeypatch)
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))

    recorders: dict[str, object] = {}
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", "camera-b"), serving, loops, tmp_path)

    def capturing_factory(camera: CameraRuntimeConfig) -> object:
        recorder = runtime._default_clip_recorder(camera)  # noqa: SLF001
        recorders[camera.camera_id] = recorder
        return recorder

    runtime._clip_recorder_factory = capturing_factory  # noqa: SLF001

    runtime.run()

    assert runtime._clip_recorder is not None, "real ClipRecorder must have started"  # noqa: SLF001
    assert set(recorders) == {"camera-a", "camera-b"}
    first, second = recorders["camera-a"], recorders["camera-b"]
    assert first is not second
    assert isinstance(first, worker_module._CameraClipRecorderView)  # noqa: SLF001
    assert isinstance(second, worker_module._CameraClipRecorderView)  # noqa: SLF001
    # Distinct per-camera objects, same shared encoder/actor underneath --
    # ClipRecorder already keys mutable per-clip state by camera_id, so this
    # sharing never cross-contaminates camera state.
    assert first.recorder is second.recorder is runtime._clip_recorder  # noqa: SLF001
    assert first.camera_id == "camera-a"
    assert second.camera_id == "camera-b"


def test_build_clip_frame_feeder_binds_the_evidence_tap_and_the_shared_clip_recorder(
    tmp_path: Path,
) -> None:
    """Production gap A: ``_build_clip_frame_feeder`` must drain
    ``bus.evidence`` -- the subscription that exists for exactly one purpose,
    feeding clip recording, and that nothing else reads -- not
    ``bus.inference``, and must share the one process-wide ``ClipRecorder``
    ``_compose_evidence_export`` resolves once, not build a fresh recorder
    per camera."""
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)
    recorder = _FakeClipRecorder()
    runtime._clip_recorder = recorder  # type: ignore[assignment]  # noqa: SLF001
    bus = BoundedFrameBus()

    feeder = runtime._build_clip_frame_feeder("camera-a", bus)  # noqa: SLF001

    assert feeder is not None
    assert feeder.camera_id == "camera-a"
    assert feeder._subscription is bus.evidence  # noqa: SLF001
    assert feeder._subscription is not bus.inference  # noqa: SLF001
    assert feeder._recorder is recorder  # noqa: SLF001


def test_build_clip_frame_feeder_returns_none_when_clip_recording_never_started(
    tmp_path: Path,
) -> None:
    """Mirrors ``_compose_evidence_export``'s non-fatal degrade path: when the
    real ``ClipRecorder`` never started (``self._clip_recorder`` stays
    ``None``), there is no point draining a subscription nobody reads."""
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a"), serving, loops, tmp_path)
    assert runtime._clip_recorder is None  # noqa: SLF001  # sanity: unset before composition runs
    bus = BoundedFrameBus()

    feeder = runtime._build_clip_frame_feeder("camera-a", bus)  # noqa: SLF001

    assert feeder is None


class _RecordingEvidenceRuntime:
    """Stand-in that records the lifecycle order the invariant depends on."""

    def __init__(self) -> None:
        self.initialize_calls: int = 0
        self.sender_starts: int = 0
        self.initialized: bool = False

    def initialize_under_lock(self) -> None:
        self.initialize_calls += 1
        self.initialized = True

    def start_sender(self) -> None:
        if not self.initialized:
            raise RuntimeError("evidence runtime must initialize before sender start")
        self.sender_starts += 1

    def stop_sender(self) -> None:
        return None

    def is_clip_held(self, clip_id: str) -> bool:
        del clip_id
        return False

    def notify_clip_finalized(self, clip_id: str) -> None:
        del clip_id


def _compose_with(
    runtime: WorkerRuntime,
    evidence_runtime: _RecordingEvidenceRuntime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run composition with the export runtime and clip store stubbed out."""
    monkeypatch.setattr(
        worker_module,
        "ClipRecorderConfig",
        lambda: SimpleNamespace(store_dir=tmp_path / "clipstore"),
    )
    monkeypatch.setattr(
        runtime,
        "_compose_evidence_delivery",
        lambda: evidence_runtime,
    )
    runtime._compose_evidence_export(  # noqa: SLF001
        SimpleNamespace(encode=SimpleNamespace())
    )


def test_clip_default_off_still_initializes_delivery_under_the_store_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorder off must not silence alert delivery.

    Skipping evidence composition entirely would leave READY rows stranded in
    the local outbox: ``start_sender()`` refuses an uninitialized runtime. With
    no ``ClipRecorder`` to own the clip-store lock, delivery composition takes
    it itself and initializes exactly once.
    """
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=False), serving, loops, tmp_path)
    evidence_runtime = _RecordingEvidenceRuntime()

    _compose_with(runtime, evidence_runtime, monkeypatch, tmp_path)

    # Delivery is composed and initialized exactly once...
    assert runtime._evidence_export_runtime is evidence_runtime  # noqa: SLF001
    assert evidence_runtime.initialize_calls == 1
    # ...and the sender can start against it.
    evidence_runtime.start_sender()
    assert evidence_runtime.sender_starts == 1
    # No recorder, and therefore no per-camera feeder.
    assert runtime._clip_recorder is None  # noqa: SLF001
    assert runtime._build_clip_frame_feeder("camera-a", BoundedFrameBus()) is None  # noqa: SLF001


def test_clip_enabled_initializes_delivery_exactly_once_through_the_recorder_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorder on keeps the existing hook path -- and still initializes once.

    ``ClipRecorder.start()`` acquires the lock and runs the startup hook before
    sweep/rotate/admission (``tests/test_evidence_export_startup.py``). Delivery
    composition must not initialize a second time on this path.
    """
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=True), serving, loops, tmp_path)
    evidence_runtime = _RecordingEvidenceRuntime()

    started: list[object] = []

    class _Recorder:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self._startup_hook = kwargs.get("startup_hook")

        def start(self) -> None:
            hook = self._startup_hook
            if hook is not None:
                hook()  # type: ignore[operator]
            started.append(self)

    monkeypatch.setattr(worker_module, "ClipRecorder", _Recorder)
    monkeypatch.setattr(worker_module, "default_services", lambda *_a, **_k: object())

    _compose_with(runtime, evidence_runtime, monkeypatch, tmp_path)

    assert started, "clip recording was enabled but no recorder started"
    assert runtime._clip_recorder is not None  # noqa: SLF001
    assert runtime._evidence_export_runtime is evidence_runtime  # noqa: SLF001
    # Exactly once, via the recorder's startup hook under the store lock.
    assert evidence_runtime.initialize_calls == 1
    evidence_runtime.start_sender()
    assert evidence_runtime.sender_starts == 1


def test_recorder_failing_after_its_startup_hook_does_not_initialize_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exactly-once invariant must survive a partial recorder start.

    ``ClipRecorder.start()`` runs the startup hook *before* it sweeps, rotates,
    and spawns its thread. A failure after the hook therefore leaves the runtime
    already initialized. Re-initializing it on the error path would break the
    invariant the disposition fixed, so composition must distinguish "failed
    before the hook" from "failed after it".
    """
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=True), serving, loops, tmp_path)
    evidence_runtime = _RecordingEvidenceRuntime()

    class _FailsAfterHook:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self._startup_hook = kwargs.get("startup_hook")

        def start(self) -> None:
            hook = self._startup_hook
            if hook is not None:
                hook()  # type: ignore[operator]
            raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(worker_module, "ClipRecorder", _FailsAfterHook)
    monkeypatch.setattr(worker_module, "default_services", lambda *_a, **_k: object())

    _compose_with(runtime, evidence_runtime, monkeypatch, tmp_path)

    # The hook already initialized the runtime; the error path must not repeat it.
    assert evidence_runtime.initialize_calls == 1
    # Clip recording is off after the failure, but delivery still works.
    assert runtime._clip_recorder is None  # noqa: SLF001
    evidence_runtime.start_sender()
    assert evidence_runtime.sender_starts == 1


def test_recorder_failing_before_its_startup_hook_still_initializes_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure before the hook leaves nobody to initialize the runtime.

    Without composition stepping in, ``start_sender()`` would refuse an
    uninitialized runtime and alerts would strand in the local outbox.
    """
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=True), serving, loops, tmp_path)
    evidence_runtime = _RecordingEvidenceRuntime()

    class _FailsBeforeHook:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("clip store locked by another process")

    monkeypatch.setattr(worker_module, "ClipRecorder", _FailsBeforeHook)
    monkeypatch.setattr(worker_module, "default_services", lambda *_a, **_k: object())

    _compose_with(runtime, evidence_runtime, monkeypatch, tmp_path)

    assert evidence_runtime.initialize_calls == 1
    assert runtime._clip_recorder is None  # noqa: SLF001
    evidence_runtime.start_sender()
    assert evidence_runtime.sender_starts == 1


def test_clip_disabled_shrinks_the_evidence_tap_built_by_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No feeder means no reason to hold a full evidence backlog per camera.

    ``bus.evidence`` is a FIFO drained only by ``ClipFrameFeeder``. With clip
    recording off no feeder is built, so a full-size tap would retain frames
    nothing will ever read -- multi-GB at fleet scale.

    This asserts on the bus that *composition actually builds*. Comparing two
    hand-constructed buses would only restate the constructor default and prove
    nothing about the production path.
    """
    captured: list[BoundedFrameBus] = []
    real_bus = worker_module.BoundedFrameBus

    def _capturing_bus(**kwargs: object) -> BoundedFrameBus:
        bus = real_bus(**kwargs)  # type: ignore[arg-type]
        captured.append(bus)
        return bus

    monkeypatch.setattr(worker_module, "BoundedFrameBus", _capturing_bus)

    serving = _FakeServingClient()
    loops = _InstantLoopFactory()

    def _evidence_capacity(*, clip_enabled: bool) -> int:
        captured.clear()
        runtime = _runtime(
            _config("camera-a", clip_enabled=clip_enabled), serving, loops, tmp_path
        )
        # _build_camera refuses to run without a fall model, and that guard
        # sits before the bus is built. A sentinel is enough: nothing after the
        # bus construction matters for this assertion.
        runtime.fall_model = object()  # type: ignore[assignment]
        # _build_camera needs a fall model; everything after the bus is
        # irrelevant here, so let it fail once the bus has been built.
        try:
            _ = runtime._build_camera(  # noqa: SLF001
                runtime.config.cameras[0],
                SimpleNamespace(extractors=()),  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 - only the constructed bus matters
            # Recorded rather than swallowed silently: if the bus was never
            # built, this is the reason the assertion below reports.
            _ = repr(exc)
        assert captured, "composition never constructed a frame bus"
        return captured[0].evidence.capacity

    off_capacity = _evidence_capacity(clip_enabled=False)
    on_capacity = _evidence_capacity(clip_enabled=True)

    assert off_capacity < on_capacity, (
        "clip-off composition still builds a full-size evidence tap: "
        f"off={off_capacity} on={on_capacity}"
    )
    assert off_capacity == 1


def test_enabled_but_misconfigured_export_fails_closed_instead_of_going_quiet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0003: a switched-on, misconfigured export must not degrade silently.

    ``EvidenceExportRuntime.from_environment`` returns ``None`` on its own when
    the env gate is off -- that is the explicit opt-out. A ``ValueError`` means
    the gate is ON and the configuration is incomplete, which used to be logged
    as a warning and turned into "no delivery". A worker that looks healthy
    while alerts pile up unsent is exactly what that decision removes.
    """
    # The gate must be ON: this test is about "switched on and broken", and
    # composition now short-circuits before building anything when it is off.
    monkeypatch.setenv("ML_WORKER_EVENT_CLIP_EXPORT_ENABLED", "1")
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=False), serving, loops, tmp_path)

    def _misconfigured(**_kwargs: object) -> object:
        raise ValueError("evidence export requires relay URL, token, and camera")

    monkeypatch.setattr(
        worker_module.EvidenceExportRuntime, "from_environment", staticmethod(_misconfigured)
    )
    monkeypatch.setattr(
        worker_module,
        "ClipRecorderConfig",
        lambda: SimpleNamespace(store_dir=tmp_path / "clipstore"),
    )

    with pytest.raises(worker_module.EvidenceDeliveryError) as captured:
        runtime._compose_evidence_export(  # noqa: SLF001
            SimpleNamespace(encode=SimpleNamespace())
        )

    message = str(captured.value)
    assert "misconfigured" in message
    # Sanitized: the relay URL and token never appear in the operator-facing text.
    assert "relay.test" not in message
    assert "relay-token" not in message


def test_startup_hook_initialization_failure_is_fatal_not_a_clip_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed delivery init inside the hook must not be demoted to "clips off".

    ``initialize_under_lock()`` runs inside ``ClipRecorder.start()``. Because the
    recorder's own failures are a deliberate non-fatal boundary, a naive broad
    catch swallows the hook's failure too -- and then ``start_sender()`` refuses
    an uninitialized runtime, so alerts pile up in the local outbox while the
    worker looks healthy. That is the exact ADR-0003 failure mode, so the hook
    re-raises as ``EvidenceDeliveryError`` and composition lets it through.
    """
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=True), serving, loops, tmp_path)

    class _FailingInit(_RecordingEvidenceRuntime):
        def initialize_under_lock(self) -> None:
            raise RuntimeError("outbox schema is newer than this worker")

    evidence_runtime = _FailingInit()

    class _RunsHook:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self._startup_hook = kwargs.get("startup_hook")

        def start(self) -> None:
            hook = self._startup_hook
            if hook is not None:
                hook()  # type: ignore[operator]

    monkeypatch.setattr(worker_module, "ClipRecorder", _RunsHook)
    monkeypatch.setattr(worker_module, "default_services", lambda *_a, **_k: object())

    with pytest.raises(worker_module.EvidenceDeliveryError) as captured:
        _compose_with(runtime, evidence_runtime, monkeypatch, tmp_path)

    assert "initialize" in str(captured.value)
    # The real cause survives for debugging without being echoed to operators.
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_sender_start_failure_is_fatal_when_delivery_is_enabled(
    tmp_path: Path,
) -> None:
    """A composed runtime that cannot start its sender must not be ignored.

    ``self._evidence_export_runtime is None`` is the explicit opt-out (export
    env gate off). A runtime that exists and then fails to start its sender
    means staged alerts accumulate in the local outbox forever while the worker
    reports healthy -- the ADR-0003 silent-degrade this closes.
    """
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=False), serving, loops, tmp_path)

    class _SenderRefuses(_RecordingEvidenceRuntime):
        def start_sender(self) -> None:
            raise RuntimeError("relay socket unavailable")

    runtime._evidence_export_runtime = _SenderRefuses()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(worker_module.EvidenceDeliveryError) as captured:
        runtime._start_export_sender()  # noqa: SLF001

    assert "sender failed to start" in str(captured.value)
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_sender_start_is_a_noop_when_export_is_not_enabled(
    tmp_path: Path,
) -> None:
    """Export gate off is an explicit opt-out, not a failure."""
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=False), serving, loops, tmp_path)

    assert runtime._evidence_export_runtime is None  # noqa: SLF001
    runtime._start_export_sender()  # noqa: SLF001


def test_clip_only_env_cannot_brick_a_worker_with_clip_and_export_both_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stray clip setting must not fail a worker that uses neither.

    ``ClipRecorderConfig`` parses clip-only environment variables and raises on
    malformed values (``int(raw)`` / ``float(raw)`` in
    ``worker/pipeline/output/evidence/clip_config.py``). Building it before the
    export gate is consulted meant that with clip recording off *and* export
    off -- the default posture -- a malformed CLIP_STORE_RETENTION_DAYS still
    aborted camera activation, for a subsystem that process never uses.
    """
    monkeypatch.setenv("CLIP_STORE_RETENTION_DAYS", "not-a-number")
    monkeypatch.delenv("ML_WORKER_EVENT_CLIP_EXPORT_ENABLED", raising=False)

    # Sanity: the malformed value really does break the config object.
    with pytest.raises(ValueError, match="invalid literal"):
        _ = worker_module.ClipRecorderConfig()

    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=False), serving, loops, tmp_path)

    # Composition must not touch that config when export is gated off.
    runtime._compose_evidence_export(  # noqa: SLF001
        SimpleNamespace(encode=SimpleNamespace())
    )

    assert runtime._evidence_export_runtime is None  # noqa: SLF001
    assert runtime._clip_recorder is None  # noqa: SLF001

def test_recorder_off_init_failure_is_typed_whatever_the_exception_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-OSError init failure must still surface as EvidenceDeliveryError.

    The recorder-off path used to catch only ``OSError``. The startup-hook path
    catches everything, so the same underlying failure was typed and sanitized
    on one path and escaped raw on the other -- and outbox initialization can
    fail with more than ``OSError`` (a locked sqlite outbox raises
    ``sqlite3.OperationalError``, which is not an ``OSError`` subclass).

    ``CLIP_STORE_DIR`` is redirected at a writable path on purpose. Left at its
    ``/var/lib/clip-store`` default, ``ClipStoreLock.acquire`` raises
    ``PermissionError`` before ``initialize_under_lock`` is ever called -- and
    ``PermissionError`` *is* an ``OSError``, so the test would have passed
    against the old narrow handler without exercising anything.

    Asserting on ``RuntimeError`` is likewise deliberate: it is not an
    ``OSError``, so this fails if the narrow handler ever comes back.
    """
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=False), serving, loops, tmp_path)

    class _InitRefuses(_RecordingEvidenceRuntime):
        def initialize_under_lock(self) -> None:
            raise RuntimeError("outbox schema is from a newer worker")

    with pytest.raises(worker_module.EvidenceDeliveryError) as captured:
        runtime._initialize_delivery_without_recorder(  # noqa: SLF001
            _InitRefuses()  # type: ignore[arg-type]
        )

    assert "failed to initialize under the clip-store lock" in str(captured.value)
    # The original failure must remain reachable for diagnosis, not be replaced.
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "newer worker" in str(captured.value.__cause__)


def test_recorder_off_init_failure_does_not_leak_the_store_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The typed error must not carry filesystem detail into operator logs.

    Same reasoning as the RTSP credential redaction: the message is what lands
    in logs and alerts, so it names the failure, not the environment. The cause
    chain still carries the detail for anyone debugging.
    """
    store_dir = tmp_path / "clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(store_dir))
    serving = _FakeServingClient()
    loops = _InstantLoopFactory()
    runtime = _runtime(_config("camera-a", clip_enabled=False), serving, loops, tmp_path)

    class _InitRefuses(_RecordingEvidenceRuntime):
        def initialize_under_lock(self) -> None:
            raise RuntimeError(f"cannot open {store_dir}/outbox.sqlite3")

    with pytest.raises(worker_module.EvidenceDeliveryError) as captured:
        runtime._initialize_delivery_without_recorder(  # noqa: SLF001
            _InitRefuses()  # type: ignore[arg-type]
        )

    assert str(store_dir) not in str(captured.value)
    # The detail is preserved on the cause, just kept out of the operator-facing
    # message -- otherwise this would be redaction by accident, not by design.
    assert str(store_dir) in str(captured.value.__cause__)
