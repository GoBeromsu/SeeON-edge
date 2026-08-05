from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from worker.adapters.model.errors import FatalAcceleratorError
from worker.adapters.model.warmup import warmup_to_ready
from worker.runtime import bootstrap
from worker.runtime.faults import FaultHandler
from worker.runtime.faults.record import WORKER_STATE_DB_FILENAME, FirstFaultRecord
from worker.runtime.lease import GpuLease, GpuLeaseUnavailableError
from worker.runtime.profile.registry import VerifyResult
from worker.runtime.watchdog import WATCHDOG_STAGE, InferenceWatchdog


class _Runner:
    def __init__(self, warmup: Callable[[], None]) -> None:
        self._warmup = warmup

    def warmup(self) -> None:
        self._warmup()


class _RecordingFaultHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[FatalAcceleratorError, FirstFaultRecord]] = []
        self.called = threading.Event()

    def handle(self, error: FatalAcceleratorError, record: FirstFaultRecord) -> None:
        self.calls.append((error, record))
        self.called.set()


def test_named_stage_failure_never_activates_a_camera(tmp_path: Path) -> None:
    activated: list[str] = []
    lease = GpuLease.acquire(tmp_path)
    context = bootstrap.BootstrapContext()
    stages = bootstrap.named_stages(
        context,
        {"ML_WORKER_PROFILE": "cpu"},
        initializers={"pose": lambda _boot: _Runner(_raise_warmup)},
        warmups={"pose": lambda runner: warmup_to_ready(runner, device="cpu")},
        activate=lambda _boot: (activated.append("camera-a"),),
        decode_probe=_successful_decode,
        acquire=lambda: lease,
    )

    with pytest.raises(bootstrap.BootstrapStageError) as exc:
        bootstrap.bootstrap_or_exit(stages, context=context, exit_fn=lambda _code: None)

    assert exc.value.exit_code == bootstrap.GENERIC_RUNTIME_EXIT_CODE
    assert activated == []
    assert lease.held is False


def test_model_backend_init_stage_failure_refuses_to_start(tmp_path: Path) -> None:
    """A model backend that refuses to initialize (e.g. the fall model's
    fail-closed ``RuntimeError`` when unconfigured) must exit with
    ``REFUSE_TO_START_EXIT_CODE``, not the generic runtime code -- this is an
    operator misconfiguration, not a transient runtime fault."""
    activated: list[str] = []
    lease = GpuLease.acquire(tmp_path)
    context = bootstrap.BootstrapContext()
    stages = bootstrap.named_stages(
        context,
        {"ML_WORKER_PROFILE": "cpu"},
        initializers={"fall": _raise_model_backend_init},
        warmups={},
        activate=lambda _boot: (activated.append("camera-a"),),
        decode_probe=_successful_decode,
        acquire=lambda: lease,
    )

    with pytest.raises(bootstrap.BootstrapStageError) as exc:
        bootstrap.bootstrap_or_exit(stages, context=context, exit_fn=lambda _code: None)

    assert exc.value.exit_code == bootstrap.REFUSE_TO_START_EXIT_CODE
    assert activated == []
    assert lease.held is False


def test_real_warmup_synchronizes_cuda_after_every_configured_forward(tmp_path: Path) -> None:
    calls: list[str] = []
    lease = GpuLease.acquire(tmp_path)
    context = bootstrap.BootstrapContext()
    tasks = ("pose", "person", "bed", "lstm")
    stages = bootstrap.named_stages(
        context,
        {"ML_WORKER_PROFILE": "cuda"},
        initializers={task: _initializer(task, calls) for task in tasks},
        warmups={
            task: lambda runner, task=task: warmup_to_ready(
                runner,
                device="cuda",
                synchronize=lambda: calls.append(f"{task}:sync"),
            )
            for task in tasks
        },
        activate=lambda _boot: (),
        deps=_cuda_dependencies(),
        decode_probe=_successful_decode,
        acquire=lambda: lease,
    )
    result = bootstrap.run_stages(stages)
    context.release_lease()

    assert calls == [
        "pose",
        "pose:sync",
        "person",
        "person:sync",
        "bed",
        "bed:sync",
        "lstm",
        "lstm:sync",
    ]
    assert result.outputs[bootstrap.WARMUP_STAGE] == tasks
    assert result.outputs[bootstrap.CAMERA_ACTIVATION_STAGE] == ()
    assert lease.held is False


def test_cpu_warmup_does_not_synchronize_and_clean_exit_releases_lease(tmp_path: Path) -> None:
    synchronizations: list[str] = []
    lease = GpuLease.acquire(tmp_path)
    context = bootstrap.BootstrapContext()
    stages = bootstrap.named_stages(
        context,
        {"ML_WORKER_PROFILE": "cpu"},
        initializers={"pose": lambda _boot: _Runner(lambda: None)},
        warmups={
            "pose": lambda runner: warmup_to_ready(
                runner,
                device="cpu",
                synchronize=lambda: synchronizations.append("unexpected"),
            )
        },
        activate=lambda _boot: (),
        decode_probe=_successful_decode,
        acquire=lambda: lease,
    )
    _ = bootstrap.run_stages(stages)
    context.release_lease()

    assert synchronizations == []
    replacement = GpuLease.acquire(tmp_path)
    assert replacement.held is True
    replacement.close()


def test_gpu_lease_rejects_a_second_process_until_clean_release(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path
        from worker.runtime.lease import GpuLease

        with GpuLease.acquire(Path(sys.argv[1])):
            print("lease-held", flush=True)
            time.sleep(10)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "lease-held"
        with pytest.raises(GpuLeaseUnavailableError):
            _ = GpuLease.acquire(tmp_path)
    finally:
        process.terminate()
        process.wait(timeout=2)

    replacement = GpuLease.acquire(tmp_path)
    assert replacement.held is True
    replacement.close()


def test_watchdog_passes_deadline_as_fatal_accelerator_fault() -> None:
    handler = _RecordingFaultHandler()
    watchdog = InferenceWatchdog(
        cast(FaultHandler, handler),
        profile="cuda",
        deadline_sec=0.02,
    )
    watchdog.start()
    watchdog.register(camera_id="camera-a", task="pose", frame_index=11)

    assert handler.called.wait(timeout=1)
    watchdog.stop()

    error, record = handler.calls[0]
    assert error.camera_id == "camera-a"
    assert error.task == "pose"
    assert record.stage == WATCHDOG_STAGE
    assert record.exit_code == 4
    assert record.frame_index == 11


def test_watchdog_subprocess_hard_exits_with_fatal_accelerator_code(tmp_path: Path) -> None:
    script = textwrap.dedent(
        f"""
        import time
        from pathlib import Path

        from worker.runtime.faults import FaultHandler
        from worker.runtime.watchdog import InferenceWatchdog

        watchdog = InferenceWatchdog(
            FaultHandler("cuda", state_dir=Path({str(tmp_path)!r})),
            profile="cuda",
            deadline_sec=0.05,
        )
        watchdog.start()
        watchdog.register(camera_id="camera-a", task="pose", frame_index=9)
        time.sleep(10)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        # 워치독 deadline이 0.05초라 정상 호스트에서는 1초 안에 끝난다.
        # 3초는 부하가 걸린 CI 러너에서 파이썬 기동 + import만으로 넘길 수
        # 있다(실제로 전체 스위트가 112초에서 318초로 늘어난 실행에서 이
        # 테스트만 TimeoutExpired로 두 번 연속 깨졌다). 판정하려는 것은
        # "워치독이 hard exit 하는가"이지 기동 속도가 아니므로 여유를 준다.
        timeout=30,
    )

    assert completed.returncode == 4
    connection = sqlite3.connect(tmp_path / WORKER_STATE_DB_FILENAME)
    try:
        cursor = connection.execute(
            "SELECT stage, exit_code, camera_id FROM faults WHERE id = 1"
        )
        stage, exit_code, camera_id = cursor.fetchone()
    finally:
        connection.close()
    assert stage == WATCHDOG_STAGE
    assert exit_code == 4
    assert camera_id == "camera-a"


@pytest.mark.real_stack
def test_watchdog_detects_a_genuinely_hanging_extractor_inside_the_real_composition(
    tmp_path: Path,
) -> None:
    """The #46 acceptance criterion the unit test above cannot prove: a forward
    pass that genuinely blocks its calling thread (a real ``time.sleep``, never
    a self-invoked ``watchdog.check()``) is caught by the watchdog's own
    background monitor thread while running inside the real
    ``WorkerRuntime`` -> ``CompositeExtractor`` -> ``CameraPipelinePump``
    composition, and hard-exits the process through the real
    ``FaultHandler``/``os._exit`` path -- exactly like
    ``test_watchdog_subprocess_hard_exits_with_fatal_accelerator_code`` above,
    except the hang is injected as a blocking "pose" extractor reached through
    the production composition rather than a direct ``watchdog.register()``
    call.

    Marked ``real_stack`` because it boots a real worker process and asserts on
    thread/wall-clock timing, not because it needs the ``mediamtx``/``ffmpeg``
    tooling that marker otherwise implies (see ``tests/AGENTS.md``): only the
    ingest *source* is faked, publishing one real ``FramePacket`` onto the real
    ``BoundedFrameBus`` and then idling. Every other seam is the genuine
    production composition: real ``WorkerRuntime`` (default, unstubbed
    ``pump_factory`` and ``hard_exit``), real ``CompositeExtractor`` with
    ``watchdog=`` wired, real ``CameraPipelinePump``, ``InferenceWatchdog``'s
    real ``"inference-watchdog"`` monitor thread (deadline shortened by
    subclassing, since ``WorkerConfig`` has no such knob), real
    ``FaultHandler``, and a real ``os._exit(4)``.
    """
    script = textwrap.dedent(
        """
        import sys
        import time
        from pathlib import Path

        import numpy as np

        import worker.runtime.worker as worker_module
        from contracts.frame import Frame
        from worker.runtime.config import WorkerConfig
        from worker.runtime.lease import GpuLease
        from worker.runtime.profile.registry import VerifyResult
        from worker.runtime.watchdog import InferenceWatchdog
        from worker.runtime.worker import WorkerRuntime
        from worker.types.frame_packet import FramePacket

        STATE_DIR = Path(sys.argv[1])


        class _FallMetadata:
            window = 2
            stride = 1
            mode = "sequence"


        class _FakeRunner:
            \"\"\"Stands in for pose/person/bed/fall's shared runner. Only
            ``.warmup()`` is ever exercised by this test: the hanging pose
            extractor is scheduled first each frame (see
            ``worker/pipeline/bus/scheduler.py``'s insertion-order iteration),
            so person/bed/fall are provisioned and warmed but never called.\"\"\"

            def __init__(self, task):
                self.task = task
                self.metadata = _FallMetadata()
                self.operating_threshold = 0.5

            def __call__(self, image):
                raise AssertionError(f"{self.task} runner must not be invoked")

            def predict(self, features):
                return 0.0

            def warmup(self):
                return None


        class _HangingPoseRunner:
            \"\"\"The genuinely blocking forward pass: a real ``time.sleep``,
            never a call to ``watchdog.check()`` -- only the watchdog's own
            monitor thread, on its own timer, can ever detect this.\"\"\"

            def __call__(self, image):
                time.sleep(30)
                raise AssertionError("hanging pose runner must never return")

            def warmup(self):
                return None


        class _FakeServingClient:
            def create(self, task, **_options):
                if task == "pose":
                    return _HangingPoseRunner()
                return _FakeRunner(task)


        class _OneFramePushLoop:
            \"\"\"Fake ingest loop: the only faked seam. Publishes exactly one
            real frame onto the real bus, then idles until stopped -- no RTSP,
            mediamtx, or ffmpeg required.\"\"\"

            def __init__(self, camera_id, bus, reporter):
                self.camera_id = camera_id
                self._bus = bus
                self._reporter = reporter
                self._stopped = False

            def run(self):
                self._reporter.mark_starting(self.camera_id)
                frame = Frame(
                    index=0,
                    time_sec=0.0,
                    image=np.zeros((360, 640, 3), dtype=np.uint8),
                )
                packet = FramePacket(
                    camera_id=self.camera_id,
                    frame=frame,
                    pts=0.0,
                    seq=0,
                    width=640,
                    height=360,
                    decode_time_ms=0.0,
                )
                self._reporter.mark_ready(self.camera_id)
                self._bus.publish(packet)
                while not self._stopped:
                    time.sleep(0.05)

            def stop(self):
                self._stopped = True


        def _loop_factory(camera, bus, reporter):
            return _OneFramePushLoop(camera.camera_id, bus, reporter)


        class _ShortDeadlineWatchdog(InferenceWatchdog):
            \"\"\"``WorkerConfig`` has no deadline knob, so the only way to keep
            this test fast is to shorten the deadline the real watchdog uses --
            the monitor thread, polling, and trip/exit path are all untouched.\"\"\"

            def __init__(self, handler, *, profile, **_ignored):
                super().__init__(handler, profile=profile, deadline_sec=0.2)


        # Bare-name rebind: `worker.py` resolves `InferenceWatchdog` from this
        # module's globals at call time (`self.watchdog = InferenceWatchdog(...)`
        # inside `_initialize_models`), so reassigning it here is picked up
        # without touching `WorkerRuntime.__init__`, which has no deadline seam.
        worker_module.InferenceWatchdog = _ShortDeadlineWatchdog


        def _fall_via_serving(self, _device):
            return self._serving.create("fall")


        # Fall model selection is fail-closed (#43) with no serving-client
        # fallback in production; pin the pre-#43 test-only behavior here so
        # this script needs no real LSTM artifact on disk.
        worker_module.WorkerRuntime._create_fall_model = _fall_via_serving

        config = WorkerConfig.model_validate(
            {
                "version": 1,
                "relay": {"url": "http://relay.test", "token": "relay-token"},
                "cameras": [
                    {
                        "camera_id": "camera-a",
                        "facility_id": "facility-a",
                        "rtsp_url": "rtsp://example.test/camera-a",
                        "heartbeat_interval_sec": 30.0,
                    }
                ],
            }
        )

        runtime = WorkerRuntime(
            config,
            env={"ML_WORKER_PROFILE": "cpu"},
            serving_client=_FakeServingClient(),
            loop_factory=_loop_factory,
            acquire_lease=lambda: GpuLease.acquire(STATE_DIR),
            decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
            state_dir=STATE_DIR,
        )
        # No `hard_exit=` override (stays the real `os._exit`) and no
        # `pump_factory=` override (stays `_default_pump_factory`, so the real
        # `CameraPipelinePump` runs on its own thread and calls the real
        # `CompositeExtractor.process()`).
        runtime.run()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 4, f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    connection = sqlite3.connect(tmp_path / WORKER_STATE_DB_FILENAME)
    try:
        cursor = connection.execute(
            "SELECT stage, exit_code, camera_id, task FROM faults WHERE id = 1"
        )
        stage, exit_code, camera_id, task = cursor.fetchone()
    finally:
        connection.close()
    assert stage == WATCHDOG_STAGE
    assert exit_code == 4
    assert camera_id == "camera-a"
    assert task == "pose"


def _raise_warmup() -> None:
    raise RuntimeError("representative forward failed")


def _raise_model_backend_init(_boot: object) -> _Runner:
    raise RuntimeError("fall model must be explicitly configured; refusing to boot")


def _successful_decode(_decode: str) -> VerifyResult:
    return VerifyResult(True, "", "decode", "available")


def _cuda_dependencies() -> bootstrap.BootDependencies:
    return bootstrap.BootDependencies(
        {"cuda": lambda: VerifyResult(True, "cuda", "device", "available")}
    )


def _initializer(task: str, calls: list[str]) -> Callable[[object], _Runner]:
    return lambda _boot: _Runner(lambda: calls.append(task))
