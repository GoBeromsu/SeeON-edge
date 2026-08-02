from __future__ import annotations

import json
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
from worker.runtime.faults.record import FirstFaultRecord
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
        timeout=3,
    )

    assert completed.returncode == 4
    record = json.loads((tmp_path / "first_fault.json").read_text(encoding="utf-8"))
    assert record["stage"] == WATCHDOG_STAGE
    assert record["exit_code"] == 4
    assert record["camera_id"] == "camera-a"


def _raise_warmup() -> None:
    raise RuntimeError("representative forward failed")


def _successful_decode(_decode: str) -> VerifyResult:
    return VerifyResult(True, "", "decode", "available")


def _cuda_dependencies() -> bootstrap.BootDependencies:
    return bootstrap.BootDependencies(
        {"cuda": lambda: VerifyResult(True, "cuda", "device", "available")}
    )


def _initializer(task: str, calls: list[str]) -> Callable[[object], _Runner]:
    return lambda _boot: _Runner(lambda: calls.append(task))
