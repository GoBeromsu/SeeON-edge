"""Issue #150: `WorkerRuntime` must boot -- and stay up -- with zero cameras.

Before this fix, a new install (no cameras registered yet) could never reach
this point at all: `WorkerConfig.cameras` required `min_length=1` and
`BackendWorkerConfigPayload.to_worker_config` raised on an empty resolved
roster, so `worker/__main__.py` exited with `CONFIG_ERROR_EXIT_CODE` before
`WorkerRuntime` was ever constructed (see
`tests/test_worker_startup_config_resolution.py` for that layer).

This module covers the next layer down, once those config-shape gates are
relaxed: `WorkerRuntime.run()` itself must not treat "zero ingest loops" as
"nothing to do, return immediately" -- that would boot the worker, pass every
gate (profile/device, decode preflight, model load), and then exit right back
out, taking the probe/MJPEG server down with it before an installer ever gets
a chance to validate a camera's RTSP URL against it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.runner import Image, RunnerResult
from worker.pipeline.ingest.lifecycle import IngestSupervisor
from worker.runtime.config import WorkerConfig
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
    """Mirrors tests/test_worker_max_frames_per_camera_composition.py's fake:
    `_initialize_models` always composes real pose/person/bed/fall runners
    through `serving_client.create(...)`, even with zero cameras, so this
    must return something usable rather than raise.
    """

    def __init__(self, task: str) -> None:
        self.task = task
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def __call__(self, _image: Image) -> RunnerResult:
        raise AssertionError("zero-camera boot tests must not run model inference")

    def predict(self, _features: NDArray[np.float32]) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _FakeServingClient:
    def create(self, task: str, **_options: object) -> _FakeRunner:
        return _FakeRunner(task)


@final
class _FastPollIngestSupervisor(IngestSupervisor):
    """Same supervisor, but polling fast enough for a test to observe a
    restart directive without a multi-second real-time wait. `_activate`
    (worker/runtime/worker.py) does not expose `restart_poll_interval_sec`
    to its caller, so this is monkeypatched in for the one test that needs
    it, exactly as tests/test_worker_ingest_lifecycle.py does directly
    against `IngestSupervisor` itself.
    """

    def __init__(self, loops: object, **kwargs: object) -> None:
        kwargs.setdefault("restart_poll_interval_sec", 0.01)
        super().__init__(loops, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _fall_model_via_serving_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests never configure a real LSTM artifact on disk; get the fall
    runner from the injected `_FakeServingClient` instead, matching the same
    scoped override tests/test_worker_composition.py and
    tests/test_worker_max_frames_per_camera_composition.py already use."""

    def _fall_via_serving(self: WorkerRuntime, _device: str) -> object:
        return self._serving.create("fall")  # noqa: SLF001

    monkeypatch.setattr(WorkerRuntime, "_create_fall_model", _fall_via_serving)


def _zero_camera_config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [],
        }
    )


def _runtime(state_dir: Path, *, restart_check: object = None) -> WorkerRuntime:
    return WorkerRuntime(
        _zero_camera_config(),
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=_FakeServingClient(),
        acquire_lease=lambda: GpuLease.acquire(state_dir),
        decode_probe=lambda _decode: VerifyResult(True, "cpu", "decode", "available"),
        hard_exit=lambda _code: None,
        restart_check=restart_check,  # type: ignore[arg-type]
    )


def _run_in_background(
    runtime: WorkerRuntime,
) -> tuple[threading.Thread, threading.Event, list[BaseException]]:
    """Drive `runtime.run()` on a background thread, capturing both
    completion and any exception -- a thread that just silently dies on an
    unhandled exception would otherwise look identical to one still blocked.
    """
    finished = threading.Event()
    errors: list[BaseException] = []

    def target() -> None:
        try:
            runtime.run()
        except BaseException as exc:  # noqa: BLE001 - captured for the test's own assertion
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, finished, errors


def test_zero_cameras_run_blocks_instead_of_exiting_immediately(tmp_path: Path) -> None:
    """The core of issue #150: with a restart watcher wired in (as
    `worker/__main__.py` always does) `run()` must stay up -- not return the
    instant `IngestSupervisor.join()` finds zero ingest threads to join --
    passing every boot gate (profile/device, decode preflight, model load,
    camera activation with an empty roster) along the way, and only return
    once something actually stops it.
    """
    runtime = _runtime(tmp_path, restart_check=lambda: False)
    thread, finished, errors = _run_in_background(runtime)
    try:
        # Given: boot has had time to finish and nothing has told it to stop.
        assert not finished.wait(timeout=0.3), (
            f"run() returned immediately with zero cameras (errors={errors!r})"
        )
        assert runtime.cameras == ()
        assert runtime._supervisor is not None  # noqa: SLF001

        # When: an external stop arrives (mirrors a SIGTERM/SIGINT handler).
        runtime.stop()

        # Then: run() unblocks and returns cleanly.
        assert finished.wait(timeout=2.0), "run() never returned after stop()"
    finally:
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    assert not errors, f"run() raised: {errors!r}"


def test_zero_cameras_run_returns_once_the_restart_watcher_observes_a_fresh_pull(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Once a camera is registered, the restart-check the boot process wired
    in (issue #150's actual pickup mechanism) reports a fresh directive; the
    zero-camera worker must exit clean rather than keep waiting forever --
    the container's `restart: unless-stopped` is what brings it back up to
    pull the new roster and start the real pipeline.
    """
    monkeypatch.setattr(worker_module.ingest, "IngestSupervisor", _FastPollIngestSupervisor)
    calls: list[None] = []

    def restart_check() -> bool:
        calls.append(None)
        # First couple of polls: no new config yet. Then: a camera showed up.
        return len(calls) >= 3

    runtime = _runtime(tmp_path, restart_check=restart_check)
    thread, finished, errors = _run_in_background(runtime)
    try:
        assert finished.wait(timeout=3.0), "run() never returned once restart_check fired"
    finally:
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    assert not errors, f"run() raised: {errors!r}"
    assert len(calls) >= 3
