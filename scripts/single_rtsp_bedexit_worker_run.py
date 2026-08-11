"""Bed-exit single-RTSP worker run, extracted from the operator script.

Lived in a `uv run python - <<'PY'` heredoc inside
`ml-worker-single-rtsp-bedexit-e2e.sh`. At 7162 bytes it is over `PIPE_BUF` on
both macOS (512) and Linux (4096), and bash 5.x writes a heredoc body into a
pipe before exec'ing the reader -- so the script hung there with no output. The
`#!/bin/bash` pin only fixes that on macOS, where /bin/bash is 3.2.57; on a
Linux edge host /bin/bash is itself a modern bash, so the body had to move into
a file. Byte-identical to the heredoc it replaced. See issue #9.

Usage: python single_rtsp_bedexit_worker_run.py <frames> <config_path>

``config_path`` is a positional argument, not the former ``EDGE_CAMERA_CONFIG``
env var: the camera roster/config path must never be provisionable through the
environment (an env var reaches the runtime via compose, and compose is
tracked in Git) -- see scripts/verify_scope_fidelity.py's ROSTER_PATTERN.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, final

import worker.runtime.worker as worker_module
from contracts.runner import Image, RunnerResult, bed_result, person_result, pose_result
from worker.runtime.config import load_worker_config
from worker.runtime.worker import WorkerRuntime


@final
class _EmptyPoseRunner:
    """No pose data: bed-exit tracking runs on person geometry alone."""

    def __call__(self, _image: Image) -> RunnerResult:
        return pose_result(poses=(), boxes=())

    def warmup(self) -> None:
        return None


# Same calibrated geometry as tests/e2e_worker_relay_fixtures.py's
# ScriptedBedExitPersonRunner, against the REAL (unconfigurable-via-YAML)
# BedExitConfig defaults worker/runtime/worker.py._bed_exit_config always
# applies: min_containment=0.35, hold_frames=2, grace_frames=3. Steps 1-2
# fully contain the person (assigns the bed by step 2); steps 3-4 partially
# overlap (still >=0.35, resets grace); step 5 onward is fully outside (grace
# increments each frame, firing once grace_frames > 3, i.e. on the 4th
# consecutive frame outside -- step 8). Matches this script's default
# MAX_FRAMES_PER_CAMERA=8; raising the frame count only holds the final
# outside position longer and does not change when the event fires.
_BED_EXIT_XS: Final = (15.0, 15.0, 40.0, 65.0, 90.0, 90.0, 90.0, 90.0)
_BED_BOX: Final = (10.0, 10.0, 90.0, 80.0, 0.9)


@final
class _WalkingPersonRunner:
    def __init__(self) -> None:
        self._index = 0
        self._lock = threading.Lock()

    def __call__(self, _image: Image) -> RunnerResult:
        with self._lock:
            index = min(self._index, len(_BED_EXIT_XS) - 1)
            self._index += 1
        x1 = _BED_EXIT_XS[index]
        return person_result(boxes=((x1, 15.0, x1 + 70.0, 75.0, 0.95),))

    def warmup(self) -> None:
        return None


@final
class _FixedBedRunner:
    def __call__(self, _image: Image) -> RunnerResult:
        return bed_result(boxes=(_BED_BOX,))

    def warmup(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 2
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@final
class _InertFallModel:
    """Never consulted -- this run's config enables bed_exit only -- but
    WorkerRuntime._initialize_models() always calls serving.create("fall")
    at boot regardless of enabled domains, so a well-formed stand-in is
    still required."""

    def __init__(self) -> None:
        self.metadata = _FallMetadata()
        self.operating_threshold = 0.5
        self.warmup_count = 0

    def predict(self, _features: object) -> float:
        return 0.0

    def warmup(self) -> None:
        self.warmup_count += 1


@final
class _CountingPersonRunner:
    """Wraps the person runner to signal once MAX_FRAMES_PER_CAMERA decoded
    frames have been pulled through the real per-frame pipeline (person is
    scheduled every frame; bed is only rescanned every 30 frames, so person
    is the correct per-frame proxy)."""

    def __init__(self, inner: _WalkingPersonRunner, *, target: int, done: threading.Event) -> None:
        self._inner = inner
        self._target = target
        self._done = done
        self._count = 0
        self._lock = threading.Lock()

    def __call__(self, image: Image) -> RunnerResult:
        result = self._inner(image)
        with self._lock:
            self._count += 1
            if self._count >= self._target:
                self._done.set()
        return result

    def warmup(self) -> None:
        self._inner.warmup()


@final
class _ScriptedServingClient:
    def __init__(self, *, target_frames: int, done: threading.Event) -> None:
        self._runners: dict[str, object] = {
            "pose": _EmptyPoseRunner(),
            "person": _CountingPersonRunner(
                _WalkingPersonRunner(), target=target_frames, done=done
            ),
            "bed": _FixedBedRunner(),
            "fall": _InertFallModel(),
        }

    def create(self, task: str, **_options: object) -> object:
        try:
            return self._runners[task]
        except KeyError as error:
            raise AssertionError(f"unexpected serving task requested: {task!r}") from error


class _FrozenDatetime(datetime):
    """Freezes worker.runtime.worker's bed-exit night-window clock.

    worker/runtime/worker.py:_build_decider hardcodes
    `clock=lambda: datetime.now(UTC)` for the bed_exit decider with no
    injection seam (it is the ONLY `datetime` use in that whole module), and
    worker.domains.registry.DOMAIN_REGISTRY is an immutable
    types.MappingProxyType, so the legacy edge_worker approach of swapping in
    a replacement DomainRegistration no longer works. Instead, monkeypatch
    the bare module-level `datetime` name worker.runtime.worker resolves at
    call time -- the same idiom tests/e2e_worker_relay_fixtures.py already
    uses for worker_module.ClipRecorderConfig. No production file is edited.
    """

    _frozen: datetime

    @classmethod
    def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
        return cls._frozen if tz is None else cls._frozen.astimezone(tz)


def _wait_until(predicate, *, timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def main() -> int:
    frames = int(sys.argv[1])
    frozen_now = datetime.fromisoformat(os.environ["BED_EXIT_NOW"])
    if frozen_now.tzinfo is None:
        print("BED_EXIT_NOW must carry a UTC offset", file=sys.stderr)
        return 1
    _FrozenDatetime._frozen = frozen_now
    worker_module.datetime = _FrozenDatetime

    config = load_worker_config(sys.argv[2])
    done = threading.Event()
    serving = _ScriptedServingClient(target_frames=frames, done=done)
    runtime = WorkerRuntime(
        config,
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=serving,
        hard_exit=lambda _code: None,
    )

    thread = threading.Thread(target=runtime.run, daemon=True, name="single-rtsp-bedexit-worker")
    thread.start()
    try:
        if not _wait_until(lambda: len(runtime.cameras) > 0, timeout=30.0):
            print("camera did not activate within 30s", file=sys.stderr)
            return 1
        if not done.wait(timeout=max(60.0, frames * 5.0)):
            print(f"did not observe {frames} decoded frames within timeout", file=sys.stderr)
            return 1
        # Event admission -> durable staging -> the async export sender's
        # relay POST (worker/pipeline/output/evidence/evidence_runtime.py)
        # run on background threads, not synchronously with frame
        # processing -- give them a settle window before tearing down.
        time.sleep(5.0)
    finally:
        runtime.stop()
        thread.join(timeout=15.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
