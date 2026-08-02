"""Real-stack: ``ML_WORKER_PROFILE=mps`` boot verification on real hardware.

Unlike ``test_worker_mps_device_probe.py`` (fakes only, deterministic, runs
everywhere) and ``test_worker_production_boot_dependencies.py`` (also fakes
``probe_mps_capability``), this test exercises the real ``import torch`` and
real ``torch.backends.mps`` calls wired into ``production_boot_dependencies``
end to end through ``resolve_profile``/``verify_device_or_raise``
(``worker/runtime/profile/boot.py``) -- the same path
``WorkerRuntime`` boot takes when ``ML_WORKER_PROFILE=mps`` is set.

Marked ``real_stack`` (see ``tests/AGENTS.md``) and deselected in CI via
``-m "not real_stack"``: the assertion (``result.ok is True``) is only true on
real Apple Silicon with an MPS-capable torch build, so it cannot run
deterministically on every dev/CI machine the way the fake-backed tests do.
Skipped outside Darwin/arm64 rather than asserted, since a negative result
there would be an environment fact, not a regression.
"""

from __future__ import annotations

import platform

import pytest

from worker.runtime.profile.boot import resolve_profile, verify_device_or_raise
from worker.runtime.worker import production_boot_dependencies

pytestmark = pytest.mark.real_stack


def test_ml_worker_profile_mps_boots_on_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("requires real Apple Silicon hardware (Darwin/arm64)")
    pytest.importorskip("torch")

    spec = resolve_profile({"ML_WORKER_PROFILE": "mps"})

    result = verify_device_or_raise(spec, production_boot_dependencies())

    assert result.ok is True
    assert result.profile == "mps"
    assert result.reason == "MPS is available"
