"""Real-stack: unified ``nvidia`` capability probe on real NVIDIA hardware.

Unlike ``test_nvidia_device_resident_prototype.py`` (fakes only, deterministic,
runs everywhere including this repo's Apple Silicon dev/CI hosts), this
exercises the real ``torch``/``pynvml`` calls wired into
``probe_device_resident_capability`` end to end -- the same path
``production_boot_dependencies`` takes when
``ML_WORKER_PROFILE=nvidia`` is set (see
``worker/runtime/worker.py:_production_device_resident_source``).

Marked ``real_stack`` (see ``tests/AGENTS.md``) and deselected in CI via
``-m "not real_stack"``: the assertion (``capability.available is True``) is
only true on a real NVIDIA host with a working driver/CUDA/torch pairing per
ADR-0002, so it cannot run deterministically on every dev/CI machine the way
the fake-backed tests do. Skipped (not asserted false) when no NVIDIA GPU is
enumerable, since a negative result there is an environment fact, not a
regression -- mirrors ``test_worker_mps_real_stack_boot.py``'s Darwin/arm64
skip-not-assert convention for the equivalent MPS case.
"""

from __future__ import annotations

import pytest

from worker.adapters.decode.nvdec_device.capability import probe_device_resident_capability
from worker.adapters.device.nvml.probe import probe_nvml_gpu_status
from worker.runtime.profile.boot import resolve_profile, verify_device_or_raise
from worker.runtime.worker import production_boot_dependencies

pytestmark = pytest.mark.real_stack


def test_device_resident_capability_probe_on_real_nvidia_hardware() -> None:
    pytest.importorskip("torch")
    nvml_status = probe_nvml_gpu_status()
    if not nvml_status.nvml_available:
        pytest.skip(f"requires real NVIDIA hardware (NVML unavailable: {nvml_status.reason})")

    capability = probe_device_resident_capability()

    assert capability.available is True, capability.reason
    assert capability.cuda.available is True
    assert capability.nvml.nvml_available is True
    assert capability.stream_event_supported is True
    assert capability.dlpack_supported is True


def test_nvidia_profile_boots_on_real_nvidia_hardware() -> None:
    """End-to-end: the same boot gate `WorkerRuntime` calls, on real hardware.

    After the NVIDIA profile unification this device-resident gate is the only
    gate for ``nvidia``; it must resolve true when real device-resident
    capability is present.
    """
    pytest.importorskip("torch")
    nvml_status = probe_nvml_gpu_status()
    if not nvml_status.nvml_available:
        pytest.skip(f"requires real NVIDIA hardware (NVML unavailable: {nvml_status.reason})")

    spec = resolve_profile({"ML_WORKER_PROFILE": "nvidia"})

    result = verify_device_or_raise(spec, production_boot_dependencies())

    assert result.ok is True
    assert result.profile == "nvidia"
