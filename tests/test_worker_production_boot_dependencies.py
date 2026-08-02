"""Tests for the composition root's real boot_dependencies (device verify) wiring.

Before this wiring existed, ``WorkerRuntime`` threaded ``boot_dependencies=None``
straight through to ``bootstrap.profile_device_stage``, which falls back to
``BootDependencies(default_verifiers())`` -- fail-closed with "CUDA capability
probe is not configured" on *every* profile, including ``cuda``.
``production_boot_dependencies`` (worker/runtime/worker.py) is the real,
hardware-checking default that fixes that for both the ``cuda`` and ``mps``
profiles, leaving ``cpu`` (always available) untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import worker.runtime.worker as worker_module
from worker.adapters.device.cuda.probe import CudaCapability
from worker.adapters.device.mps.probe import MpsCapability
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.worker import WorkerRuntime, production_boot_dependencies


class _FakeServingClient:
    def create(self, task: str) -> object:
        raise AssertionError(f"unexpected serving client call for task {task!r}")


def _config() -> WorkerConfig:
    return WorkerConfig.model_validate(
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


def test_production_boot_dependencies_cuda_true_when_capability_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(
        worker_module,
        "probe_cuda_capability",
        lambda: CudaCapability(True, "cuda available", device_count=1, arch_list=("sm_90",)),
    )

    # When
    result = production_boot_dependencies().verifiers["cuda"]()

    # Then
    assert result.ok is True
    assert result.profile == "cuda"
    assert result.stage == "device"
    assert result.reason == "cuda available"


def test_production_boot_dependencies_cuda_false_fails_closed_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given -- the real signal on this repo's macOS dev machines: no CUDA device
    monkeypatch.setattr(
        worker_module,
        "probe_cuda_capability",
        lambda: CudaCapability(
            False, "torch.cuda.is_available() is False and no CUDA devices are visible"
        ),
    )

    # When
    result = production_boot_dependencies().verifiers["cuda"]()

    # Then
    assert result.ok is False
    assert result.profile == "cuda"
    assert result.reason == "torch.cuda.is_available() is False and no CUDA devices are visible"


def test_production_boot_dependencies_cpu_always_available() -> None:
    """The cpu profile must not regress: it never consults a device source."""
    result = production_boot_dependencies().verifiers["cpu"]()

    assert result.ok is True
    assert result.profile == "cpu"
    assert result.reason == "CPU is available"


def test_production_boot_dependencies_mps_true_when_capability_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given -- Apple Silicon with a torch build that supports MPS
    monkeypatch.setattr(
        worker_module,
        "probe_mps_capability",
        lambda: MpsCapability(True, "mps available", is_built=True),
    )

    # When
    result = production_boot_dependencies().verifiers["mps"]()

    # Then -- MpsProbeSource is a bare Callable[[], bool], so only the
    # available flag crosses into the verifier; the reason is the registry's
    # own generic string, not the probe's detailed diagnostic.
    assert result.ok is True
    assert result.profile == "mps"
    assert result.stage == "device"
    assert result.reason == "MPS is available"


def test_production_boot_dependencies_mps_false_fails_closed_without_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given -- e.g. an NVIDIA machine or a torch wheel without MPS support
    monkeypatch.setattr(
        worker_module,
        "probe_mps_capability",
        lambda: MpsCapability(False, "MPS not usable: torch.backends.mps.is_built() is False"),
    )

    # When
    result = production_boot_dependencies().verifiers["mps"]()

    # Then
    assert result.ok is False
    assert result.profile == "mps"
    assert result.reason == "MPS is unavailable"


def test_worker_runtime_defaults_boot_dependencies_to_production_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WorkerRuntime must wire the real cuda verifier itself when none is injected.

    Every test elsewhere in this suite injects a stub ``boot_dependencies`` on
    purpose; this test is the one place that must *not*, to prove the
    composition root's own default reaches the real, hardware-checking probe
    and not ``None`` threaded through to bootstrap's fail-closed fallback.
    """
    monkeypatch.setattr(
        worker_module,
        "probe_cuda_capability",
        lambda: CudaCapability(True, "cuda available", device_count=1, arch_list=("sm_90",)),
    )

    runtime = WorkerRuntime(
        _config(),
        serving_client=_FakeServingClient(),
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
    )

    result = runtime._boot_dependencies.verifiers["cuda"]()  # noqa: SLF001
    assert result.ok is True
    assert result.reason == "cuda available"
