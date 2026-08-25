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
from contracts.runner import RunnerProtocol
from worker.adapters.device.mps.probe import MpsCapability
from worker.native.deepstream.preflight import DeepStreamPreflightError
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.worker import WorkerRuntime, production_boot_dependencies


class _FakeServingClient:
    def create(
        self,
        task: str,
        **_kwargs: str | int | float | bool | None,
    ) -> RunnerProtocol:
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


def test_production_boot_dependencies_expose_no_plain_cuda_verifier() -> None:
    """Plain CUDA no longer gates a public profile after the nvidia unification.

    The single `nvidia` profile is gated by the device-resident probe, so a
    bare `cuda`/`nvidia-host-bridge` verifier key must not survive: leaving one
    wired would let a host that only passes `torch.cuda.is_available()` boot a
    profile whose descriptor promises device-resident concrete stages.
    """
    verifiers = production_boot_dependencies().verifiers

    assert "cuda" not in verifiers
    assert "nvidia-host-bridge" not in verifiers
    assert "nvidia-device-experimental" not in verifiers
    assert "nvidia" in verifiers


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


def test_production_boot_dependencies_nvidia_true_when_preflight_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(
        worker_module,
        "run_configured_deepstream_preflight",
        lambda: {"status": "ok"},
    )

    # When
    result = production_boot_dependencies().verifiers["nvidia"]()

    # Then
    assert result.ok is True
    assert result.profile == "nvidia"
    assert result.stage == "device"
    assert result.reason == "pinned DeepStream preflight passed"


def test_production_boot_dependencies_nvidia_fails_closed_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    def fail_preflight() -> dict[str, str]:
        raise DeepStreamPreflightError("gpu_absent", "NVIDIA device unavailable")

    monkeypatch.setattr(worker_module, "run_configured_deepstream_preflight", fail_preflight)

    # When
    result = production_boot_dependencies().verifiers["nvidia"]()

    # Then
    assert result.ok is False
    assert result.profile == "nvidia"
    assert "NVIDIA device unavailable" in result.reason


def test_production_boot_dependencies_nvidia_fails_closed_on_plain_cuda_only_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host that only passes plain CUDA must not satisfy the `nvidia` gate.

    After unification there is a single NVIDIA profile whose descriptor claims
    device-resident concrete stages, so its verifier must consult the
    device-resident probe -- never `probe_cuda_capability` alone.
    """
    def fail_preflight() -> dict[str, str]:
        raise DeepStreamPreflightError("plugin_missing", "nvstreammux unavailable")

    monkeypatch.setattr(worker_module, "run_configured_deepstream_preflight", fail_preflight)

    result = production_boot_dependencies().verifiers["nvidia"]()

    assert result.ok is False
    assert result.profile == "nvidia"
    assert "nvstreammux unavailable" in result.reason


def test_worker_runtime_defaults_boot_dependencies_to_production_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WorkerRuntime must wire the real nvidia verifier itself when none is injected.

    Every test elsewhere in this suite injects a stub ``boot_dependencies`` on
    purpose; this test is the one place that must *not*, to prove the
    composition root's own default reaches the real, hardware-checking probe
    and not ``None`` threaded through to bootstrap's fail-closed fallback.
    """
    monkeypatch.setattr(
        worker_module,
        "run_configured_deepstream_preflight",
        lambda: {"status": "ok"},
    )

    runtime = WorkerRuntime(
        _config(),
        serving_client=_FakeServingClient(),
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
    )

    result = runtime._boot_dependencies.verifiers["nvidia"]()  # noqa: SLF001
    assert result.ok is True
    assert result.reason == "pinned DeepStream preflight passed"
