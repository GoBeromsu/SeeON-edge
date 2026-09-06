"""6-stage 2-tier bootstrap gate injection (ADR-0002, acceptance 4), worker edition.

worker's successor to the legacy flat `runtime.pipeline_bootstrap` module is
`worker.runtime.bootstrap`: six *named* global stages (gpu_lease -> profile_device
-> decode_capability -> model_backend_init -> real_warmup -> camera_activation)
instead of edge's caller-assembled two-stage list, plus a `FatalAcceleratorError`
override that outranks a stage's own exit code (worker/runtime/bootstrap.py:1-36).

Disposition correction (evidence-based; differs from the initial GpuUnavailableError
-> {GpuLeaseUnavailableError, FatalAcceleratorError} hypothesis): edge's
`GpuUnavailableError` (edge/runners/device.py:26, raised by `require_gpu_device`)
was edge's *boot-time, CUDA-unavailable, fail-fast, no-CPU-fallback* signal. Its
real worker successor is `ProfileVerifyError`, raised by
`verify_device_or_raise` (worker/runtime/profile/boot.py:50-61) when a profile's
device verifier reports `ok=False` -- already covered unit-level by
tests/test_profile_boot.py (test_cuda_profile_verify_false,
test_verifier_exception_becomes_profile_verify_error, test_cuda_profile_verify_true,
test_cpu_profile_ok). `GpuLeaseUnavailableError` (worker/runtime/lease.py:43) and
`FatalAcceleratorError` (worker/adapters/model/errors.py:17) are *not* alternate
spellings of that same boot-time check -- they cover two failure modes edge's
pipeline_bootstrap.py never had at all: pre-CUDA-touch advisory-lease contention
between repo processes, and mid-inference CUDA runtime faults. Both are exercised
here only for their bootstrap-stage *wiring* (gpu_lease_stage's REFUSE_TO_START
mapping; run_camera_stage's re-raise-don't-degrade special case), since that
wiring is worker-only code with no edge analog and, per the residue check below,
no other existing test hits it directly.

Residue check performed against tests/test_worker_runtime_safety.py (which drives
the full named_stages() sequence end-to-end) and tests/test_worker_gpu_lease.py
(which drives GpuLease.acquire() directly, not via the bootstrap Stage wrapper):
neither exercises `gpu_lease_stage`'s own failure path, `profile_device_stage`'s
own failure path, or `run_camera_stage` at all -- ML_WORKER_PROFILE=cpu keeps
their profile/device stage trivially passing (test_cpu_profile_ok-equivalent), so
edge's `test_gpu_verify_stage_honors_explicit_cpu_optin` is superseded by that
existing cpu-profile coverage rather than re-ported: worker has no "explicit
device overrides a fail-fast check" parameter at all (ML_WORKER_PROFILE is a
required enum, not an optional override), so the *mechanism* being tested no
longer exists, only the *outcome* (choosing cpu never fails fast) does, and that
outcome is already asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.adapters.model.errors import FatalAcceleratorError
from worker.runtime import bootstrap as boot
from worker.runtime.lease import GpuLeaseUnavailableError
from worker.runtime.profile.registry import BootDependencies, VerifyResult


def test_run_stages_runs_in_order_and_collects_outputs() -> None:
    calls: list[str] = []
    stages = (
        boot.Stage("gpu_verify", lambda: (calls.append("gpu"), "cuda")[1]),
        boot.Stage("backend_init", lambda: (calls.append("backend"), "bundle")[1]),
        boot.Stage("warmup", lambda: (calls.append("warmup"), "ready")[1]),
    )
    result = boot.run_stages(stages)
    assert calls == ["gpu", "backend", "warmup"]
    assert result.outputs == {"gpu_verify": "cuda", "backend_init": "bundle", "warmup": "ready"}


def test_stage_failure_raises_bootstrap_stage_error_naming_the_stage() -> None:
    def boom() -> None:
        raise RuntimeError("backend down")

    stages = (boot.Stage("gpu_verify", lambda: "cuda"), boot.Stage("backend_init", boom))
    with pytest.raises(boot.BootstrapStageError) as exc:
        boot.run_stages(stages)
    assert exc.value.stage == "backend_init"
    assert "backend_init" in str(exc.value)
    assert "backend down" in str(exc.value)


def test_bootstrap_or_exit_exits_with_stage_exit_code_on_failure() -> None:
    exit_codes: list[int] = []

    def boom() -> None:
        raise RuntimeError("no compiled arch kernels (sm_120 missing); see ADR-0002")

    stages = (boot.Stage("gpu_verify", boom),)
    with pytest.raises(boot.BootstrapStageError):
        boot.bootstrap_or_exit(stages, exit_fn=exit_codes.append)
    assert exit_codes == [boot.GENERIC_RUNTIME_EXIT_CODE]
    assert boot.GENERIC_RUNTIME_EXIT_CODE != 0


def test_bootstrap_or_exit_returns_result_when_all_pass() -> None:
    exit_codes: list[int] = []
    stages = (boot.Stage("gpu_verify", lambda: "cuda"), boot.Stage("warmup", lambda: "ready"))
    result = boot.bootstrap_or_exit(stages, exit_fn=exit_codes.append)
    assert exit_codes == []
    assert result.outputs["gpu_verify"] == "cuda"


def test_gpu_lease_stage_fails_fast_when_lease_unavailable(tmp_path: Path) -> None:
    context = boot.BootstrapContext()

    def refuse() -> boot.GpuLease:
        raise GpuLeaseUnavailableError(tmp_path / ".gpu.lease")

    stage = boot.gpu_lease_stage(context, acquire=refuse)
    with pytest.raises(boot.BootstrapStageError) as exc:
        boot.run_stages((stage,))
    assert exc.value.exit_code == boot.REFUSE_TO_START_EXIT_CODE
    assert context.lease is None


def test_profile_device_stage_fails_fast_and_publishes_no_profile_on_failure() -> None:
    context = boot.BootstrapContext()
    deps = BootDependencies({"flow": lambda: VerifyResult(True, "flow", "device", "available")})
    stage = boot.profile_device_stage(context, {"ML_WORKER_PROFILE": "nvidia"}, deps)

    with pytest.raises(boot.BootstrapStageError) as exc:
        boot.run_stages((stage,))
    assert exc.value.exit_code == boot.REFUSE_TO_START_EXIT_CODE
    assert "ADR-0002: unsupported ML_WORKER_PROFILE 'nvidia'; set flow" in str(exc.value)
    assert context.profile is None  # never published on a failed verify


def test_per_camera_ordinary_failure_degrades_only_that_camera() -> None:
    def boom() -> None:
        raise RuntimeError("nvdec spawn failed")

    ok = boot.run_camera_stage("camA", lambda: None)
    degraded = boot.run_camera_stage("camB", boom)
    assert ok.ok is True and ok.reason is None
    assert degraded.ok is False and "nvdec" in degraded.reason
    assert degraded.camera_id == "camB"


def test_per_camera_fatal_accelerator_error_is_reraised_not_degraded() -> None:
    def boom() -> None:
        raise FatalAcceleratorError("CUDA error: device-side assert triggered", camera_id="camC")

    with pytest.raises(FatalAcceleratorError):
        boot.run_camera_stage("camC", boom)
