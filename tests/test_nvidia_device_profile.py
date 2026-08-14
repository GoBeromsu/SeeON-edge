from __future__ import annotations

import pytest

from worker.runtime.profile.boot import resolve_boot_context, resolve_profile
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    ML_WORKER_PROFILE_ENV,
    BootDependencies,
    ProfileVerifyError,
    VerifyResult,
    default_verifiers,
)


def _decode_ok(backend: str) -> VerifyResult:
    return VerifyResult(True, "nvidia", "decode", f"{backend} available")


def _encode_ok() -> VerifyResult:
    return VerifyResult(True, "nvidia", "encode", "nvenc available")


def _cuda_ok() -> CudaProbe:
    return CudaProbe(True, "cuda available", device_count=1, arch_list=("sm_90",))


def test_device_resident_profile_is_never_selected_without_exact_opt_in() -> None:
    assert resolve_profile({}).name == "cpu-host"
    assert resolve_profile({ML_WORKER_PROFILE_ENV: "cuda"}).name == "nvidia-host-bridge"
    experimental = resolve_profile({ML_WORKER_PROFILE_ENV: "nvidia-device-experimental"})
    assert experimental.name == "nvidia-device-experimental"
    assert experimental.accepted_names == ("nvidia-device-experimental",)


def test_plain_cuda_capability_cannot_promote_or_unblock_device_resident_profile() -> None:
    plain_cuda_only = BootDependencies(default_verifiers(cuda_source=_cuda_ok))
    production = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "cuda"},
        plain_cuda_only,
        _decode_ok,
        _encode_ok,
    )
    assert production.canonical_profile == "nvidia-host-bridge"
    assert production.runtime_profile.device_resident_after_decode is False

    with pytest.raises(
        ProfileVerifyError,
        match="device-resident capability probe is not configured",
    ):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia-device-experimental"},
            plain_cuda_only,
            _decode_ok,
            _encode_ok,
        )

    blocked = BootDependencies(
        default_verifiers(
            cuda_source=_cuda_ok,
            device_resident_source=lambda: VerifyResult(
                False,
                "nvidia-device-experimental",
                "device",
                "DLPack handoff unavailable",
            ),
        )
    )
    with pytest.raises(ProfileVerifyError, match="DLPack handoff unavailable"):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia-device-experimental"},
            blocked,
            _decode_ok,
            _encode_ok,
        )
