from __future__ import annotations

import re

import pytest

import worker.runtime.worker as worker_module
from worker.adapters.decode.nvdec_device.capability import DeviceResidentCapability
from worker.adapters.device.cuda.probe import CudaCapability
from worker.adapters.device.nvml.probe import NvmlGpuStatus
from worker.runtime.profile.boot import resolve_boot_context, resolve_profile
from worker.runtime.profile.device import CudaProbe
from worker.runtime.profile.registry import (
    CANONICAL_PROFILE_REGISTRY,
    ML_WORKER_PROFILE_ENV,
    PROFILE_ALIASES,
    PROFILE_REGISTRY,
    BootDependencies,
    ProfileError,
    ProfileVerifyError,
    VerifyResult,
    default_verifiers,
)
from worker.runtime.worker import production_boot_dependencies


def _decode_ok(backend: str) -> VerifyResult:
    return VerifyResult(True, "nvidia", "decode", f"{backend} available")


def _encode_ok() -> VerifyResult:
    return VerifyResult(True, "nvidia", "encode", "nvenc available")


def _cuda_ok() -> CudaProbe:
    return CudaProbe(True, "cuda available", device_count=1, arch_list=("sm_90",))


def _resident_ok() -> VerifyResult:
    return VerifyResult(True, "nvidia", "device", "device-resident stages available")


def test_canonical_choices_are_cpu_intel_apple_and_nvidia() -> None:
    assert set(CANONICAL_PROFILE_REGISTRY) == {
        "cpu-host",
        "intel-vaapi-host",
        "apple-mps-host",
        "nvidia",
    }
    assert dict(PROFILE_ALIASES) == {
        "cpu": "cpu-host",
        "igpu": "intel-vaapi-host",
        "mps": "apple-mps-host",
    }
    assert set(PROFILE_REGISTRY) == {
        "cpu-host",
        "cpu",
        "intel-vaapi-host",
        "igpu",
        "apple-mps-host",
        "mps",
        "nvidia",
    }


@pytest.mark.parametrize(
    "rejected",
    (
        "cuda",
        "nvidia-host-bridge",
        "nvidia-device-experimental",
        "NVIDIA",
        "Nvidia",
        "gpu",
        "tpu",
    ),
)
def test_old_and_malformed_nvidia_names_raise_profile_error(rejected: str) -> None:
    with pytest.raises(ProfileError, match="unknown ML_WORKER_PROFILE"):
        resolve_profile({ML_WORKER_PROFILE_ENV: rejected})


def test_nvidia_is_never_selected_without_exact_opt_in() -> None:
    assert resolve_profile({}).name == "cpu-host"
    spec = resolve_profile({ML_WORKER_PROFILE_ENV: "nvidia"})
    assert spec.name == "nvidia"
    assert spec.accepted_names == ("nvidia",)
    assert spec.device == "cuda"
    assert spec.decode == "nvdec"
    assert spec.preprocess == "cuda-nv12-to-rgb24"
    assert spec.inference == "tensorrt"
    assert spec.overlay == "cuda-device"
    assert spec.encode == "h264_nvenc"
    assert spec.concrete_stages_available is False
    assert spec.encode_fallback is None


def test_other_vendor_aliases_remain_byte_identical() -> None:
    cpu = resolve_profile({ML_WORKER_PROFILE_ENV: "cpu"})
    assert cpu.name == "cpu-host"
    assert cpu.accepted_names == ("cpu-host", "cpu")
    assert cpu.device == "cpu"
    assert cpu.decode == "opencv"
    assert cpu.preprocess == "numpy-rgb24"
    assert cpu.inference == "cpu"
    assert cpu.overlay == "numpy-host"
    assert cpu.encode == "libx264"

    igpu = resolve_profile({ML_WORKER_PROFILE_ENV: "igpu"})
    assert igpu.name == "intel-vaapi-host"
    assert igpu.accepted_names == ("intel-vaapi-host", "igpu")
    assert igpu.device == "cpu"
    assert igpu.decode == "vaapi"
    assert igpu.decode_fallback == "opencv"

    mps = resolve_profile({ML_WORKER_PROFILE_ENV: "mps"})
    assert mps.name == "apple-mps-host"
    assert mps.accepted_names == ("apple-mps-host", "mps")
    assert mps.device == "mps"
    assert mps.decode == "opencv"
    assert mps.inference == "mps"


def test_plain_cuda_capability_cannot_unblock_nvidia_profile() -> None:
    plain_cuda_only = BootDependencies(default_verifiers(cuda_source=_cuda_ok))
    with pytest.raises(
        ProfileVerifyError,
        match="device-resident capability probe is not configured",
    ):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia"},
            plain_cuda_only,
            _decode_ok,
            _encode_ok,
        )


def test_nvidia_fails_closed_on_dlpack_reason() -> None:
    blocked = BootDependencies(
        default_verifiers(
            cuda_source=_cuda_ok,
            device_resident_source=lambda: VerifyResult(
                False,
                "nvidia",
                "device",
                "DLPack handoff unavailable",
            ),
        )
    )
    with pytest.raises(ProfileVerifyError, match="DLPack handoff unavailable"):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia"},
            blocked,
            _decode_ok,
            _encode_ok,
        )


def test_nvidia_reports_device_resident_runtime_path_when_probe_passes() -> None:
    deps = BootDependencies(default_verifiers(device_resident_source=_resident_ok))
    context = resolve_boot_context(
        {ML_WORKER_PROFILE_ENV: "nvidia"},
        deps,
        _decode_ok,
        _encode_ok,
    )
    report = context.runtime_profile
    assert context.requested_profile == "nvidia"
    assert context.canonical_profile == "nvidia"
    assert report.requested_profile == "nvidia"
    assert report.canonical_profile == "nvidia"
    assert report.requested_preprocess_backend == "cuda-nv12-to-rgb24"
    assert report.requested_inference_backend == "tensorrt"
    assert report.requested_overlay_backend == "cuda-device"
    assert report.requested_encode_backend == "h264_nvenc"
    assert report.memory_path == (
        "cuda-device/nv12",
        "cuda-device/nv12",
        "cuda-device/rgb24",
        "cuda-device/rgb24",
        "cuda-device/rgb24",
    )
    assert report.converter_chain == ("cuda-nv12-to-rgb24",)
    assert report.device_resident_after_decode is True
    assert report.concrete_stages_available is False
    assert report.full_frame_h2d_count == 0
    assert report.full_frame_d2h_count == 0


def _unavailable_capability(reason: str) -> DeviceResidentCapability:
    return DeviceResidentCapability(
        available=False,
        reason=reason,
        cuda=CudaCapability(True, "cuda available", device_count=1, arch_list=("sm_90",)),
        nvml=NvmlGpuStatus(True, "ok", driver_version="580.1", device_name="RTX 5070 Ti"),
        stream_event_supported=False,
        dlpack_supported=False,
    )


@pytest.mark.parametrize(
    "reason",
    (
        "torch build has no DLPack (from_dlpack/__dlpack__) support",
        "nvml device identity unavailable: NVML init failed",
        "torch.cuda.Stream/Event construction failed",
    ),
)
def test_nvidia_profile_verify_error_carries_exact_residency_reason(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "probe_device_resident_capability",
        lambda: _unavailable_capability(reason),
    )
    with pytest.raises(ProfileVerifyError, match=re.escape(reason)):
        resolve_boot_context(
            {ML_WORKER_PROFILE_ENV: "nvidia"},
            production_boot_dependencies(),
            _decode_ok,
            _encode_ok,
        )
