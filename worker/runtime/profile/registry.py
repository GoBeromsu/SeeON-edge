from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from worker.runtime.profile.device import CudaProbeSource, probe_cuda

DevicePolicy: TypeAlias = Literal["cuda", "mps", "cpu"]
DecodePolicy: TypeAlias = Literal["nvdec", "opencv", "vaapi"]
EncodePolicy: TypeAlias = Literal["h264_nvenc", "libx264"]
MpsProbeSource: TypeAlias = Callable[[], bool]

ML_WORKER_PROFILE_ENV: Final = "ML_WORKER_PROFILE"
# Issue #133: the worker must boot with zero env vars. "cpu" is the only
# profile in PROFILE_REGISTRY whose device verifier (`_verify_cpu`) always
# succeeds with no injected capability probe -- "cuda"/"mps" fail closed
# without one (`default_verifiers()` wires no probe source by default), so
# defaulting to either would make an unconfigured boot device-dependent
# (it would pass on a real GPU/Apple-Silicon host and fail everywhere else,
# including CI). Real deployments still set ML_WORKER_PROFILE explicitly
# per target (compose.edge.yaml); this default only governs the zero-config
# local/dev/CI boot path.
DEFAULT_PROFILE_NAME: Final = "cpu"


class ProfileError(RuntimeError):
    pass


class ProfileVerifyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    profile: str
    stage: str
    reason: str


DeviceVerifier: TypeAlias = Callable[[], VerifyResult]
DecodeProbe: TypeAlias = Callable[[str], VerifyResult]
# Encode has only one non-trivial backend to preflight (h264_nvenc; libx264
# ships with virtually every ffmpeg build), so unlike DecodeProbe this takes
# no backend-name argument -- it always answers "is NVENC usable".
EncodeProbe: TypeAlias = Callable[[], VerifyResult]


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    name: str
    device: DevicePolicy
    decode: DecodePolicy
    encode: EncodePolicy


PROFILE_REGISTRY: Final[Mapping[str, ProfileSpec]] = MappingProxyType(
    {
        "cuda": ProfileSpec("cuda", "cuda", "nvdec", "h264_nvenc"),
        "mps": ProfileSpec("mps", "mps", "opencv", "libx264"),
        "cpu": ProfileSpec("cpu", "cpu", "opencv", "libx264"),
        # Intel iGPU RTSP decode via VAAPI (decode only -- inference stays on
        # CPU here; OpenVINO GPU/NPU inference is a separate follow-up). Device
        # stays "cpu" because torch has no XPU kernels on this repo's pinned
        # cu130 index -- only the decode leg moves to the iGPU.
        "igpu": ProfileSpec("igpu", "cpu", "vaapi", "libx264"),
    }
)


@dataclass(frozen=True, slots=True)
class BootDependencies:
    verifiers: Mapping[str, DeviceVerifier]


def _verify_cuda(source: CudaProbeSource | None) -> VerifyResult:
    result = probe_cuda(source)
    return VerifyResult(result.available, "cuda", "device", result.reason)


def _verify_mps(source: MpsProbeSource | None) -> VerifyResult:
    if source is None:
        return VerifyResult(False, "mps", "device", "MPS capability probe is not configured")
    available = source()
    reason = "MPS is available" if available else "MPS is unavailable"
    return VerifyResult(available, "mps", "device", reason)


def _verify_cpu() -> VerifyResult:
    return VerifyResult(True, "cpu", "device", "CPU is available")


def _verify_igpu_device() -> VerifyResult:
    """Device check for the "igpu" profile, which always succeeds like ``_verify_cpu``.

    ``verify_device_or_raise`` keys ``BootDependencies.verifiers`` by
    ``spec.name`` (``"cuda"``, ``"mps"``, ``"cpu"`` today, all of which equal
    their own ``spec.device``) -- the "igpu" profile's *name* is ``"igpu"``
    but its *device* is ``"cpu"`` (only decode moves to the iGPU in this PR;
    inference stays on CPU), so it needs its own registered verifier key
    rather than reusing ``"cpu"``'s. Its actual capability check is identical
    to ``_verify_cpu``'s (torch on CPU always succeeds) -- the real,
    hardware-touching VAAPI capability check lives in ``DecodeProbe``
    (``worker.adapters.decode.vaapi.probe.probe_vaapi_capability``, wired in
    by ``worker/runtime/worker.py``), not here.
    """
    return VerifyResult(True, "igpu", "device", "CPU is available (decode targets the iGPU)")


def default_verifiers(
    *,
    cuda_source: CudaProbeSource | None = None,
    mps_source: MpsProbeSource | None = None,
) -> Mapping[str, DeviceVerifier]:
    """Build fail-closed device verifiers from injected capability probes."""
    return MappingProxyType(
        {
            "cuda": lambda: _verify_cuda(cuda_source),
            "mps": lambda: _verify_mps(mps_source),
            "cpu": _verify_cpu,
            "igpu": _verify_igpu_device,
        }
    )


def default_decode_probe(
    decode: str,
    probes: Mapping[str, Callable[[], VerifyResult]] | None = None,
) -> VerifyResult:
    """Run an injected decode probe, failing closed when one is unavailable."""
    if probes is None or decode not in probes:
        return VerifyResult(
            False,
            "cuda" if decode == "nvdec" else "",
            "decode",
            f"{decode} capability probe is not configured",
        )
    return probes[decode]()


def default_encode_probe() -> VerifyResult:
    """Fail-closed default NVENC probe, used when boot wiring injects none.

    This package holds policy, not hardware access (worker/runtime/AGENTS.md):
    the real ffmpeg-build probe (`worker.adapters.device.cuda.probe.probe_nvenc_capability`)
    is only ever wired in by the composition root (`worker/runtime/worker.py`),
    mirroring `default_decode_probe`'s injection pattern above. Failing closed
    here is safe by construction because the caller
    (`worker.runtime.profile.boot.resolve_encode_or_fallback`) never raises on
    a failed probe -- it demotes to `libx264` with a WARNING instead of
    aborting boot, unlike `preflight_decode_or_raise`.
    """
    return VerifyResult(
        False, "cuda", "encode", "h264_nvenc capability probe is not configured"
    )


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "ML_WORKER_PROFILE_ENV",
    "PROFILE_REGISTRY",
    "BootDependencies",
    "DecodePolicy",
    "DecodeProbe",
    "DevicePolicy",
    "DeviceVerifier",
    "EncodePolicy",
    "EncodeProbe",
    "ProfileError",
    "ProfileSpec",
    "ProfileVerifyError",
    "VerifyResult",
    "default_decode_probe",
    "default_encode_probe",
    "default_verifiers",
]
