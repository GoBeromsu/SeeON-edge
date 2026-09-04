from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from worker.runtime.profile.descriptor import (
    MemoryPathStep,
    ProfileConverter,
    ProfileStage,
    RuntimeProfileDescriptor,
    RuntimeProfileEdge,
)
from worker.runtime.profile.device import CudaProbeSource
from worker.types import FrameCapability, MemoryKind, PipelineProfile, PixelFormat

DevicePolicy: TypeAlias = Literal["cuda", "mps", "cpu"]
DecodePolicy: TypeAlias = Literal["nvdec", "opencv", "vaapi"]
EncodePolicy: TypeAlias = Literal["h264_nvenc", "libx264"]
MpsProbeSource: TypeAlias = Callable[[], bool]
# `nvidia`'s own concrete-stage capability check -- distinct from
# `CudaProbeSource` (plain `torch.cuda` usability): this source answers
# whether NVDEC, NVML identity, CUDA stream/event, and DLPack are present
# and must never be satisfied by a host that only passes the plain CUDA check.
DeviceResidentProbeSource: TypeAlias = Callable[[], "VerifyResult"]

ML_WORKER_PROFILE_ENV: Final = "ML_WORKER_PROFILE"
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
EncodeProbe: TypeAlias = Callable[[], VerifyResult]


_HOST_RGB = FrameCapability(MemoryKind.HOST, PixelFormat.RGB24)
_CUDA_RGB = FrameCapability(MemoryKind.CUDA_DEVICE, PixelFormat.RGB24)
_CUDA_NV12 = FrameCapability(MemoryKind.CUDA_DEVICE, PixelFormat.NV12)
_MPS_RGB = FrameCapability(MemoryKind.MPS_DEVICE, PixelFormat.RGB24)


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """One canonical infrastructure profile plus its accepted legacy names."""

    name: str
    accepted_names: tuple[str, ...]
    device: DevicePolicy
    decode: DecodePolicy
    preprocess: str
    inference: str
    overlay: str
    encode: EncodePolicy
    pipeline: PipelineProfile | None = None
    decode_fallback: DecodePolicy | None = None
    encode_fallback: EncodePolicy | None = None
    concrete_stages_available: bool = True

    def __post_init__(self) -> None:
        if not self.accepted_names or self.accepted_names[0] != self.name:
            raise ValueError("canonical profile name must be the first accepted name")
        if len(self.accepted_names) != len(set(self.accepted_names)):
            raise ValueError(f"profile {self.name!r} contains duplicate accepted names")

    def with_pipeline(self, pipeline: PipelineProfile) -> ProfileSpec:
        return ProfileSpec(
            self.name,
            self.accepted_names,
            self.device,
            self.decode,
            self.preprocess,
            self.inference,
            self.overlay,
            self.encode,
            pipeline,
            self.decode_fallback,
            self.encode_fallback,
            self.concrete_stages_available,
        )


_CPU_HOST = ProfileSpec(
    "cpu-host",
    ("cpu-host", "cpu"),
    "cpu",
    "opencv",
    "numpy-rgb24",
    "cpu",
    "numpy-host",
    "libx264",
    None,
)
_NVIDIA = ProfileSpec(
    "nvidia",
    ("nvidia",),
    "cuda",
    "nvdec",
    "cuda-nv12-to-rgb24",
    "tensorrt",
    "cuda-device",
    "h264_nvenc",
    None,
    concrete_stages_available=True,
)
_FLOW = ProfileSpec(
    "flow",
    ("flow",),
    "cuda",
    "nvdec",
    "deepstream-flow",
    "onnxruntime",
    "deepstream-flow",
    "h264_nvenc",
    None,
    concrete_stages_available=True,
)
_INTEL_VAAPI_HOST = ProfileSpec(
    "intel-vaapi-host",
    ("intel-vaapi-host", "igpu"),
    "cpu",
    "vaapi",
    "numpy-rgb24",
    "cpu",
    "numpy-host",
    "libx264",
    None,
    decode_fallback="opencv",
)
_APPLE_MPS_HOST = ProfileSpec(
    "apple-mps-host",
    ("apple-mps-host", "mps"),
    "mps",
    "opencv",
    "mps-tensor-upload",
    "mps",
    "numpy-host",
    "libx264",
    None,
)
CANONICAL_PROFILE_REGISTRY: Final[Mapping[str, ProfileSpec]] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            _CPU_HOST,
            _INTEL_VAAPI_HOST,
            _APPLE_MPS_HOST,
            _NVIDIA,
            _FLOW,
        )
    }
)
PROFILE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        accepted: spec.name
        for spec in CANONICAL_PROFILE_REGISTRY.values()
        for accepted in spec.accepted_names
        if accepted != spec.name
    }
)
PROFILE_REGISTRY: Final[Mapping[str, ProfileSpec]] = MappingProxyType(
    {
        accepted: spec
        for spec in CANONICAL_PROFILE_REGISTRY.values()
        for accepted in spec.accepted_names
    }
)


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    requested_name: str
    spec: ProfileSpec

    @property
    def canonical_name(self) -> str:
        return self.spec.name


@dataclass(frozen=True, slots=True)
class BootDependencies:
    verifiers: Mapping[str, DeviceVerifier]


def requested_profile_name(env: Mapping[str, str]) -> str:
    raw = env.get(ML_WORKER_PROFILE_ENV)
    return DEFAULT_PROFILE_NAME if raw is None or not raw.strip() else raw.strip()


def select_profile(
    env: Mapping[str, str],
    registry: Mapping[str, ProfileSpec] = PROFILE_REGISTRY,
) -> ProfileSelection:
    requested = requested_profile_name(env)
    try:
        return ProfileSelection(requested, registry[requested])
    except KeyError as error:
        choices = "|".join(sorted(registry))
        raise ProfileError(f"unknown ML_WORKER_PROFILE {requested!r}; set {choices}") from error


def _edge(
    source_stage: ProfileStage,
    target_stage: ProfileStage,
    source: FrameCapability,
    target: FrameCapability,
    converter_name: str | None = None,
) -> RuntimeProfileEdge:
    return RuntimeProfileEdge(source_stage, target_stage, source, target, converter_name)


def _memory_path_for(
    spec: ProfileSpec,
    decode: DecodePolicy,
    encode: EncodePolicy,
) -> tuple[
    tuple[MemoryPathStep, ...],
    tuple[ProfileConverter, ...],
    tuple[RuntimeProfileEdge, ...],
]:
    del decode, encode  # Device vs host path is selected by spec.name, not these.
    if spec.name == "apple-mps-host":
        return (
            (
                MemoryPathStep("decode", MemoryKind.HOST, PixelFormat.RGB24),
                MemoryPathStep("preprocess", MemoryKind.HOST, PixelFormat.RGB24),
                MemoryPathStep("inference", MemoryKind.MPS_DEVICE, PixelFormat.RGB24),
                MemoryPathStep("overlay", MemoryKind.HOST, PixelFormat.RGB24),
                MemoryPathStep("encode", MemoryKind.HOST, PixelFormat.RGB24),
            ),
            (ProfileConverter("mps-inference-host-input-upload", _HOST_RGB, _MPS_RGB, "h2d"),),
            (
                _edge("decode", "preprocess", _HOST_RGB, _HOST_RGB),
                _edge(
                    "preprocess",
                    "inference",
                    _HOST_RGB,
                    _MPS_RGB,
                    "mps-inference-host-input-upload",
                ),
                _edge("decode", "overlay", _HOST_RGB, _HOST_RGB),
                _edge("overlay", "encode", _HOST_RGB, _HOST_RGB),
            ),
        )
    if spec.name in {"nvidia", "flow"}:
        device_stages: tuple[tuple[ProfileStage, PixelFormat], ...] = (
            ("decode", PixelFormat.NV12),
            ("preprocess", PixelFormat.NV12),
            ("inference", PixelFormat.RGB24),
            ("overlay", PixelFormat.RGB24),
            ("encode", PixelFormat.RGB24),
        )
        return (
            tuple(
                MemoryPathStep(stage, MemoryKind.CUDA_DEVICE, format_)
                for stage, format_ in device_stages
            ),
            (ProfileConverter("cuda-nv12-to-rgb24", _CUDA_NV12, _CUDA_RGB, "none"),),
            (
                _edge("decode", "preprocess", _CUDA_NV12, _CUDA_NV12),
                _edge(
                    "preprocess",
                    "inference",
                    _CUDA_NV12,
                    _CUDA_RGB,
                    "cuda-nv12-to-rgb24",
                ),
                _edge("inference", "overlay", _CUDA_RGB, _CUDA_RGB),
                _edge("overlay", "encode", _CUDA_RGB, _CUDA_RGB),
            ),
        )
    host_stages: tuple[ProfileStage, ...] = (
        "decode",
        "preprocess",
        "inference",
        "overlay",
        "encode",
    )
    return (
        tuple(MemoryPathStep(stage, MemoryKind.HOST, PixelFormat.RGB24) for stage in host_stages),
        (),
        (
            _edge("decode", "preprocess", _HOST_RGB, _HOST_RGB),
            _edge("preprocess", "inference", _HOST_RGB, _HOST_RGB),
            _edge("decode", "overlay", _HOST_RGB, _HOST_RGB),
            _edge("overlay", "encode", _HOST_RGB, _HOST_RGB),
        ),
    )


def runtime_descriptor_for(
    spec: ProfileSpec,
    *,
    requested_profile: str,
    effective_decode: DecodePolicy | None = None,
    effective_encode: EncodePolicy | None = None,
    degraded_reasons: tuple[str, ...] = (),
) -> RuntimeProfileDescriptor:
    decode = effective_decode or spec.decode
    encode = effective_encode or spec.encode
    requested_steps, requested_converters, _requested_edges = _memory_path_for(
        spec, spec.decode, spec.encode
    )
    effective_steps, effective_converters, effective_edges = _memory_path_for(spec, decode, encode)

    return RuntimeProfileDescriptor(
        requested_profile=requested_profile,
        canonical_profile=spec.name,
        requested_decode_backend=spec.decode,
        effective_decode_backend=decode,
        requested_preprocess_backend=spec.preprocess,
        effective_preprocess_backend=spec.preprocess,
        requested_inference_backend=spec.inference,
        effective_inference_backend=spec.inference,
        requested_overlay_backend=spec.overlay,
        effective_overlay_backend=spec.overlay,
        requested_encode_backend=spec.encode,
        effective_encode_backend=encode,
        requested_memory_steps=requested_steps,
        effective_memory_steps=effective_steps,
        requested_converters=requested_converters,
        effective_converters=effective_converters,
        effective_edges=effective_edges,
        degraded_reasons=degraded_reasons,
        device_resident_after_decode=(spec.name in {"nvidia", "flow"} and decode == "nvdec"),
        concrete_stages_available=spec.concrete_stages_available,
    )


def _verify_mps(source: MpsProbeSource | None) -> VerifyResult:
    if source is None:
        return VerifyResult(False, "mps", "device", "MPS capability probe is not configured")
    available = source()
    return VerifyResult(
        available,
        "mps",
        "device",
        "MPS is available" if available else "MPS is unavailable",
    )


def _verify_cpu() -> VerifyResult:
    return VerifyResult(True, "cpu", "device", "CPU is available")


def _verify_igpu_device() -> VerifyResult:
    return VerifyResult(True, "igpu", "device", "CPU is available (decode targets the iGPU)")


def _verify_device_resident(source: DeviceResidentProbeSource | None) -> VerifyResult:
    if source is None:
        return VerifyResult(
            False,
            "nvidia",
            "device",
            "device-resident capability probe is not configured",
        )
    return source()


def default_verifiers(
    *,
    cuda_source: CudaProbeSource | None = None,
    mps_source: MpsProbeSource | None = None,
    device_resident_source: DeviceResidentProbeSource | None = None,
) -> Mapping[str, DeviceVerifier]:
    del cuda_source  # Plain CUDA no longer gates a public profile.

    def mps() -> VerifyResult:
        return _verify_mps(mps_source)

    def device_resident() -> VerifyResult:
        return _verify_device_resident(device_resident_source)

    def flow() -> VerifyResult:
        result = device_resident()
        return VerifyResult(result.ok, "flow", result.stage, result.reason)

    return MappingProxyType(
        {
            "cpu": _verify_cpu,
            "cpu-host": _verify_cpu,
            "nvidia": device_resident,
            "flow": flow,
            "mps": mps,
            "apple-mps-host": mps,
            "igpu": _verify_igpu_device,
            "intel-vaapi-host": _verify_igpu_device,
        }
    )


def default_decode_probe(
    decode: str,
    probes: Mapping[str, Callable[[], VerifyResult]] | None = None,
) -> VerifyResult:
    if probes is None or decode not in probes:
        return VerifyResult(
            False,
            "cuda" if decode == "nvdec" else "",
            "decode",
            f"{decode} capability probe is not configured",
        )
    return probes[decode]()


def default_encode_probe() -> VerifyResult:
    return VerifyResult(False, "cuda", "encode", "h264_nvenc capability probe is not configured")


__all__ = [
    "CANONICAL_PROFILE_REGISTRY",
    "DEFAULT_PROFILE_NAME",
    "ML_WORKER_PROFILE_ENV",
    "PROFILE_ALIASES",
    "PROFILE_REGISTRY",
    "BootDependencies",
    "DecodePolicy",
    "DecodeProbe",
    "DevicePolicy",
    "DeviceResidentProbeSource",
    "DeviceVerifier",
    "EncodePolicy",
    "EncodeProbe",
    "ProfileError",
    "ProfileSelection",
    "ProfileSpec",
    "ProfileVerifyError",
    "VerifyResult",
    "default_decode_probe",
    "default_encode_probe",
    "default_verifiers",
    "requested_profile_name",
    "runtime_descriptor_for",
    "select_profile",
]
