from __future__ import annotations

from dataclasses import replace

import pytest

from worker.runtime.profile.capability_graph import (
    CapabilityMismatchError,
    validate_capability_graph,
    validate_runtime_profile_descriptor,
)
from worker.runtime.profile.descriptor import ProfileConverter
from worker.runtime.profile.registry import PROFILE_REGISTRY, runtime_descriptor_for
from worker.types import (
    ConverterCapabilities,
    FrameCapability,
    MemoryKind,
    PipelineProfile,
    PixelFormat,
    StageCapabilities,
)

_HOST_RGB = FrameCapability(MemoryKind.HOST, PixelFormat.RGB24)
_HOST_BGR = FrameCapability(MemoryKind.HOST, PixelFormat.BGR24)
_CUDA_NV12 = FrameCapability(MemoryKind.CUDA_DEVICE, PixelFormat.NV12)


def _stage(
    name: str,
    *,
    accepts: tuple[FrameCapability, ...],
    produces: FrameCapability,
) -> StageCapabilities:
    return StageCapabilities(name=name, accepts=frozenset(accepts), produces=produces)


def test_compatible_host_pipeline_validates_without_a_copy() -> None:
    profile = PipelineProfile(
        name="cpu-host",
        stages=(
            _stage("decode", accepts=(_HOST_RGB,), produces=_HOST_RGB),
            _stage("inference", accepts=(_HOST_RGB,), produces=_HOST_RGB),
            _stage("output", accepts=(_HOST_RGB,), produces=_HOST_RGB),
        ),
    )

    graph = validate_capability_graph(profile)

    assert graph.profile_name == "cpu-host"
    assert graph.converter_names == ()
    assert graph.full_frame_copy_count == 0


def test_memory_or_format_mismatch_fails_boot_without_named_converter() -> None:
    profile = PipelineProfile(
        name="broken-device-path",
        stages=(
            _stage("decode", accepts=(_CUDA_NV12,), produces=_CUDA_NV12),
            _stage("inference", accepts=(_HOST_RGB,), produces=_HOST_RGB),
        ),
    )

    with pytest.raises(CapabilityMismatchError, match="decode.*inference"):
        validate_capability_graph(profile)


def test_named_converter_closes_mismatch_and_is_counted() -> None:
    profile = PipelineProfile(
        name="explicit-host-bridge",
        stages=(
            _stage("decode", accepts=(_CUDA_NV12,), produces=_CUDA_NV12),
            _stage("inference", accepts=(_HOST_RGB,), produces=_HOST_RGB),
            _stage("output", accepts=(_HOST_BGR,), produces=_HOST_BGR),
        ),
    )
    converters = (
        ConverterCapabilities(
            name="nv12-device-to-rgb-host-materializer",
            source=_CUDA_NV12,
            target=_HOST_RGB,
            copies_frame=True,
        ),
        ConverterCapabilities(
            name="rgb-to-bgr-host-converter",
            source=_HOST_RGB,
            target=_HOST_BGR,
            copies_frame=True,
        ),
    )

    graph = validate_capability_graph(profile, converters=converters)

    assert graph.converter_names == (
        "nv12-device-to-rgb-host-materializer",
        "rgb-to-bgr-host-converter",
    )
    assert graph.full_frame_copy_count == 2


@pytest.mark.parametrize(
    "source_kind",
    [MemoryKind.CUDA_DEVICE, MemoryKind.VAAPI_SURFACE, MemoryKind.DMABUF],
)
def test_device_to_host_transition_cannot_lie_about_copy_semantics(
    source_kind: MemoryKind,
) -> None:
    source = FrameCapability(source_kind, PixelFormat.NV12)
    profile = PipelineProfile(
        name="untruthful-host-bridge",
        stages=(
            _stage("decode", accepts=(source,), produces=source),
            _stage("inference", accepts=(_HOST_RGB,), produces=_HOST_RGB),
        ),
    )

    with pytest.raises(ValueError, match="must declare copies_frame=True"):
        validate_capability_graph(
            profile,
            converters=(ConverterCapabilities("lying-bridge", source, _HOST_RGB, False),),
        )


def test_memory_transition_copy_is_derived_and_counted() -> None:
    converter = ConverterCapabilities("device-host", _CUDA_NV12, _HOST_RGB, True)
    assert converter.requires_memory_copy is True
    assert converter.effective_copies_frame is True


@pytest.mark.parametrize("edge_index", range(4))
@pytest.mark.parametrize("forged_endpoint", ["source", "target"])
def test_runtime_descriptor_rejects_forged_cpu_cuda_nv12_edge_endpoint(
    forged_endpoint: str,
    edge_index: int,
) -> None:
    descriptor = runtime_descriptor_for(
        PROFILE_REGISTRY["cpu"],
        requested_profile="cpu",
    )
    original_edge = descriptor.effective_edges[edge_index]
    if forged_endpoint == "source":
        converter = ProfileConverter(
            "forged-cuda-nv12-download",
            _CUDA_NV12,
            _HOST_RGB,
            "d2h",
        )
        edge = replace(
            original_edge,
            source=_CUDA_NV12,
            converter_name=converter.name,
        )
    else:
        converter = ProfileConverter(
            "forged-cuda-nv12-upload",
            _HOST_RGB,
            _CUDA_NV12,
            "h2d",
        )
        edge = replace(
            original_edge,
            target=_CUDA_NV12,
            converter_name=converter.name,
        )
    edges = list(descriptor.effective_edges)
    edges[edge_index] = edge
    forged = replace(
        descriptor,
        effective_converters=(converter,),
        effective_edges=tuple(edges),
    )

    with pytest.raises(
        CapabilityMismatchError,
        match=rf"{forged_endpoint} endpoint.*effective memory step",
    ):
        validate_runtime_profile_descriptor(forged)


def test_converter_route_is_deterministic_and_rejects_duplicate_names() -> None:
    profile = PipelineProfile(
        name="device-host",
        stages=(
            _stage("decode", accepts=(_CUDA_NV12,), produces=_CUDA_NV12),
            _stage("output", accepts=(_HOST_BGR,), produces=_HOST_BGR),
        ),
    )
    duplicate_name = (
        ConverterCapabilities("convert", _CUDA_NV12, _HOST_RGB, True),
        ConverterCapabilities("convert", _HOST_RGB, _HOST_BGR, True),
    )

    with pytest.raises(ValueError, match="converter name.*convert"):
        validate_capability_graph(profile, converters=duplicate_name)
