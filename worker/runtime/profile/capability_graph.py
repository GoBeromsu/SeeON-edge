from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from worker.runtime.profile.descriptor import RuntimeProfileDescriptor
from worker.types.capabilities import (
    HOST_PIPELINE_PROFILE,
    ConverterCapabilities,
    FrameCapability,
    PipelineProfile,
)


class CapabilityMismatchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedCapabilityEdge:
    source_stage: str
    target_stage: str
    converter_names: tuple[str, ...]
    validated: bool = True


@dataclass(frozen=True, slots=True)
class ValidatedCapabilityGraph:
    profile_name: str
    converter_names: tuple[str, ...]
    full_frame_copy_count: int
    edges: tuple[ValidatedCapabilityEdge, ...] = ()


def validate_capability_graph(
    profile: PipelineProfile,
    *,
    converters: tuple[ConverterCapabilities, ...] = (),
) -> ValidatedCapabilityGraph:
    names = tuple(converter.name for converter in converters)
    if len(names) != len(set(names)):
        duplicate = next(name for name in names if names.count(name) > 1)
        raise ValueError(f"converter name {duplicate!r} is configured more than once")

    selected: list[ConverterCapabilities] = []
    validated_edges: list[ValidatedCapabilityEdge] = []
    for producer, consumer in zip(profile.stages, profile.stages[1:], strict=False):
        route = _find_route(producer.produces, consumer.accepts, converters)
        if route is None:
            source = producer.produces
            accepted = ", ".join(
                f"{item.memory_kind.value}/{item.pixel_format.value}"
                for item in sorted(
                    consumer.accepts,
                    key=lambda item: (item.memory_kind.value, item.pixel_format.value),
                )
            )
            message = " ".join(
                (
                    f"capability graph mismatch from {producer.name!r} to {consumer.name!r}:",
                    f"produces {source.memory_kind.value}/{source.pixel_format.value},",
                    f"accepts [{accepted}], and no named converter closes the edge",
                )
            )
            raise CapabilityMismatchError(message)
        selected.extend(route)
        validated_edges.append(
            ValidatedCapabilityEdge(
                producer.name,
                consumer.name,
                tuple(converter.name for converter in route),
            )
        )

    return ValidatedCapabilityGraph(
        profile_name=profile.name,
        converter_names=tuple(converter.name for converter in selected),
        full_frame_copy_count=sum(converter.effective_copies_frame for converter in selected),
        edges=tuple(validated_edges),
    )


def validate_runtime_profile_descriptor(
    descriptor: RuntimeProfileDescriptor,
) -> ValidatedCapabilityGraph:
    converters = {converter.name: converter for converter in descriptor.effective_converters}
    stage_capabilities = {step.stage: step.capability for step in descriptor.effective_memory_steps}
    selected_names: list[str] = []
    validated_edges: list[ValidatedCapabilityEdge] = []
    for edge in descriptor.effective_edges:
        expected_source = stage_capabilities[edge.source_stage]
        if edge.source != expected_source:
            raise CapabilityMismatchError(
                " ".join(
                    (
                        f"runtime profile edge {edge.source_stage!r}->{edge.target_stage!r}",
                        "source endpoint does not match its effective memory step:",
                        f"declared {edge.source.memory_kind.value}/"
                        f"{edge.source.pixel_format.value},",
                        "expected",
                        f"{expected_source.memory_kind.value}/{expected_source.pixel_format.value}",
                    )
                )
            )
        expected_target = stage_capabilities[edge.target_stage]
        if edge.target != expected_target:
            raise CapabilityMismatchError(
                " ".join(
                    (
                        f"runtime profile edge {edge.source_stage!r}->{edge.target_stage!r}",
                        "target endpoint does not match its effective memory step:",
                        f"declared {edge.target.memory_kind.value}/"
                        f"{edge.target.pixel_format.value},",
                        "expected",
                        f"{expected_target.memory_kind.value}/{expected_target.pixel_format.value}",
                    )
                )
            )
        if edge.source == edge.target and edge.converter_name is None:
            names: tuple[str, ...] = ()
        else:
            converter = converters.get(edge.converter_name or "")
            if converter is None:
                message = " ".join(
                    (
                        f"runtime profile edge {edge.source_stage!r}->{edge.target_stage!r}",
                        "requires a named converter",
                    )
                )
                raise CapabilityMismatchError(message)
            if converter.source != edge.source or converter.target != edge.target:
                message = " ".join(
                    (
                        f"runtime profile converter {converter.name!r} does not close edge",
                        f"{edge.source_stage!r}->{edge.target_stage!r}",
                    )
                )
                raise CapabilityMismatchError(message)
            names = (converter.name,)
            selected_names.append(converter.name)
        validated_edges.append(ValidatedCapabilityEdge(edge.source_stage, edge.target_stage, names))

    declared_names = tuple(converter.name for converter in descriptor.effective_converters)
    if tuple(selected_names) != declared_names:
        raise CapabilityMismatchError(
            "runtime profile converter chain does not match its validated edges"
        )
    return ValidatedCapabilityGraph(
        profile_name=descriptor.canonical_profile,
        converter_names=declared_names,
        full_frame_copy_count=(descriptor.full_frame_h2d_count + descriptor.full_frame_d2h_count),
        edges=tuple(validated_edges),
    )


def _find_route(
    source: FrameCapability,
    targets: frozenset[FrameCapability],
    converters: tuple[ConverterCapabilities, ...],
) -> tuple[ConverterCapabilities, ...] | None:
    if source in targets:
        return ()
    pending: deque[tuple[FrameCapability, tuple[ConverterCapabilities, ...]]] = deque(
        [(source, ())]
    )
    visited = {source}
    while pending:
        capability, route = pending.popleft()
        for converter in converters:
            if converter.source != capability or converter.target in visited:
                continue
            candidate = (*route, converter)
            if converter.target in targets:
                return candidate
            visited.add(converter.target)
            pending.append((converter.target, candidate))
    return None


HOST_VALIDATED_CAPABILITY_GRAPH = validate_capability_graph(HOST_PIPELINE_PROFILE)


__all__ = [
    "CapabilityMismatchError",
    "HOST_VALIDATED_CAPABILITY_GRAPH",
    "ValidatedCapabilityEdge",
    "ValidatedCapabilityGraph",
    "validate_capability_graph",
    "validate_runtime_profile_descriptor",
]
