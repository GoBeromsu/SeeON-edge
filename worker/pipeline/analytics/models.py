from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import TypeAlias

from contracts.runner import Image, RunnerProtocol, RunnerResult
from worker.interfaces.serving import ServingClient
from worker.types import FramePacket, ModuleResult

ServingOption: TypeAlias = str | int | float | bool | None
Clock: TypeAlias = Callable[[], float]
RunnerCall: TypeAlias = Callable[[Image], RunnerResult]


class ExtractorNameConflictError(ValueError):
    module_name: str

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        super().__init__(f"extractor module name is configured more than once: {module_name!r}")


class InvalidExtractorSpecError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class ExtractorSpec:
    """Name one analytics result and the serving task that provisions it."""

    module_name: str
    task: str
    options: tuple[tuple[str, ServingOption], ...] = ()


@dataclass(frozen=True, slots=True)
class NamedExtractor:
    """Adapt one shared runner while preserving identity and output semantics."""

    module_name: str
    runner: RunnerProtocol = field(compare=False, hash=False, repr=False)
    _call: RunnerCall = field(compare=False, hash=False, repr=False)
    _clock: Clock = field(compare=False, hash=False, repr=False)
    output_adapter: str | None = None

    def extract(self, packet: FramePacket) -> ModuleResult:
        started_at = self._clock()
        result = self._call(packet.borrow_host_frame().image)
        elapsed_ms = max(0.0, (self._clock() - started_at) * 1000.0)
        return ModuleResult(
            module_name=self.module_name,
            result=result,
            elapsed_ms=elapsed_ms,
            output_adapter=self.output_adapter,
        )


def provision_extractors(
    serving_client: ServingClient,
    specs: Sequence[ExtractorSpec],
    *,
    clock: Clock = perf_counter,
) -> tuple[NamedExtractor, ...]:
    """Validate all names, then provision each shared runner exactly once."""
    frozen_specs = tuple(specs)
    ensure_unique_module_names(tuple(spec.module_name for spec in frozen_specs))
    for spec in frozen_specs:
        _validate_spec(spec)

    extractors: list[NamedExtractor] = []
    for spec in frozen_specs:
        runner = serving_client.create(spec.task, **dict(spec.options))
        extractors.append(
            NamedExtractor(
                module_name=spec.module_name,
                runner=runner,
                _call=_runner_call(runner),
                _clock=clock,
                output_adapter=spec.module_name,
            )
        )
    return tuple(extractors)


def ensure_unique_module_names(names: Sequence[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ExtractorNameConflictError(name)
        seen.add(name)


def _validate_spec(spec: ExtractorSpec) -> None:
    if not spec.module_name:
        raise InvalidExtractorSpecError("extractor module name must be non-empty")
    if not spec.task:
        raise InvalidExtractorSpecError(
            f"serving task must be non-empty for module {spec.module_name!r}"
        )
    option_names = tuple(name for name, _value in spec.options)
    if len(option_names) != len(set(option_names)):
        raise InvalidExtractorSpecError(
            f"serving options contain duplicate keys for module {spec.module_name!r}"
        )


def _runner_call(runner: RunnerProtocol) -> RunnerCall:
    if callable(runner):
        return runner
    return runner.run


__all__ = [
    "Clock",
    "ExtractorNameConflictError",
    "ExtractorSpec",
    "InvalidExtractorSpecError",
    "NamedExtractor",
    "ServingOption",
    "provision_extractors",
]
