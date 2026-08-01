from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Generic, Protocol, TypeVar

import numpy as np
from typing_extensions import override

from contracts.runner import Image


@dataclass(slots=True)
class InvalidWarmupFrameError(ValueError):
    width: int
    height: int

    @override
    def __str__(self) -> str:
        return f"warmup frame dimensions must be positive, received {self.width}x{self.height}"


@dataclass(frozen=True, slots=True)
class WarmupFrameSpec:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise InvalidWarmupFrameError(width=self.width, height=self.height)


DEFAULT_WARMUP_FRAME: Final = WarmupFrameSpec(width=640, height=360)


class WarmupRunner(Protocol):
    def warmup(self) -> None: ...


WarmupRunnerT = TypeVar("WarmupRunnerT", bound=WarmupRunner)


@dataclass(frozen=True, slots=True)
class WarmupResult(Generic[WarmupRunnerT]):
    runner: WarmupRunnerT
    ready: bool
    synchronized: bool


def synthetic_rgb_frame(spec: WarmupFrameSpec) -> Image:
    return np.zeros((spec.height, spec.width, 3), dtype=np.uint8)


def warmup_runner(runner: WarmupRunnerT) -> WarmupRunnerT:
    runner.warmup()
    return runner


def _default_cuda_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def warmup_to_ready(
    runner: WarmupRunnerT,
    *,
    device: str,
    synchronize: Callable[[], None] | None = None,
) -> WarmupResult[WarmupRunnerT]:
    warmed = warmup_runner(runner)
    synchronized = False
    if device.split(":", maxsplit=1)[0] == "cuda":
        sync = _default_cuda_synchronize if synchronize is None else synchronize
        sync()
        synchronized = True
    return WarmupResult(runner=warmed, ready=True, synchronized=synchronized)


__all__ = [
    "DEFAULT_WARMUP_FRAME",
    "InvalidWarmupFrameError",
    "WarmupFrameSpec",
    "WarmupResult",
    "WarmupRunner",
    "synthetic_rgb_frame",
    "warmup_runner",
    "warmup_to_ready",
]
