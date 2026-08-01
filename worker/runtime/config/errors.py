from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import override


@dataclass(slots=True)
class ConfigValidationError(ValueError):
    message: str

    @override
    def __str__(self) -> str:
        return self.message


@dataclass(slots=True)
class WorkerConfigError(Exception):
    message: str

    @override
    def __str__(self) -> str:
        return self.message


__all__ = ["ConfigValidationError", "WorkerConfigError"]
