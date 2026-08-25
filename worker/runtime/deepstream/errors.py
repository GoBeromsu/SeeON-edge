"""Typed dark-child startup and fatal containment errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import override


@dataclass(frozen=True, slots=True)
class ChildStartupError(Exception):
    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ChildFatalError(Exception):
    exit_code: int
    category: str
    first_fault_path: Path

    @override
    def __str__(self) -> str:
        return f"DeepStream child exited {self.exit_code}: {self.category}"


__all__ = ["ChildFatalError", "ChildStartupError"]
