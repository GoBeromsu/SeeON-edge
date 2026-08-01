from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, override

from worker.interfaces.decision import Decider

DomainDetector: TypeAlias = Decider


@dataclass(frozen=True, slots=True)
class AuditContext:
    model_version: str | None
    operating_threshold: float | None


@dataclass(frozen=True, slots=True)
class DomainAuditSnapshot:
    model_version: str | None
    operating_threshold: float | None


@dataclass(slots=True)
class DomainDependencyError(TypeError):
    domain: str
    dependency_type: str

    @override
    def __str__(self) -> str:
        return f"{self.domain} cannot use {self.dependency_type} dependencies"


__all__ = [
    "AuditContext",
    "DomainAuditSnapshot",
    "DomainDependencyError",
    "DomainDetector",
]
