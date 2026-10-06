"""Audit process session identity shared by the PostgreSQL session owner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuditSession:
    session_id: str


__all__ = ["AuditSession"]
