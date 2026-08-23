"""Retired evidence outbox schema marker.

Durable evidence is represented solely by delivery-queue entries.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final = 0
MIGRATIONS: Final[tuple[tuple[str, ...], ...]] = ()
