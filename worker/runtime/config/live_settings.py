"""Thread-safe worker policy updated by normal config polling."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass(slots=True)
class LiveClipExportPolicy:
    _enabled: bool = False
    _version: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def snapshot(self) -> tuple[bool, int]:
        with self._lock:
            return self._enabled, self._version

    def apply(self, *, enabled: bool, version: int) -> bool:
        """Apply a non-stale snapshot and report whether effective state changed."""
        with self._lock:
            if version < self._version:
                return False
            changed = enabled != self._enabled or version != self._version
            self._enabled = enabled
            self._version = version
            return changed


__all__ = ["LiveClipExportPolicy"]
