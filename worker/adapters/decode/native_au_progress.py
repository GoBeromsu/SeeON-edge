"""Exact accepted-AU progress signals for bounded QA and shutdown coordination."""

from __future__ import annotations

import threading
from typing import final


@final
class NativeAuProgress:
    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._counts: dict[str, int] = {}

    def count(self, camera_id: str) -> int:
        with self._changed:
            return self._counts.get(camera_id, 0)

    def wait(self, camera_id: str, target: int, timeout: float) -> bool:
        with self._changed:
            return self._changed.wait_for(
                lambda: self._counts.get(camera_id, 0) >= target,
                timeout=timeout,
            )

    def accept(self, camera_id: str) -> None:
        with self._changed:
            self._counts[camera_id] = self._counts.get(camera_id, 0) + 1
            self._changed.notify_all()


__all__ = ["NativeAuProgress"]
