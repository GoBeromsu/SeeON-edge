"""Domain-neutral detection-window primitive.

Originally lived only under ``worker.domains.bed_exit`` as ``NightWindow``.
Generalizing per-domain detection windows (issue #24) needs this class
available to any domain and to the runtime composition root without
crossing the ``worker.domains`` -> ``worker.runtime`` import-linter boundary
(``worker.domains`` must never import ``worker.runtime``), so the canonical
implementation lives here, domain-neutral, and
``worker.domains.bed_exit.night_window`` re-exports it as ``NightWindow``
for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo


class DetectionWindowError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DetectionWindow:
    start: str
    end: str
    tz: str

    def contains(self, now: datetime) -> bool:
        if now.tzinfo is None or now.utcoffset() is None:
            raise DetectionWindowError("now must be timezone-aware")
        local_time = now.astimezone(ZoneInfo(self.tz)).time()
        start = _parse_hhmm(self.start)
        end = _parse_hhmm(self.end)
        if start <= end:
            return start <= local_time < end
        return local_time >= start or local_time < end


def _parse_hhmm(value: str) -> time:
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise DetectionWindowError("night window time must use HH:MM") from exc
    return parsed.time()


__all__ = ["DetectionWindow", "DetectionWindowError"]
