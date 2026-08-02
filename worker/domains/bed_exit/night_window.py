"""Backward-compatible re-export shim.

``NightWindow`` moved to ``worker.domains.detection_window.DetectionWindow``
(domain-neutral, so any domain's window can share it). This module keeps the
old import path and name working for existing callers/tests.
"""

from __future__ import annotations

from worker.domains.detection_window import DetectionWindow as NightWindow

__all__ = ["NightWindow"]
