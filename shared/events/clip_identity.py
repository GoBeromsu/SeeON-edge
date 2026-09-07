"""Shared clip identifier grammar for backend-worker boundaries."""

from __future__ import annotations

import re

CLIP_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def is_clip_id(value: str) -> bool:
    return CLIP_ID_PATTERN.fullmatch(value) is not None


__all__ = ["CLIP_ID_PATTERN", "is_clip_id"]
