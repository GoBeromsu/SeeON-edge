"""Stable environment names and retention policy for derivative evidence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

CLIP_STORE_DIR_ENV: Final = "CLIP_STORE_DIR"
CLIP_STORE_RETENTION_DAYS_ENV: Final = "CLIP_STORE_RETENTION_DAYS"
CLIP_RETENTION_DAYS_ENV: Final = "CLIP_RETENTION_DAYS"
CLIP_STORE_MAX_USAGE_ENV: Final = "CLIP_STORE_MAX_USAGE"
CLIP_DISK_HIGH_WATERMARK_ENV: Final = "CLIP_DISK_HIGH_WATERMARK"
CLIP_FFMPEG_BIN_ENV: Final = "CLIP_FFMPEG_BIN"
DEFAULT_CLIP_STORE_DIR: Final = "/var/lib/clip-store"
DEFAULT_FFMPEG_BIN: Final = "ffmpeg"
MIN_RETENTION_DAYS: Final = 60
DEFAULT_RETENTION_DAYS: Final = MIN_RETENTION_DAYS
DEFAULT_DISK_HIGH_WATERMARK: Final = 0.80


def configured_store_dir(store_dir: Path | None = None) -> Path:
    """Resolve an injected portable root or the baked production root."""
    return Path(DEFAULT_CLIP_STORE_DIR) if store_dir is None else store_dir


def configured_ffmpeg_bin() -> str:
    value = os.environ.get(CLIP_FFMPEG_BIN_ENV, "").strip()
    return DEFAULT_FFMPEG_BIN if value == "" else value


def configured_retention_days() -> int:
    raw = os.environ.get(
        CLIP_STORE_RETENTION_DAYS_ENV,
        os.environ.get(CLIP_RETENTION_DAYS_ENV, ""),
    )
    if raw.strip() == "":
        return DEFAULT_RETENTION_DAYS
    return max(MIN_RETENTION_DAYS, int(raw))


def configured_disk_high_watermark() -> float:
    raw = os.environ.get(
        CLIP_STORE_MAX_USAGE_ENV,
        os.environ.get(CLIP_DISK_HIGH_WATERMARK_ENV, ""),
    )
    if raw.strip() == "":
        return DEFAULT_DISK_HIGH_WATERMARK
    value = float(raw)
    if value > 1.0:
        value /= 100.0
    return min(max(value, 0.01), 0.99)


__all__ = [
    "CLIP_DISK_HIGH_WATERMARK_ENV",
    "CLIP_FFMPEG_BIN_ENV",
    "CLIP_RETENTION_DAYS_ENV",
    "CLIP_STORE_DIR_ENV",
    "CLIP_STORE_MAX_USAGE_ENV",
    "CLIP_STORE_RETENTION_DAYS_ENV",
    "DEFAULT_CLIP_STORE_DIR",
    "DEFAULT_DISK_HIGH_WATERMARK",
    "DEFAULT_FFMPEG_BIN",
    "DEFAULT_RETENTION_DAYS",
    "MIN_RETENTION_DAYS",
    "configured_disk_high_watermark",
    "configured_ffmpeg_bin",
    "configured_retention_days",
    "configured_store_dir",
]
