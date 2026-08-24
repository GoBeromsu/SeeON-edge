"""Value types and source-boundary validation for native child control."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import override
from urllib.parse import urlsplit

_MAX_SOURCE_URI_BYTES = 4_096


@dataclass(frozen=True, slots=True)
class ControlIdentity:
    worker_boot_id: uuid.UUID
    child_instance_id: uuid.UUID
    transform_id: str


@dataclass(frozen=True, slots=True)
class NativeStatus:
    metadata_published: int
    metadata_overwritten: int
    wake_dropped: int
    source_failures: int
    malformed_frames: int
    source_count: int
    custom_transform_available: bool


@dataclass(frozen=True, slots=True)
class ChildControlError(Exception):
    code: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def parse_source_uri(raw: str) -> str:
    if raw == "" or len(raw.encode()) > _MAX_SOURCE_URI_BYTES:
        raise ChildControlError("source_uri_invalid", "bounds")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise ChildControlError("source_uri_invalid", "control_character")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"rtsp", "loopback"}:
        raise ChildControlError("source_uri_scheme", "unsupported")
    if parsed.scheme == "rtsp" and parsed.hostname is None:
        raise ChildControlError("source_uri_invalid", "host_required")
    return raw


__all__ = ["ChildControlError", "ControlIdentity", "NativeStatus", "parse_source_uri"]
