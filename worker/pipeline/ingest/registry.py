from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_MIME_PREFIXES = ("video/",)
DEFAULT_MAX_DURATION_SEC = 120.0
DEFAULT_SOURCE_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "api-sources"
LIVE_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://", "camera:", "device:")


class SourceRegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    path: Path
    duration_sec: float
    mime_type: str
    kind: Literal["stored", "live"] = "stored"
    trusted_live: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    record: SourceRecord
    path: Path
    is_live: bool


class SourceRegistry:
    def __init__(
        self,
        base_dir: Path = DEFAULT_SOURCE_BASE_DIR,
        records: dict[str, SourceRecord] | None = None,
        max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    ) -> None:
        self.base_dir = base_dir.resolve()
        self.max_duration_sec = max_duration_sec
        self._records = records or self._load_records_from_env()

    def _load_records_from_env(self) -> dict[str, SourceRecord]:
        manifest = os.environ.get("SERVING_SOURCE_MANIFEST")
        if not manifest:
            return {}
        data = json.loads(Path(manifest).read_text())
        if not isinstance(data, dict):
            raise SourceRegistryError("SERVING_SOURCE_MANIFEST must contain an object")
        records: dict[str, SourceRecord] = {}
        for key, value in data.items():
            if not isinstance(value, dict):
                raise SourceRegistryError(f"source manifest entry {key!r} must be an object")
            records[key] = SourceRecord(
                source_id=key,
                path=Path(str(value["path"])),
                duration_sec=float(value["duration_sec"]),
                mime_type=str(
                    value.get("mime_type") or mimetypes.guess_type(str(value["path"]))[0] or ""
                ),
                kind="live" if value.get("kind") == "live" else "stored",
                trusted_live=bool(value.get("trusted_live", False)),
            )
        return records

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, dict[str, Any]],
        base_dir: Path = DEFAULT_SOURCE_BASE_DIR,
        max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    ) -> SourceRegistry:
        records = {
            key: SourceRecord(
                source_id=key,
                path=Path(str(value["path"])),
                duration_sec=float(value["duration_sec"]),
                mime_type=str(
                    value.get("mime_type") or mimetypes.guess_type(str(value["path"]))[0] or ""
                ),
                kind="live" if value.get("kind") == "live" else "stored",
                trusted_live=bool(value.get("trusted_live", False)),
            )
            for key, value in mapping.items()
        }
        return cls(base_dir=base_dir, records=records, max_duration_sec=max_duration_sec)

    def resolve(self, source_id: str | None = None, upload_id: str | None = None) -> ResolvedSource:
        provided = [value for value in (source_id, upload_id) if value]
        if len(provided) != 1:
            raise SourceRegistryError("exactly one of source_id or upload_id is required")
        key = provided[0]
        self._reject_descriptor(key)
        record = self._records.get(key)
        if record is None:
            raise SourceRegistryError(f"unknown source id {key!r}")
        if record.kind == "live":
            if not record.trusted_live:
                raise SourceRegistryError("live sources require an allowlisted trusted source_id")
            return ResolvedSource(record=record, path=record.path, is_live=True)
        path = self._safe_path(record.path)
        self._validate_stored_record(record, path)
        return ResolvedSource(record=record, path=path, is_live=False)

    def _reject_descriptor(self, value: str) -> None:
        lowered = value.strip().lower()
        if lowered.isdigit():
            raise SourceRegistryError(
                "device indexes are not accepted; reference an allowlisted source_id instead"
            )
        if any(lowered.startswith(scheme) for scheme in LIVE_SCHEMES):
            raise SourceRegistryError(
                "raw live descriptors are not accepted; reference an allowlisted source_id instead"
            )
        if ".." in Path(value).parts or value.startswith(("/", "~")):
            raise SourceRegistryError(
                "raw paths and traversal are not accepted; reference a registered source_id instead"
            )

    def _safe_path(self, raw_path: Path) -> Path:
        path = raw_path if raw_path.is_absolute() else self.base_dir / raw_path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.base_dir):
            raise SourceRegistryError("resolved source escapes api base directory")
        return resolved

    def _validate_stored_record(self, record: SourceRecord, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise SourceRegistryError(f"source file does not exist: {record.source_id}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise SourceRegistryError(f"unsupported video extension {path.suffix!r}")
        guessed = mimetypes.guess_type(path.name)[0] or ""
        mime = record.mime_type or guessed
        if not mime.startswith(SUPPORTED_MIME_PREFIXES):
            raise SourceRegistryError(f"unsupported source MIME type {mime!r}")
        if record.duration_sec <= 0:
            raise SourceRegistryError("source duration metadata must be positive")
        if record.duration_sec > self.max_duration_sec:
            raise SourceRegistryError(
                "source duration "
                f"{record.duration_sec:.3f}s exceeds registry maximum "
                f"{self.max_duration_sec:.3f}s"
            )


def get_source_registry() -> SourceRegistry:
    return SourceRegistry()


__all__ = [
    "DEFAULT_MAX_DURATION_SEC",
    "DEFAULT_SOURCE_BASE_DIR",
    "LIVE_SCHEMES",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MIME_PREFIXES",
    "ResolvedSource",
    "SourceRecord",
    "SourceRegistry",
    "SourceRegistryError",
    "get_source_registry",
]
