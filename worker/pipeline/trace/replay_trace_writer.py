"""Bounded append-only production replay trace sink."""

from __future__ import annotations

import hashlib
from pathlib import Path

from contracts.replay_trace import ReplayRow, ReplayTraceHeader, encode_jsonl

DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_ROTATION_COUNT = 3


class ReplayTraceWriter:
    """Append rows to one camera file while retaining a bounded rotated history."""

    def __init__(
        self,
        directory: Path,
        camera_id: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        rotation_count: int = DEFAULT_ROTATION_COUNT,
    ) -> None:
        if max_bytes <= 0 or rotation_count < 0:
            raise ValueError("replay trace bounds must be positive")
        self._directory = directory.resolve()
        self._path = self._within_root(
            self._directory / f"{hashlib.sha256(camera_id.encode()).hexdigest()[:16]}.jsonl"
        )
        self._max_bytes = max_bytes
        self._rotation_count = rotation_count
        self.written_rows_total = 0
        self.dropped_rows_total = 0

    def append(self, row: ReplayRow) -> bool:
        payload = encode_jsonl(ReplayTraceHeader(), [row])
        header = encode_jsonl(ReplayTraceHeader(), [])
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._within_root(self._path)
        existing_size = path.stat().st_size if path.exists() else 0
        addition = payload[len(header) :]
        required_size = (len(header) if existing_size == 0 else existing_size) + len(addition)
        if required_size > self._max_bytes:
            self._rotate()
            if len(header) + len(addition) > self._max_bytes:
                self.dropped_rows_total += 1
                return False
            existing_size = 0
        with path.open("a", encoding="utf-8") as trace:
            if existing_size == 0:
                trace.write(header)
            trace.write(addition)
        self.written_rows_total += 1
        return True

    def _within_root(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self._directory):
            raise ValueError("replay trace path escapes configured root")
        return resolved

    def _rotate(self) -> None:
        if self._rotation_count == 0:
            self._within_root(self._path).unlink(missing_ok=True)
            return
        oldest = self._within_root(
            self._path.with_suffix(self._path.suffix + f".{self._rotation_count}")
        )
        oldest.unlink(missing_ok=True)
        for index in range(self._rotation_count - 1, 0, -1):
            source = self._within_root(self._path.with_suffix(self._path.suffix + f".{index}"))
            if source.exists():
                source.replace(
                    self._within_root(self._path.with_suffix(self._path.suffix + f".{index + 1}"))
                )
        path = self._within_root(self._path)
        if path.exists():
            path.replace(self._within_root(self._path.with_suffix(self._path.suffix + ".1")))


__all__ = ["DEFAULT_MAX_BYTES", "DEFAULT_ROTATION_COUNT", "ReplayTraceWriter"]
