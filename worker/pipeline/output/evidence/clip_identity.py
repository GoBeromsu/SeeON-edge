from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, final
from uuid import uuid4

from worker.pipeline.output.evidence.durability import fsync_directory
from worker.pipeline.output.evidence.evidence_outbox_types import ClipId


class ClipIdFactory(Protocol):
    def __call__(self, camera_id: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ClipReservation:
    clip_id: ClipId
    camera_id: str
    staging_dir: Path
    final_dir: Path


@dataclass(slots=True)
class ClipIdCollisionError(Exception):
    camera_id: str
    attempts: int

    def __str__(self) -> str:
        return f"unable to reserve a collision-free clip id after {self.attempts} attempts"


@final
class ClipIdAllocator:
    def __init__(
        self,
        store_dir: Path,
        *,
        id_factory: ClipIdFactory | None = None,
        max_attempts: int = 8,
    ) -> None:
        self._clips_dir = store_dir / "clips"
        self._staging_root = self._clips_dir / ".staging"
        self._id_factory = _new_clip_id if id_factory is None else id_factory
        self._max_attempts = max_attempts
        self.collision_count = 0

    def reserve(self, camera_id: str) -> ClipReservation:
        self._staging_root.mkdir(parents=True, exist_ok=True)
        for _ in range(self._max_attempts):
            clip_id = ClipId(self._id_factory(camera_id))
            reservation = self._reserve(camera_id, clip_id)
            if reservation is not None:
                return reservation
        raise ClipIdCollisionError(camera_id, self._max_attempts)

    def reserve_existing(self, camera_id: str, clip_id: str) -> ClipReservation:
        """Reserve the identity assigned by an external recorder exactly once."""
        if not clip_id:
            raise ValueError("clip id must not be empty")
        self._staging_root.mkdir(parents=True, exist_ok=True)
        reservation = self._reserve(camera_id, ClipId(clip_id))
        if reservation is None:
            raise ClipIdCollisionError(camera_id, 1)
        return reservation

    def _reserve(self, camera_id: str, clip_id: ClipId) -> ClipReservation | None:
        final_dir = self._clips_dir / clip_id
        staging_dir = self._staging_root / clip_id
        if final_dir.exists():
            self.collision_count += 1
            return None
        try:
            staging_dir.mkdir(exist_ok=False)
        except FileExistsError:
            self.collision_count += 1
            return None
        if final_dir.exists():
            shutil.rmtree(staging_dir)
            fsync_directory(self._staging_root)
            self.collision_count += 1
            return None
        fsync_directory(self._staging_root)
        return ClipReservation(clip_id, camera_id, staging_dir, final_dir)


def _new_clip_id(camera_id: str) -> str:
    safe_camera = (
        "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in camera_id
        )
        or "camera"
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_camera}-{timestamp}-{uuid4().hex[:12]}"


__all__ = [
    "ClipIdAllocator",
    "ClipIdCollisionError",
    "ClipIdFactory",
    "ClipReservation",
]
