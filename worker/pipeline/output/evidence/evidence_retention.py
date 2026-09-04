"""Hold-aware, verified local evidence retention."""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal, Protocol, assert_never, final

from pydantic import BaseModel, ConfigDict, ValidationError


class DiskUsage(Protocol):
    @property
    def total(self) -> int: ...

    @property
    def used(self) -> int: ...

    @property
    def free(self) -> int: ...


class PurgeResult(StrEnum):
    PURGED = "PURGED"
    HELD = "HELD"
    MISSING = "MISSING"
    UNVERIFIABLE = "UNVERIFIABLE"
    DELETE_FAILED = "DELETE_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"


@dataclass(frozen=True, slots=True)
class PurgeCandidate:
    clip_id: str
    clip_dir: Path
    finalized_at: datetime


@dataclass(frozen=True, slots=True)
class RotationReport:
    purged_clip_ids: tuple[str, ...]
    held_clip_ids: tuple[str, ...]
    failure_clip_ids: tuple[str, ...]
    pressure_blocked: bool


class _RetentionManifest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True, strict=True)

    clip_id: str
    finalized: bool = False
    manifest_schema_version: int | None = None
    finalized_at: str | None = None
    path: str | None = None
    media_relpath: str | None = None
    video_available: bool | None = None
    local_state: Literal["AWAITING_FINALIZE", "VERIFIED", "UNAVAILABLE", "CORRUPT"] | None = None
    state: Literal["READY", "UNAVAILABLE"] | None = None


@final
class EvidenceRetention:
    def __init__(
        self,
        store_dir: Path,
        *,
        is_held: Callable[[str], bool],
        disk_usage_provider: Callable[[Path], DiskUsage],
        begin_purge: Callable[[str], bool] | None = None,
        complete_purge: Callable[[str], None] | None = None,
        fail_purge: Callable[[str, str], None] | None = None,
    ) -> None:
        self._store_dir = store_dir
        self._clips_dir = store_dir / "clips"
        self._is_held = is_held
        self._disk_usage_provider = disk_usage_provider
        self._begin_purge = begin_purge
        self._complete_purge = complete_purge
        self._fail_purge = fail_purge

    def is_held(self, clip_id: str) -> bool:
        return self._is_held(clip_id)

    def preflight(self, candidate: PurgeCandidate) -> PurgeResult | None:
        """Verify hold, ownership, containment, and immutable media without deleting."""
        if self._is_held(candidate.clip_id):
            return PurgeResult.HELD
        return self._verify_candidate(candidate)

    def purge(self, candidate: PurgeCandidate) -> PurgeResult:
        verification = self.preflight(candidate)
        if verification is not None:
            return verification
        if self._begin_purge is not None:
            try:
                if not self._begin_purge(candidate.clip_id):
                    return PurgeResult.HELD
            except Exception:  # noqa: BLE001 - retention must fail closed at the DB boundary
                return PurgeResult.VERIFICATION_FAILED
        try:
            # Open and verify again after the intent hook, then remove only by
            # the verified clips-root descriptor. A preflight is intentionally
            # non-authoritative: a root can be replaced between commands.
            with self._opened_verified_candidate(candidate) as clips_directory:
                shutil.rmtree(candidate.clip_id, dir_fd=clips_directory)
                try:
                    os.stat(
                        candidate.clip_id,
                        dir_fd=clips_directory,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    self._record_failure(candidate.clip_id, "DELETE_NOT_DURABLE")
                    return PurgeResult.VERIFICATION_FAILED
        except FileNotFoundError:
            self._record_failure(candidate.clip_id, "MISSING_DURING_DELETE")
            return PurgeResult.MISSING
        except (OSError, ValueError):
            self._record_failure(candidate.clip_id, "DELETE_FAILED")
            return PurgeResult.DELETE_FAILED
        if self._complete_purge is not None:
            try:
                self._complete_purge(candidate.clip_id)
            except Exception:  # noqa: BLE001 - pending tombstone completes on restart
                return PurgeResult.VERIFICATION_FAILED
        return PurgeResult.PURGED

    def _record_failure(self, clip_id: str, reason: str) -> None:
        if self._fail_purge is None:
            return
        try:
            self._fail_purge(clip_id, reason)
        except Exception:  # noqa: BLE001 - preserve the original purge failure result
            return

    def rotate(
        self,
        candidates: Iterable[PurgeCandidate],
        *,
        retention_cutoff: datetime,
        disk_high_watermark: float,
    ) -> RotationReport:
        ordered = sorted(candidates, key=lambda candidate: candidate.finalized_at)
        held: list[str] = []
        failures: list[str] = []
        purged: list[str] = []

        def record(candidate: PurgeCandidate, result: PurgeResult) -> None:
            match result:
                case PurgeResult.PURGED:
                    purged.append(candidate.clip_id)
                    return
                case PurgeResult.HELD:
                    held.append(candidate.clip_id)
                    return
                case (
                    PurgeResult.MISSING
                    | PurgeResult.UNVERIFIABLE
                    | PurgeResult.DELETE_FAILED
                    | PurgeResult.VERIFICATION_FAILED
                ):
                    failures.append(candidate.clip_id)
                    return
            assert_never(result)

        for candidate in ordered:
            if self._is_held(candidate.clip_id):
                held.append(candidate.clip_id)
                continue
            if candidate.finalized_at > retention_cutoff:
                continue
            record(candidate, self.purge(candidate))

        return RotationReport(
            purged_clip_ids=tuple(purged),
            held_clip_ids=tuple(held),
            failure_clip_ids=tuple(failures),
            pressure_blocked=self._over_watermark(disk_high_watermark),
        )

    def _over_watermark(self, high_watermark: float) -> bool:
        usage = self._disk_usage_provider(self._store_dir)
        return usage.total > 0 and usage.used / usage.total > high_watermark

    def _verify_candidate(self, candidate: PurgeCandidate) -> PurgeResult | None:
        try:
            with self._opened_verified_candidate(candidate):
                return None
        except _CandidateMissing:
            return PurgeResult.MISSING
        except (FileNotFoundError, OSError, ValidationError, ValueError):
            return PurgeResult.UNVERIFIABLE

    @contextmanager
    def _opened_verified_candidate(self, candidate: PurgeCandidate) -> Generator[int, None, None]:
        """Yield a no-symlink clips-root descriptor after complete verification.

        Every path component is opened relative to an already verified parent
        with ``O_NOFOLLOW``. The destructive caller retains the clips descriptor
        so it never resolves the attacker-controlled pathname again.
        """
        expected = self._clips_dir / candidate.clip_id
        if candidate.clip_dir != expected or candidate.clip_id in {"", ".", ".."}:
            raise ValueError("clip candidate escapes governed root")
        try:
            root = _open_directory_path(self._store_dir)
        except FileNotFoundError as error:
            raise _CandidateMissing from error
        clips_directory: int | None = None
        clip_directory: int | None = None
        try:
            try:
                clips_directory = _open_directory_entry(root, "clips")
                clip_directory = _open_directory_entry(clips_directory, candidate.clip_id)
            except FileNotFoundError as error:
                raise _CandidateMissing from error
            manifest = _read_manifest(clip_directory)
            _validate_manifest(manifest, candidate.clip_id)
            media_relpath = manifest.path or manifest.media_relpath
            if media_relpath is not None:
                parts = PurePosixPath(media_relpath).parts
                if len(parts) != 3 or parts[:2] != ("clips", candidate.clip_id):
                    raise ValueError("media path escapes governed clip directory")
                _verify_regular_file(clip_directory, parts[2])
            else:
                unavailable = (
                    manifest.video_available is False
                    or manifest.local_state == "UNAVAILABLE"
                    or manifest.state == "UNAVAILABLE"
                )
                if not unavailable:
                    raise ValueError("available clip has no contained media")
            yield clips_directory
        finally:
            if clip_directory is not None:
                os.close(clip_directory)
            if clips_directory is not None:
                os.close(clips_directory)
            os.close(root)


class _CandidateMissing(FileNotFoundError):
    """The governed clips root exists but this candidate does not."""


def _open_directory_path(path: Path) -> int:
    """Open an absolute store path only when every component is a real directory."""
    absolute = path.absolute()
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            child = _open_directory_entry(descriptor, component)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    else:
        return descriptor


def _open_directory_entry(parent: int, name: str) -> int:
    entry = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise ValueError(f"governed path component is not a real directory: {name}")
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


def _read_manifest(directory: int) -> _RetentionManifest:
    descriptor = _open_regular_file(directory, "manifest.json")
    with os.fdopen(descriptor, "rb") as manifest_file:
        return _RetentionManifest.model_validate_json(manifest_file.read())


def _verify_regular_file(directory: int, name: str) -> None:
    descriptor = _open_regular_file(directory, name)
    os.close(descriptor)


def _open_regular_file(directory: int, name: str) -> int:
    entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
        raise ValueError(f"governed path component is not a regular file: {name}")
    return os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)


def _validate_manifest(manifest: _RetentionManifest, clip_id: str) -> None:
    if manifest.clip_id != clip_id:
        raise ValueError("manifest clip identity differs from governed directory")
    finalized_v2 = manifest.manifest_schema_version == 2 and manifest.finalized_at is not None
    if not manifest.finalized and not finalized_v2:
        raise ValueError("manifest is not finalized")
    if manifest.local_state == "CORRUPT":
        raise ValueError("corrupt evidence cannot be purged")


__all__ = [
    "DiskUsage",
    "EvidenceRetention",
    "PurgeCandidate",
    "PurgeResult",
    "RotationReport",
]
