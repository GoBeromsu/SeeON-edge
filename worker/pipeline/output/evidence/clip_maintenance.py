from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
)
from worker.pipeline.output.evidence.evidence_retention import (
    DiskUsage,
    EvidenceRetention,
    PurgeCandidate,
    PurgeResult,
)


class ClipMaintenance:
    def __init__(
        self,
        config: ClipRecorderConfig,
        stats: ClipRecorderStats,
        *,
        is_clip_held: Callable[[str], bool],
        disk_usage_provider: Callable[[Path], DiskUsage],
        begin_clip_purge: Callable[[str], bool] | None = None,
        complete_clip_purge: Callable[[str], None] | None = None,
        fail_clip_purge: Callable[[str, str], None] | None = None,
        operator_delete_preflight: Callable[[str], PurgeResult | None] | None = None,
    ) -> None:
        self._config = config
        self._stats = stats
        self._retention = EvidenceRetention(
            config.store_dir,
            is_held=is_clip_held,
            disk_usage_provider=disk_usage_provider,
            begin_purge=begin_clip_purge,
            complete_purge=complete_clip_purge,
            fail_purge=fail_clip_purge,
        )
        self._operator_delete_preflight = operator_delete_preflight
        self._last_rotate_monotonic: float | None = None

    def sweep_stale_staging(self) -> None:
        staging_root = self._config.store_dir / "clips" / ".staging"
        cutoff = time.time() - max(0.0, self._config.stale_staging_seconds)
        cleaned = 0
        for staging_dir in staging_root.iterdir():
            try:
                if self._retention.is_held(staging_dir.name):
                    continue
                if not staging_dir.is_dir() or staging_dir.stat().st_mtime > cutoff:
                    continue
                shutil.rmtree(staging_dir)
                cleaned += 1
            except OSError:
                continue
        self._stats.stale_staging_cleaned += cleaned

    def preflight_clip(self, clip_id: str) -> PurgeResult | None:
        """Check an operator deletion without mutating files or durable state."""
        clips_dir = self._config.store_dir / "clips"
        clip_dir = clips_dir / clip_id
        if clip_id in {"", ".", ".."} or clip_dir.parent != clips_dir:
            return PurgeResult.UNVERIFIABLE
        if self._operator_delete_preflight is not None:
            result = self._operator_delete_preflight(clip_id)
            if result is not None:
                return result
        finalized_at = _operator_finalized_at(clip_dir)
        candidate = PurgeCandidate(
            clip_id=clip_id,
            clip_dir=clip_dir,
            finalized_at=finalized_at,
        )
        verification = self._retention.preflight(candidate)
        if verification is not None:
            return verification
        if finalized_at > datetime.now(UTC) - timedelta(days=self._config.retention_days):
            return PurgeResult.HELD
        return None

    def purge_clip(self, clip_id: str) -> PurgeResult:
        """Delete one specific finalized primary clip on operator request.

        Reuses ``EvidenceRetention.purge`` unchanged: the same manifest/
        containment/symlink verification, hold check, and begin/complete/fail
        DB hooks that automatic age/pressure ``rotate()`` already relies on.
        ``finalized_at`` is irrelevant here -- ``purge()`` never reads it, only
        ``rotate()``'s own cutoff filter does -- so ``time.time()`` is a fine
        placeholder.

        Unlike ``rotate()`` -- whose candidates only ever come from
        enumerating real ``clips/*/manifest.json`` directories -- this method
        builds a path directly from an operator-supplied ``clip_id``, so it
        has to reject a traversal/escape attempt (``".."``, ``"../x"``,
        ``"a/b"``) itself, before ever constructing a ``PurgeCandidate``
        ``EvidenceRetention``'s own containment check would otherwise trust.
        """
        preflight = self.preflight_clip(clip_id)
        if preflight is not None:
            return preflight
        clip_dir = self._config.store_dir / "clips" / clip_id
        candidate = PurgeCandidate(
            clip_id=clip_id,
            clip_dir=clip_dir,
            finalized_at=datetime.now(UTC),
        )
        return self._retention.purge(candidate)

    def rotate(self, *, force: bool = False) -> None:
        now_monotonic = time.monotonic()
        if (
            not force
            and self._last_rotate_monotonic is not None
            and now_monotonic - self._last_rotate_monotonic
            < self._config.rotate_min_interval_seconds
        ):
            return
        self._last_rotate_monotonic = now_monotonic
        report = self._retention.rotate(
            (
                PurgeCandidate(
                    clip_id=clip_dir.name,
                    clip_dir=clip_dir,
                    finalized_at=finalized_at,
                )
                for finalized_at, clip_dir in finalized_clips(self._config.store_dir)
            ),
            retention_cutoff=datetime.now(UTC)
            - timedelta(days=self._config.retention_days),
            disk_high_watermark=self._config.disk_high_watermark,
        )
        self._stats.held_clips = len(report.held_clip_ids)
        self._stats.purge_failures += len(report.failure_clip_ids)
        self._stats.recording_suspended = report.pressure_blocked


def finalized_clips(store_dir: Path) -> list[tuple[datetime, Path]]:
    root = store_dir / "clips"
    if not root.exists():
        return []
    clips: list[tuple[datetime, Path]] = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("finalized") is not True:
            continue
        finalized_at = _parse_utc(str(payload.get("finalized_at", "")))
        if finalized_at is None:
            try:
                finalized_at = datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC)
            except OSError:
                continue
        clips.append((finalized_at, manifest_path.parent))
    return clips


def _operator_finalized_at(clip_dir: Path) -> datetime:
    manifest_path = clip_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = payload.get("finalized_at", payload.get("started_at", ""))
        parsed = _parse_utc(str(raw))
        if parsed is not None:
            return parsed
        return datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC)
    except (OSError, json.JSONDecodeError):
        return datetime.now(UTC)


def _parse_utc(value: str) -> datetime | None:
    if value == "":
        return None
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError:
        return None


def default_disk_usage(path: Path) -> DiskUsage:
    return shutil.disk_usage(path)


__all__ = ["ClipMaintenance", "default_disk_usage", "finalized_clips"]
