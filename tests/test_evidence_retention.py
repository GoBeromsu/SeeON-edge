from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from contracts.frame import Frame
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder, ClipRecorderConfig
from worker.pipeline.output.evidence.evidence_retention import (
    EvidenceRetention,
    PurgeCandidate,
    PurgeResult,
)
from worker.types import BusinessEvent, FramePacket


class DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def _clip(
    root: Path,
    clip_id: str,
    finalized_at: datetime,
    *,
    media: bool = True,
) -> PurgeCandidate:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    media_path = clip_dir / "clip.mp4"
    if media:
        media_path.write_bytes(b"verified-media")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "finalized": True,
                "started_at": finalized_at.isoformat().replace("+00:00", "Z"),
                "video_available": media,
                "path": f"clips/{clip_id}/clip.mp4" if media else None,
            }
        ),
        encoding="utf-8",
    )
    return PurgeCandidate(clip_id=clip_id, clip_dir=clip_dir, finalized_at=finalized_at)


def _retention(
    root: Path,
    held_ids: set[str],
    usages: list[DiskUsage] | None = None,
) -> EvidenceRetention:
    values = iter(usages or [DiskUsage(total=100, used=10, free=90)])
    return EvidenceRetention(
        root,
        is_held=lambda clip_id: clip_id in held_ids,
        disk_usage_provider=lambda _path: next(values),
    )


def test_pressure_never_purges_59_day_unheld_clip_and_blocks(tmp_path: Path) -> None:
    # Given: an unheld clip is one day younger than the legal retention floor.
    now = datetime.now(UTC)
    young = _clip(tmp_path, "young-59-days", now - timedelta(days=59))
    retention = _retention(
        tmp_path,
        set(),
        [DiskUsage(100, 95, 5), DiskUsage(100, 95, 5)],
    )

    # When: disk pressure remains above the high watermark.
    report = retention.rotate(
        (young,),
        retention_cutoff=now - timedelta(days=60),
        disk_high_watermark=0.80,
    )

    # Then: legal retention wins over disk pressure and recording must suspend.
    assert report.purged_clip_ids == ()
    assert report.pressure_blocked is True
    assert young.clip_dir.exists()


def test_pressure_purges_only_60_day_and_older_verified_unheld_clips(
    tmp_path: Path,
) -> None:
    # Given: candidates immediately before, at, and after the legal age boundary.
    now = datetime.now(UTC)
    older = _clip(tmp_path, "older-61-days", now - timedelta(days=61))
    boundary = _clip(tmp_path, "boundary-60-days", now - timedelta(days=60))
    young = _clip(tmp_path, "young-59-days", now - timedelta(days=59))
    retention = _retention(tmp_path, set(), [DiskUsage(100, 70, 30)])

    # When: retention runs with disk pressure cleared after eligible purges.
    report = retention.rotate(
        (young, boundary, older),
        retention_cutoff=now - timedelta(days=60),
        disk_high_watermark=0.80,
    )

    # Then: the exact 60-day boundary is eligible, but younger evidence survives.
    assert report.purged_clip_ids == (older.clip_id, boundary.clip_id)
    assert report.pressure_blocked is False
    assert not older.clip_dir.exists()
    assert not boundary.clip_dir.exists()
    assert young.clip_dir.exists()


def test_age_and_pressure_never_delete_active_hold(tmp_path: Path) -> None:
    # Given: the only clip is old, remotely unacknowledged, and disk is full.
    now = datetime.now(UTC)
    held = _clip(tmp_path, "pending-held", now - timedelta(days=90))
    retention = _retention(
        tmp_path,
        {held.clip_id},
        [DiskUsage(100, 100, 0), DiskUsage(100, 100, 0)],
    )

    # When: both ordinary retention and pressure selection apply.
    report = retention.rotate(
        (held,),
        retention_cutoff=now - timedelta(days=60),
        disk_high_watermark=0.80,
    )

    # Then: evidence remains and callers are told to suspend new recording.
    assert report.purged_clip_ids == ()
    assert report.held_clip_ids == (held.clip_id,)
    assert report.pressure_blocked is True
    assert held.clip_dir.exists()


def test_purge_verification_failure_is_reported_and_evidence_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a verified, remotely acknowledged clip and a silent delete failure.
    candidate = _clip(tmp_path, "delete-noop", datetime.now(UTC))
    retention = _retention(tmp_path, set())
    monkeypatch.setattr(shutil, "rmtree", lambda *_args, **_kwargs: None)

    # When: purge runs and the directory still exists afterward.
    result = retention.purge(candidate)

    # Then: the false success is rejected and the evidence is not marked purged.
    assert result is PurgeResult.VERIFICATION_FAILED
    assert candidate.clip_dir.exists()


def test_corrupt_manifest_state_is_never_purgeable(tmp_path: Path) -> None:
    # Given: a released candidate whose manifest records corrupt local evidence.
    candidate = _clip(tmp_path, "corrupt-held", datetime.now(UTC))
    manifest_path = candidate.clip_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["local_state"] = "CORRUPT"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    retention = _retention(tmp_path, set())

    # When: purge is requested despite the lost integrity guarantee.
    result = retention.purge(candidate)

    # Then: corrupt evidence is retained for repair or operator review.
    assert result is PurgeResult.UNVERIFIABLE
    assert candidate.clip_dir.exists()


def test_missing_candidate_is_not_treated_as_successful_purge(tmp_path: Path) -> None:
    # Given: a catalog candidate whose evidence directory disappeared externally.
    candidate = PurgeCandidate(
        clip_id="missing",
        clip_dir=tmp_path / "clips" / "missing",
        finalized_at=datetime.now(UTC),
    )
    retention = _retention(tmp_path, set())

    # When: purge attempts to verify and remove it.
    result = retention.purge(candidate)

    # Then: missing evidence is reported for reconciliation, never called purged.
    assert result is PurgeResult.MISSING


@pytest.mark.parametrize("case", ("symlink", "path-escape"))
def test_symlink_and_path_escape_are_unverifiable_and_never_deleted(
    tmp_path: Path, case: str
) -> None:
    # Given: a candidate that does not name an owned, real clip directory.
    outside = tmp_path.parent / f"outside-{tmp_path.name}-{case}"
    outside.mkdir()
    (outside / "sentinel").write_text("preserve", encoding="utf-8")
    if case == "symlink":
        clips = tmp_path / "clips"
        clips.mkdir()
        candidate_path = clips / "escape"
        candidate_path.symlink_to(outside, target_is_directory=True)
    else:
        candidate_path = outside
    candidate = PurgeCandidate(
        clip_id="escape",
        clip_dir=candidate_path,
        finalized_at=datetime.now(UTC),
    )
    retention = _retention(tmp_path, set())

    # When: the retention boundary evaluates it.
    result = retention.purge(candidate)

    # Then: path ambiguity is held for repair and outside bytes survive.
    assert result is PurgeResult.UNVERIFIABLE
    assert (outside / "sentinel").read_text(encoding="utf-8") == "preserve"
    shutil.rmtree(outside)


def test_restart_reacquires_lock_and_preserves_pending_evidence(tmp_path: Path) -> None:
    # Given: a first recorder sees full disk with only held evidence.
    now = datetime.now(UTC)
    held = _clip(tmp_path, "restart-held", now - timedelta(days=90))
    config = ClipRecorderConfig(
        store_dir=tmp_path,
        retention_days=60,
        disk_high_watermark=0.80,
    )
    first = ClipRecorder(
        config,
        is_clip_held=lambda clip_id: clip_id == held.clip_id,
        disk_usage_provider=lambda _path: DiskUsage(100, 100, 0),
    )
    first.start()
    try:
        assert first.rotate_once()
        assert first.stats.recording_suspended is True
    finally:
        first.stop()

    # When: the process-local recorder is reconstructed after restart.
    second = ClipRecorder(
        config,
        is_clip_held=lambda clip_id: clip_id == held.clip_id,
        disk_usage_provider=lambda _path: DiskUsage(100, 100, 0),
    )
    trigger_packet = FramePacket(
        "cam-1",
        Frame(1, now.timestamp(), np.zeros((8, 8, 3), dtype=np.uint8)),
        now.timestamp(),
        1,
        8,
        8,
        0.1,
        worker_boot_id="restart-boot",
        stream_epoch=1,
    )
    second.start()
    try:
        assert second.rotate_once()
        new_clip = second.on_event(
            trigger_packet,
            BusinessEvent(
                "fall",
                "fall.detected",
                "event-new",
                "cam-1",
                "facility-1",
                now.timestamp(),
                0.9,
            ),
        )
    finally:
        second.stop()
        trigger_packet.release()

    # Then: restart does not clear the hold, and pressure refuses new evidence.
    assert new_clip is None
    assert held.clip_dir.exists()


def test_stale_staged_evidence_is_held_until_reconciliation_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: stale-looking staging belongs to pending evidence with no release proof.
    staging = tmp_path / "clips" / ".staging" / "pending-staged"
    staging.mkdir(parents=True)
    old_time = datetime.now().timestamp() - 3600
    staging.chmod(0o700)
    os.utime(staging, (old_time, old_time))
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=tmp_path, stale_staging_seconds=1.0),
        disk_usage_provider=lambda _path: DiskUsage(100, 10, 90),
    )

    # When: the lock-aware worker starts before a reconciler releases the hold.
    recorder.start()
    recorder.stop()

    # Then: startup does not silently delete pending staging evidence.
    assert staging.exists()
