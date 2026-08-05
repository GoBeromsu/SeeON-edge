"""Issue #131: AuditLogStore size-based rotation and archive retention."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from backend.app.features.clips.audit_log import (
    API_AUDIT_ARCHIVE_RETENTION_DAYS_ENV,
    API_AUDIT_LOG_MAX_BYTES_ENV,
    MIN_AUDIT_ARCHIVE_RETENTION_DAYS,
    AuditLogStore,
    _configured_audit_archive_retention_days,
)


@pytest.fixture(autouse=True)
def _no_backend_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_BACKEND_CLIP_EVENTS_URL", raising=False)


def test_append_rotates_to_a_timestamped_archive_once_max_bytes_is_exceeded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_AUDIT_LOG_MAX_BYTES_ENV, "200")
    store = AuditLogStore(tmp_path / "audit.jsonl")

    for _ in range(20):
        store.append(actor="admin", action="list", clip_id="-")

    archives = sorted(tmp_path.glob("audit-*.jsonl"))
    assert archives, "expected at least one rotated archive file"

    # Every entry across the archive(s) plus the still-live file must survive
    # rotation -- none may be dropped.
    total_lines = sum(
        len(path.read_text(encoding="utf-8").splitlines()) for path in [*archives, store.path]
    )
    assert total_lines == 20


def test_archive_retention_env_override_is_clamped_to_the_60_day_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_AUDIT_ARCHIVE_RETENTION_DAYS_ENV, "5")

    assert _configured_audit_archive_retention_days() == MIN_AUDIT_ARCHIVE_RETENTION_DAYS


def test_archive_retention_env_override_above_the_floor_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_AUDIT_ARCHIVE_RETENTION_DAYS_ENV, "90")

    assert _configured_audit_archive_retention_days() == 90


def test_rotation_prunes_archives_older_than_the_retention_window(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_AUDIT_LOG_MAX_BYTES_ENV, "1")
    monkeypatch.setenv(API_AUDIT_ARCHIVE_RETENTION_DAYS_ENV, "60")
    store = AuditLogStore(tmp_path / "audit.jsonl")

    stale_archive = tmp_path / "audit-stale.jsonl"
    stale_archive.write_text('{"stale": true}\n', encoding="utf-8")
    stale_mtime = time.time() - (61 * 86400)
    os.utime(stale_archive, (stale_mtime, stale_mtime))

    fresh_archive = tmp_path / "audit-fresh.jsonl"
    fresh_archive.write_text('{"fresh": true}\n', encoding="utf-8")
    fresh_mtime = time.time() - (10 * 86400)
    os.utime(fresh_archive, (fresh_mtime, fresh_mtime))

    # The first append creates the (still empty) live file; the second sees
    # it non-empty and past the 1-byte threshold, triggering a rotation --
    # which sweeps stale archives as a side effect.
    store.append(actor="admin", action="list", clip_id="-")
    store.append(actor="admin", action="list", clip_id="-")

    assert not stale_archive.exists()
    assert fresh_archive.exists()


def test_rotation_is_a_no_op_when_the_live_file_does_not_exist_yet(tmp_path) -> None:
    store = AuditLogStore(tmp_path / "audit.jsonl")

    entry = store.append(actor="admin", action="list", clip_id="-")

    assert entry["actor"] == "admin"
    assert store.path.exists()
    assert list(tmp_path.glob("audit-*.jsonl")) == []


def test_append_degrades_gracefully_and_logs_when_the_log_is_unwritable(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #152: ``list_clips``/``clip_video``/``label_clip`` all call
    ``append()`` after their own primary action already succeeded (or, for
    ``list_clips``, as the *only* side effect of a pure read) -- an unwritable
    audit log must never turn any of those into a 500. The failure must still
    be visible in the logs rather than silently swallowed."""
    store = AuditLogStore(tmp_path / "audit.jsonl")
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        if self == store.path:
            raise PermissionError("audit log mount unavailable")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    with caplog.at_level("WARNING"):
        entry = store.append(actor="admin", action="list", clip_id="-")

    assert entry == {
        "ts": entry["ts"],
        "actor": "admin",
        "action": "list",
        "clip_id": "-",
    }
    assert not store.path.exists()
    assert any(
        "clip audit log unavailable" in record.message for record in caplog.records
    )
