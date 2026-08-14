from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from worker.pipeline.output.evidence.clip_recorder import ClipRecorder, ClipRecorderConfig
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLock, ClipStoreLockedError


def test_second_process_is_refused_while_first_holds_store_lock(tmp_path: Path) -> None:
    # Given: another worker process holds the clip store lock for its lifetime.
    script = """
import sys
from pathlib import Path
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLock

lock = ClipStoreLock.acquire(Path(sys.argv[1]))
print("locked", flush=True)
sys.stdin.read()
lock.close()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"

    try:
        # When: a second worker attempts to acquire the same lifetime lock.
        with pytest.raises(ClipStoreLockedError):
            ClipStoreLock.acquire(tmp_path)
    finally:
        process.terminate()
        process.wait(timeout=5.0)

    # Then: process exit releases the advisory lock without deleting its inode.
    lock_path = tmp_path / ".worker.lock"
    assert lock_path.exists()
    with ClipStoreLock.acquire(tmp_path):
        assert lock_path.exists()


def test_stale_lock_file_does_not_block_restarted_worker(tmp_path: Path) -> None:
    # Given: an unlocked file remains after a prior process closed normally.
    first = ClipStoreLock.acquire(tmp_path)
    first.close()
    lock_path = tmp_path / ".worker.lock"
    before = lock_path.stat().st_ino

    # When: a restarted worker acquires the existing lock file.
    with ClipStoreLock.acquire(tmp_path):
        during = lock_path.stat().st_ino

    # Then: ownership depends on flock state, not file existence or a stale PID.
    assert during == before


def test_second_recorder_refuses_before_stale_evidence_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a first recorder owns the store, then stale-looking staging appears.
    config = ClipRecorderConfig(store_dir=tmp_path, stale_staging_seconds=0.0)
    first = ClipRecorder(config)
    first.start()
    stale = tmp_path / "clips" / ".staging" / "must-survive"
    stale.mkdir()

    second = ClipRecorder(config)
    try:
        # When: an overlapping recorder attempts startup against the same store.
        with pytest.raises(ClipStoreLockedError):
            second.start()
    finally:
        second.stop()
        first.stop()

    # Then: lock refusal happens before any sweep/finalize/retention mutation.
    assert stale.exists()
