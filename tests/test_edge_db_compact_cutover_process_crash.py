from __future__ import annotations

import multiprocessing
import os
import signal
import sqlite3
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from compact_cutover_fixtures import cutover_request, sha256

from backend.app.edge_db import compact_cutover
from backend.app.edge_db.compact_cutover import (
    CompactCutoverRequest,
    CutoverPhase,
    run_compact_cutover,
)

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")

_BOUNDARIES = tuple(CutoverPhase)


def _cutover_child(request: CompactCutoverRequest, channel: Connection) -> None:
    compact_cutover._runtime_sqlite_version = lambda: (3, 51, 3)

    def pause(phase: CutoverPhase) -> None:
        channel.send(phase.value)
        channel.recv()

    try:
        run_compact_cutover(request, on_phase=pause)
    finally:
        channel.close()


@pytest.mark.parametrize("boundary", _BOUNDARIES, ids=lambda phase: phase.value)
def test_real_process_kill_resumes_each_durable_boundary(
    tmp_path: Path, boundary: CutoverPhase
) -> None:
    root = tmp_path / boundary.value
    root.mkdir()
    request = cutover_request(root)
    source_hash = sha256(request.source)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_cutover_child, args=(request, child))
    process.start()
    child.close()
    try:
        while True:
            assert parent.poll(10), f"child did not reach {boundary.value}"
            observed = CutoverPhase(parent.recv())
            if observed is boundary:
                os.kill(process.pid, signal.SIGKILL)
                break
            parent.send("continue")
        process.join(10)
        assert process.exitcode == -signal.SIGKILL
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()

    assert sha256(request.source) == source_hash
    if request.archive.exists():
        assert sha256(request.archive) == source_hash
    with sqlite3.connect(request.live) as connection:
        interrupted_version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert interrupted_version in {17, 18}

    result = run_compact_cutover(request)

    assert result.source_sha256 == source_hash
    assert sha256(request.source) == source_hash == sha256(request.archive)
    assert not request.candidate.exists()
    with sqlite3.connect(request.live) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
