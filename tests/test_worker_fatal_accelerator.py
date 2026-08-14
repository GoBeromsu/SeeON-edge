"""Tests for todo 25: fatal accelerator fault containment.

All tests are hardware-free — CUDA errors are injected via fakes.
"""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
import threading
import time as time_module
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import final

import numpy as np
import pytest

from shared.edge_db.connection import RuntimeActor, open_runtime_database
from shared.edge_db.migrator import migrate_database
from worker.adapters.model.errors import FatalAcceleratorError, ModelInputError
from worker.adapters.model.yolo_api import (
    YoloPredictOptions,
    _classify_or_reraise,
    predict_one,
)
from worker.pipeline.output.evidence.evidence_outbox_database import open_connection
from worker.runtime.faults.handler import FATAL_ACCELERATOR_EXIT_CODE, FaultHandler
from worker.runtime.faults.record import (
    WORKER_STATE_DB_FILENAME,
    FirstFaultRecord,
    make_fault_record,
    persist_first_fault,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(**kwargs) -> FirstFaultRecord:
    defaults = dict(
        pid=1,
        boot_time_iso="2026-01-01T00:00:00Z",
        profile="cuda",
        task="pose",
        stage="inference",
        camera_id="cam-1",
        frame_index=42,
        pts=1.0,
        frame_shape=(480, 640, 3),
        frame_hash_sha256=None,
        model_artifact_digest="abc123",
        invocation_seq=1,
        exception_type="RuntimeError",
        exception_message="CUDA error: device-side assert triggered",
        exit_code=4,
        fault_time_iso="2026-01-01T00:00:01Z",
    )
    defaults.update(kwargs)
    return FirstFaultRecord(**defaults)


@final
@dataclass
class _FakeLoop:
    stopped: bool = False

    def stop(self) -> None:
        self.stopped = True


# ---------------------------------------------------------------------------
# FatalAcceleratorError classification
# ---------------------------------------------------------------------------


def test_cuda_keyword_raises_fatal_accelerator_error() -> None:
    exc = RuntimeError("CUDA error: device-side assert triggered")
    with pytest.raises(FatalAcceleratorError) as exc_info:
        _classify_or_reraise(exc, task="pose", camera_id="cam-1")
    assert exc_info.value.task == "pose"
    assert exc_info.value.camera_id == "cam-1"


def test_non_cuda_error_raises_yolo_forward_error() -> None:
    from worker.adapters.model.yolo_api import YoloForwardError

    exc = OSError("model file not found")
    with pytest.raises(YoloForwardError):
        _classify_or_reraise(exc, task="pose", camera_id="cam-1")


def test_cuda_classification_is_case_insensitive() -> None:
    for msg in ("cuda launch failed", "CUDA error", "cuLaunch error", "device lost"):
        exc = RuntimeError(msg)
        with pytest.raises(FatalAcceleratorError):
            _classify_or_reraise(exc, task="pose", camera_id="cam-1")


def test_non_fatal_validation_error_stays_isolated() -> None:
    """A ModelInputError (shape mismatch) is NOT a fatal accelerator fault."""
    err = ModelInputError("wrong frame shape")
    assert not isinstance(err, FatalAcceleratorError)


# ---------------------------------------------------------------------------
# predict_one wrapping
# ---------------------------------------------------------------------------


def test_predict_one_raises_fatal_on_cuda_error() -> None:
    options = YoloPredictOptions(task="pose", confidence=0.05, device="cuda")

    class _BadModel:
        def predict(self, **_kwargs):
            raise RuntimeError("CUDA error: illegal memory access")

    with pytest.raises(FatalAcceleratorError):
        predict_one(_BadModel(), np.zeros((4, 4, 3), dtype=np.uint8), options, camera_id="cam-1")


def test_predict_one_raises_forward_error_on_non_cuda() -> None:
    from worker.adapters.model.yolo_api import YoloForwardError

    options = YoloPredictOptions(task="bed", confidence=0.25, device="cpu")

    class _BadModel:
        def predict(self, **_kwargs):
            raise OSError("onnx model corrupt")

    with pytest.raises(YoloForwardError):
        predict_one(_BadModel(), np.zeros((4, 4, 3), dtype=np.uint8), options)


# ---------------------------------------------------------------------------
# First-fault record persistence
# ---------------------------------------------------------------------------


def test_persist_first_fault_writes_exactly_one_record_to_faults_table(tmp_path: Path) -> None:
    # Module-level _written flag is per-import, so we reset it between tests.
    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    rec = _record()
    wrote_first = persist_first_fault(rec, state_dir=tmp_path)
    wrote_second = persist_first_fault(rec, state_dir=tmp_path)

    assert wrote_first is True
    assert wrote_second is False  # second call is a no-op

    connection = sqlite3.connect(tmp_path / WORKER_STATE_DB_FILENAME)
    try:
        cursor = connection.execute("SELECT * FROM faults")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
    finally:
        connection.close()

    assert len(rows) == 1  # exactly one record, per the "first fault wins" contract
    row = dict(zip(columns, rows[0], strict=True))
    assert row["id"] == 1
    assert row["exit_code"] == 4
    assert row["camera_id"] == "cam-1"
    assert row["exception_message"] == "CUDA error: device-side assert triggered"


def test_persist_first_fault_degrades_to_false_when_database_parent_is_uncreatable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A storage failure (e.g. an unwritable/uncreatable state dir) must
    degrade to False rather than raise. persist_first_fault runs on
    FaultHandler's hard-exit boundary and must never prevent the process
    from exiting -- see test_fault_handler_exits_even_when_fault_storage_is_unavailable
    for the handler-level version of this contract."""
    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    blocker = tmp_path / "blocker-file"
    blocker.write_text("not a directory")
    state_dir = blocker / "state"

    rec = _record()
    with caplog.at_level("WARNING"):
        written = persist_first_fault(rec, state_dir=state_dir)

    assert written is False
    assert "first-fault record unavailable" in caplog.text


def test_persist_first_fault_does_not_wait_out_a_concurrent_writer_lock(tmp_path: Path) -> None:
    """The evidence outbox or config LKG store may hold worker-state.sqlite3's
    write lock when a fault fires. persist_first_fault must fail fast --
    SQLITE_BUSY surfacing immediately via busy_timeout_ms=0 -- rather than
    wait out open_connection's normal 5-second busy timeout, which would
    delay the hard-exit boundary a watchdog trip may itself be racing a
    deadline against."""
    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    database_path = tmp_path / WORKER_STATE_DB_FILENAME
    # Pre-create the schema (including `faults`) at the normal busy timeout.
    open_connection(database_path).close()

    # Hold the write lock on a second connection, the same lock shape the
    # outbox writer or config LKG store would hold mid-write.
    holder = sqlite3.connect(database_path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        rec = _record()
        started = time_module.monotonic()
        written = persist_first_fault(rec, state_dir=tmp_path)
        elapsed = time_module.monotonic() - started
    finally:
        holder.rollback()
        holder.close()

    assert written is False
    assert elapsed < 1.0  # fails fast; must not wait out a multi-second timeout


def _hold_edge_worker_write(database: str, channel: Connection) -> None:
    """Child process: hold BEGIN IMMEDIATE until the parent signals release."""
    connection = open_runtime_database(Path(database), actor=RuntimeActor.WORKER)
    try:
        connection.execute("BEGIN IMMEDIATE")
        channel.send("LOCKED")
        assert channel.recv() == "RELEASE"
        connection.rollback()
        channel.send("RELEASED")
    finally:
        connection.close()
        channel.close()


def test_persist_first_fault_writes_to_production_named_edge_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production path resolves EDGE_DATABASE_PATH (edge.sqlite3) and must
    upsert the faults row through the central runtime writer ownership."""
    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    database_path = tmp_path / "edge-state" / "edge.sqlite3"
    migrate_database(database_path)
    monkeypatch.setattr(mod, "EDGE_DATABASE_PATH", database_path)

    rec = _record(camera_id="cam-edge-1", exception_message="CUDA error: edge path")
    wrote_first = persist_first_fault(rec)  # state_dir defaults -> EDGE_DATABASE_PATH
    wrote_second = persist_first_fault(rec)

    assert wrote_first is True
    assert wrote_second is False

    connection = sqlite3.connect(database_path)
    try:
        row = connection.execute(
            "SELECT camera_id, exception_message, exit_code FROM faults WHERE id = 1"
        ).fetchone()
    finally:
        connection.close()

    assert row == ("cam-edge-1", "CUDA error: edge path", 4)


def test_persist_first_fault_edge_sqlite_returns_immediately_under_held_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Production edge.sqlite3 must use BusyPolicy.ZERO_WAIT via
    best_effort_zero_wait_write: under a real held SQLite write lock the
    fatal path returns immediately (no default 5s bound), logs truthfully,
    and leaves the faults table untouched. Synchronization uses pipe
    barriers only -- no sleeps."""
    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    database_path = tmp_path / "edge-state" / "edge.sqlite3"
    migrate_database(database_path)
    monkeypatch.setattr(mod, "EDGE_DATABASE_PATH", database_path)

    context = multiprocessing.get_context("spawn")
    parent_channel, child_channel = context.Pipe()
    holder = context.Process(
        target=_hold_edge_worker_write,
        args=(os.fspath(database_path), child_channel),
    )
    holder.start()
    assert parent_channel.poll(10), "worker holder did not acquire the write lock"
    assert parent_channel.recv() == "LOCKED"

    try:
        rec = _record()
        started = time_module.monotonic()
        with caplog.at_level("WARNING"):
            written = persist_first_fault(rec)
        elapsed = time_module.monotonic() - started
    finally:
        parent_channel.send("RELEASE")
        assert parent_channel.poll(10), "worker holder did not release"
        assert parent_channel.recv() == "RELEASED"
        holder.join(10)

    assert holder.exitcode == 0
    assert written is False
    assert elapsed < 1.0  # near-immediate; must never wait the default 5s bound
    assert "first-fault record unavailable" in caplog.text
    assert "zero-wait write failed" in caplog.text

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("SELECT count(*) FROM faults").fetchone() == (0,)
    finally:
        connection.close()


def test_persist_first_fault_includes_frame_hash(tmp_path: Path) -> None:
    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    rec = make_fault_record(
        RuntimeError("CUDA error"),
        profile="cuda",
        task="pose",
        stage="inference",
        camera_id="cam-1",
        image=image,
    )
    assert rec.frame_hash_sha256 is not None
    assert len(rec.frame_hash_sha256) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# FaultHandler — stop all cameras, exit with code 4
# ---------------------------------------------------------------------------


def test_fault_handler_stops_all_loops_and_exits(tmp_path: Path) -> None:
    exits: list[int] = []
    handler = FaultHandler("cuda", hard_exit=exits.append, state_dir=tmp_path)
    loop_a = _FakeLoop()
    loop_b = _FakeLoop()
    handler.register_loop(loop_a)
    handler.register_loop(loop_b)

    rec = _record()
    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    handler.handle(FatalAcceleratorError("CUDA error"), rec)

    assert loop_a.stopped
    assert loop_b.stopped
    assert exits == [FATAL_ACCELERATOR_EXIT_CODE]


def test_fault_handler_is_idempotent(tmp_path: Path) -> None:
    """Concurrent calls from two camera threads must trigger exactly one exit."""
    exits: list[int] = []
    handler = FaultHandler("cuda", hard_exit=exits.append, state_dir=tmp_path)
    loop = _FakeLoop()
    handler.register_loop(loop)

    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    rec = _record()
    t1 = threading.Thread(target=handler.handle, args=(FatalAcceleratorError("CUDA error"), rec))
    t2 = threading.Thread(target=handler.handle, args=(FatalAcceleratorError("CUDA error"), rec))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert exits == [FATAL_ACCELERATOR_EXIT_CODE]


def test_fault_handler_exits_even_when_fault_storage_is_unavailable(tmp_path: Path) -> None:
    """The never-blocks-exit contract at the handler level: a storage failure
    inside persist_first_fault (unwritable/uncreatable state dir) must not
    prevent FaultHandler.handle() from stopping every camera loop and hard-
    exiting with FATAL_ACCELERATOR_EXIT_CODE."""
    import worker.runtime.faults.record as mod

    mod._written = False  # noqa: SLF001

    blocker = tmp_path / "blocker-file"
    blocker.write_text("not a directory")
    state_dir = blocker / "state"

    exits: list[int] = []
    handler = FaultHandler("cuda", hard_exit=exits.append, state_dir=state_dir)
    loop = _FakeLoop()
    handler.register_loop(loop)

    rec = _record()
    handler.handle(FatalAcceleratorError("CUDA error"), rec)

    assert loop.stopped
    assert exits == [FATAL_ACCELERATOR_EXIT_CODE]


def test_fatal_exit_code_is_4() -> None:
    assert FATAL_ACCELERATOR_EXIT_CODE == 4


# ---------------------------------------------------------------------------
# lifecycle.py: FatalAcceleratorError propagates through ingest loop
# ---------------------------------------------------------------------------


def test_fatal_accelerator_propagates_through_ingest_loop() -> None:
    """FatalAcceleratorError must not be swallowed by the broad except in lifecycle."""

    class _FatalBus:
        def publish(self, _packet) -> None:
            raise FatalAcceleratorError("CUDA error: illegal memory access")

    class _SimpleLoop:
        camera_id = "cam-1"
        _ready = False

        def run(self) -> None:
            bus = _FatalBus()
            for _ in range(1):
                if not self._ready:
                    self._ready = True
                try:
                    bus.publish(None)
                except FatalAcceleratorError:
                    raise
                except Exception:  # noqa: BLE001 S110
                    pass

    loop = _SimpleLoop()
    with pytest.raises(FatalAcceleratorError):
        loop.run()
