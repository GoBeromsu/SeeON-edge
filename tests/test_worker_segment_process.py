"""Exercises the real subprocess plumbing in ``segment_process.py`` --
specifically the stderr drain that keeps a long-lived ffmpeg process from
blocking on a full stderr pipe (see #105).

``test_worker_segment_encoder.py`` covers ``FFmpegSegmentEncoder`` against a
fake, in-process ``EncoderProcess``; these tests instead spawn real child
processes (via ``sys.executable -c ...``, not ffmpeg, so they're fast and
don't depend on ffmpeg being installed) to exercise the real pipe-backpressure
behavior that a fake process can't reproduce.
"""

from __future__ import annotations

import logging
import sys
import threading

import pytest

from worker.adapters.encode.segment_process import (
    _STDERR_TAIL_MAX_BYTES,  # pyright: ignore[reportPrivateUsage]
    _PopenEncoderProcess,  # pyright: ignore[reportPrivateUsage]
    spawn_encoder_process,
)

# Reads stdin in small chunks and, for every chunk received, writes a much
# larger chunk to stderr -- mirroring an ffmpeg process that keeps consuming
# frames on stdin while emitting a sustained trickle of stderr diagnostics.
# Only exits once stdin is closed (EOF), same as the real long-lived encoder.
_STDERR_FLOOD_CHILD = (
    "import sys\n"
    "while True:\n"
    "    chunk = sys.stdin.buffer.read(256)\n"
    "    if not chunk:\n"
    "        break\n"
    "    sys.stderr.buffer.write(b'e' * 4096)\n"
    "    sys.stderr.buffer.flush()\n"
    "sys.exit(0)\n"
)


def test_sustained_stderr_output_does_not_block_stdin_writes() -> None:
    """Regression test for #105: without continuous stderr draining, the
    child blocks on a full stderr pipe, stops reading stdin, and this write
    loop hangs -- confirmed by running this exact scenario against the
    pre-fix code, which reliably blocks after a few hundred KB written.
    2000 iterations push ~500KB through stdin and ~8MB through stderr, well
    beyond any OS pipe buffer, while the process stays alive throughout."""
    process = spawn_encoder_process((sys.executable, "-c", _STDERR_FLOOD_CHILD))
    assert isinstance(process, _PopenEncoderProcess)

    completed = threading.Event()
    errors: list[BaseException] = []

    def _drive_writes() -> None:
        try:
            for _ in range(2000):
                process.write(b"x" * 256)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test thread below
            errors.append(exc)
        finally:
            completed.set()

    writer = threading.Thread(target=_drive_writes, daemon=True)
    writer.start()
    finished = completed.wait(timeout=15.0)
    writer.join(timeout=1.0)

    assert finished, "stdin writes hung -- stderr pipe backpressure was not drained"
    assert errors == []

    # The bounded tail buffer must never grow to hold the full ~800KB of
    # stderr the child emitted.
    drain = process._stderr_drain  # pyright: ignore[reportPrivateUsage]
    assert drain is not None
    assert len(drain.tail()) <= _STDERR_TAIL_MAX_BYTES

    assert process.reap() == 0


def test_reap_logs_the_stderr_tail_at_warning_on_nonzero_exit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The observable logging contract on a failed encoder process must be
    unchanged: a nonzero exit still logs the buffered stderr tail at WARNING,
    even though it's now collected by a background thread instead of a
    single blocking read in reap()."""
    script = "import sys\nsys.stderr.write('boom: encoder failure')\nsys.exit(7)\n"
    process = spawn_encoder_process((sys.executable, "-c", script))

    with caplog.at_level(logging.WARNING, logger="worker.adapters.encode.segment_process"):
        returncode = process.reap()

    assert returncode == 7
    records = caplog.records
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    message = records[0].getMessage()
    assert "boom: encoder failure" in message
    assert "7" in message


def test_reap_does_not_log_anything_on_a_clean_exit(caplog: pytest.LogCaptureFixture) -> None:
    script = "import sys\nsys.stderr.write('should not be logged')\nsys.exit(0)\n"
    process = spawn_encoder_process((sys.executable, "-c", script))

    with caplog.at_level(logging.WARNING, logger="worker.adapters.encode.segment_process"):
        returncode = process.reap()

    assert returncode == 0
    assert caplog.records == []


def test_stderr_drain_thread_terminates_after_reap() -> None:
    process = spawn_encoder_process((sys.executable, "-c", "import sys; sys.exit(0)"))
    assert isinstance(process, _PopenEncoderProcess)
    drain = process._stderr_drain  # pyright: ignore[reportPrivateUsage]
    assert drain is not None

    process.reap()

    assert not drain.is_alive()
