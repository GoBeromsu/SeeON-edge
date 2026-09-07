from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from time import monotonic

import pytest


def _wait_gone(pid: int) -> None:
    deadline = monotonic() + 5
    while monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        threading.Event().wait(0.01)
    raise AssertionError(f"process {pid} survived parent lifetime boundary")


def test_expected_parent_mismatch_self_terminates_immediately(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    code = (
        "from worker.runtime.clip_analysis_subprocess import bootstrap_child;"
        f"bootstrap_child(expected_parent={os.getpid() + 100000}, "
        f"cpu_index={next(iter(os.sched_getaffinity(0)))}, control_fd={read_fd})"
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code], pass_fds=(read_fd,), cwd=Path.cwd()
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert process.wait(timeout=3) == 3


def test_bootstrap_sets_cpu_and_thread_limits_before_work() -> None:
    read_fd, write_fd = os.pipe()
    cpu = next(iter(os.sched_getaffinity(0)))
    code = (
        "import os,time;"
        "from worker.runtime.clip_analysis_subprocess import bootstrap_child;"
        f"bootstrap_child(expected_parent={os.getpid()}, cpu_index={cpu}, control_fd={read_fd});"
        "print(sorted(os.sched_getaffinity(0)), os.environ['OMP_NUM_THREADS'], "
        "os.environ['OPENBLAS_NUM_THREADS'], os.environ['MKL_NUM_THREADS'], flush=True);"
        "time.sleep(10)"
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code], pass_fds=(read_fd,), stdout=subprocess.PIPE, text=True
        )
        assert process.stdout is not None
        assert process.stdout.readline().strip() == f"[{cpu}] 1 1 1"
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert process.wait(timeout=3) == 3


@pytest.mark.heavy
def test_parent_sigkill_terminates_busy_child_with_zero_survivors() -> None:
    read_fd, write_fd = os.pipe()
    cpu = next(iter(os.sched_getaffinity(0)))
    child_code = (
        "import os;"
        "from worker.runtime.clip_analysis_subprocess import bootstrap_child;"
        f"bootstrap_child(expected_parent=os.getppid(), cpu_index={cpu}, control_fd={read_fd});"
        "print(os.getpid(), flush=True);"
        "\nwhile True: pass"
    )
    parent_code = (
        "import os,subprocess,sys;"
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}], pass_fds=({read_fd},), "
        "stdout=subprocess.PIPE, text=True);"
        "print(child.stdout.readline().strip(), flush=True);"
        "\nwhile True: pass"
    )
    sentinel = subprocess.Popen(["sleep", "10"])
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        pass_fds=(read_fd,),
        stdout=subprocess.PIPE,
        text=True,
        cwd=Path.cwd(),
    )
    os.close(read_fd)
    try:
        assert parent.stdout is not None
        child_pid = int(parent.stdout.readline().strip())
        os.kill(parent.pid, 9)
        assert parent.wait(timeout=3) == -9
        _wait_gone(child_pid)
        assert sentinel.poll() is None
    finally:
        os.close(write_fd)
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if sentinel.poll() is None:
            sentinel.terminate()
            sentinel.wait()


@pytest.mark.heavy
def test_pipe_eof_terminates_busy_child() -> None:
    read_fd, write_fd = os.pipe()
    cpu = next(iter(os.sched_getaffinity(0)))
    code = (
        "import os;"
        "from worker.runtime.clip_analysis_subprocess import bootstrap_child;"
        f"bootstrap_child(expected_parent={os.getpid()}, cpu_index={cpu}, control_fd={read_fd});"
        "print('ready', flush=True);"
        "\nwhile True: pass"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        pass_fds=(read_fd,),
        stdout=subprocess.PIPE,
        text=True,
        cwd=Path.cwd(),
    )
    os.close(read_fd)
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        os.close(write_fd)
        write_fd = -1
        assert process.wait(timeout=5) == 3
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        if process.poll() is None:
            process.kill()
            process.wait()
