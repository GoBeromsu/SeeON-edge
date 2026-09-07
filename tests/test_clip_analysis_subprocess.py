from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
