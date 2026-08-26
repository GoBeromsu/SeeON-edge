"""C5 native child supervision contracts."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

import worker.runtime.deepstream.supervisor as supervisor
from worker.runtime.deepstream.readiness import wait_for_ready
from worker.runtime.deepstream.source_control import SourceReadinessError
from worker.runtime.lease import GpuLease, GpuLeaseUnavailableError


def test_non_nvidia_profile_constructs_no_child() -> None:
    # Given / When / Then
    assert supervisor.configured_dark_supervisors({"ML_WORKER_PROFILE": "cpu"}) == ()


def test_nvidia_profile_admits_exactly_one_gpu_zero_child_without_dark_switch() -> None:
    # Given
    environment = {
        "ML_WORKER_PROFILE": "nvidia",
        "NVIDIA_VISIBLE_DEVICES": "0",
        "SEEON_DEEPSTREAM_ENGINE_CACHE": "/var/cache/seeon/tensorrt/c7-fixture",
    }

    # When
    configured = supervisor.configured_dark_supervisors(environment)

    # Then
    assert len(configured) == 1
    assert configured[0].socket_dir.name == "gpu-0"
    with pytest.raises(supervisor.DarkChildConfigError) as failed:
        _ = supervisor.configured_dark_supervisors(
            {
                "ML_WORKER_PROFILE": "nvidia",
                "NVIDIA_VISIBLE_DEVICES": "0,2",
                "SEEON_DEEPSTREAM_ENGINE_CACHE": "/var/cache/seeon/tensorrt/c7-fixture",
            }
        )
    assert failed.value.code == "unsupported_gpu"


def test_supervisor_uses_ready_fd_not_misleading_stdout(tmp_path: Path) -> None:
    # Given
    executable = tmp_path / "misleading-child"
    _ = executable.write_text(
        "#!/bin/sh\nprintf 'READY\\n'\nexec sleep 2\n",
        encoding="utf-8",
    )
    _ = executable.chmod(0o700)
    child = supervisor.DeepStreamChildSupervisor(
        supervisor.ChildConfig(
            executable=executable,
            worker_boot_id=uuid.uuid4(),
            socket_dir=tmp_path / "ipc",
            first_fault_path=tmp_path / "first-fault.bin",
            startup_timeout_sec=0.1,
            stop_timeout_sec=0.2,
        )
    )

    # When / Then
    with pytest.raises(supervisor.ChildStartupError) as failed:
        child.start()
    assert failed.value.code == "ready_failed"
    child.stop()
    child.stop()
    assert child.pid is None


def test_ready_handshake_supports_a_descriptor_above_fd_setsize() -> None:
    descriptors = [os.open(os.devnull, os.O_RDONLY) for _ in range(1_100)]
    ready_fd, writer_fd = os.pipe()

    class RunningProcess:
        @staticmethod
        def poll() -> None:
            return None

    try:
        assert ready_fd > 1_024
        _ = os.write(writer_fd, b"R")
        assert wait_for_ready(ready_fd, RunningProcess(), 1.0) is True  # type: ignore[arg-type]
    finally:
        os.close(writer_fd)
        for descriptor in descriptors:
            os.close(descriptor)


def test_au_rebuild_failure_enters_real_fatal_path(tmp_path: Path) -> None:
    child = supervisor.DeepStreamChildSupervisor(
        supervisor.ChildConfig(
            executable=tmp_path / "unused",
            worker_boot_id=uuid.uuid4(),
            socket_dir=tmp_path / "ipc",
            first_fault_path=tmp_path / "fault.json",
            lease_state_dir=tmp_path,
        )
    )

    class FailingSources:
        def rebuild(self, camera_id: str, category: str) -> None:
            raise SourceReadinessError(category, camera_id)

    child._sources = FailingSources()  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001
    child._handle_au_gap("camera-a", "parser")  # noqa: SLF001

    assert child._fatal_received.is_set()  # noqa: SLF001
    assert child._fatal_category == "au_epoch_rebuild_failed"  # noqa: SLF001
    assert child._config.first_fault_path.is_file()  # noqa: SLF001


def test_supervisor_refuses_before_spawn_when_python_gpu_lease_is_held(tmp_path: Path) -> None:
    # Given
    lease_state = tmp_path / "lease"
    child = supervisor.DeepStreamChildSupervisor(
        supervisor.ChildConfig(
            executable=tmp_path / "must-not-spawn",
            worker_boot_id=uuid.uuid4(),
            socket_dir=tmp_path / "ipc",
            first_fault_path=tmp_path / "fault",
            lease_state_dir=lease_state,
        )
    )

    # When / Then
    with GpuLease.acquire(lease_state):
        with pytest.raises(GpuLeaseUnavailableError):
            child.start()
    assert child.pid is None
