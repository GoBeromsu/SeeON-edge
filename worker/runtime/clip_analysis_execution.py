"""Child execution and completion stages for stored clip analysis."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from time import monotonic

from shared.events.clip_analysis_wire import MAX_CLIP_ANALYSIS_OUTPUT_BYTES, decode_clip_analysis
from worker.adapters.model.errors import ModelLoadError
from worker.pipeline.output.evidence.clip_analysis_artifact import (
    ClipAnalysisArtifactError,
    ClipAnalysisArtifactIdentity,
    publish_clip_analysis,
    publish_failed_outcome,
)
from worker.runtime import clip_analysis_process
from worker.runtime.clip_analysis_supervisor_status import ClipAnalysisStatus

LOGGER = logging.getLogger(__name__)
_TERMINAL_CHILD_EXITS = frozenset({"child_exit_2", "child_exit_3"})


def execute_job(
    job: clip_analysis_process.ClipAnalysisJob,
    python_executable: str,
    cpu_index: int,
    deadline_s: float,
    launch: Callable[..., subprocess.Popen[bytes]],
    set_process: Callable[[subprocess.Popen[bytes]], bool],
    cancelled: Callable[[], bool],
    terminate: Callable[[subprocess.Popen[bytes]], None],
    clock: Callable[[], float],
) -> ClipAnalysisStatus:
    try:
        status, _ = run_job(
            job,
            python_executable=python_executable,
            cpu_index=cpu_index,
            deadline_s=deadline_s,
            launch=launch,
            set_process=set_process,
            cancelled=cancelled,
            terminate=terminate,
            clock=clock,
        )
    except ModelLoadError as exc:
        return ClipAnalysisStatus("failed", clip_analysis_process.reason(exc))
    except Exception as exc:  # noqa: BLE001 - supervisor must remain available
        return ClipAnalysisStatus("failed", clip_analysis_process.reason(exc))
    return status


def settle_job(
    job: clip_analysis_process.ClipAnalysisJob,
    status: ClipAnalysisStatus,
    process: subprocess.Popen[bytes] | None,
    *,
    identity: Callable[[clip_analysis_process.ClipAnalysisJob], ClipAnalysisArtifactIdentity],
    terminate: Callable[[subprocess.Popen[bytes]], None],
) -> tuple[ClipAnalysisStatus, bool]:
    teardown_failed = False
    if process is not None:
        try:
            terminate(process)
        except Exception as exc:  # noqa: BLE001
            status, teardown_failed = (
                ClipAnalysisStatus("failed", clip_analysis_process.reason(exc)),
                True,
            )
    if status.reason in _TERMINAL_CHILD_EXITS:
        try:
            publish_failed_outcome(job.clip_path, identity(job), status.reason)
        except (ModelLoadError, OSError):
            LOGGER.exception(
                "clip analysis failure outcome persistence failed clip=%s", job.clip_id
            )
    return status, teardown_failed


def run_job(
    job: clip_analysis_process.ClipAnalysisJob,
    *,
    python_executable: str,
    cpu_index: int,
    deadline_s: float,
    launch: Callable[..., subprocess.Popen[bytes]],
    set_process: Callable[[subprocess.Popen[bytes]], bool],
    cancelled: Callable[[], bool],
    terminate: Callable[[subprocess.Popen[bytes]], None],
    clock: Callable[[], float] = monotonic,
) -> tuple[ClipAnalysisStatus, subprocess.Popen[bytes] | None]:
    child: clip_analysis_process.ClipAnalysisChild | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        child = clip_analysis_process.launch_child(
            launch,
            python_executable=python_executable,
            job=job,
            cpu_index=cpu_index,
        )
        process = child.process
        scratch = child.scratch
        if set_process(process):
            terminate(process)
            return ClipAnalysisStatus("failed", "cancelled"), process
        expires_at = clock() + deadline_s
        try:
            exit_code = process.wait(timeout=max(0.0, expires_at - clock()))
        except subprocess.TimeoutExpired:
            terminate(process)
            return ClipAnalysisStatus("failed", "timeout"), process
        if cancelled():
            terminate(process)
            return ClipAnalysisStatus("failed", "cancelled"), process
        if clock() >= expires_at:
            terminate(process)
            return ClipAnalysisStatus("failed", "timeout"), process
        if exit_code != 0:
            return ClipAnalysisStatus("failed", f"child_exit_{exit_code}"), process
        if scratch.stat().st_size > MAX_CLIP_ANALYSIS_OUTPUT_BYTES:
            return ClipAnalysisStatus("failed", "output_too_large"), process
        result = decode_clip_analysis(scratch.read_bytes())
        if not clip_analysis_process.identity_matches(job, result):
            return ClipAnalysisStatus("failed", "identity_mismatch"), process
        identity = ClipAnalysisArtifactIdentity(
            job.clip_id,
            job.clip_sha256,
            result.pose_model_sha256,
            result.bed_model_sha256,
            job.profile_sha256,
            result.decoder_identity,
        )
        if cancelled():
            return ClipAnalysisStatus("failed", "cancelled"), process
        if clock() >= expires_at:
            return ClipAnalysisStatus("failed", "timeout"), process
        publish_clip_analysis(job.clip_path, scratch, identity)
        return ClipAnalysisStatus("available"), process
    except (
        ClipAnalysisArtifactError,
        ModelLoadError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        return ClipAnalysisStatus("failed", clip_analysis_process.reason(exc)), process
    finally:
        if child is not None:
            child.cleanup()
