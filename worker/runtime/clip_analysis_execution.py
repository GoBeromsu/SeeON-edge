"""Child execution stage for stored clip analysis."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from time import monotonic

from shared.events.clip_analysis_wire import MAX_CLIP_ANALYSIS_OUTPUT_BYTES, decode_clip_analysis
from worker.pipeline.output.evidence.clip_analysis_artifact import (
    ClipAnalysisArtifactError,
    ClipAnalysisArtifactIdentity,
    publish_clip_analysis,
)
from worker.runtime import clip_analysis_process
from worker.runtime.clip_analysis_supervisor_status import ClipAnalysisStatus


def run_job(
    job: clip_analysis_process.ClipAnalysisJob,
    *,
    python_executable: str,
    cpu_index: int,
    deadline_s: float,
    launch: Callable[..., subprocess.Popen[bytes]],
    set_process: Callable[[subprocess.Popen[bytes]], None],
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
        set_process(process)
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
    except (ClipAnalysisArtifactError, OSError, ValueError, subprocess.SubprocessError) as exc:
        return ClipAnalysisStatus("failed", clip_analysis_process.reason(exc)), process
    finally:
        if child is not None:
            child.cleanup()
