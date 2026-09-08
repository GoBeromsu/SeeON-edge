"""Pre-execution admission for stored clip analysis."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected
from worker.adapters.model.errors import ModelLoadError
from worker.pipeline.output.evidence.clip_analysis_artifact import (
    ClipAnalysisArtifactIdentity,
    has_current_artifact,
    has_failed_outcome,
)
from worker.runtime import clip_analysis_process
from worker.runtime.clip_analysis_subprocess import (
    ClipAnalysisPdeathsigUnavailable,
    require_pdeathsig,
)
from worker.runtime.clip_analysis_supervisor_status import ClipAnalysisStatus

LOGGER = logging.getLogger(__name__)


class ClipAnalysisSupervisorError(RuntimeError):
    """A supervisor operation could not safely start."""


def validate_runtime(cpu_index: int | None) -> int:
    """Validate the process-wide execution boundary before serving admissions."""
    if cpu_index is None:
        raise ClipAnalysisSupervisorError("clip_analysis_cpu_required")
    available = os.sched_getaffinity(0)
    if len(available) <= 1:
        raise ClipAnalysisSupervisorError("clip_analysis_cpu_unavailable")
    if cpu_index not in available:
        raise ClipAnalysisSupervisorError("clip_analysis_cpu_invalid")
    try:
        require_pdeathsig()
    except ClipAnalysisPdeathsigUnavailable as exc:
        raise ClipAnalysisSupervisorError("pdeathsig_unavailable") from exc
    return cpu_index


def log_eviction(clip_id: str) -> None:
    LOGGER.warning("clip analysis evicted automatic clip=%s for manual admission", clip_id)


def build_job(
    clip_id: str,
    clip_path: Path,
    clip_sha256: str,
    pose_model_path: Path,
    bed_model_path: Path,
    profile: object,
    profile_sha256: str,
    size_bytes: int,
    duration_ms: int,
    width: int,
    height: int,
    front: bool,
    pose_model_sha256: str,
    bed_model_sha256: str,
    generation: int,
) -> clip_analysis_process.ClipAnalysisJob:
    return clip_analysis_process.ClipAnalysisJob(
        clip_id,
        clip_path,
        clip_sha256,
        pose_model_path,
        bed_model_path,
        profile,
        profile_sha256,
        "",
        front,
        size_bytes,
        duration_ms,
        width,
        height,
        pose_model_sha256,
        bed_model_sha256,
        generation,
    )


def prepare_job(
    pending: clip_analysis_process.ClipAnalysisJob,
    *,
    profile: object,
    pose_model_path: Path,
    bed_model_path: Path,
    profile_sha256: str,
    probe: Callable[[Path], clip_analysis_process.ClipMediaFacts],
) -> tuple[clip_analysis_process.ClipAnalysisJob | None, ClipAnalysisStatus]:
    """Probe and validate a queued clip before it occupies the execution slot."""
    try:
        facts = probe(pending.clip_path)
        _pre_admission(profile, pending.size_bytes, pending.duration_ms, facts.width, facts.height)
        job = replace(
            pending,
            decoder_identity=facts.decoder_identity,
            width=facts.width,
            height=facts.height,
        )
        identity = artifact_identity(job, profile_sha256)
        if has_current_artifact(
            job.clip_path.parent,
            clip_id=identity.clip_id,
            clip_sha256=identity.clip_sha256,
            pose_model_sha256=identity.pose_model_sha256,
            bed_model_sha256=identity.bed_model_sha256,
            analysis_profile_sha256=identity.analysis_profile_sha256,
            decoder_identity=identity.decoder_identity,
        ):
            return None, ClipAnalysisStatus("available")
        if not job.front and has_failed_outcome(job.clip_path, identity):
            return None, ClipAnalysisStatus("failed", "previous_failure")
        return job, ClipAnalysisStatus("running")
    except (ClipAnalysisRejected, ModelLoadError, OSError, ValueError) as exc:
        return None, ClipAnalysisStatus("failed", clip_analysis_process.reason(exc))


def artifact_identity(
    job: clip_analysis_process.ClipAnalysisJob,
    profile_sha256: str,
) -> ClipAnalysisArtifactIdentity:
    return ClipAnalysisArtifactIdentity(
        job.clip_id,
        job.clip_sha256,
        job.pose_model_sha256,
        job.bed_model_sha256,
        profile_sha256,
        job.decoder_identity,
    )


def _pre_admission(
    profile: object, size_bytes: int, duration_ms: int, width: int, height: int
) -> None:
    if any(
        not isinstance(value, int) or value < 0
        for value in (size_bytes, duration_ms, width, height)
    ):
        raise ClipAnalysisRejected("manifest_facts")
    if size_bytes > profile.max_input_bytes:
        raise ClipAnalysisRejected("input_bytes")
    if duration_ms > profile.max_duration_s * 1000:
        raise ClipAnalysisRejected("duration")
    if width == 0 or height == 0 or width * height > profile.max_pixels:
        raise ClipAnalysisRejected("resolution")
