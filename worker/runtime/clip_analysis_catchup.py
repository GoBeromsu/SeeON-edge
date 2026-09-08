"""One-shot, bounded boot catch-up for published clips awaiting analysis."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected
from worker.interfaces.clip_analysis import ClipAnalysisSupervisor
from worker.pipeline.output.evidence.clip_identity import bounded_clip_roots
from worker.pipeline.output.evidence.evidence_manifest import (
    ClipEvidenceError,
    parse_manifest_content,
)
from worker.pipeline.output.evidence.manifest_models import ReadyClipManifest

LOGGER = logging.getLogger(__name__)
_CANDIDATE_LIMIT = 256
_DISCOVERY_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class _Candidate:
    clip_id: str
    clip_path: Path
    clip_sha256: str
    size_bytes: int
    duration_ms: int
    mtime: float


def start_clip_analysis_catchup(
    store_dir: Path, supervisor: ClipAnalysisSupervisor, stop: threading.Event | None = None
) -> threading.Thread:
    thread = threading.Thread(
        target=catch_up_clip_analysis,
        args=(store_dir, supervisor, stop or threading.Event()),
        name="clip-analysis-catchup",
        daemon=True,
    )
    thread.start()
    return thread


def catch_up_clip_analysis(
    store_dir: Path, supervisor: ClipAnalysisSupervisor, stop: threading.Event | None = None
) -> None:
    stop = stop or threading.Event()
    deadline = monotonic() + _DISCOVERY_SECONDS
    candidates = _ready_candidates(store_dir, stop, deadline)
    accepted = skipped = rejected = 0
    for candidate in sorted(candidates, key=lambda item: item.mtime, reverse=True):
        if stop.is_set() or monotonic() >= deadline:
            break
        try:
            admission = supervisor.enqueue(
                candidate.clip_id,
                candidate.clip_path,
                candidate.clip_sha256,
                size_bytes=candidate.size_bytes,
                duration_ms=candidate.duration_ms,
                width=0,
                height=0,
            )
        except ClipAnalysisRejected:
            rejected += 1
            continue
        if admission in ("queue_full", "stopped"):
            break
        accepted += 1
    LOGGER.info(
        "clip analysis catch-up complete accepted=%d skipped=%d rejected=%d discovered=%d",
        accepted,
        skipped,
        rejected,
        len(candidates),
    )


def _ready_candidates(store_dir: Path, stop: threading.Event, deadline: float) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for clips_root in bounded_clip_roots(store_dir):
        if stop.is_set() or monotonic() >= deadline:
            break
        try:
            directories = clips_root.iterdir()
            for clip_dir in directories:
                if stop.is_set() or monotonic() >= deadline or len(candidates) >= _CANDIDATE_LIMIT:
                    return candidates
                manifest_path = clip_dir / "manifest.json"
                clip_path = clip_dir / "clip.mp4"
                if not clip_dir.is_dir() or not clip_path.is_file():
                    continue
                candidate = _manifest_candidate(clip_path, manifest_path)
                if candidate is not None:
                    candidates.append(candidate)
        except OSError:
            continue
    return candidates


def _manifest_candidate(clip_path: Path, manifest_path: Path) -> _Candidate | None:
    try:
        manifest, _, _ = parse_manifest_content(manifest_path)
        mtime = manifest_path.stat().st_mtime
    except (ClipEvidenceError, OSError):
        return None
    if not isinstance(manifest, ReadyClipManifest) or manifest.clip_id != clip_path.parent.name:
        return None
    return _Candidate(
        manifest.clip_id,
        clip_path,
        manifest.sha256,
        manifest.size_bytes,
        manifest.duration_ms,
        mtime,
    )
