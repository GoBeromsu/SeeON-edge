"""One-shot boot catch-up for published clips awaiting analysis."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from time import monotonic

from worker.adapters.model.clip_reanalysis import ClipAnalysisRejected
from worker.interfaces.clip_analysis import ClipAnalysisSupervisor
from worker.pipeline.output._clip_analysis_http import manifest_facts
from worker.pipeline.output.evidence.clip_identity import bounded_clip_roots
from worker.pipeline.output.evidence.evidence_manifest import (
    ClipEvidenceError,
    parse_manifest_content,
)
from worker.pipeline.output.evidence.manifest_models import ReadyClipManifest

LOGGER = logging.getLogger(__name__)


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
    if stop is None:
        stop = threading.Event()
    deadline = monotonic() + 30
    candidates = sorted(
        _ready_clips(store_dir),
        key=lambda path: path.with_name("manifest.json").stat().st_mtime,
        reverse=True,
    )
    accepted = skipped = rejected = 0
    for clip_path in candidates[:256]:
        if stop.is_set() or monotonic() >= deadline:
            break
        try:
            facts = manifest_facts(clip_path)
        except OSError:
            skipped += 1
            continue
        if facts is None:
            skipped += 1
            continue
        clip_sha256, size_bytes, duration_ms, width, height = facts
        try:
            queued = supervisor.enqueue(
                clip_path.parent.name,
                clip_path,
                clip_sha256,
                size_bytes=size_bytes,
                duration_ms=duration_ms,
                width=width,
                height=height,
            )
        except ClipAnalysisRejected:
            rejected += 1
            continue
        if queued in ("queue_full", "stopped"):
            break
        accepted += 1
    LOGGER.info(
        "clip analysis catch-up complete accepted=%d skipped=%d rejected=%d remaining=%d",
        accepted,
        skipped,
        rejected,
        max(0, len(candidates) - accepted - skipped - rejected),
    )


def _ready_clips(store_dir: Path) -> list[Path]:
    clips: list[Path] = []
    for clips_root in bounded_clip_roots(store_dir):
        for manifest_path in clips_root.glob("*/manifest.json"):
            clip_path = manifest_path.with_name("clip.mp4")
            if not clip_path.is_file():
                continue
            try:
                manifest, _, _ = parse_manifest_content(manifest_path)
            except (ClipEvidenceError, OSError):
                continue
            if (
                isinstance(manifest, ReadyClipManifest)
                and manifest.clip_id == clip_path.parent.name
            ):
                clips.append(clip_path)
    return clips
