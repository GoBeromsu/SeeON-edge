"""Terminal corruption publication after partial clip commit."""

from __future__ import annotations

import hashlib

from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.pipeline.output.evidence.clip_publication_types import (
    ClipPublicationMetadata,
    PublishedClip,
)
from worker.pipeline.output.evidence.evidence_manifest import parse_manifest
from worker.pipeline.output.evidence.terminal_outcome import (
    TerminalClipOutcome,
    TerminalClipState,
    commit_corrupt_terminal_outcome,
)


def publish_existing_corrupt(
    reservation: ClipReservation,
    metadata: ClipPublicationMetadata,
) -> PublishedClip | None:
    manifest_path = reservation.final_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = parse_manifest(manifest_path)
    _ = commit_corrupt_terminal_outcome(
        reservation.final_dir,
        TerminalClipOutcome(
            str(reservation.clip_id),
            tuple(str(value) for value in metadata.event_refs),
            TerminalClipState.CORRUPT,
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        ),
    )
    video_path = reservation.final_dir / "clip.mp4"
    return PublishedClip(
        reservation.clip_id,
        manifest,
        manifest_path,
        video_path if video_path.is_file() else None,
    )


__all__ = ["publish_existing_corrupt"]
