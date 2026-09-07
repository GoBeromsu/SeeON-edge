from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from shared.events.clip_analysis_wire import (
    ClipAnalysisFrame,
    ClipAnalysisResult,
    ClipAnalysisTimeBase,
    encode_clip_analysis,
)
from worker.pipeline.output.evidence.clip_analysis_artifact import (
    ClipAnalysisArtifactError,
    ClipAnalysisArtifactIdentity,
    artifact_path,
    load_clip_analysis,
    publish_clip_analysis,
)

_SHA = "a" * 64


def _identity(decoder: str = "pyav-16/h264") -> ClipAnalysisArtifactIdentity:
    return ClipAnalysisArtifactIdentity("event-1", _SHA, "b" * 64, "c" * 64, "d" * 64, decoder)


def _payload(identity: ClipAnalysisArtifactIdentity) -> bytes:
    return encode_clip_analysis(
        ClipAnalysisResult(
            source="clip_reanalysis",
            clip_id=identity.clip_id,
            clip_sha256=identity.clip_sha256,
            pose_model_sha256=identity.pose_model_sha256,
            bed_model_sha256=identity.bed_model_sha256,
            decoder_identity=identity.decoder_identity,
            analysis_profile_sha256=identity.analysis_profile_sha256,
            image_width=10,
            image_height=10,
            time_base=ClipAnalysisTimeBase(1, 1),
            frames=(ClipAnalysisFrame(0, "no_evidence"),),
        )
    )


def test_publish_once_and_load_identity_bound_artifact(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    scratch = tmp_path / "out.json"
    identity = _identity()
    scratch.write_bytes(_payload(identity))

    first = publish_clip_analysis(clip, scratch, identity)
    scratch.write_bytes(_payload(identity))
    second = publish_clip_analysis(clip, scratch, identity)

    assert first == second == artifact_path(clip)
    assert load_clip_analysis(clip, identity) is not None
    assert first.read_bytes() == _payload(identity)
    assert (
        first.with_name("clip.analysis.json.sha256").read_text().strip()
        == sha256(_payload(identity)).hexdigest()
    )


def test_identity_mismatch_is_rejected_and_never_served(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    scratch = tmp_path / "out.json"
    identity = _identity()
    scratch.write_bytes(_payload(identity))
    publish_clip_analysis(clip, scratch, identity)

    assert load_clip_analysis(clip, _identity("pyav-16/hevc")) is None
    with pytest.raises(ClipAnalysisArtifactError, match="identity_mismatch"):
        publish_clip_analysis(clip, scratch, _identity("pyav-16/hevc"))
