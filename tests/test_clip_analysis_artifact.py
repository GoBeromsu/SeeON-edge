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
    has_current_artifact,
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

    assert first == second == artifact_path(clip, identity)
    assert first.read_bytes() == _payload(identity)
    assert (
        first.with_name(f"{first.name}.sha256").read_text().strip()
        == sha256(_payload(identity)).hexdigest()
    )


def test_current_artifact_matches_exact_decoder_identity(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    scratch = tmp_path / "out.json"
    identity = _identity()
    scratch.write_bytes(_payload(identity))
    publish_clip_analysis(clip, scratch, identity)

    assert has_current_artifact(
        tmp_path,
        clip_id=identity.clip_id,
        clip_sha256=identity.clip_sha256,
        pose_model_sha256=identity.pose_model_sha256,
        bed_model_sha256=identity.bed_model_sha256,
        analysis_profile_sha256=identity.analysis_profile_sha256,
        decoder_identity=identity.decoder_identity,
    )
    assert not has_current_artifact(
        tmp_path,
        clip_id=identity.clip_id,
        clip_sha256=identity.clip_sha256,
        pose_model_sha256=identity.pose_model_sha256,
        bed_model_sha256=identity.bed_model_sha256,
        analysis_profile_sha256=identity.analysis_profile_sha256,
        decoder_identity="pyav-16/hevc",
    )


def test_identity_mismatch_is_rejected_and_never_served(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    scratch = tmp_path / "out.json"
    identity = _identity()
    scratch.write_bytes(_payload(identity))
    publish_clip_analysis(clip, scratch, identity)

    with pytest.raises(ClipAnalysisArtifactError, match="identity_mismatch"):
        publish_clip_analysis(clip, scratch, _identity("pyav-16/hevc"))


def test_sidecarless_matching_target_is_repaired_without_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "clip.mp4"
    scratch = tmp_path / "out.json"
    identity = _identity()
    payload = _payload(identity)
    scratch.write_bytes(payload)
    target = artifact_path(clip, identity)
    target.write_bytes(payload)
    writes: list[Path] = []
    from worker.pipeline.output.evidence import clip_analysis_artifact

    original = clip_analysis_artifact._atomic_write

    def record(path: Path, content: bytes) -> None:
        writes.append(path)
        original(path, content)

    monkeypatch.setattr(clip_analysis_artifact, "_atomic_write", record)

    assert publish_clip_analysis(clip, scratch, identity) == target
    assert writes == [target.with_name(f"{target.name}.sha256")]


def test_corrupt_target_is_quarantined_before_republish(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    scratch = tmp_path / "out.json"
    identity = _identity()
    scratch.write_bytes(_payload(identity))
    target = artifact_path(clip, identity)
    target.write_bytes(b"corrupt")

    publish_clip_analysis(clip, scratch, identity)

    assert target.read_bytes() == _payload(identity)
    quarantined = tuple(tmp_path.glob(f"{target.name}.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"corrupt"


def test_verified_different_target_is_identity_collision_and_never_overwritten(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip.mp4"
    scratch = tmp_path / "out.json"
    identity = _identity()
    scratch.write_bytes(_payload(identity))
    target = artifact_path(clip, identity)
    foreign = _payload(_identity("pyav-16/hevc"))
    target.write_bytes(foreign)
    target.with_name(f"{target.name}.sha256").write_text(sha256(foreign).hexdigest())

    with pytest.raises(ClipAnalysisArtifactError, match="identity_collision"):
        publish_clip_analysis(clip, scratch, identity)

    assert target.read_bytes() == foreign
