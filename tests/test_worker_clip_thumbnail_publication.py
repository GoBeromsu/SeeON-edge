from __future__ import annotations

import importlib
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator, ClipReservation
from worker.pipeline.output.evidence.clip_publication import (
    ClipPublicationMetadata,
    ClipPublisher,
    PublicationBarrier,
    PublicationStage,
    PublishedClip,
)
from worker.pipeline.output.evidence.evidence_media import MediaFacts
from worker.pipeline.output.evidence.evidence_outbox_types import EdgeEventId

START = datetime(2026, 8, 9, 1, 2, 3, tzinfo=UTC)


def _thumbnail_module() -> ModuleType:
    spec = importlib.util.find_spec("worker.adapters.encode.thumbnail")
    assert spec is not None, "worker FFmpeg thumbnail adapter is missing"
    return importlib.import_module("worker.adapters.encode.thumbnail")


def _jpeg_bytes() -> bytes:
    encoded, payload = cv2.imencode(
        ".jpg",
        np.zeros((360, 640, 3), dtype=np.uint8),
    )
    assert encoded
    return payload.tobytes()


def _metadata() -> ClipPublicationMetadata:
    return ClipPublicationMetadata(
        camera_id="camera-1",
        event_refs=(EdgeEventId("00000000-0000-4000-8000-000000000001"),),
        event_type="fall",
        clip_start_at=START,
        clip_end_at=START + timedelta(seconds=10),
        finalized_at=START + timedelta(seconds=11),
        started_at=START,
        detected_at=START + timedelta(seconds=30),
        duration_s=10.0,
        encoder="libx264",
    )


def _fixed_clip_id(value: str, camera_id: str) -> str:
    del camera_id
    return value


class _ThumbnailRunner:
    def __init__(self, payload: bytes | None = None) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []
        self.payload = _jpeg_bytes() if payload is None else payload

    def __call__(self, args: tuple[str, ...], timeout_s: float):
        self.commands.append(args)
        self.timeouts.append(timeout_s)
        return _thumbnail_module().ThumbnailCommandResult(0, self.payload)


class _ThumbnailGenerator:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure

    def generate(
        self,
        video_path: Path,
        thumbnail_path: Path,
        duration_s: float,
    ) -> Path:
        del video_path, duration_s
        if self.failure is not None:
            raise self.failure
        thumbnail_path.write_bytes(_jpeg_bytes())
        return thumbnail_path


def _ready_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generator: _ThumbnailGenerator,
    barrier: PublicationBarrier,
) -> tuple[ClipReservation, PublishedClip]:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda camera_id: "clip-ready",
    ).reserve("camera-1")
    artifact = reservation.staging_dir / "clip.mp4"
    artifact.write_bytes(b"derivative-media")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, 16, 10_000),
    )
    published = ClipPublisher(
        tmp_path,
        thumbnail_generator=generator,
        barrier=barrier,
    ).publish_ready(reservation, artifact, _metadata())
    return reservation, published


def test_ready_publication_places_thumbnail_before_unchanged_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages: list[PublicationStage] = []
    generator = _ThumbnailGenerator()

    reservation, published = _ready_publication(
        tmp_path,
        monkeypatch,
        generator,
        lambda stage, _path: stages.append(stage),
    )
    manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))

    assert stages.index(PublicationStage.THUMBNAIL_RENAMED) < stages.index(
        PublicationStage.MANIFEST_RENAMED
    )
    assert (reservation.final_dir / "thumbnail.jpg").is_file()
    assert "thumbnail" not in manifest
    assert manifest["state"] == "READY"


def test_thumbnail_failure_does_not_invalidate_ready_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _thumbnail_module()
    generator = _ThumbnailGenerator(module.ThumbnailPayloadError(0))

    reservation, published = _ready_publication(
        tmp_path,
        monkeypatch,
        generator,
        lambda _stage, _path: None,
    )
    manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))

    assert published.video_path == reservation.final_dir / "clip.mp4"
    assert manifest["state"] == "READY"
    assert manifest["video_available"] is True
    assert not (reservation.final_dir / "thumbnail.jpg").exists()


def test_expected_thumbnail_failures_publish_ready_without_log_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _thumbnail_module()
    failures = (
        module.ThumbnailTimeoutError(1.0),
        module.ThumbnailPayloadError(0),
        module.ThumbnailSecurityError("publish", "OSError"),
    )

    for index, failure in enumerate(failures):
        clip_id = f"clip-{index}"
        reservation = ClipIdAllocator(
            tmp_path,
            id_factory=partial(_fixed_clip_id, clip_id),
        ).reserve("camera-1")
        artifact = reservation.staging_dir / "clip.mp4"
        artifact.write_bytes(b"derivative-media")
        monkeypatch.setattr(
            "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
            lambda _path, **_kwargs: MediaFacts("a" * 64, 16, 10_000),
        )
        metadata = _metadata()
        metadata = ClipPublicationMetadata(
            camera_id="camera\r\nforged",
            event_refs=metadata.event_refs,
            event_type=metadata.event_type,
            clip_start_at=metadata.clip_start_at,
            clip_end_at=metadata.clip_end_at,
            finalized_at=metadata.finalized_at,
            started_at=metadata.started_at,
            duration_s=metadata.duration_s,
            encoder=metadata.encoder,
            detected_at=metadata.detected_at,
        )

        published = ClipPublisher(
            tmp_path,
            thumbnail_generator=_ThumbnailGenerator(failure),
        ).publish_ready(reservation, artifact, metadata)

        manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))
        assert manifest["state"] == "READY"

    messages = [record.getMessage() for record in caplog.records]
    assert messages
    assert all("\r" not in message and "\n" not in message for message in messages)


def test_unknown_thumbnail_failure_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda camera_id: "clip-unknown",
    ).reserve("camera-1")
    artifact = reservation.staging_dir / "clip.mp4"
    artifact.write_bytes(b"derivative-media")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, 16, 10_000),
    )

    with pytest.raises(RuntimeError, match="unknown thumbnail bug"):
        ClipPublisher(
            tmp_path,
            thumbnail_generator=_ThumbnailGenerator(RuntimeError("unknown thumbnail bug")),
        ).publish_ready(reservation, artifact, _metadata())


def test_publication_retry_after_thumbnail_rename_reuses_durable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _thumbnail_module()
    runner = _ThumbnailRunner()
    generator = module.FFmpegThumbnailGenerator(runner=runner)
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda camera_id: "clip-retry",
    ).reserve("camera-1")
    artifact = reservation.staging_dir / "clip.mp4"
    artifact.write_bytes(b"derivative-media")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, 16, 10_000),
    )
    interrupted = False

    def interrupt(stage: PublicationStage, _path: Path) -> None:
        nonlocal interrupted
        if stage is PublicationStage.THUMBNAIL_RENAMED and not interrupted:
            interrupted = True
            raise OSError("interrupted after thumbnail rename")

    with pytest.raises(OSError, match="thumbnail rename"):
        _ = ClipPublisher(
            tmp_path,
            thumbnail_generator=generator,
            barrier=interrupt,
        ).publish_ready(reservation, artifact, _metadata())

    assert (reservation.final_dir / "clip.mp4").is_file()
    assert (reservation.final_dir / "thumbnail.jpg").is_file()
    assert not (reservation.final_dir / "manifest.json").exists()

    published = ClipPublisher(
        tmp_path,
        thumbnail_generator=generator,
    ).publish_ready(reservation, artifact, _metadata())

    assert published.manifest_path.is_file()
    assert len(runner.commands) == 1
