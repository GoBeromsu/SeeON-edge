from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.features.clips.catalog import strict_manifest_records
from backend.app.features.clips.store import ClipStore
from worker.pipeline.output.evidence import clip_publication
from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator, ClipReservation
from worker.pipeline.output.evidence.clip_publication import (
    ClipPublicationMetadata,
    ClipPublisher,
    ClipTimeOrigin,
    PublicationStage,
)
from worker.pipeline.output.evidence.evidence_media import MediaFacts
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    EdgeEventId,
    EvidenceReasonCode,
)

EVENT_ONE = EdgeEventId("00000000-0000-4000-8000-000000000001")
EVENT_TWO = EdgeEventId("00000000-0000-4000-8000-000000000002")
START = datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC)
RUNTIME_MANIFEST_SHA256 = "b" * 64


def _metadata() -> ClipPublicationMetadata:
    return ClipPublicationMetadata(
        camera_id="camera-1",
        event_refs=(EVENT_ONE, EVENT_ONE, EVENT_TWO),
        event_type="fall",
        clip_start_at=START,
        clip_end_at=START + timedelta(seconds=1),
        finalized_at=START + timedelta(seconds=2),
        started_at=START,
        detected_at=START + timedelta(seconds=30),
        duration_s=1.0,
        encoder="libx264",
        runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
    )


def test_process_kill_after_manifest_reconstructs_exact_event_outcomes(tmp_path: Path) -> None:
    script = textwrap.dedent(
        f"""
        import os
        from datetime import UTC, datetime, timedelta
        from pathlib import Path
        from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator
        from worker.pipeline.output.evidence.clip_publication import (
            ClipPublicationMetadata, ClipPublisher, PublicationStage,
        )
        from worker.pipeline.output.evidence.evidence_outbox_types import (
            EdgeEventId, EvidenceReasonCode,
        )
        root = Path({str(tmp_path)!r})
        reservation = ClipIdAllocator(
            root, id_factory=lambda _camera: 'killed-clip'
        ).reserve('camera-1')
        start = datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC)
        metadata = ClipPublicationMetadata(
            'camera-1',
            (EdgeEventId('{EVENT_ONE}'), EdgeEventId('{EVENT_TWO}')),
            'fall', start, start + timedelta(seconds=1), start + timedelta(seconds=2),
            start, start + timedelta(seconds=30), 1.0, 'libx264', '{RUNTIME_MANIFEST_SHA256}',
        )
        def barrier(stage, _path):
            if stage is PublicationStage.MANIFEST_RENAMED:
                os._exit(91)
        ClipPublisher(root, barrier=barrier).publish_unavailable(
            reservation, metadata, EvidenceReasonCode.ENCODER_FAILED,
        )
        """
    )
    killed = subprocess.run([sys.executable, "-c", script], check=False)
    assert killed.returncode == 91
    reservation = ClipReservation(
        ClipId("killed-clip"),
        "camera-1",
        tmp_path / "clips/.staging/killed-clip",
        tmp_path / "clips/killed-clip",
    )

    publisher = ClipPublisher(tmp_path)
    _ = publisher.publish_unavailable(reservation, _metadata(), EvidenceReasonCode.ENCODER_FAILED)
    _ = publisher.publish_unavailable(reservation, _metadata(), EvidenceReasonCode.ENCODER_FAILED)

    outcomes = tuple((reservation.final_dir / "terminal-outcomes").glob("*.json"))
    assert len(outcomes) == 2
    assert {json.loads(path.read_text())["event_id"] for path in outcomes} == {
        str(EVENT_ONE),
        str(EVENT_TWO),
    }


def test_reservation_skips_collisions_and_keeps_the_caller_visible_identity(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "clips" / "collision"
    existing.mkdir(parents=True)
    ids = iter(("collision", "reserved"))
    allocator = ClipIdAllocator(tmp_path, id_factory=lambda _camera: next(ids))

    reservation = allocator.reserve("camera-1")

    assert reservation.clip_id == "reserved"
    assert reservation.staging_dir.is_dir()
    assert existing.is_dir()
    assert allocator.collision_count == 1


@pytest.mark.parametrize(
    "failure_stage",
    (
        PublicationStage.MEDIA_FSYNCED,
        PublicationStage.MEDIA_RENAMED,
        PublicationStage.MANIFEST_RENAMED,
    ),
)
def test_publication_retry_recovers_each_durability_boundary_with_same_clip_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: PublicationStage,
) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "stable-clip-id",
    ).reserve("camera-1")
    artifact = reservation.staging_dir / "clip.mp4"
    artifact.write_bytes(b"derivative-media")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, len(b"derivative-media"), 1000),
    )
    failed = False

    def interrupt(stage: PublicationStage, _path: Path) -> None:
        nonlocal failed
        if stage is failure_stage and not failed:
            failed = True
            raise OSError(f"interrupted after {stage}")

    with pytest.raises(OSError, match="interrupted"):
        ClipPublisher(tmp_path, barrier=interrupt).publish_ready(
            reservation,
            artifact,
            _metadata(),
        )

    published = ClipPublisher(tmp_path).publish_ready(
        reservation,
        artifact,
        _metadata(),
    )
    payload = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    records = strict_manifest_records(ClipStore(tmp_path))

    assert failed
    assert published.clip_id == "stable-clip-id"
    assert payload["event_refs"] == [EVENT_ONE, EVENT_TWO]
    assert payload["state"] == "READY"
    assert payload["path"] == "clips/stable-clip-id/clip.mp4"
    assert payload["runtime_manifest_sha256"] == RUNTIME_MANIFEST_SHA256
    assert records[0].manifest.clip_id == "stable-clip-id"
    assert records[0].payload["runtime_manifest_sha256"] == RUNTIME_MANIFEST_SHA256
    assert not reservation.staging_dir.exists()


def test_strict_manifest_accepts_exact_remux_translation_and_rejects_nonuniform_ticks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "translation-clip",
    ).reserve("camera-1")
    artifact = reservation.staging_dir / "clip.mp4"
    artifact.write_bytes(b"source-packets")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, len(b"source-packets"), 1000),
    )
    metadata = replace(
        _metadata(),
        source_media={
            "configuration_id": "configuration-1",
            "timestamp_translation_seconds": "-1/1536",
            "streams": [
                {
                    "index": 0,
                    "time_base": "1/15360",
                    "packet_count": 25,
                    "timestamp_translation_ticks": -10,
                },
                {
                    "index": 1,
                    "time_base": "1/48000",
                    "packet_count": 0,
                    "timestamp_translation_ticks": None,
                },
            ],
        },
    )
    published = ClipPublisher(tmp_path).publish_ready(reservation, artifact, metadata)

    records = strict_manifest_records(ClipStore(tmp_path))
    assert records[0].payload["source_media"] == metadata.source_media

    payload = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    payload["source_media"]["streams"][0]["timestamp_translation_ticks"] = -11
    published.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="nonuniform remux timestamp translation"):
        strict_manifest_records(ClipStore(tmp_path))


@pytest.mark.parametrize("invalid", ("A" * 64, "a" * 63, "a" * 65))
def test_publication_metadata_rejects_noncanonical_runtime_manifest_hash(
    invalid: str,
) -> None:
    with pytest.raises(ValueError, match="runtime_manifest_sha256"):
        replace(_metadata(), runtime_manifest_sha256=invalid)


def test_publication_omits_runtime_manifest_when_explicitly_absent(tmp_path: Path) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "legacy-no-runtime-manifest",
    ).reserve("camera-1")

    published = ClipPublisher(tmp_path).publish_unavailable(
        reservation,
        replace(_metadata(), runtime_manifest_sha256=None),
        EvidenceReasonCode.ENCODER_FAILED,
    )
    payload = json.loads(published.manifest_path.read_text(encoding="utf-8"))

    assert "runtime_manifest_sha256" not in payload


def test_publication_records_deterministic_event_pts_to_media_time_mapping(
    tmp_path: Path,
) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "time-origin-clip",
    ).reserve("camera-1")
    metadata = _metadata()
    metadata = ClipPublicationMetadata(
        camera_id=metadata.camera_id,
        event_refs=metadata.event_refs,
        event_type=metadata.event_type,
        clip_start_at=metadata.clip_start_at,
        clip_end_at=metadata.clip_end_at,
        finalized_at=metadata.finalized_at,
        started_at=metadata.started_at,
        detected_at=metadata.detected_at,
        duration_s=metadata.duration_s,
        encoder=metadata.encoder,
        runtime_manifest_sha256=metadata.runtime_manifest_sha256,
        time_origin=ClipTimeOrigin(
            worker_boot_id="boot-1",
            camera_id="camera-1",
            stream_epoch=4,
            generation=7,
            media_origin_pts_sec=40.0,
            event_pts_sec=42.25,
            requested_start_pts_sec=12.25,
            requested_end_pts_sec=72.25,
        ),
    )

    published = ClipPublisher(tmp_path).publish_unavailable(
        reservation,
        metadata,
        EvidenceReasonCode.ENCODER_FAILED,
    )
    payload = json.loads(published.manifest_path.read_text(encoding="utf-8"))

    assert payload["time_origin"] == {
        "camera_id": "camera-1",
        "event_media_time_ms": 2250.0,
        "event_pts_sec": 42.25,
        "generation": 7,
        "media_origin_pts_sec": 40.0,
        "requested_end_pts_sec": 72.25,
        "requested_start_pts_sec": 12.25,
        "stream_epoch": 4,
        "worker_boot_id": "boot-1",
    }


def test_unavailable_publication_persists_reason_without_video(tmp_path: Path) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "unavailable-clip-id",
    ).reserve("camera-1")

    published = ClipPublisher(tmp_path).publish_unavailable(
        reservation,
        _metadata(),
        EvidenceReasonCode.ENCODER_FAILED,
    )
    payload = json.loads(published.manifest_path.read_text(encoding="utf-8"))

    assert published.video_path is None
    assert payload["state"] == "UNAVAILABLE"
    assert payload["reason_code"] == "ENCODER_FAILED"
    assert payload["detected_at"] == "2026-07-16T01:02:33Z"
    assert all(not key.startswith("scene") for key in payload)
    assert payload["path"] is None
    assert payload["video_available"] is False
    assert not (reservation.final_dir / "clip.mp4").exists()
    assert not reservation.staging_dir.exists()


def test_ready_publication_fsyncs_before_renames_and_staging_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "durable-clip-id",
    ).reserve("camera-1")
    artifact = reservation.staging_dir / "derivative.mp4"
    artifact.write_bytes(b"derivative-media")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, len(b"derivative-media"), 1000),
    )
    operations: list[tuple[str, str, str]] = []
    real_replace = os.replace
    real_rmtree = shutil.rmtree

    def record_file(path: Path) -> None:
        operations.append(("fsync-file", path.name, ""))

    def record_directory(path: Path) -> None:
        operations.append(("fsync-directory", path.name, ""))

    def record_replace(source: Path, target: Path) -> None:
        operations.append(("replace", source.name, target.name))
        real_replace(source, target)

    def record_rmtree(path: Path) -> None:
        operations.append(("rmtree", path.name, ""))
        real_rmtree(path)

    monkeypatch.setattr(clip_publication, "fsync_file", record_file)
    monkeypatch.setattr(clip_publication, "fsync_directory", record_directory)
    monkeypatch.setattr(clip_publication.os, "replace", record_replace)
    monkeypatch.setattr(clip_publication.shutil, "rmtree", record_rmtree)

    _ = ClipPublisher(tmp_path).publish_ready(reservation, artifact, _metadata())

    assert operations[:8] == [
        ("fsync-directory", "clips", ""),
        ("fsync-file", "derivative.mp4", ""),
        ("replace", "derivative.mp4", "clip.mp4"),
        ("fsync-file", "clip.mp4", ""),
        ("fsync-directory", "durable-clip-id", ""),
        ("replace", "manifest.json.tmp", "manifest.json"),
        ("fsync-file", "manifest.json", ""),
        ("fsync-directory", "durable-clip-id", ""),
    ]
    terminal_targets = [
        target for operation, _source, target in operations if operation == "replace"
    ]
    assert "terminal-outcome.json" in terminal_targets
    assert sum(target.endswith(".json") for target in terminal_targets) == 4
    assert operations[-2:] == [
        ("rmtree", "durable-clip-id", ""),
        ("fsync-directory", ".staging", ""),
    ]
