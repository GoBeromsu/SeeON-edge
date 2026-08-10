"""Config/maintenance and macOS-environment-floor coverage for the new
``worker.pipeline.output.evidence.clip_recorder.ClipRecorder``, replacing the
deleted ``edge.evidence.clip_recorder`` module this file used to test.

The old ``edge`` ``ClipRecorder`` was a single ~1000-line actor that owned
frame buffering, ffmpeg segment writing, event coalescing, clip finalization,
manifest publication, and retention in one class, tested by white-box
reaching into its private attributes (``_state``, ``_active_clips``,
``_open_clip``, ``_handle_event``, ...). The new architecture decomposes the
same responsibilities into single-purpose collaborators under
``worker/pipeline/output/evidence/``: ``ClipAdmission`` (frame/event intake
and queueing), ``ClipActor`` (coalescing, expiry, finalize orchestration),
``ClipRecordingCoordinator``/``FFmpegSegmentEncoder``/``FFmpegConcatFinalizer``
(segment capture and remux), ``ClipPublisher`` (durable publish), and
``EvidenceRetention``/``ClipMaintenance`` (rotation/pressure). ``ClipRecorder``
itself is now a thin queue-driven facade over these. Most of the original
white-box tests are superseded by tests written directly against the
collaborator that now owns the property; this file ports only what still has
no equivalent coverage, plus the two tests that pin this repo's macOS
environment floor (no ``/proc``, so ``os.readlink("/proc/self/fd/N")``
fails) so CI continues to fail closed on the same signature rather than
silently losing the regression check.

Ported as-is (config/maintenance surface unchanged in the new module):

* ``test_retention_defaults_to_60_days``
* ``test_clip_window_defaults_to_30_seconds_before_and_after``
* ``test_retention_environment_enforces_60_day_minimum``
* ``test_direct_config_below_60_days_is_clamped``
* ``test_clip_recorder_start_sweeps_old_staging_directories``
* ``test_legacy_retention_clock_uses_manifest_mtime_not_started_at``

Preserved to pin the macOS ``/proc`` environment floor (see module docstring
above and each test's own docstring for the exact failure signature this
repo's sandboxed macOS CI observes; both are rooted in ``os.readlink(
"/proc/self/fd/N")`` raising ``FileNotFoundError`` since macOS has no
``/proc``, propagating up as an ``OSError`` that ``ClipActor._finalize``
(``worker/pipeline/output/evidence/clip_actor.py:212-218``) catches and
counts as a failed write rather than a partial manifest -- a stricter
behavior than edge's, which wrote a manifest even on this failure):

* ``test_clip_recorder_finalizes_atomic_manifest_with_pre_and_post_window``
* ``test_clip_recorder_fsyncs_media_and_manifest_before_staging_cleanup``

Superseded (same property, different call site, cited by file:line):

* ``test_clip_recorder_coalesces_event_refs_into_one_active_clip`` and
  ``test_clip_recorder_keeps_coalesced_event_refs_ordered_and_unique`` --
  ``tests/test_worker_clip_actor.py::
  test_coalesced_refs_remain_ordered_and_unique`` (line 188).

* ``test_clip_recorder_drops_attach_only_event_after_queued_frame_finalizes``
  -- ``tests/test_worker_clip_actor.py::
  test_attach_only_event_after_frame_finalization_cannot_mutate_clip``
  (line 219).

* ``test_clip_recorder_reserves_collision_free_id_and_preserves_identity`` --
  ``tests/test_worker_clip_publication.py::
  test_reservation_skips_collisions_and_keeps_the_caller_visible_identity``
  (line 42).

* ``test_clip_recorder_stop_works_when_the_queue_is_full`` --
  ``tests/test_worker_clip_recorder_lifecycle.py::
  test_stop_drains_full_queue_after_blocked_frame_write`` (line 164): stop()
  still drains a blocked/full-queue in-flight write and publishes before the
  thread exits.

* ``test_clip_recorder_stop_drains_accepted_event_and_writes_manifest`` --
  ``tests/test_worker_clip_recorder.py::
  test_stop_drains_an_accepted_event_before_releasing_store_lock``
  (line 164).

* ``test_clip_recorder_rolls_back_reserved_clip_id_after_open_failure`` --
  ``tests/test_worker_clip_recorder_lifecycle.py::
  test_recorder_cleans_failed_reservation_before_retrying_event``
  (line 193): a failed reservation is cleaned up and the next event gets a
  fresh, distinct clip id.

* ``test_clip_recorder_coalesce_keeps_first_event_cutoff`` --
  ``tests/test_worker_clip_actor.py::
  test_coalesced_event_keeps_the_first_event_cutoff`` (line 203).

* ``test_clip_recorder_uses_registered_camera_fps`` --
  ``tests/test_worker_clip_recorder.py::
  test_camera_fps_registration_reaches_coordinator`` (line 184).

* ``test_clip_recorder_forces_finalize_after_stream_time_resets`` --
  ``tests/test_worker_clip_actor.py::
  test_expiry_forces_finalize_when_stream_time_stalls`` (line 234).

* ``test_clip_recorder_writes_manifest_when_video_append_fails`` and
  ``test_clip_recorder_writes_manifest_when_no_codec_opens`` --
  ``tests/test_worker_clip_publication.py::
  test_unavailable_publication_persists_reason_without_video`` (line 113)
  proves the UNAVAILABLE-manifest-without-video contract at the publisher
  boundary; ``worker/pipeline/output/evidence/evidence_manifest.py``'s
  closed manifest model makes a video-bearing yet UNAVAILABLE-reasoned
  manifest structurally unrepresentable, so the specific encoder-failure and
  no-codec triggers edge exercised no longer need separate coverage.

* ``test_clip_recorder_throttles_automatic_rotation`` --
  ``tests/test_worker_clip_maintenance.py::
  test_clip_maintenance_throttles_automatic_rotation`` (line 24), against
  ``ClipMaintenance.rotate`` directly (``clip_maintenance.py:55-80``), which
  is what ``ClipRecorder._rotate`` now delegates to.

* ``test_startup_checks_pressure_before_accepting_new_clip`` --
  ``tests/test_evidence_retention.py::
  test_restart_reacquires_lock_and_preserves_pending_evidence`` (line 218):
  ``ClipRecorder.start()`` still runs ``_rotate(force=True)`` synchronously
  before accepting events (``clip_recorder.py:108-109`` precede
  ``self._admission.set_accepting(True)`` at line 134), so pressure detected
  at startup suspends recording before any event is admitted.

* ``test_clip_recorder_rotation_uses_finalized_at_at_60_day_boundary`` --
  ``tests/test_evidence_retention.py::
  test_pressure_purges_only_60_day_and_older_verified_unheld_clips``
  (line 89).

* ``test_clip_recorder_queue_full_drops_without_blocking`` --
  ``tests/test_worker_clip_recorder.py::
  test_bounded_queue_rejects_without_blocking`` (line 176).

* ``test_camera_worker_records_frames_and_admitted_events_without_blocking``
  and ``test_camera_worker_uses_monotonic_time_for_incident_admission`` --
  ``edge.runtime.camera_worker.CameraWorker`` no longer exists; frame/event
  routing and monotonic-clocked incident admission are proven at their new
  call sites by ``tests/test_worker_event_sink.py::
  test_event_sink_stages_then_binds_the_admitted_business_event`` and
  ``tests/test_worker_incident_manager.py::
  test_duplicate_within_cooldown_has_zero_output_side_effects`` (line 212,
  constructs ``EventAggregator(monotonic=lambda: ...)`` and proves cooldown
  gating tracks the injected clock rather than event-embedded or wall-clock
  time); see ``tests/test_camera_worker_clip_id.py`` for the full
  supersession ledger of every other ``CameraWorker`` test this file used to
  carry.

Obsolete by design (encoder resolution is no longer runtime-probed):

* ``test_clip_recorder_selects_nvenc_when_probe_succeeds``,
  ``test_clip_recorder_falls_back_to_libx264_when_nvenc_probe_fails``, and
  ``test_clip_recorder_nvenc_probe_uses_minimum_supported_resolution`` --
  ``worker.pipeline.output.evidence.clip_recorder_services.resolve_encoder``
  (line 32-33) unconditionally returns ``"libx264"``; there is no
  ``_probe_nvenc``, no ``subprocess.run`` probe, and no per-process encoder
  cache left to test. Encoder/profile selection is now a boot-time
  configuration concern, not a runtime capability probe.

Genuine production gap found, reported per task instructions, NOT fixed and
NOT asserted as passing behavior by any test here:

* ``test_clip_recorder_ffmpeg_args_enforce_browser_playable_h264`` asserted
  an explicit ``-profile:v`` flag forcing a browser-playable H.264 profile.
  The new encode-args builders --
  ``worker/adapters/encode/segment_process.py:79`` (live segment capture,
  forces ``-pix_fmt yuv420p`` and ``+faststart``) and
  ``worker/adapters/encode/clip_finalizer.py:53-96`` (concat remux, forces
  ``+faststart``) -- never pass ``-profile:v``, relying on libx264's default
  profile (High, since only pixel format is forced) instead of an explicit
  baseline/main profile. ``tests/test_worker_segment_encoder.py:237-253``,
  ``tests/test_worker_clip_finalizer.py:58-69``, and
  ``tests/test_worker_encode_real_ffmpeg.py:108-129`` cover the args and
  pixel format/codec that do exist, but none assert a profile flag, because
  the code sets none. This is a real, uncovered narrowing of the
  "browser-playable" guarantee versus edge and should be tracked, not
  silently dropped by omission.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from contracts.frame import Frame
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder, ClipRecorderConfig


class DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def _low_disk_usage(_path: Path) -> DiskUsage:
    return DiskUsage(total=100, used=10, free=90)


def _frame(index: int, time_sec: float) -> Frame:
    value = index % 255
    return Frame(index, time_sec, np.full((16, 16, 3), value, dtype=np.uint8))


def test_retention_defaults_to_60_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLIP_STORE_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("CLIP_RETENTION_DAYS", raising=False)

    assert ClipRecorderConfig(store_dir=tmp_path).retention_days == 60


def test_clip_window_defaults_to_30_seconds_before_and_after(tmp_path: Path) -> None:
    config = ClipRecorderConfig(store_dir=tmp_path)

    assert config.pre_event_seconds == 30.0
    assert config.post_event_seconds == 30.0


@pytest.mark.parametrize(("configured", "expected"), (("59", 60), ("60", 60)))
def test_retention_environment_enforces_60_day_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: int,
) -> None:
    monkeypatch.setenv("CLIP_STORE_RETENTION_DAYS", configured)
    monkeypatch.delenv("CLIP_RETENTION_DAYS", raising=False)

    assert ClipRecorderConfig(store_dir=tmp_path).retention_days == expected


def test_direct_config_below_60_days_is_clamped(tmp_path: Path) -> None:
    config = ClipRecorderConfig(store_dir=tmp_path, retention_days=1)

    assert config.retention_days == 60


def test_clip_recorder_start_sweeps_old_staging_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: one staging directory old enough to be orphaned crash residue,
    # and one too fresh to be safely assumed abandoned.
    staging_root = tmp_path / "clips" / ".staging"
    old_staging = staging_root / "old"
    fresh_staging = staging_root / "fresh"
    old_staging.mkdir(parents=True)
    fresh_staging.mkdir(parents=True)
    old_time = datetime.now().timestamp() - 11
    os.utime(old_staging, (old_time, old_time))
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.clip_recorder._resolve_encoder", lambda: "libx264"
    )
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=tmp_path, stale_staging_seconds=10.0),
        disk_usage_provider=_low_disk_usage,
        is_clip_held=lambda _clip_id: False,
    )

    # When: the recorder starts.
    recorder.start()
    try:
        # Then: only the stale directory is swept, and the encoder resolved
        # for this run is recorded on stats.
        assert not old_staging.exists()
        assert fresh_staging.exists()
        assert recorder.stats.stale_staging_cleaned == 1
        assert recorder.stats.encoder == "libx264"
    finally:
        recorder.stop()


def _finalized_clip_dir(
    root: Path,
    clip_id: str,
    *,
    finalized_at: datetime,
) -> Path:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"video")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "cam-1",
                "finalized": True,
                "finalized_at": finalized_at.isoformat().replace("+00:00", "Z"),
                "path": f"clips/{clip_id}/clip.mp4",
                "video_available": True,
            }
        ),
        encoding="utf-8",
    )
    return clip_dir


def test_legacy_retention_clock_uses_manifest_mtime_not_started_at(tmp_path: Path) -> None:
    # Given: a manifest predating the "finalized_at" field, so retention has
    # only the manifest file's own mtime to judge its age by.
    import worker.pipeline.output.evidence.clip_recorder as recorder_mod

    now = datetime.now(UTC)
    legacy_dir = tmp_path / "clips" / "legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "clip.mp4").write_bytes(b"video")
    manifest_path = legacy_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clip_id": "legacy",
                "camera_id": "cam-1",
                "finalized": True,
                "path": "clips/legacy/clip.mp4",
                "video_available": True,
            }
        ),
        encoding="utf-8",
    )
    conservative_time = now - timedelta(days=10)
    os.utime(manifest_path, (conservative_time.timestamp(), conservative_time.timestamp()))

    # When: the module-level scan (module alias for
    # ``clip_maintenance.finalized_clips``) walks the store.
    clips = recorder_mod._finalized_clips(tmp_path)

    # Then: the manifest's own mtime stands in for the missing timestamp.
    assert clips == [(datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC), legacy_dir)]


_REQUIRES_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="requires the ffmpeg binary for real-encode clip tests",
)

# event_refs are validated as canonical UUIDv4 by
# worker/pipeline/output/evidence/manifest_models.py:125-143, unlike edge's
# looser event_ref strings.
EVENT_ONE = "00000000-0000-4000-8000-000000000001"
EVENT_THREE = "00000000-0000-4000-8000-000000000003"


@_REQUIRES_FFMPEG
def test_clip_recorder_finalizes_atomic_manifest_with_pre_and_post_window(
    tmp_path: Path,
) -> None:
    """Ported unchanged in intent: pins this repo's macOS ``/proc`` floor.

    This exercises the real ``ClipRecordingCoordinator`` /
    ``FFmpegSegmentEncoder`` / ``FFmpegConcatFinalizer`` / ``ClipPublisher``
    stack end to end with the real ``ffmpeg``/``ffprobe`` binaries -- no
    monkeypatching -- and asserts the full READY-manifest contract, exactly
    as edge's original did. On a host with a working ``/proc`` (e.g. Linux
    CI) this passes outright.

    On this repo's sandboxed macOS host it fails, and the root cause is in
    *production* code, not test instrumentation:
    ``_probe_h264_video_only`` (``worker/pipeline/output/evidence/
    evidence_media.py:66-74``) hands ffprobe the open media descriptor via
    the hardcoded, TOCTOU-safe path ``f"/proc/self/fd/{descriptor}"``
    (line 74) rather than the file's real path. macOS has no ``/proc``, so
    ffprobe cannot open that path and exits nonzero, which
    ``inspect_finalized_media`` surfaces as ``ClipEvidenceError``.
    ``ClipActor._finalize`` (``clip_actor.py:212-218``) catches that
    alongside ``OSError`` and only increments ``stats.failed_writes`` --
    unlike edge's recorder, which still wrote an ``UNAVAILABLE`` manifest on
    this same ffprobe failure, so no manifest is ever published here. That
    makes the very first assertion below (``manifest_path.exists()``) the
    one that fails on macOS, with ``AssertionError: assert False`` -- the
    same ``/proc/self/fd``-rooted cause as edge's
    ``AssertionError: assert 'UNAVAILABLE' == 'READY'``, just surfaced one
    step earlier because the new code never gets far enough to write a
    degraded manifest at all. This ``/proc/self/fd`` probing design is
    carried forward unchanged from edge (not a new regression); it is a
    pre-existing Linux-only assumption in the fd-based, symlink-race-safe
    media inspection this repo's evidence pipeline relies on.
    """
    notifications: list[str] = []

    def finalized(clip_id: str) -> None:
        assert (tmp_path / "clips" / clip_id / "manifest.json").exists()
        assert not (tmp_path / "clips" / ".staging" / clip_id).exists()
        notifications.append(clip_id)

    recorder = ClipRecorder(
        ClipRecorderConfig(
            store_dir=tmp_path,
            segment_seconds=2.0,
            pre_event_seconds=10.0,
            post_event_seconds=20.0,
            fps=2.0,
        ),
        disk_usage_provider=_low_disk_usage,
        on_clip_finalized=finalized,
    )
    recorder.start()
    try:
        clip_id: str | None = None
        for index in range(69):
            time_sec = index * 0.5
            assert recorder.on_frame("cam-1", _frame(index, time_sec))
            if time_sec == 12.0:
                clip_id = recorder.on_event("cam-1", EVENT_ONE)
                assert clip_id is not None
        assert recorder.flush()
    finally:
        recorder.stop()

    assert clip_id is not None
    manifest_path = tmp_path / "clips" / clip_id / "manifest.json"
    video_path = tmp_path / "clips" / clip_id / "clip.mp4"
    assert manifest_path.exists()
    assert video_path.exists()
    assert not (tmp_path / "clips" / ".staging" / clip_id).exists()
    assert list((tmp_path / "clips" / clip_id).glob("*.tmp")) == []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 2
    assert manifest["state"] == "READY"
    assert manifest["state_version"] == 2
    assert manifest["camera_id"] == "cam-1"
    assert manifest["clip_id"] == clip_id
    assert manifest["codec"] == "h264"
    assert manifest["mime_type"] == "video/mp4"
    assert manifest["event_ref"] == EVENT_ONE
    assert manifest["event_refs"] == [EVENT_ONE]
    assert manifest["finalized"] is True
    assert manifest["path"] == f"clips/{clip_id}/{video_path.name}"
    assert manifest["video_available"] is True
    assert len(manifest["sha256"]) == 64
    assert manifest["size_bytes"] == video_path.stat().st_size
    assert 1 <= manifest["duration_ms"] <= 120_000
    assert manifest["clip_start_at"].endswith("Z")
    assert manifest["clip_end_at"].endswith("Z")
    assert manifest["finalized_at"].endswith("Z")
    assert manifest["started_at"].endswith("Z")
    assert 29.0 <= manifest["duration_s"] <= 31.5
    assert recorder.stats.finalized_clips == 1
    assert notifications == [clip_id]


def test_clip_recorder_fsyncs_media_and_manifest_before_staging_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ported unchanged in intent: pins this repo's macOS ``/proc`` floor.

    Edge's original test intercepted ``os.fsync`` and used
    ``os.readlink("/proc/self/fd/N")`` purely as test-only instrumentation
    to identify *which* file/directory each fsync call targeted. The trace may
    contain duplicate basenames from media inspection, so phase lookups anchor
    to the relevant replace operations: media, thumbnail, manifest, and staging
    cleanup. That instrumentation
    technique ports unchanged -- the new code's own fsync calls
    (``worker/pipeline/output/evidence/durability.py:8-21``'s
    ``fsync_file``/``fsync_directory``, invoked from ``clip_publication.py``
    and ``clip_identity.py``) still take a bare file descriptor, and
    ``/proc/self/fd/N`` is still how the test resolves it back to a path --
    and the ordering assertions below preserve edge's original durability
    intent while covering the thumbnail publication phases.

    On a host with a working ``/proc`` (e.g. Linux CI), every event is
    recorded and the full ordering assertion passes. On this repo's
    sandboxed macOS host, ``/proc`` does not exist at all, so the *very
    first* fsync this test's wrapper intercepts -- not one of the
    publication-time fsyncs the ordering assertion below targets, but the
    earlier, synchronous ``fsync_directory(self._staging_root)`` that
    ``ClipIdAllocator.reserve`` runs on every clip-id reservation
    (``worker/pipeline/output/evidence/clip_identity.py:69``, called
    directly from ``recorder.on_event()`` on the caller's own thread, not
    the background actor) -- raises ``FileNotFoundError`` inside the test's
    own wrapper. Because ``reserve`` does not catch that exception, it
    propagates straight out of the ``recorder.on_event(...)`` call below
    (index 2 in the loop) rather than being absorbed by
    ``ClipActor._finalize``'s broad ``except (..., OSError)``
    (``clip_actor.py:212-218``), which only guards the later
    finalize/publish path. The test still fails on the identical
    ``/proc/self/fd``-rooted cause edge's original hit, just surfaced at
    reservation time instead of at the ordering-assertion lookup.
    """
    recorder = ClipRecorder(
        ClipRecorderConfig(
            store_dir=tmp_path,
            segment_seconds=2.0,
            pre_event_seconds=0.0,
            post_event_seconds=1.0,
            fps=2.0,
        ),
        disk_usage_provider=_low_disk_usage,
    )
    events: list[tuple[str, str, str]] = []
    real_replace = os.replace
    real_fsync = os.fsync
    real_rmtree = shutil.rmtree

    def _record_replace(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        events.append(("replace", Path(source).name, Path(target).name))
        real_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def _record_fsync(descriptor: int) -> None:
        target = Path(os.readlink(f"/proc/self/fd/{descriptor}")).name
        events.append(("fsync", target, ""))
        real_fsync(descriptor)

    def _record_rmtree(path: str | os.PathLike[str]) -> None:
        events.append(("rmtree", Path(path).name, ""))
        real_rmtree(path)

    monkeypatch.setattr(
        "worker.pipeline.output.evidence.clip_publication.os.replace", _record_replace
    )
    monkeypatch.setattr("worker.pipeline.output.evidence.durability.os.fsync", _record_fsync)
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.clip_publication.shutil.rmtree", _record_rmtree
    )
    recorder.start()
    clip_id: str | None = None
    try:
        for index in range(8):
            assert recorder.on_frame("cam-1", _frame(index, index * 0.5))
            if index == 2:
                clip_id = recorder.on_event("cam-1", EVENT_THREE)
        assert recorder.flush()
    finally:
        recorder.stop()

    assert clip_id is not None
    clips_dir_fsync = events.index(("fsync", "clips", ""))
    media_replace = events.index(("replace", "clip.mp4", "clip.mp4"))
    media_fsync = events.index(("fsync", "clip.mp4", ""), media_replace)
    media_dir_fsync = events.index(("fsync", clip_id, ""), media_fsync)
    thumbnail_replace = next(
        index
        for index, event in enumerate(events)
        if event[0] == "replace" and event[2] == "thumbnail.jpg"
    )
    thumbnail_temp = events[thumbnail_replace][1]
    thumbnail_temp_fsync = next(
        index
        for index, event in enumerate(events[:thumbnail_replace])
        if event == ("fsync", thumbnail_temp, "")
    )
    thumbnail_dir_fsync = events.index(("fsync", clip_id, ""), thumbnail_replace)
    manifest_replace = events.index(("replace", "manifest.json.tmp", "manifest.json"))
    manifest_fsync = events.index(("fsync", "manifest.json", ""), manifest_replace)
    manifest_dir_fsync = events.index(("fsync", clip_id, ""), manifest_fsync)
    staging_cleanup = events.index(("rmtree", clip_id, ""))
    staging_dir_fsync = events.index(("fsync", ".staging", ""), staging_cleanup)
    assert (
        clips_dir_fsync
        < media_replace
        < media_fsync
        < media_dir_fsync
        < thumbnail_temp_fsync
        < thumbnail_replace
        < thumbnail_dir_fsync
        < manifest_replace
        < manifest_fsync
        < manifest_dir_fsync
        < staging_cleanup
        < staging_dir_fsync
    )
