from __future__ import annotations

import importlib
import importlib.util
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

import worker.__main__ as worker_main
from worker.pipeline.output.evidence.clip_config import (
    DEFAULT_CLIP_STORE_DIR,
    configured_store_dir,
)


def _backfill_module() -> ModuleType:
    spec = importlib.util.find_spec("worker.pipeline.output.evidence.thumbnail_backfill")
    assert spec is not None, "worker thumbnail backfill is missing"
    return importlib.import_module("worker.pipeline.output.evidence.thumbnail_backfill")


def _jpeg_bytes() -> bytes:
    encoded, payload = cv2.imencode(
        ".jpg",
        np.zeros((360, 640, 3), dtype=np.uint8),
    )
    assert encoded
    return payload.tobytes()


class _ThumbnailGenerator:
    def __init__(self, failing_video_names: set[str] | None = None) -> None:
        self.failing_video_names = failing_video_names or set()
        self.calls: list[str] = []

    def generate(
        self,
        video_path: Path,
        thumbnail_path: Path,
        duration_s: float,
    ) -> Path:
        del duration_s
        self.calls.append(video_path.parent.name)
        if video_path.parent.name in self.failing_video_names:
            error_type = importlib.import_module(
                "worker.adapters.encode.adapter_errors"
            ).ThumbnailGenerationError
            raise error_type("thumbnail extraction failed\r\nforged")
        thumbnail_path.write_bytes(_jpeg_bytes())
        return thumbnail_path


class _BlockingThumbnailGenerator:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(
        self,
        video_path: Path,
        thumbnail_path: Path,
        duration_s: float,
    ) -> Path:
        del video_path, duration_s
        self.entered.set()
        assert self.release.wait(timeout=5.0)
        thumbnail_path.write_bytes(_jpeg_bytes())
        return thumbnail_path


def _write_playable_clip(root: Path, clip_id: str) -> Path:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"video")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "camera-1",
                "event_ref": f"event-{clip_id}",
                "started_at": "2026-08-09T00:00:00Z",
                "duration_s": 8.0,
                "path": f"clips/{clip_id}/clip.mp4",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    return clip_dir


def test_backfill_scans_bounded_layouts_resumes_and_reports_remaining_failures(
    tmp_path: Path,
) -> None:
    module = _backfill_module()
    root = tmp_path / "clip-store"
    root_clip = _write_playable_clip(root, "clip-root")
    nested_clip = _write_playable_clip(root / "archive", "clip-one")
    deep_clip = _write_playable_clip(root / "external" / "drive", "clip-two")
    nested_clip.joinpath("thumbnail.jpg").write_bytes(_jpeg_bytes())
    generator = _ThumbnailGenerator({"clip-two"})

    first = module.backfill_thumbnails(root, generator)

    assert first.scanned == 3
    assert first.playable == 3
    assert first.generated == 1
    assert first.skipped == 1
    assert first.failed == 1
    assert first.missing == 1
    assert root_clip.joinpath("thumbnail.jpg").is_file()
    assert not deep_clip.joinpath("thumbnail.jpg").exists()

    generator.failing_video_names.clear()
    second = module.backfill_thumbnails(root, generator)

    assert second.generated == 1
    assert second.skipped == 2
    assert second.failed == 0
    assert second.missing == 0
    assert generator.calls.count("clip-root") == 1
    assert generator.calls.count("clip-one") == 0


def test_backfill_refuses_active_nested_recorder_before_any_mutation(
    tmp_path: Path,
) -> None:
    module = _backfill_module()
    lock_module = importlib.import_module("worker.pipeline.output.evidence.clip_store_lock")
    root = tmp_path / "clip-store"
    root_clip = _write_playable_clip(root, "clip-root")
    nested_root = root / "archive"
    nested_clip = _write_playable_clip(nested_root, "clip-nested")
    generator = _ThumbnailGenerator()

    with lock_module.ClipStoreLock.acquire(nested_root):
        with pytest.raises(lock_module.ClipStoreLockedError):
            module.backfill_thumbnails(root, generator)

    assert not root_clip.joinpath("thumbnail.jpg").exists()
    assert not nested_clip.joinpath("thumbnail.jpg").exists()
    assert generator.calls == []


def test_concurrent_backfill_is_refused_while_first_writer_holds_lock(
    tmp_path: Path,
) -> None:
    module = _backfill_module()
    lock_module = importlib.import_module("worker.pipeline.output.evidence.clip_store_lock")
    root = tmp_path / "clip-store"
    _write_playable_clip(root, "clip-root")
    generator = _BlockingThumbnailGenerator()

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(module.backfill_thumbnails, root, generator)
        assert generator.entered.wait(timeout=5.0)
        try:
            with pytest.raises(lock_module.ClipStoreLockedError):
                module.backfill_thumbnails(root, _ThumbnailGenerator())
        finally:
            generator.release.set()

        report = first.result(timeout=5.0)

    assert report.generated == 1


def test_backfill_failure_log_escapes_control_characters(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _backfill_module()
    root = tmp_path / "clip\r\nstore"
    _write_playable_clip(root, "clip-fail")

    report = module.backfill_thumbnails(root, _ThumbnailGenerator({"clip-fail"}))

    assert report.failed == 1
    messages = [record.getMessage() for record in caplog.records]
    assert messages
    assert all("\r" not in message and "\n" not in message for message in messages)


def test_configured_store_dir_uses_injected_or_baked_root_not_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = tmp_path / "portable-clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "retired-environment-root"))

    assert configured_store_dir(injected) == injected
    assert configured_store_dir() == Path(DEFAULT_CLIP_STORE_DIR)


def test_worker_cli_backfill_needs_no_camera_config_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "clip-store"
    monkeypatch.delenv("EDGE_CAMERA_CONFIG", raising=False)

    exit_code = worker_main.main(["--backfill-thumbnails", "--clip-store-dir", str(root)])

    assert exit_code == 0
    assert root.joinpath(".worker.lock").is_file()
    assert capsys.readouterr().out.strip() == (
        "thumbnail backfill: scanned=0 playable=0 generated=0 skipped=0 failed=0 missing=0"
    )


def test_worker_cli_backfill_defaults_to_baked_store_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _backfill_module()
    report = module.BackfillReport(0, 0, 0, 0, 0, 0)
    roots: list[Path] = []

    def _backfill(root: Path, generator: object) -> object:
        del generator
        roots.append(root)
        return report

    monkeypatch.setattr(worker_main, "backfill_thumbnails", _backfill)

    assert worker_main.main(["--backfill-thumbnails"]) == 0
    assert roots == [Path(DEFAULT_CLIP_STORE_DIR)]


def test_worker_cli_rejects_clip_store_dir_without_backfill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_runtime(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("backfill-only option must not start the worker")

    monkeypatch.setattr(worker_main.WorkerRuntime, "__init__", _fail_runtime)

    assert worker_main.main(["--clip-store-dir", str(tmp_path)]) == 2


def test_worker_cli_help_documents_portable_backfill_store_seam() -> None:
    help_text = worker_main._build_parser().format_help()  # noqa: SLF001

    assert "--clip-store-dir" in help_text
    assert "backfill" in help_text


def test_worker_cli_backfill_exits_nonzero_while_playable_clip_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _backfill_module()
    report = module.BackfillReport(1, 1, 0, 0, 1, 1)
    monkeypatch.setattr(
        worker_main,
        "backfill_thumbnails",
        lambda root, generator: report,
    )

    exit_code = worker_main.main(
        ["--backfill-thumbnails", "--clip-store-dir", str(tmp_path / "clip-store")]
    )

    assert exit_code == 1
    assert capsys.readouterr().out.strip().endswith("failed=1 missing=1")


def test_worker_cli_backfill_refuses_active_recorder_with_nonzero_exit(
    tmp_path: Path,
) -> None:
    lock_module = importlib.import_module("worker.pipeline.output.evidence.clip_store_lock")
    root = tmp_path / "clip-store"

    with lock_module.ClipStoreLock.acquire(root):
        exit_code = worker_main.main(["--backfill-thumbnails", "--clip-store-dir", str(root)])

    assert exit_code == 1
