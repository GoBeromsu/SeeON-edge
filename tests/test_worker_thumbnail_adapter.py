from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest


def _thumbnail_module() -> ModuleType:
    return importlib.import_module("worker.adapters.encode.thumbnail")


def _artifact_module() -> ModuleType:
    return importlib.import_module("worker.adapters.encode.thumbnail_artifact")


def _jpeg_bytes(*, height: int = 360, width: int = 640) -> bytes:
    encoded, payload = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    assert encoded
    return payload.tobytes()


class _ThumbnailRunner:
    def __init__(self, payload: bytes | None = None) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []
        self.payload = _jpeg_bytes() if payload is None else payload

    def __call__(self, args: tuple[str, ...], timeout_s: float):
        self.commands.append(args)
        self.timeouts.append(timeout_s)
        return _thumbnail_module().ThumbnailCommandResult(0, self.payload)


def test_ffmpeg_thumbnail_adapter_extracts_midpoint_to_atomic_640x360_jpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _thumbnail_module()
    artifact = _artifact_module()
    runner = _ThumbnailRunner()
    video_path = tmp_path / "clip.mp4"
    thumbnail_path = tmp_path / "thumbnail.jpg"
    video_path.write_bytes(b"video")
    temporary_open_flags: list[int] = []
    relative_replaces: list[tuple[int | None, int | None]] = []
    real_open = artifact.os.open
    real_replace = artifact.os.replace

    def record_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is not None and str(path).endswith(".tmp"):
            temporary_open_flags.append(flags)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def record_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        relative_replaces.append((src_dir_fd, dst_dir_fd))
        real_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(artifact.os, "open", record_open)
    monkeypatch.setattr(artifact.os, "replace", record_replace)
    generator = module.FFmpegThumbnailGenerator(
        ffmpeg_bin="ffmpeg-custom",
        runner=runner,
    )

    generated = generator.generate(video_path, thumbnail_path, 10.0)

    args = runner.commands[0]
    assert args[0] == "ffmpeg-custom"
    assert args[args.index("-ss") + 1] == "5.000000"
    assert "scale=640:360" in args[args.index("-vf") + 1]
    assert args[args.index("-f") + 1] == "image2pipe"
    assert args[-1] == "pipe:1"
    assert runner.timeouts == [module.THUMBNAIL_TIMEOUT_SECONDS]
    assert generated.read_bytes() == _jpeg_bytes()
    assert temporary_open_flags[0] & os.O_CREAT
    assert temporary_open_flags[0] & os.O_EXCL
    assert temporary_open_flags[0] & os.O_NOFOLLOW
    assert relative_replaces[0][0] is not None
    assert relative_replaces[0][0] == relative_replaces[0][1]
    assert generator.generate(video_path, thumbnail_path, 10.0) == thumbnail_path
    assert len(runner.commands) == 1


def test_predictable_thumbnail_temp_symlink_never_overwrites_target(tmp_path: Path) -> None:
    module = _thumbnail_module()
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive", encoding="utf-8")
    predictable = tmp_path / ".thumbnail.jpg.tmp"
    predictable.symlink_to(outside)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    generated = module.FFmpegThumbnailGenerator(runner=_ThumbnailRunner()).generate(
        video_path,
        tmp_path / "thumbnail.jpg",
        10.0,
    )

    assert generated.read_bytes() == _jpeg_bytes()
    assert outside.read_text(encoding="utf-8") == "must survive"
    assert predictable.is_symlink()


def test_random_temp_collision_is_typed_and_never_follows_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _thumbnail_module()
    artifact = _artifact_module()
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive", encoding="utf-8")
    temporary = tmp_path / ".thumbnail.jpg.fixed.tmp"
    temporary.symlink_to(outside)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(artifact.secrets, "token_hex", lambda _size: "fixed")

    with pytest.raises(module.ThumbnailSecurityError):
        module.FFmpegThumbnailGenerator(runner=_ThumbnailRunner()).generate(
            video_path,
            tmp_path / "thumbnail.jpg",
            10.0,
        )

    assert outside.read_text(encoding="utf-8") == "must survive"
    assert temporary.is_symlink()


def test_ffmpeg_timeout_is_finite_and_typed() -> None:
    module = _thumbnail_module()
    timeout_s = 0.01
    command = (sys.executable, "-c", "import time; time.sleep(5)")

    with pytest.raises(module.ThumbnailTimeoutError) as raised:
        module.run_ffmpeg_thumbnail(command, timeout_s)

    assert raised.value.timeout_s == timeout_s


def test_ffmpeg_stdout_is_rejected_at_the_byte_limit() -> None:
    module = _thumbnail_module()
    command = (
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'x' * (3 * 1024 * 1024))",
    )

    with pytest.raises(module.ThumbnailPayloadError):
        module.run_ffmpeg_thumbnail(command, module.THUMBNAIL_TIMEOUT_SECONDS)


def test_hostile_jpeg_dimensions_are_rejected_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact_module()
    payload = bytearray(_jpeg_bytes())
    start_of_frame = payload.index(b"\xff\xc0")
    payload[start_of_frame + 5 : start_of_frame + 7] = (65535).to_bytes(2, "big")
    payload[start_of_frame + 7 : start_of_frame + 9] = (65535).to_bytes(2, "big")
    decoded = False

    def record_decode(*_args, **_kwargs):
        nonlocal decoded
        decoded = True
        return None

    monkeypatch.setattr(artifact.cv2, "imdecode", record_decode)

    assert artifact.is_valid_jpeg(bytes(payload)) is False
    assert decoded is False


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not-a-jpeg",
        _jpeg_bytes()[:-2],
        b"x" * (2 * 1024 * 1024 + 1),
        _jpeg_bytes(height=180, width=320),
    ],
    ids=["empty", "corrupt", "truncated", "oversized", "wrong-shape"],
)
def test_invalid_generated_thumbnail_payload_is_typed_and_never_published(
    tmp_path: Path,
    payload: bytes,
) -> None:
    module = _thumbnail_module()
    thumbnail_path = tmp_path / "thumbnail.jpg"

    with pytest.raises(module.ThumbnailPayloadError):
        module.FFmpegThumbnailGenerator(runner=_ThumbnailRunner(payload)).generate(
            tmp_path / "clip.mp4",
            thumbnail_path,
            10.0,
        )

    assert not thumbnail_path.exists()


@pytest.mark.parametrize(
    "payload",
    [b"", b"not-a-jpeg", _jpeg_bytes()[:-2], b"x" * (2 * 1024 * 1024 + 1)],
    ids=["empty", "corrupt", "truncated", "oversized"],
)
def test_existing_malformed_thumbnail_validation_is_total(
    tmp_path: Path,
    payload: bytes,
) -> None:
    module = _thumbnail_module()
    thumbnail_path = tmp_path / "thumbnail.jpg"
    thumbnail_path.write_bytes(payload)

    assert module.is_valid_thumbnail(thumbnail_path) is False
    generated = module.FFmpegThumbnailGenerator(runner=_ThumbnailRunner()).generate(
        tmp_path / "clip.mp4",
        thumbnail_path,
        10.0,
    )
    assert module.is_valid_thumbnail(generated) is True


def test_thumbnail_validation_contains_cv2_decode_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _thumbnail_module()
    artifact = _artifact_module()
    thumbnail_path = tmp_path / "thumbnail.jpg"
    thumbnail_path.write_bytes(_jpeg_bytes())
    monkeypatch.setattr(
        artifact.cv2,
        "imdecode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(cv2.error("decode failed")),
    )

    assert module.is_valid_thumbnail(thumbnail_path) is False
