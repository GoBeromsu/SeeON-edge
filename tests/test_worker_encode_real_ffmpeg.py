from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from contracts.frame import Frame
from worker.adapters.encode import (
    ClipRemuxError,
    EncoderGeometry,
    FFmpegConcatFinalizer,
    FFmpegSegmentEncoder,
    SegmentEncoderConfig,
)
from worker.adapters.encode.csv_segment_index import CsvSegmentIndex
from worker.adapters.encode.encoder_setup import parse_encode_policy
from worker.adapters.encode.segment_process import INPUT_PIX_FMT
from worker.runtime.profile import PROFILE_REGISTRY
from worker.types import BusinessEvent, FramePacket

_REQUIRES_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires the ffmpeg/ffprobe binaries for real-encode clip tests",
)

_RED_RGB = (255, 0, 0)


def _red_packet(seq: int, *, width: int, height: int) -> FramePacket:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = _RED_RGB[0]
    frame = Frame(index=seq, time_sec=float(seq) * 0.2, image=image)
    return FramePacket("cam-1", frame, float(seq) * 0.2, seq, width, height, 1.0)


def _event() -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", 7, "cam-1", "facility-1", 2.0, 0.9)


def _ffprobe(path: Path) -> dict[str, str]:
    completed = subprocess.run(
        (
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, _, value = line.partition("=")
        fields[key] = value
    return fields


def _centre_pixel_rgb(path: Path, tmp_path: Path) -> tuple[int, int, int]:
    frame_png = tmp_path / "probe.png"
    subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            str(frame_png),
        ),
        check=True,
    )
    import cv2

    image = cv2.cvtColor(cv2.imread(str(frame_png)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    pixel = image[height // 2, width // 2]
    return (int(pixel[0]), int(pixel[1]), int(pixel[2]))


def test_every_profile_encode_policy_is_an_accepted_encoder() -> None:
    resolved = {name: parse_encode_policy(spec.encode) for name, spec in PROFILE_REGISTRY.items()}

    assert resolved == {
        "cpu-host": "libx264",
        "cpu": "libx264",
        "nvidia": "h264_nvenc",
        "flow": "h264_nvenc",
        "intel-vaapi-host": "libx264",
        "igpu": "libx264",
        "apple-mps-host": "libx264",
        "mps": "libx264",
    }


def test_frame_packet_images_are_declared_to_ffmpeg_as_rgb() -> None:
    # FramePacket carries RGB; a bgr24 declaration silently swaps red and blue.
    assert INPUT_PIX_FMT == "rgb24"


@_REQUIRES_FFMPEG
def test_real_encode_then_remux_preserves_colour_without_re_encoding(tmp_path: Path) -> None:
    geometry = EncoderGeometry(64, 64, 5.0)
    encoder = FFmpegSegmentEncoder(
        SegmentEncoderConfig(store_dir=tmp_path / "store", segment_seconds=1.0)
    )
    session = encoder.open("cam-1", "libx264", geometry)
    for seq in range(1, 26):
        session.write(_red_packet(seq, width=64, height=64))
    list_path = session.segment_list_path
    session.close()

    segments = CsvSegmentIndex(list_path, 1).refresh()
    assert len(segments) >= 2
    for segment in segments:
        assert segment.path.exists()

    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips")
    artifact = finalizer.finalize(segments, _event())

    probed = _ffprobe(artifact.path)
    assert probed["codec_name"] == "h264"
    assert probed["pix_fmt"] == "yuv420p"
    assert probed["width"] == "64"
    assert probed["height"] == "64"

    payload = artifact.path.read_bytes()
    assert payload.index(b"moov") < payload.index(b"mdat")

    red, green, blue = _centre_pixel_rgb(artifact.path, tmp_path)
    assert red > 200
    assert green < 60
    assert blue < 60


@_REQUIRES_FFMPEG
def test_remux_failure_yields_a_typed_outcome_and_no_clip(tmp_path: Path) -> None:
    geometry = EncoderGeometry(64, 64, 5.0)
    encoder = FFmpegSegmentEncoder(
        SegmentEncoderConfig(store_dir=tmp_path / "store", segment_seconds=1.0)
    )
    session = encoder.open("cam-1", "libx264", geometry)
    for seq in range(1, 16):
        session.write(_red_packet(seq, width=64, height=64))
    list_path = session.segment_list_path
    session.close()

    segments = CsvSegmentIndex(list_path, 1).refresh()
    assert segments
    for segment in segments:
        segment.path.write_bytes(b"not-a-media-file")

    finalizer = FFmpegConcatFinalizer(output_dir=tmp_path / "clips")

    with pytest.raises(ClipRemuxError):
        finalizer.finalize(segments, _event())

    assert not (tmp_path / "clips" / "clip.mp4").exists()
