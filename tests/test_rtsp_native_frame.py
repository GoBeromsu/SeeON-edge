from __future__ import annotations

import subprocess

import pytest

from worker.adapters.media.rtsp_native_frame import NativeFrameUnavailable, grab_native_jpeg


def test_grab_native_jpeg_returns_first_decoded_frame() -> None:
    calls: list[tuple[object, ...]] = []

    def run(command: tuple[object, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert kwargs == {"check": False, "capture_output": True, "timeout": 8.0}
        return subprocess.CompletedProcess(command, 0, stdout=b"\xff\xd8jpeg\xff\xd9")

    assert (
        grab_native_jpeg("rtsp://user:secret@camera.example/stream", run=run)
        == b"\xff\xd8jpeg\xff\xd9"
    )
    assert calls == [
        (
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            "rtsp://user:secret@camera.example/stream",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "2",
            "-",
        )
    ]


@pytest.mark.parametrize(
    "run",
    [
        lambda _command, **_kwargs: subprocess.CompletedProcess(
            (), 1, stdout=b"", stderr=b"rtsp://user:secret@camera.example/stream"
        ),
        lambda _command, **_kwargs: subprocess.CompletedProcess((), 0, stdout=b""),
        lambda _command, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(("ffmpeg",), 8.0)
        ),
    ],
)
def test_grab_native_jpeg_failure_never_exposes_camera_url(run: object) -> None:
    url = "rtsp://user:secret@camera.example/stream"
    with pytest.raises(NativeFrameUnavailable) as raised:
        grab_native_jpeg(url, run=run)  # type: ignore[arg-type]
    assert url not in str(raised.value)
