from __future__ import annotations

import socket
import struct

from worker.adapters.decode.native_preview_receiver import NativePreviewReceiver
from worker.pipeline.output.live_view import LatestFrameStore

_HEADER = struct.Struct("<4sIQQHI")


def test_native_preview_publishes_real_jpeg_to_latest_frame_store() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    store = LatestFrameStore()
    receiver = NativePreviewReceiver(parent, store)
    receiver.start()
    camera, jpeg = b"camera-a", b"\xff\xd8native-jpeg\xff\xd9"
    body = camera + jpeg
    child.sendall(_HEADER.pack(b"SJP1", len(body), 7, 90_000, len(camera), len(jpeg)) + body)

    frame = store.wait_for_latest("camera-a", previous=None, timeout=2.0)

    receiver.close()
    child.close()
    assert frame is not None
    assert frame.jpeg == jpeg
    assert frame.seq == 7
