from __future__ import annotations

import threading
import time

from worker.pipeline.output.live_view import LatestFrameStore

# worker unified edge's two per-purpose buffers (raw-frame handoff
# LatestFrameBuffer and dev-view OverlayFrameBuffer) into one camera-keyed
# LatestFrameStore (worker/pipeline/output/live_view.py:25-76). It trades the
# old bounded queue.Queue put()/take()-with-stop_event contract for a plain
# dict guarded by a threading.Condition: publish_jpeg() always overwrites the
# stored value (no "queue full, evict oldest" branch is needed since there is
# no bounded slot to fill), and get_latest()/wait_for_latest() are
# non-consuming reads. This file is the direct unit-level successor of both
# edge's test_runtime_latest_frame.py (LatestFrameBuffer overwrite semantics)
# and the buffer-level assertion in the legacy mjpeg-server suite
# (test_mjpeg_buffer_is_camera_keyed_non_consuming), since both now exercise
# this same class.


def test_latest_frame_store_is_camera_keyed_and_non_consuming() -> None:
    store = LatestFrameStore()
    store.register_camera("camera-b")
    store.publish_jpeg("camera-a", b"jpeg-1", frame_index=1)

    first = store.get_latest("camera-a")
    second = store.get_latest("camera-a")
    assert first is not None
    assert second is not None
    assert first.jpeg == b"jpeg-1"
    assert second.jpeg == b"jpeg-1"
    assert store.get_latest("camera-b") is None


def test_latest_frame_store_is_known_before_and_after_registration() -> None:
    store = LatestFrameStore()
    assert store.is_known("camera-a") is False

    store.register_camera("camera-a")
    assert store.is_known("camera-a") is True
    assert store.get_latest("camera-a") is None  # known but nothing published yet


def test_latest_frame_store_publish_overwrites_previous_value() -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", b"jpeg-1", frame_index=1)
    store.publish_jpeg("camera-a", b"jpeg-2", frame_index=2)

    latest = store.get_latest("camera-a")
    assert latest is not None
    assert latest.jpeg == b"jpeg-2"
    assert latest.frame_index == 2


def test_latest_frame_store_seq_defaults_to_frame_index_but_is_overridable() -> None:
    store = LatestFrameStore()
    store.publish_jpeg("camera-a", b"jpeg-1", frame_index=7)
    assert store.get_latest("camera-a").seq == 7  # type: ignore[union-attr]

    store.publish_jpeg("camera-a", b"jpeg-2", frame_index=8, seq=99)
    latest = store.get_latest("camera-a")
    assert latest is not None
    assert latest.frame_index == 8
    assert latest.seq == 99


def test_latest_frame_store_wait_for_latest_unblocks_on_publish() -> None:
    store = LatestFrameStore()
    results: list[object] = []

    def wait_for_publish() -> None:
        results.append(store.wait_for_latest("camera-a", previous=None, timeout=1.0))

    waiter = threading.Thread(target=wait_for_publish)
    waiter.start()
    time.sleep(0.05)  # give the waiter time to block on the condition
    store.publish_jpeg("camera-a", b"jpeg-1", frame_index=1)
    waiter.join(timeout=1.0)

    assert len(results) == 1
    assert results[0] is not None
    assert results[0].jpeg == b"jpeg-1"  # type: ignore[union-attr]


def test_latest_frame_store_wait_for_latest_times_out_without_publish() -> None:
    store = LatestFrameStore()
    assert store.wait_for_latest("camera-a", previous=None, timeout=0.05) is None


def test_latest_frame_store_viewer_counter_tracks_open_stream_connections() -> None:
    store = LatestFrameStore()
    assert store.has_viewers("camera-a") is False

    store.mark_viewer_connected("camera-a")
    assert store.has_viewers("camera-a") is True

    store.mark_viewer_connected("camera-a")  # a second concurrent viewer
    store.mark_viewer_disconnected("camera-a")
    assert store.has_viewers("camera-a") is True  # first viewer is still connected

    store.mark_viewer_disconnected("camera-a")
    assert store.has_viewers("camera-a") is False
    assert store.has_viewers("camera-b") is False  # untouched camera is unaffected


def test_latest_frame_store_viewer_counter_never_goes_negative() -> None:
    """An extra disconnect (e.g. a duplicate ``finally`` firing) must not underflow."""
    store = LatestFrameStore()
    store.mark_viewer_disconnected("camera-a")
    assert store.has_viewers("camera-a") is False

    store.mark_viewer_connected("camera-a")
    store.mark_viewer_disconnected("camera-a")
    store.mark_viewer_disconnected("camera-a")
    assert store.has_viewers("camera-a") is False


def test_latest_frame_store_snapshot_demand_is_a_one_shot_flag() -> None:
    store = LatestFrameStore()
    assert store.consume_snapshot_demand("camera-a") is False

    store.request_snapshot_refresh("camera-a")
    assert store.consume_snapshot_demand("camera-a") is True
    assert store.consume_snapshot_demand("camera-a") is False  # consumed exactly once
    assert store.consume_snapshot_demand("camera-b") is False  # untouched camera


def test_latest_frame_store_mode_defaults_none_and_is_per_camera() -> None:
    store = LatestFrameStore()
    assert store.get_mode("camera-a") == "none"

    store.set_mode("camera-a", "fall")
    assert store.get_mode("camera-a") == "fall"
    assert store.get_mode("camera-b") == "none"  # untouched camera stays off

    store.set_mode("camera-a", "bedexit")
    assert store.get_mode("camera-a") == "bedexit"
