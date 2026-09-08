from __future__ import annotations

from pathlib import Path

from worker.runtime.clip_analysis_process import ClipAnalysisJob
from worker.runtime.clip_analysis_queue import Admission, ClipAnalysisQueue


def _job(clip_id: str, *, manual: bool = False) -> ClipAnalysisJob:
    return ClipAnalysisJob(
        clip_id=clip_id,
        clip_path=Path(f"/{clip_id}.mp4"),
        clip_sha256="a" * 64,
        pose_model_path=Path("/pose.onnx"),
        bed_model_path=Path("/bed.onnx"),
        profile=object(),
        profile_sha256="b" * 64,
        decoder_identity="pyav-16/h264",
        front=manual,
    )


def test_fifo_order_and_manual_head_insertion() -> None:
    queue = ClipAnalysisQueue(capacity=3)
    assert queue.add(_job("automatic-1"), front=False) == Admission.QUEUED
    assert queue.add(_job("automatic-2"), front=False) == Admission.QUEUED
    assert queue.add(_job("manual", manual=True), front=True) == Admission.QUEUED

    assert [queue.take().clip_id for _ in range(3)] == ["manual", "automatic-1", "automatic-2"]


def test_manual_promotes_existing_queued_duplicate() -> None:
    queue = ClipAnalysisQueue()
    queue.add(_job("first"), front=False)
    queue.add(_job("second"), front=False)

    assert queue.add(_job("second", manual=True), front=True) == Admission.ALREADY_QUEUED
    assert [queue.take().clip_id for _ in range(2)] == ["second", "first"]


def test_full_queue_evicts_catchup_tail_for_manual_never_manual_or_running() -> None:
    queue = ClipAnalysisQueue(capacity=2)
    queue.add(_job("automatic"), front=False)
    queue.add(_job("manual-queued", manual=True), front=True)

    assert queue.add(_job("manual-new", manual=True), front=True) == Admission.QUEUED
    assert [queue.take().clip_id for _ in range(2)] == ["manual-new", "manual-queued"]


def test_full_queue_rejects_automatic_with_queue_full() -> None:
    queue = ClipAnalysisQueue(capacity=1)
    queue.add(_job("manual", manual=True), front=True)

    assert queue.add(_job("automatic"), front=False) == Admission.QUEUE_FULL
