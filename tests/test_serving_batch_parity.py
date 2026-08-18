"""Single-frame vs batched pose parity on the real weights (todo 7).

Batching is only a safe substitution for per-camera forwards if a frame's
batched result is the SAME result it would have received alone. Ultralytics
letterboxes and pads a list source as one tensor, so this is a real risk, not
a formality -- and it is exactly the kind of defect that a shape-only
assertion would miss. Every assertion below compares the actual numeric
payload (keypoint coordinates, box coordinates, confidences), never just
counts or shapes.

The fixed corpus repeats a packaged real image with people across 8 rows and
4 pretend cameras. A non-vacuity guard requires real detected keypoint arrays.

- CPU parity runs in CI whenever the pose weights are present (they are
  gitignored, so a machine without them reports skipped, never a false pass).
- GPU parity is ``real_stack``-marked and runs locally on the NVIDIA host.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from ultralytics import ASSETS

from contracts.frame import Frame
from contracts.runner import Image, PoseRunnerResult
from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import default_registry
from worker.types import FramePacket

REPO_ROOT = Path(__file__).resolve().parents[1]
_POSE_WEIGHTS = REPO_ROOT / "models" / "pose" / "yolo26n-pose.pt"
_FRAME_HEIGHT = 640
_FRAME_WIDTH = 640
_ASSET_FRAME = Path(ASSETS) / "bus.jpg"


def _require_weights() -> None:
    if not _POSE_WEIGHTS.is_file():
        pytest.skip(f"pose weights are not present at {_POSE_WEIGHTS}")


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")


def _corpus() -> tuple[FramePacket, ...]:
    """Eight deterministic real-image rows spread across four cameras."""
    source = cv2.imread(str(_ASSET_FRAME), cv2.IMREAD_COLOR)
    assert source is not None, f"fixed parity asset is missing: {_ASSET_FRAME}"
    image: Image = cv2.resize(source, (_FRAME_WIDTH, _FRAME_HEIGHT))
    images = tuple(image.copy() for _ in range(8))
    packets = []
    for index, image in enumerate(images):
        camera_id = f"camera-{index % 4 + 1}"
        packets.append(
            FramePacket(
                camera_id=camera_id,
                frame=Frame(index=index, time_sec=float(index), image=image),
                pts=float(index),
                seq=100 + index,
                width=_FRAME_WIDTH,
                height=_FRAME_HEIGHT,
                decode_time_ms=0.0,
            )
        )
    return tuple(packets)


def _assert_pose_parity(
    single: Sequence[PoseRunnerResult],
    batched: Sequence[PoseRunnerResult],
    frames: Sequence[FramePacket],
) -> None:
    assert len(single) == len(batched) == len(frames)
    compared_keypoints = 0
    for packet, alone, together in zip(frames, single, batched, strict=True):
        where = f"{packet.camera_id}#{packet.seq}"
        assert len(alone.poses) == len(together.poses), f"pose count differs at {where}"
        assert len(alone.boxes) == len(together.boxes), f"box count differs at {where}"
        for alone_pose, together_pose in zip(alone.poses, together.poses, strict=True):
            left = np.asarray(alone_pose, dtype=np.float64)
            right = np.asarray(together_pose, dtype=np.float64)
            assert left.shape == right.shape
            np.testing.assert_allclose(
                left[:, :2], right[:, :2], atol=1.0, err_msg=f"keypoint coordinates at {where}"
            )
            np.testing.assert_allclose(
                left[:, 2], right[:, 2], atol=0.02, err_msg=f"keypoint confidence at {where}"
            )
            compared_keypoints += left.size
        for alone_box, together_box in zip(alone.boxes, together.boxes, strict=True):
            left = np.asarray(alone_box, dtype=np.float64)
            right = np.asarray(together_box, dtype=np.float64)
            np.testing.assert_allclose(left[:4], right[:4], atol=1.0, err_msg=f"box at {where}")
            np.testing.assert_allclose(left[4], right[4], atol=0.02, err_msg=f"conf at {where}")
    # Guard against a vacuous pass: an all-empty corpus would compare nothing.
    assert compared_keypoints > 0, "parity corpus produced no detections to compare"


def _run_parity(device: str) -> None:
    serving = InProcessServingClient(default_registry())
    client = serving.batch_serving_client
    runner = client.create("pose", device=device)
    frames = _corpus()

    single = [runner.run(packet.borrow_host_frame().image) for packet in frames]
    batched = client.infer_batch("pose", frames, device=device)

    assert all(isinstance(result, PoseRunnerResult) for result in single)
    assert all(isinstance(result, PoseRunnerResult) for result in batched)
    _assert_pose_parity(single, batched, frames)


def test_batched_pose_matches_single_frame_pose_on_cpu() -> None:
    _require_weights()
    _run_parity("cpu")


def test_batched_pose_uses_the_same_runner_as_the_single_frame_path() -> None:
    """Parity is only meaningful while both paths share one model instance."""
    _require_weights()
    serving = InProcessServingClient(default_registry())
    client = serving.batch_serving_client

    runner = client.create("pose", device="cpu")

    assert client.create("pose", device="cpu") is runner


@pytest.mark.real_stack
def test_batched_pose_matches_single_frame_pose_on_cuda_parity() -> None:
    _require_weights()
    _require_cuda()
    _run_parity("cuda")
