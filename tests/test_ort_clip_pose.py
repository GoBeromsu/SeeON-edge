from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from worker.adapters.model.ort_clip_pose import OrtClipPoseRunner


class _Input:
    name = "images"


class _Session:
    def get_inputs(self):
        return [_Input()]

    def run(self, names, feed):
        rows = np.zeros((1, 300, 57), dtype=np.float32)
        rows[0, 0, :6] = (64, 32, 320, 160, 0.8, 0)
        rows[0, 1, :6] = (1, 1, 2, 2, 0.2, 0)
        rows[0, 2, :6] = (1, 1, 2, 2, 0.9, 1)
        return [rows]


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "pose.onnx"
    path.write_bytes(b"fake")
    path.with_suffix(".onnx.sha256").write_text(hashlib.sha256(b"fake").hexdigest() + "\n")
    return path


def test_right_bottom_letterbox_unletterboxes_and_filters(tmp_path: Path) -> None:
    runner = OrtClipPoseRunner(
        _model(tmp_path), 0.25, session_factory=lambda path, providers: _Session()
    )
    boxes = runner.detect_persons(np.zeros((320, 640, 3), dtype=np.uint8))
    assert boxes == ((64.0, 32.0, 320.0, 160.0, pytest.approx(0.8)),)
