from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.ort_bed_seg import OrtBedSegRunner


class _Session:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, output_names: object, input_feed: dict[str, np.ndarray]) -> list[object]:
        assert output_names is None
        self.feeds.append(input_feed)
        return self.outputs


def _model(tmp_path: Path, payload: bytes = b"onnx") -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(payload)
    path.with_suffix(".onnx.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )
    return path


def _outputs() -> list[object]:
    rows = np.zeros((1, 1, 38), dtype=np.float32)
    rows[0, 0, :6] = (0, 0, 640, 640, 0.9, 59)
    rows[0, 0, 6] = 1.0
    return [rows, np.ones((1, 32, 160, 160), dtype=np.float32)]


def test_ort_runner_refuses_non_cpu_provider_and_tampered_artifact(tmp_path: Path) -> None:
    path = _model(tmp_path)
    with pytest.raises(ModelLoadError, match="CPUExecutionProvider"):
        OrtBedSegRunner(str(path), providers=("CUDAExecutionProvider",))

    path.write_bytes(b"tampered")
    with pytest.raises(ModelLoadError, match="digest mismatch"):
        OrtBedSegRunner(str(path), session_factory=lambda _path, _providers: _Session(_outputs()))


def test_ort_runner_uses_rgb_letterbox_and_returns_bed_result(tmp_path: Path) -> None:
    session = _Session(_outputs())
    runner = OrtBedSegRunner(
        str(_model(tmp_path)), session_factory=lambda _path, _providers: session
    )

    result = runner.run(np.zeros((320, 640, 3), dtype=np.uint8))

    assert runner.device == "cpu"
    assert runner.preprocessing_identity == "rgb24-to-bed-regions.v1"
    assert result.kind == "bed"
    assert next(iter(result.boxes))[:5] == (0, 0, 640, 320, np.float32(0.9))
    assert session.feeds[0]["images"].shape == (1, 3, 640, 640)


def test_ort_runner_import_does_not_import_torch_or_ultralytics() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import worker.adapters.model.ort_bed_seg; "
                "assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(
    not Path("models/bed/yolo26m-seg.pt").is_file(), reason="real bed weights are unavailable"
)
def test_real_weights_ort_matches_ultralytics_on_synthetic_image() -> None:
    from worker.adapters.model.yolo_bed_seg import YoloBedSegRunner

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[160:480, 80:560] = (160, 120, 80)
    expected = tuple(YoloBedSegRunner().run(image).boxes)
    actual = tuple(OrtBedSegRunner().run(image).boxes)
    assert len(actual) == len(expected)
    if actual:
        expected_top, actual_top = expected[0], actual[0]
        assert np.max(np.abs(np.asarray(expected_top[:4]) - np.asarray(actual_top[:4]))) <= 2
        assert _polygon_iou(expected_top[5], actual_top[5]) >= 0.9


def _polygon_iou(first: object, second: object) -> float:
    first_points = np.asarray(first, dtype=np.intp)
    second_points = np.asarray(second, dtype=np.intp)
    if len(first_points) == 0 or len(second_points) == 0:
        return float(len(first_points) == len(second_points))
    min_x = min(first_points[:, 0].min(), second_points[:, 0].min())
    min_y = min(first_points[:, 1].min(), second_points[:, 1].min())
    max_x = max(first_points[:, 0].max(), second_points[:, 0].max())
    max_y = max(first_points[:, 1].max(), second_points[:, 1].max())
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    first_mask = _inside(xx, yy, first_points)
    second_mask = _inside(xx, yy, second_points)
    union = np.logical_or(first_mask, second_mask).sum()
    return 0.0 if union == 0 else float(np.logical_and(first_mask, second_mask).sum() / union)


def _inside(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(x.shape, dtype=bool)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        crosses = (start[1] > y) != (end[1] > y)
        inside ^= crosses & (
            x < (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0]
        )
    return inside
