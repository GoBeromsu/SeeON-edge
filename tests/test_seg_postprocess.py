from __future__ import annotations

import numpy as np

from worker.adapters.model.seg_postprocess import (
    decode_end_to_end_segmentation,
    letterbox_rgb,
)


def test_decodes_known_mask_to_largest_polygon_and_ignores_non_bed_rows() -> None:
    _, letterbox = letterbox_rgb(np.zeros((320, 640, 3), dtype=np.uint8))
    rows = np.zeros((1, 2, 38), dtype=np.float32)
    rows[0, 0, :6] = (0, 160, 640, 480, 0.9, 59)
    rows[0, 0, 6] = 1.0
    rows[0, 1, :6] = (0, 0, 640, 640, 0.99, 0)
    prototypes = np.full((1, 32, 160, 160), -10.0, dtype=np.float32)
    prototypes[0, 0, 40:120, 40:120] = 10.0

    instances = decode_end_to_end_segmentation(
        rows, prototypes, letterbox, confidence=0.25, max_points=4
    )

    assert instances == (
        (0, 0, 640, 320, np.float32(0.9), ((161, 0), (480, 1), (479, 320), (160, 319))),
    )


def test_letterbox_inverse_removes_asymmetric_padding() -> None:
    image = np.zeros((101, 200, 3), dtype=np.uint8)
    tensor, letterbox = letterbox_rgb(image)
    assert tensor.shape == (1, 3, 640, 640)
    assert (letterbox.resized_width, letterbox.resized_height) == (640, 323)
    assert (letterbox.pad_left, letterbox.pad_top) == (0, 158)

    rows = np.zeros((1, 1, 38), dtype=np.float32)
    rows[0, 0, :6] = (0, 158, 640, 481, 0.9, 59)
    rows[0, 0, 6] = 1.0
    prototypes = np.ones((1, 32, 160, 160), dtype=np.float32)
    assert decode_end_to_end_segmentation(
        rows, prototypes, letterbox, confidence=0.25, max_points=48
    )[0][:4] == (0, 0, 200, 100)
