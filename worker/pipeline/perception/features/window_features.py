"""Deterministic 45-value pose-window features used by trained fall models.

The feature order is model-facing: 34 keypoint velocities, 2 centroid
displacements, 3 box-aspect statistics, 1 torso tilt, 2 vertical velocities,
2 torso-height statistics, and 1 motion-energy value.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from contracts.model import DEFAULT_FALL_CONFIDENCE_THRESHOLD

_LEFT_SHOULDER: Final = 5
_RIGHT_SHOULDER: Final = 6
_LEFT_HIP: Final = 11
_RIGHT_HIP: Final = 12
_D: Final = 45


def extract_window_features(window: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a finite ``float32[45]`` feature vector for a COCO-17 window."""
    normalized: NDArray[np.float32] = np.asarray(window, dtype=np.float32)
    time_steps = len(normalized)
    keypoint_count = len(normalized[0]) if time_steps > 0 else 0
    x_coordinates: NDArray[np.float32] = normalized[:, :, 0]
    y_coordinates: NDArray[np.float32] = normalized[:, :, 1]
    confidence: NDArray[np.float32] = normalized[:, :, 2]
    valid: NDArray[np.bool_] = confidence > DEFAULT_FALL_CONFIDENCE_THRESHOLD

    features: list[float] = []
    valid_step: NDArray[np.bool_] = valid[:-1] & valid[1:]
    absolute_x_delta: NDArray[np.float32] = np.abs(np.diff(x_coordinates, axis=0))
    absolute_y_delta: NDArray[np.float32] = np.abs(np.diff(y_coordinates, axis=0))
    for keypoint_index in range(keypoint_count):
        mask: NDArray[np.bool_] = valid_step[:, keypoint_index]
        if mask.any():
            features.append(float(absolute_x_delta[:, keypoint_index][mask].mean()))
            features.append(float(absolute_y_delta[:, keypoint_index][mask].mean()))
        else:
            features.extend((0.0, 0.0))

    valid_counts: NDArray[np.float32] = valid.sum(axis=1).astype(np.float32)
    denominator: NDArray[np.float32] = np.maximum(valid_counts, 1.0)
    centroid_x: NDArray[np.float32] = np.where(
        valid_counts > 0,
        (x_coordinates * valid).sum(axis=1) / denominator,
        0.0,
    )
    centroid_y: NDArray[np.float32] = np.where(
        valid_counts > 0,
        (y_coordinates * valid).sum(axis=1) / denominator,
        0.0,
    )
    step_distances: NDArray[np.float32] = np.sqrt(
        np.diff(centroid_x) ** 2 + np.diff(centroid_y) ** 2
    )
    features.append(float(step_distances.sum()))
    features.append(float(step_distances.max()) if len(step_distances) > 0 else 0.0)

    aspect_ratios: list[float] = []
    for time_index in range(time_steps):
        frame_mask: NDArray[np.bool_] = valid[time_index]
        if int(frame_mask.sum()) < 2:
            continue
        width = float(
            x_coordinates[time_index, frame_mask].max()
            - x_coordinates[time_index, frame_mask].min()
        )
        height = float(
            y_coordinates[time_index, frame_mask].max()
            - y_coordinates[time_index, frame_mask].min()
        )
        aspect_ratios.append(width / height if height > 0.0 else 0.0)
    if aspect_ratios:
        aspect_array: NDArray[np.float32] = np.asarray(aspect_ratios, dtype=np.float32)
        features.extend(
            (
                float(aspect_array.mean()),
                float(aspect_array.std()),
                float(aspect_array.max() - aspect_array.min()),
            )
        )
    else:
        features.extend((0.0, 0.0, 0.0))

    tilt_values: list[float] = []
    for time_index in range(time_steps):
        shoulder_x = [
            float(x_coordinates[time_index, index])
            for index in (_LEFT_SHOULDER, _RIGHT_SHOULDER)
            if valid[time_index, index]
        ]
        shoulder_y = [
            float(y_coordinates[time_index, index])
            for index in (_LEFT_SHOULDER, _RIGHT_SHOULDER)
            if valid[time_index, index]
        ]
        hip_x = [
            float(x_coordinates[time_index, index])
            for index in (_LEFT_HIP, _RIGHT_HIP)
            if valid[time_index, index]
        ]
        hip_y = [
            float(y_coordinates[time_index, index])
            for index in (_LEFT_HIP, _RIGHT_HIP)
            if valid[time_index, index]
        ]
        if not shoulder_x or not hip_x:
            continue
        horizontal_delta = sum(hip_x) / len(hip_x) - sum(shoulder_x) / len(shoulder_x)
        vertical_delta = sum(hip_y) / len(hip_y) - sum(shoulder_y) / len(shoulder_y)
        magnitude = math.sqrt(horizontal_delta**2 + vertical_delta**2)
        if magnitude > 0.0:
            tilt_values.append(abs(horizontal_delta) / magnitude)
    features.append(float(np.mean(tilt_values)) if tilt_values else 0.0)

    absolute_centroid_y_delta: NDArray[np.float32] = np.abs(np.diff(centroid_y))
    features.append(
        float(absolute_centroid_y_delta.mean()) if len(absolute_centroid_y_delta) > 0 else 0.0
    )
    features.append(
        float(absolute_centroid_y_delta.max()) if len(absolute_centroid_y_delta) > 0 else 0.0
    )

    torso_height_values: list[float] = []
    for time_index in range(time_steps):
        shoulder_y = [
            float(y_coordinates[time_index, index])
            for index in (_LEFT_SHOULDER, _RIGHT_SHOULDER)
            if valid[time_index, index]
        ]
        hip_y = [
            float(y_coordinates[time_index, index])
            for index in (_LEFT_HIP, _RIGHT_HIP)
            if valid[time_index, index]
        ]
        if shoulder_y and hip_y:
            torso_height_values.append(
                abs(sum(shoulder_y) / len(shoulder_y) - sum(hip_y) / len(hip_y))
            )
    if torso_height_values:
        torso_height_array: NDArray[np.float32] = np.asarray(
            torso_height_values,
            dtype=np.float32,
        )
        features.extend((float(torso_height_array.mean()), float(torso_height_array.std())))
    else:
        features.extend((0.0, 0.0))

    features.append(float((step_distances**2).sum()))
    result: NDArray[np.float32] = np.asarray(features, dtype=np.float32)
    if len(result) != _D:
        message = f"feature dimension mismatch: expected {_D}, got {len(result)}"
        raise RuntimeError(message)
    return result


__all__ = ["extract_window_features"]
