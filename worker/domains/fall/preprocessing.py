from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, TypeAlias

from contracts.model import DEFAULT_FALL_CONFIDENCE_THRESHOLD

Pose: TypeAlias = tuple[tuple[int, int, float], ...]
# One normalized COCO-17 keypoint window row: (x, y, confidence), each
# frame-relative x/y in [0, 1] and confidence carried through unchanged.
# Plain nested tuples -- no ndarray -- so this stays a pure numeric type
# usable directly as `worker.types.FallModelInput`'s "sequence" row shape.
NormalizedPose: TypeAlias = tuple[tuple[float, float, float], ...]

_KEYPOINT_COUNT: Final = 17
_LEFT_SHOULDER: Final = 5
_RIGHT_SHOULDER: Final = 6
_LEFT_HIP: Final = 11
_RIGHT_HIP: Final = 12
_FEATURE_DIMENSION: Final = 45
_ZERO_TRIPLE: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)


def normalize_pose(pose: Pose, frame_width: int, frame_height: int) -> NormalizedPose:
    normalized: list[tuple[float, float, float]] = [_ZERO_TRIPLE] * _KEYPOINT_COUNT
    for index, (x_coordinate, y_coordinate, confidence) in enumerate(pose):
        if index >= _KEYPOINT_COUNT:
            break
        if confidence >= DEFAULT_FALL_CONFIDENCE_THRESHOLD:
            normalized[index] = (
                float(x_coordinate) / frame_width,
                float(y_coordinate) / frame_height,
                float(confidence),
            )
    return tuple(normalized)


def extract_window_features(
    window: Sequence[NormalizedPose],
) -> tuple[float, ...]:
    frame_count = len(window)
    valid: list[list[bool]] = [
        [
            pose[keypoint][2] > DEFAULT_FALL_CONFIDENCE_THRESHOLD
            for keypoint in range(_KEYPOINT_COUNT)
        ]
        for pose in window
    ]
    features: list[float] = []

    for keypoint_index in range(_KEYPOINT_COUNT):
        x_deltas: list[float] = []
        y_deltas: list[float] = []
        for frame_index in range(frame_count - 1):
            if valid[frame_index][keypoint_index] and valid[frame_index + 1][keypoint_index]:
                x_deltas.append(
                    abs(
                        window[frame_index + 1][keypoint_index][0]
                        - window[frame_index][keypoint_index][0]
                    )
                )
                y_deltas.append(
                    abs(
                        window[frame_index + 1][keypoint_index][1]
                        - window[frame_index][keypoint_index][1]
                    )
                )
        if x_deltas:
            features.append(_mean(x_deltas))
            features.append(_mean(y_deltas))
        else:
            features.extend((0.0, 0.0))

    valid_keypoints_by_frame = [
        [keypoint for keypoint in range(_KEYPOINT_COUNT) if valid[frame_index][keypoint]]
        for frame_index in range(frame_count)
    ]
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    for frame_index, valid_keypoints in enumerate(valid_keypoints_by_frame):
        if valid_keypoints:
            denominator = float(len(valid_keypoints))
            centroid_x.append(
                sum(window[frame_index][keypoint][0] for keypoint in valid_keypoints) / denominator
            )
            centroid_y.append(
                sum(window[frame_index][keypoint][1] for keypoint in valid_keypoints) / denominator
            )
        else:
            centroid_x.append(0.0)
            centroid_y.append(0.0)

    step_distances: list[float] = [
        math.sqrt(
            (centroid_x[frame_index + 1] - centroid_x[frame_index]) ** 2
            + (centroid_y[frame_index + 1] - centroid_y[frame_index]) ** 2
        )
        for frame_index in range(frame_count - 1)
    ]
    features.append(sum(step_distances) if step_distances else 0.0)
    features.append(max(step_distances) if step_distances else 0.0)

    aspect_ratios: list[float] = []
    for frame_index, valid_keypoints in enumerate(valid_keypoints_by_frame):
        if len(valid_keypoints) >= 2:
            frame_x = [window[frame_index][keypoint][0] for keypoint in valid_keypoints]
            frame_y = [window[frame_index][keypoint][1] for keypoint in valid_keypoints]
            width = max(frame_x) - min(frame_x)
            height = max(frame_y) - min(frame_y)
            aspect_ratios.append(width / height if height > 0.0 else 0.0)
    if aspect_ratios:
        features.extend(
            (
                _mean(aspect_ratios),
                _std(aspect_ratios),
                max(aspect_ratios) - min(aspect_ratios),
            )
        )
    else:
        features.extend((0.0, 0.0, 0.0))

    tilt_values: list[float] = []
    torso_vertical_values: list[float] = []
    for frame_index in range(frame_count):
        shoulder_x = [
            window[frame_index][index][0]
            for index in (_LEFT_SHOULDER, _RIGHT_SHOULDER)
            if valid[frame_index][index]
        ]
        shoulder_y = [
            window[frame_index][index][1]
            for index in (_LEFT_SHOULDER, _RIGHT_SHOULDER)
            if valid[frame_index][index]
        ]
        hip_x = [
            window[frame_index][index][0]
            for index in (_LEFT_HIP, _RIGHT_HIP)
            if valid[frame_index][index]
        ]
        hip_y = [
            window[frame_index][index][1]
            for index in (_LEFT_HIP, _RIGHT_HIP)
            if valid[frame_index][index]
        ]
        if shoulder_x and hip_x:
            x_delta = sum(hip_x) / len(hip_x) - sum(shoulder_x) / len(shoulder_x)
            y_delta = sum(hip_y) / len(hip_y) - sum(shoulder_y) / len(shoulder_y)
            magnitude = math.sqrt(x_delta * x_delta + y_delta * y_delta)
            if magnitude > 0.0:
                tilt_values.append(abs(x_delta) / magnitude)
            torso_vertical_values.append(
                abs(sum(shoulder_y) / len(shoulder_y) - sum(hip_y) / len(hip_y))
            )
    features.append(_mean(tilt_values) if tilt_values else 0.0)

    absolute_centroid_y_delta: list[float] = [
        abs(centroid_y[frame_index + 1] - centroid_y[frame_index])
        for frame_index in range(frame_count - 1)
    ]
    features.append(_mean(absolute_centroid_y_delta) if absolute_centroid_y_delta else 0.0)
    features.append(max(absolute_centroid_y_delta) if absolute_centroid_y_delta else 0.0)

    if torso_vertical_values:
        features.extend((_mean(torso_vertical_values), _std(torso_vertical_values)))
    else:
        features.extend((0.0, 0.0))
    features.append(sum(distance * distance for distance in step_distances))

    result = tuple(features)
    if len(result) != _FEATURE_DIMENSION:
        message = f"fall feature shape must be ({_FEATURE_DIMENSION},), received ({len(result)},)"
        raise RuntimeError(message)
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


__all__ = ["NormalizedPose", "extract_window_features", "normalize_pose"]
