"""Diagnostic-only ONNX Runtime batch-parity check for pose exports."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def _letterbox(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = 640 / max(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    canvas.paste(resized, (0, 0))
    return np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _scored_rows(output: np.ndarray, threshold: float) -> np.ndarray:
    rows = np.asarray(output, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise ValueError(f"expected pose rows [N, >=6], received {rows.shape}")
    rows = rows[rows[:, 4] >= threshold]
    return rows[np.argsort(-rows[:, 4], kind="stable")]


def _compare(reference: np.ndarray, batched: np.ndarray) -> tuple[float, float, list[str]]:
    violations: list[str] = []
    if len(reference) != len(batched):
        return 0.0, 0.0, [f"row count differs: b1={len(reference)} batch={len(batched)}"]
    if not len(reference):
        return 0.0, 0.0, violations
    coordinates = np.concatenate((reference[:, :4], reference[:, 5:]), axis=1)
    compared_coordinates = np.concatenate((batched[:, :4], batched[:, 5:]), axis=1)
    coordinate_diff = float(np.max(np.abs(coordinates - compared_coordinates)))
    score_diff = float(np.max(np.abs(reference[:, 4] - batched[:, 4])))
    if coordinate_diff > 1e-3:
        violations.append(f"coordinate/keypoint max abs diff {coordinate_diff:.8g} exceeds 0.001")
    if score_diff > 1e-4:
        violations.append(f"score max abs diff {score_diff:.8g} exceeds 0.0001")
    return coordinate_diff, score_diff, violations


def _run_group(
    session: ort.InferenceSession,
    input_name: str,
    tensors: list[np.ndarray],
    references: list[np.ndarray],
    indexes: list[int],
    threshold: float,
) -> tuple[float, float, list[dict[str, object]]]:
    output = session.run(None, {input_name: np.stack([tensors[index] for index in indexes])})[0]
    if output.shape[0] != len(indexes):
        raise ValueError(f"batch output has {output.shape[0]} frames for {len(indexes)} inputs")
    max_coordinate = 0.0
    max_score = 0.0
    violations: list[dict[str, object]] = []
    for position, index in enumerate(indexes):
        coordinate_diff, score_diff, problems = _compare(
            references[index], _scored_rows(output[position], threshold)
        )
        max_coordinate = max(max_coordinate, coordinate_diff)
        max_score = max(max_score, score_diff)
        for problem in problems:
            violations.append({"frame": index, "batch_position": position, "reason": problem})
    return max_coordinate, max_score, violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    frame_paths = sorted(args.frames.glob("*.png"))
    if not frame_paths:
        parser.error("--frames contains no PNG files")

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX model has no inputs")
    input_name = inputs[0].name
    tensors = [_letterbox(path) for path in frame_paths]
    references = [
        _scored_rows(session.run(None, {input_name: tensor[None, ...]})[0][0], args.threshold)
        for tensor in tensors
    ]

    max_coordinate = 0.0
    max_score = 0.0
    violations: list[dict[str, object]] = []
    batches = 0
    for start in range(0, len(tensors), args.batch):
        indexes = list(range(start, min(start + args.batch, len(tensors))))
        coordinate_diff, score_diff, problems = _run_group(
            session, input_name, tensors, references, indexes, args.threshold
        )
        max_coordinate = max(max_coordinate, coordinate_diff)
        max_score = max(max_score, score_diff)
        violations.extend(problems)
        batches += 1
    if len(tensors) >= args.batch:
        indexes = list(range(args.batch))
        random.Random(503).shuffle(indexes)
        coordinate_diff, score_diff, problems = _run_group(
            session, input_name, tensors, references, indexes, args.threshold
        )
        max_coordinate = max(max_coordinate, coordinate_diff)
        max_score = max(max_score, score_diff)
        violations.extend(problems)
        batches += 1

    receipt = {
        "frames": len(tensors),
        "batches": batches,
        "batch_size": args.batch,
        "threshold": args.threshold,
        "max_coordinate_keypoint_abs_diff": max_coordinate,
        "max_score_abs_diff": max_score,
        "violations": violations,
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
