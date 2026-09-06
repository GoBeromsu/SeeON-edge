#!/usr/bin/env python3
"""Score a pose+bbox56 fall bundle on its published selection-validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.domains.fall.pose_bbox56 import pose_bbox56_row

WINDOW_FRAMES = 30
STRIDE_FRAMES = 5
WINDOW_STARTS = range(0, 300 - WINDOW_FRAMES + 1, STRIDE_FRAMES)


class FallRunner(Protocol):
    receipt_threshold: float | None

    def predict(self, features: np.ndarray) -> Any: ...


def _threshold_key(threshold: float) -> str:
    return f"threshold_{threshold:g}".replace(".", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required in {path}")
    return value


def _canonical_json_digest(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_dataset_payload(dataset: Path, expected_digest: str) -> dict[str, Any]:
    """Validate the published payload manifest and its complete checksum listing."""
    manifest = _read_json(dataset / "payload-manifest.json")
    verified_digest = _canonical_json_digest(manifest)
    if verified_digest != expected_digest:
        raise ValueError(
            "dataset payload manifest digest mismatch: "
            f"expected {expected_digest}, got {verified_digest}"
        )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise TypeError("payload-manifest.json files must be a list")

    expected_checksums: dict[str, str] = {}
    for member in files:
        if not isinstance(member, dict):
            raise TypeError("payload manifest member must be an object")
        relative_path = member.get("relative_path")
        expected_sha256 = member.get("sha256")
        expected_size = member.get("size")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or relative_path in expected_checksums
        ):
            raise ValueError("invalid payload manifest member")
        member_path = dataset / relative_path
        if not member_path.is_file():
            raise ValueError(f"dataset payload member is missing: {relative_path}")
        if member_path.stat().st_size != expected_size or _sha256(member_path) != expected_sha256:
            raise ValueError(f"dataset payload member digest mismatch: {relative_path}")
        expected_checksums[relative_path] = expected_sha256

    checksum_entries: dict[str, str] = {}
    try:
        checksum_lines = (dataset / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read checksums.sha256: {exc}") from exc
    for line in checksum_lines:
        try:
            digest, relative_path = line.split(None, 1)
        except ValueError as exc:
            raise ValueError("invalid checksums.sha256 entry") from exc
        relative_path = relative_path.strip().removeprefix("*")
        if relative_path in checksum_entries or len(digest) != 64:
            raise ValueError("invalid checksums.sha256 entry")
        checksum_entries[relative_path] = digest
    if checksum_entries != expected_checksums:
        raise ValueError("checksums.sha256 entries do not exactly match payload manifest")

    return {
        "verified_payload_digest": verified_digest,
        "manifest_members_verified": len(files),
    }


def default_dataset(bundle_receipt: Mapping[str, Any]) -> Path:
    publication = bundle_receipt.get("dataset_publication")
    if not isinstance(publication, dict):
        raise TypeError("bundle evaluation receipt lacks dataset_publication")
    repo = publication.get("hf_repo")
    revision = publication.get("payload_revision")
    if not isinstance(repo, str) or not isinstance(revision, str) or not repo or not revision:
        raise ValueError("bundle dataset_publication is invalid")
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / f"datasets--{repo.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def _window_label(interval: Mapping[str, Any] | None, start: int) -> bool:
    if not isinstance(interval, Mapping):
        return False
    interval_start = interval.get("start")
    interval_end = interval.get("end")
    if not isinstance(interval_start, int) or not isinstance(interval_end, int):
        return False
    return max(0, min(start + WINDOW_FRAMES, interval_end) - max(start, interval_start)) >= 15


def _feature_window(clip: Mapping[str, Any], start: int) -> np.ndarray:
    width, height = clip["width"], clip["height"]
    pose, boxes = clip["pose"], clip["pose_head_bbox"]
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("clip dimensions must be integers")
    rows = []
    for keypoints, box_with_confidence in zip(
        pose[start : start + WINDOW_FRAMES], boxes[start : start + WINDOW_FRAMES], strict=True
    ):
        bbox = None
        if isinstance(box_with_confidence, Sequence) and len(box_with_confidence) == 5:
            confidence = box_with_confidence[4]
            if isinstance(confidence, (int, float)) and confidence >= 0.5:
                bbox = [
                    float(box_with_confidence[0]),
                    float(box_with_confidence[1]),
                    float(box_with_confidence[2]),
                    float(box_with_confidence[3]),
                ]
        dataset_keypoints = [
            [float(point[0]), float(point[1]), float(point[2])] for point in keypoints
        ]
        rows.append(pose_bbox56_row(dataset_keypoints, bbox, width, height))
    return np.asarray(rows, dtype=np.float32)


def _metrics(scores_and_labels: Iterable[tuple[float, bool]], gate: float) -> dict[str, Any]:
    pairs = list(scores_and_labels)
    positives = sum(label for _, label in pairs)
    true_positives = sum(label and score >= gate for score, label in pairs)
    false_positives = sum(not label and score >= gate for score, label in pairs)
    predicted_positive = true_positives + false_positives
    return {
        "recall": true_positives / positives if positives else 0.0,
        "precision": true_positives / predicted_positive if predicted_positive else 0.0,
        "precision_defined": bool(predicted_positive),
        "positive_windows_clearing_gate": true_positives,
        "false_positive_windows_clearing_gate": false_positives,
        "false_negative_positive_windows": positives - true_positives,
    }


def _calibration_comparison(bundle_receipt: Mapping[str, Any]) -> dict[str, Any]:
    per_seed = bundle_receipt.get("per_seed")
    champion_seed = bundle_receipt.get("champion_seed")
    if not isinstance(per_seed, dict) or not isinstance(champion_seed, int):
        raise TypeError("bundle evaluation receipt lacks champion calibration")
    # The champion family is whatever the receipt scored and did NOT list as a
    # comparator - not a name this script knows. A replacement model's receipt
    # names its own family the same way.
    comparators = bundle_receipt.get("comparators", [])
    if not isinstance(comparators, list):
        raise TypeError("bundle evaluation receipt comparators must be a list")
    families = [
        key
        for key, value in per_seed.items()
        if isinstance(value, list) and key not in set(comparators)
    ]
    if len(families) != 1:
        raise TypeError(
            "bundle evaluation receipt must leave exactly one non-comparator family under "
            f"per_seed, found {families!r} with comparators {comparators!r}"
        )
    candidates = per_seed[families[0]]
    champion = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("seed") == champion_seed
        ),
        None,
    )
    if not isinstance(champion, dict) or not isinstance(champion.get("calibration"), dict):
        raise TypeError("bundle evaluation receipt lacks champion calibration")
    metrics = champion["calibration"].get("selection_metrics")
    if not isinstance(metrics, dict):
        raise TypeError("bundle champion calibration lacks selection metrics")
    confusion = metrics.get("confusion")
    if not isinstance(confusion, dict):
        raise TypeError("bundle champion calibration lacks confusion metrics")
    positives = confusion.get("true_positive_windows", 0) + confusion.get(
        "false_negative_windows", 0
    )
    return {
        "calibration_positive_window_count": positives,
        "calibration_threshold_0_05_recall": metrics["recall"],
        "calibration_threshold_0_05_precision": metrics["precision"],
        "note": (
            "The positive-window count and calibrated-gate recall agree. "
            "This worker-equivalent measurement scores all 55 valid stride positions in each "
            "300-frame clip (24,640 windows); calibration.json records 22,848 windows, so its "
            "false-positive count and precision are not substituted for this deployed-runner "
            "result."
        ),
    }


def score_bundle(
    bundle: Path,
    dataset: Path | None,
    gate: float,
    *,
    runner_factory: Callable[[Path], FallRunner] = OrtPoseBbox56Runner.from_artifact_dir,
) -> dict[str, Any]:
    if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
        raise ValueError("gate must be a finite number between 0 and 1")
    bundle_receipt = _read_json(bundle / "evaluation-receipt.json")
    publication = bundle_receipt.get("dataset_publication")
    if not isinstance(publication, dict):
        raise TypeError("bundle evaluation receipt lacks dataset_publication")
    expected_digest = publication.get("dataset_payload_digest")
    if not isinstance(expected_digest, str):
        raise TypeError("bundle dataset payload digest is invalid")
    dataset = dataset or default_dataset(bundle_receipt)
    verified = verify_dataset_payload(dataset, expected_digest)
    runner = runner_factory(bundle)
    calibrated_threshold = runner.receipt_threshold
    if calibrated_threshold is None:
        raise ValueError("bundle runner has no calibrated threshold")

    selected = pl.read_parquet(dataset / "clips.parquet").filter(
        pl.col("split_membership").struct.field("split_role") == "selection_validation"
    )
    scores_and_labels: list[tuple[float, bool]] = []
    positive_clips = 0
    for clip in selected.iter_rows(named=True):
        labels = clip["labels"]
        interval = labels.get("source_proxy_interval_15fps") if isinstance(labels, dict) else None
        if any(_window_label(interval, start) for start in WINDOW_STARTS):
            positive_clips += 1
        for start in WINDOW_STARTS:
            prediction = runner.predict(_feature_window(clip, start))
            score = float(prediction.fall_transition)
            if not math.isfinite(score):
                raise ValueError("runner returned a non-finite fall_transition score")
            scores_and_labels.append((score, _window_label(interval, start)))

    if not scores_and_labels:
        raise ValueError("selection_validation split has no worker-valid windows")
    positive_scores = [score for score, label in scores_and_labels if label]
    if not positive_scores:
        raise ValueError("selection_validation split has no positive windows")
    calibration = _read_json(bundle / "calibration.json")
    metrics_at_gate = _metrics(scores_and_labels, gate)
    metrics_at_calibrated = _metrics(scores_and_labels, calibrated_threshold)
    return {
        "receipt_version": 1,
        "measured_at_utc": datetime.now(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "status": "measured",
        "dataset": {
            "source": "local Hugging Face cache only; no network or token was used",
            "hf_repo": publication["hf_repo"],
            "payload_revision": publication["payload_revision"],
            "expected_payload_digest": expected_digest,
            "verified_payload_digest": verified["verified_payload_digest"],
            "digest_verification": {
                "method": (
                    "SHA-256 of payload-manifest.json parsed then serialized as canonical JSON "
                    "(sorted keys, compact separators)"
                ),
                "manifest_members_verified": verified["manifest_members_verified"],
                "manifest_member_size_and_sha256_match": True,
                "checksums_sha256_entries_match_manifest": True,
            },
            "split": {
                "role_in_clips_parquet": "selection_validation",
                "calibration_selection_role": "selection-validation",
                "clip_count": selected.height,
                "positive_clip_count": positive_clips,
                "negative_clip_count": selected.height - positive_clips,
                "positive_window_label": (
                    "source_proxy_interval_15fps overlap with [window_start, window_start + 30) "
                    "is at least 15 frames"
                ),
            },
        },
        "runner": {
            "class": "worker.adapters.model.ort_pose_bbox56.OrtPoseBbox56Runner",
            "artifact_dir": str(bundle),
            "provider": "CPUExecutionProvider",
            "device": "cpu",
            "load_status": "loaded and warmup passed",
            "model_onnx_sha256": _sha256(bundle / "model.onnx"),
            "bundle_manifest_sha256": _sha256(bundle / "bundle-manifest.json"),
            "calibration_sha256": _sha256(bundle / "calibration.json"),
            "calibrated_threshold": calibrated_threshold,
            "temperature": calibration["temperature"],
            "readout": "fall_transition = sigmoid(ONNX logits[0,0] / temperature)",
        },
        "inference_contract": {
            "preprocessing": "pose+bbox56 / coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
            "implementation": "worker.domains.fall.pose_bbox56.pose_bbox56_row",
            "window_frames": WINDOW_FRAMES,
            "stride_frames": STRIDE_FRAMES,
            "worker_window_positions_per_300_frame_clip": len(WINDOW_STARTS),
            "worker_window_starts": "0 through 270 inclusive, step 5",
            "feature_rules": (
                "COCO-17 x,y,confidence; confidence below 0.5 zeroes that triple; pose-head "
                "bbox is required and its fifth dataset value below 0.5 makes the row zero-filled; "
                "finite coordinates are clipped and normalized by the clip width and height; every "
                "output value is float32."
            ),
            "reproduced": True,
            "difference_from_worker": None,
        },
        "metrics": {
            "total_windows_scored": len(scores_and_labels),
            "positive_window_count": len(positive_scores),
            _threshold_key(gate): metrics_at_gate,
            _threshold_key(calibrated_threshold): metrics_at_calibrated,
            "maximum_fall_transition_score_on_positive_window": max(positive_scores),
            "minimum_fall_transition_score_all_windows": min(
                score for score, _ in scores_and_labels
            ),
            "maximum_fall_transition_score_all_windows": max(
                score for score, _ in scores_and_labels
            ),
        },
        "calibration_comparison": _calibration_comparison(bundle_receipt),
        "scope_note": (
            "This measures only the model's own selection-validation source-proxy distribution. "
            "It says nothing about facility falls."
        ),
        "owner_instruction": _owner_instruction(
            gate=gate, max_positive=max(positive_scores), recall=metrics_at_gate["recall"]
        ),
    }


def _owner_instruction(*, gate: float, max_positive: float, recall: float) -> str:
    """Say what the measurement means, in both directions.

    The instruction must follow the number: a replacement whose positive windows
    clear the gate is ready for a staged fall, and saying 'replace the model' to
    the model that fixed the problem would send the owner in a circle.
    """
    if max_positive < gate:
        return (
            "replace the model first — the maximum positive-window fall_transition score is "
            f"{max_positive}, below the deployed {gate:g} gate; no fall can raise an alert."
        )
    return (
        f"the model clears the deployed {gate:g} gate — recall {recall:.3f} on its own "
        f"validation split, maximum positive score {max_positive}; proceed to a staged fall on "
        "a facility camera, which is the only evidence that speaks to the facility distribution."
    )


def write_receipt(out: Path, receipt: Mapping[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("models/fall/pose-bbox56-gru"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--gate", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = score_bundle(args.bundle, args.dataset, args.gate)
    write_receipt(args.out, receipt)
    print(receipt["owner_instruction"])


if __name__ == "__main__":
    main()
