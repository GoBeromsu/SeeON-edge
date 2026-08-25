"""Export the shipped pose/person/bed artifacts to ONNX and bind their identity.

The manifest is the contract between an exported engine and the native parser:
it binds the source artifact SHA, the exporter/library version, input/output
names, the dynamic batch profile, the precision and the parser/preprocess
digests. C2's preflight refuses to boot on any drift.

FP32 is the only accepted initial precision. FP16 stays rejected until the full
parity suite independently passes -- a half-precision engine changes confidences
in the last mantissa bits, which silently perturbs the strict ``score > 0.05``
pose cut and every downstream event timeline that depends on it.

Running the ONNX export requires the model artifacts and ultralytics on disk, so
``main`` is a developer/build-step entry point, never an import-time action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from worker.native.deepstream.parity.geometry import (
    BOX_INVERSE_IDENTITY,
    KEYPOINT_INVERSE_IDENTITY,
    LETTERBOX_PAD_VALUE,
    LETTERBOX_SIZE,
    LETTERBOX_STRIDE,
)
from worker.native.deepstream.parity.parse import (
    BED_PROTOTYPE_SHAPE,
    BED_ROW_STRIDE,
    MAX_ROWS,
    PERSON_ROW_STRIDE,
    POSE_ROW_STRIDE,
    POSE_SCORE_THRESHOLD,
)
from worker.native.deepstream.parity.preprocess import (
    NATIVE_CHANNEL_ORDER,
    PREPROCESS_IDENTITY,
    TENSOR_PRECISION,
)

MANIFEST_SCHEMA_VERSION: Final = 1
ACCEPTED_PRECISION: Final = "fp32"
MAX_DYNAMIC_BATCH: Final = 8
ONNX_OPSET: Final = 17
INPUT_NAME: Final = "images"


@dataclass(frozen=True, slots=True)
class ModelExportSpec:
    """One model's export identity and the tensor shape its parser expects."""

    task: str
    weight_path: str
    output_name: str
    output_shape: tuple[int, int]
    preprocessing_identity: str
    prototype_name: str | None = None
    prototype_shape: tuple[int, int, int] | None = None


MODEL_EXPORTS: Final[dict[str, ModelExportSpec]] = {
    "pose": ModelExportSpec(
        task="pose",
        weight_path="models/pose/yolo26n-pose.pt",
        output_name="output0",
        output_shape=(MAX_ROWS, POSE_ROW_STRIDE),
        preprocessing_identity="rgb24-to-coco17.v1",
    ),
    "person": ModelExportSpec(
        task="person",
        weight_path="models/person/yolo26n.pt",
        output_name="output0",
        output_shape=(MAX_ROWS, PERSON_ROW_STRIDE),
        preprocessing_identity="rgb24-to-person-boxes.v1",
    ),
    "bed": ModelExportSpec(
        task="bed",
        weight_path="models/bed/yolo26m-seg.pt",
        output_name="output0",
        output_shape=(MAX_ROWS, BED_ROW_STRIDE),
        preprocessing_identity="rgb24-to-bed-regions.v1",
        prototype_name="output1",
        prototype_shape=BED_PROTOTYPE_SHAPE,
    ),
}


def validate_precision(precision: str) -> str:
    """Accept only FP32; every other precision is a typed refusal.

    FP16 is named explicitly because it is the tempting one: it is faster, it
    "looks" equivalent, and it is exactly what silently breaks the strict pose
    threshold. It stays rejected until the full parity suite passes on its own.
    """
    if precision != ACCEPTED_PRECISION:
        raise ValueError(
            f"precision {precision!r} is rejected: {ACCEPTED_PRECISION} is the only accepted "
            "initial precision (fp16 remains rejected until the full parity suite passes)"
        )
    return precision


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser_digest() -> str:
    """Hash the parser/preprocess constants the native code must agree on.

    Any change to a threshold, stride, pad value or inverse rule moves this
    digest, so a stale engine paired with a newer parser fails preflight instead
    of silently producing different detections.
    """
    payload = json.dumps(
        {
            "box_inverse": BOX_INVERSE_IDENTITY,
            "channel_order": NATIVE_CHANNEL_ORDER,
            "keypoint_inverse": KEYPOINT_INVERSE_IDENTITY,
            "letterbox_pad_value": LETTERBOX_PAD_VALUE,
            "letterbox_size": LETTERBOX_SIZE,
            "letterbox_stride": LETTERBOX_STRIDE,
            "max_rows": MAX_ROWS,
            "pose_score_threshold": POSE_SCORE_THRESHOLD,
            "preprocess_identity": PREPROCESS_IDENTITY,
            "row_strides": {
                "bed": BED_ROW_STRIDE,
                "person": PERSON_ROW_STRIDE,
                "pose": POSE_ROW_STRIDE,
            },
            "tensor_precision": TENSOR_PRECISION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_export_manifest(
    spec: ModelExportSpec,
    *,
    weight_path: Path,
    onnx_path: Path,
    exporter_version: str,
    precision: str = ACCEPTED_PRECISION,
    max_batch: int = MAX_DYNAMIC_BATCH,
) -> dict[str, Any]:
    """Bind one exported artifact to the parser that is allowed to read it."""
    outputs: list[dict[str, Any]] = [
        {"name": spec.output_name, "shape": list(spec.output_shape)}
    ]
    if spec.prototype_name is not None and spec.prototype_shape is not None:
        outputs.append(
            {"name": spec.prototype_name, "shape": list(spec.prototype_shape)}
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task": spec.task,
        "source_artifact": {
            "path": str(spec.weight_path),
            "sha256": artifact_sha256(weight_path),
        },
        "exported_artifact": {
            "path": onnx_path.name,
            "sha256": artifact_sha256(onnx_path) if onnx_path.is_file() else None,
        },
        "exporter": {"library": "ultralytics", "version": exporter_version, "opset": ONNX_OPSET},
        "input": {
            "name": INPUT_NAME,
            "layout": "NCHW",
            "channel_order": NATIVE_CHANNEL_ORDER,
            "normalization": "divide_255",
        },
        "outputs": outputs,
        "dynamic_batch": {"min": 1, "opt": 1, "max": max_batch},
        "precision": validate_precision(precision),
        "preprocess": {
            "identity": PREPROCESS_IDENTITY,
            "adapter_identity": spec.preprocessing_identity,
            "letterbox_size": LETTERBOX_SIZE,
            "letterbox_stride": LETTERBOX_STRIDE,
            "pad_value": LETTERBOX_PAD_VALUE,
        },
        "parser_digest": parser_digest(),
    }


def export_onnx(
    spec: ModelExportSpec,
    *,
    repo_root: Path,
    output_dir: Path,
    precision: str = ACCEPTED_PRECISION,
    max_batch: int = MAX_DYNAMIC_BATCH,
) -> dict[str, Any]:
    """Export one shipped artifact to ONNX and write its manifest beside it."""
    validate_precision(precision)
    weight_path = repo_root / spec.weight_path
    if not weight_path.is_file():
        raise FileNotFoundError(f"{spec.task} weights are missing: {weight_path}")
    from ultralytics import YOLO
    from ultralytics import __version__ as exporter_version

    output_dir.mkdir(parents=True, exist_ok=True)
    export_weight = output_dir / weight_path.name
    _ = shutil.copyfile(weight_path, export_weight)
    exported = Path(
        YOLO(str(export_weight)).export(
            format="onnx",
            imgsz=LETTERBOX_SIZE,
            dynamic=True,
            simplify=True,
            opset=ONNX_OPSET,
            half=False,
        )
    )
    onnx_path = output_dir / f"{spec.task}.onnx"
    if exported.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(exported.read_bytes())
    manifest = build_export_manifest(
        spec,
        weight_path=weight_path,
        onnx_path=onnx_path,
        exporter_version=exporter_version,
        precision=precision,
        max_batch=max_batch,
    )
    manifest_path = output_dir / f"{spec.task}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", default=ACCEPTED_PRECISION)
    parser.add_argument("--tasks", default=",".join(sorted(MODEL_EXPORTS)))
    arguments = parser.parse_args(None if argv is None else list(argv))
    manifests = [
        export_onnx(
            MODEL_EXPORTS[task],
            repo_root=arguments.repo_root,
            output_dir=arguments.output_dir,
            precision=arguments.precision,
        )
        for task in arguments.tasks.split(",")
    ]
    print(json.dumps(manifests, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTED_PRECISION",
    "INPUT_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "MAX_DYNAMIC_BATCH",
    "MODEL_EXPORTS",
    "ONNX_OPSET",
    "ModelExportSpec",
    "artifact_sha256",
    "build_export_manifest",
    "export_onnx",
    "parser_digest",
    "validate_precision",
]
