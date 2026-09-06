"""Export the bed YOLO26 segmentation weights to a digest-pinned ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

from contracts.artifacts import bed_seg_weight_path
from worker.adapters.model.errors import ModelLoadError


def export_bed_seg_onnx(model_path: Path | None = None, *, force: bool = False) -> str:
    """Export the fixed 640px ONNX model and return its SHA-256 digest."""
    source = (bed_seg_weight_path() if model_path is None else model_path).expanduser().resolve()
    if not source.is_file():
        raise ModelLoadError(f"bed segmentation weights do not exist: {source}")
    destination = source.with_suffix(".onnx")
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    with tempfile.TemporaryDirectory(dir=source.parent, prefix="bed-seg-export-") as work:
        temporary_weights = Path(work) / source.name
        shutil.copy2(source, temporary_weights)
        try:
            from ultralytics import YOLO

            exported = Path(YOLO(temporary_weights).export(format="onnx", imgsz=640, opset=17))
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise ModelLoadError(f"cannot export bed segmentation ONNX model: {exc}") from exc
        payload = exported.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    existing_digest = _existing_digest(destination)
    if existing_digest is not None and existing_digest != digest and not force:
        raise ModelLoadError("yolo26m-seg.onnx differs; pass --force to overwrite it")
    destination.write_bytes(payload)
    sidecar.write_text(f"{digest}\n", encoding="ascii")
    return digest


def _existing_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ModelLoadError(f"cannot read existing bed segmentation ONNX model: {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?", type=Path, default=bed_seg_weight_path())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(export_bed_seg_onnx(args.model_path, force=args.force))


if __name__ == "__main__":
    main()
