"""Export the nano pose weights to a dynamic-batch, digest-pinned ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from contracts.artifacts import pose_weight_path
from worker.adapters.model.errors import ModelLoadError
from worker.runtime.flow.onnx_shape import batch_axis_is_dynamic, input_dims

Exporter = Callable[[Path], Path]


def _export(weights: Path) -> Path:
    from ultralytics import YOLO

    return Path(YOLO(weights).export(format="onnx", imgsz=640, opset=17, dynamic=True))


def _canonicalize(payload: bytes) -> bytes:
    """Drop the two export outputs that vary between identical runs.

    ultralytics stamps ``date`` into ``metadata_props`` and the inferred
    ``value_info`` annotations differ from run to run; neither affects the
    graph, and dropping them makes the digest reproducible per source weight.
    """
    import onnx

    model = onnx.load_from_string(payload)
    retained = [entry for entry in model.metadata_props if entry.key != "date"]
    del model.metadata_props[:]
    model.metadata_props.extend(retained)
    del model.graph.value_info[:]
    return model.SerializeToString()


def _self_check(path: Path) -> None:
    import numpy
    import onnxruntime

    dims = input_dims(path)
    if not batch_axis_is_dynamic(dims):
        raise ModelLoadError(f"exported pose ONNX has fixed batch dimension {dims[0]}")
    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    frames = numpy.random.default_rng(0).random((2, 3, 640, 640), dtype=numpy.float32)
    if numpy.array_equal(frames[0], frames[1]):
        raise ModelLoadError("exported pose self-check frames must differ")
    output = session.run(None, {session.get_inputs()[0].name: frames})[0]
    if output.shape != (2, 300, 57):
        raise ModelLoadError(
            f"exported pose ONNX output shape is {output.shape}, expected (2, 300, 57)"
        )


def export_pose_onnx(
    model_path: Path | None = None, *, force: bool = False, exporter: Exporter | None = None
) -> str:
    """Export the nano pose model, self-check it at B=2, and return its digest."""
    source = (pose_weight_path("n") if model_path is None else model_path).expanduser().resolve()
    if not source.is_file():
        raise ModelLoadError(f"pose weights do not exist: {source}")
    destination = source.with_suffix(".onnx")
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    try:
        with tempfile.TemporaryDirectory(dir=source.parent, prefix="pose-export-") as work:
            temporary_weights = Path(work) / source.name
            shutil.copy2(source, temporary_weights)
            exported = (exporter or _export)(temporary_weights)
            payload = _canonicalize(exported.read_bytes())
            checked = Path(work) / "checked.onnx"
            checked.write_bytes(payload)
            _self_check(checked)
    except ModelLoadError:
        raise
    except Exception as error:
        raise ModelLoadError(f"cannot export pose ONNX model: {error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    if (
        destination.exists()
        and hashlib.sha256(destination.read_bytes()).hexdigest() != digest
        and not force
    ):
        raise ModelLoadError("yolo26n-pose.onnx differs; pass --force to overwrite it")
    destination.write_bytes(payload)
    sidecar.write_text(f"{digest}\n", encoding="ascii")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?", type=Path, default=pose_weight_path("n"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(export_pose_onnx(args.model_path, force=args.force))


if __name__ == "__main__":
    main()
