"""Build one immutable TensorRT engine after model provisioning."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path


class EngineBuildError(RuntimeError):
    pass


def _input_shape(onnx_path: Path) -> tuple[str, tuple[int, ...]]:
    # Read the graph through onnxruntime rather than the `onnx` package: the
    # runtime is already a Flow dependency because the fall and bed models are
    # served with it, and this tool runs inside the shipped image.
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise EngineBuildError("ONNX model must expose exactly one input tensor")
    tensor = inputs[0]
    if not tensor.shape:
        raise EngineBuildError(f"ONNX input tensor {tensor.name!r} has no shape")
    shape: list[int] = []
    for dimension in tensor.shape[1:]:
        if not isinstance(dimension, int) or dimension <= 0:
            raise EngineBuildError(
                f"ONNX input tensor {tensor.name!r} has a non-static non-batch dimension"
            )
        shape.append(dimension)
    if not tensor.name or not shape:
        raise EngineBuildError("ONNX input must have a name and at least one non-batch dimension")
    batch = tensor.shape[0]
    static_batch = batch if isinstance(batch, int) and batch > 0 else None
    return tensor.name, tuple(shape), static_batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_infer_config(template: Path, destination: Path, engine: Path, batch_size: int) -> None:
    text = template.read_text(encoding="utf-8")
    text, engine_count = re.subn(
        r"(?m)^model-engine-file=.*$",
        f"model-engine-file={engine}",
        text,
    )
    text, batch_count = re.subn(r"(?m)^batch-size=.*$", f"batch-size={batch_size}", text)
    if engine_count != 1 or batch_count != 1:
        raise EngineBuildError("Flow infer config must define one model-engine-file and batch-size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def identity_for(
    *,
    engine: Path,
    onnx: Path,
    parser_lib: Path,
    infer_config: Path,
    tracker_config: Path,
    tracker_library: Path,
    image_digest: str,
    batch_size: int,
) -> dict[str, str]:
    if not image_digest:
        raise EngineBuildError("image_digest must not be empty")
    if batch_size <= 0:
        raise EngineBuildError("batch_size must be positive")
    return {
        "engine_sha256": sha256(engine),
        "onnx_sha256": sha256(onnx),
        "parser_lib_sha256": sha256(parser_lib),
        "infer_config_sha256": sha256(infer_config),
        "tracker_config_sha256": sha256(tracker_config),
        "tracker_library_sha256": sha256(tracker_library),
        "image_digest": image_digest,
        "batch_size": str(batch_size),
    }


def build_engine(
    *,
    onnx: Path,
    engine: Path,
    identity_path: Path,
    parser_lib: Path,
    infer_config: Path,
    tracker_config: Path,
    tracker_library: Path,
    image_digest: str,
    batch_size: int,
    served_infer_config: Path | None = None,
    force: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    required = (onnx, parser_lib, infer_config, tracker_config, tracker_library)
    missing = next((path for path in required if not path.is_file()), None)
    if missing is not None:
        raise EngineBuildError(f"required Flow build artifact is absent: {missing}")
    if batch_size <= 0:
        raise EngineBuildError("batch_size must be positive")
    active_infer_config = served_infer_config or infer_config
    if served_infer_config is not None:
        _render_infer_config(infer_config, served_infer_config, engine, batch_size)
    if engine.exists() and identity_path.exists() and not force:
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and existing == identity_for(
            engine=engine,
            onnx=onnx,
            parser_lib=parser_lib,
            infer_config=active_infer_config,
            tracker_config=tracker_config,
            tracker_library=tracker_library,
            image_digest=image_digest,
            batch_size=batch_size,
        ):
            return existing
        # A changed deployment batch is a cache miss. Rebuild it before any
        # source can activate rather than booting against a stale engine.
    engine.parent.mkdir(parents=True, exist_ok=True)
    input_name, input_dimensions, static_batch = _input_shape(onnx)
    if static_batch is not None:
        # The model itself fixes the batch, and TensorRT refuses explicit
        # shapes for it. Serving a roster larger than that batch would make
        # nvinfer rebuild at runtime, so refuse here instead.
        if static_batch < batch_size:
            raise EngineBuildError(
                f"ONNX input {input_name!r} fixes batch {static_batch}, which cannot serve the "
                f"deployed roster batch {batch_size}; export the model with a dynamic batch"
            )
        shape_arguments: list[str] = []
    else:
        min_shape = "x".join(map(str, (1, *input_dimensions)))
        deployed_shape = "x".join(map(str, (batch_size, *input_dimensions)))
        shape_arguments = [
            f"--minShapes={input_name}:{min_shape}",
            f"--optShapes={input_name}:{deployed_shape}",
            f"--maxShapes={input_name}:{deployed_shape}",
        ]
    result = run(
        [
            "trtexec",
            f"--onnx={onnx}",
            f"--saveEngine={engine}",
            "--fp16",
            *shape_arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EngineBuildError(f"trtexec failed: {result.stderr.strip() or result.stdout.strip()}")
    if not engine.is_file():
        raise EngineBuildError("trtexec succeeded without creating the requested engine")
    identity = identity_for(
        engine=engine,
        onnx=onnx,
        parser_lib=parser_lib,
        infer_config=active_infer_config,
        tracker_config=tracker_config,
        tracker_library=tracker_library,
        image_digest=image_digest,
        batch_size=batch_size,
    )
    identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(prog="edge-engine-build")
    for name in (
        "onnx",
        "engine",
        "identity",
        "parser-lib",
        "infer-config",
        "tracker-config",
        "tracker-library",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--served-infer-config")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_engine(
        onnx=Path(args.onnx),
        engine=Path(args.engine),
        identity_path=Path(args.identity),
        parser_lib=Path(args.parser_lib),
        infer_config=Path(args.infer_config),
        tracker_config=Path(args.tracker_config),
        tracker_library=Path(args.tracker_library),
        image_digest=args.image_digest,
        batch_size=args.batch_size,
        served_infer_config=(
            None if args.served_infer_config is None else Path(args.served_infer_config)
        ),
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
