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


def _verify_served_infer_config(infer_config: Path, engine: Path, batch_size: int) -> None:
    text = infer_config.read_text(encoding="utf-8")
    engine_values = re.findall(r"(?m)^model-engine-file=(.*)$", text)
    batch_values = re.findall(r"(?m)^batch-size=(.*)$", text)
    if engine_values != [str(engine)] or batch_values != [str(batch_size)]:
        raise EngineBuildError(
            "served Flow infer config must name the requested engine and deployed batch"
        )


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


def _nvinfer_build_command(*, infer_config: Path, batch_size: int) -> list[str]:
    command = [
        "gst-launch-1.0",
        "-e",
        "nvstreammux",
        "name=mux",
        f"batch-size={batch_size}",
        "width=640",
        "height=640",
        "live-source=0",
        "batched-push-timeout=40000",
        "!",
        "nvinfer",
        f"config-file-path={infer_config}",
        "!",
        "fakesink",
        "sync=false",
    ]
    for source_index in range(batch_size):
        command.extend(
            [
                "videotestsrc",
                "num-buffers=1",
                "pattern=black",
                "!",
                "video/x-raw,format=I420,width=640,height=640,framerate=30/1",
                "!",
                "nvvideoconvert",
                "!",
                "video/x-raw(memory:NVMM),format=NV12",
                "!",
                f"mux.sink_{source_index}",
            ]
        )
    return command


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
    _verify_served_infer_config(active_infer_config, engine, batch_size)
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
    input_name, _, static_batch = _input_shape(onnx)
    # The model itself fixes the batch, and TensorRT refuses explicit shapes for
    # it. Serving a roster larger than that batch would make nvinfer rebuild at
    # runtime, so refuse here instead.
    if static_batch is not None and static_batch < batch_size:
        raise EngineBuildError(
            f"ONNX input {input_name!r} fixes batch {static_batch}, which cannot serve the "
            f"deployed roster batch {batch_size}; export the model with a dynamic batch"
        )
    engine.unlink(missing_ok=True)
    try:
        result = run(
            _nvinfer_build_command(infer_config=active_infer_config, batch_size=batch_size),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise EngineBuildError(f"nvinfer build pipeline could not start: {error}") from error
    if result.returncode != 0:
        raise EngineBuildError(
            "nvinfer build pipeline failed: "
            f"{result.stderr.strip() or result.stdout.strip() or result.returncode}"
        )
    if not engine.is_file():
        # When nvinfer builds rather than deserialises, it writes the engine
        # beside the ONNX under its own name and ignores `model-engine-file`.
        # That file is the artefact that serves - verified on hardware - so
        # adopt it into the configured path instead of failing.
        produced = onnx.with_name(f"{onnx.name}_b{batch_size}_gpu0_fp16.engine")
        if not produced.is_file():
            raise EngineBuildError(
                "nvinfer build pipeline succeeded without creating an engine at either "
                f"{engine} or {produced}"
            )
        engine.write_bytes(produced.read_bytes())
        produced.unlink()
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
