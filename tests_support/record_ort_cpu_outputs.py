"""Record existing Python CPU outputs for the Rust binding, never GPU goldens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MANIFEST_SHA256 = "1184583122b90f5ecbe83a5f5b9c59e740a4c4701e4600b1a038c5db728cb82b"
MANIFEST = (
    Path(__file__).resolve().parents[1] / "worker/runtime/rust/tests/fixtures/gpu/manifest.json"
)
MODELS = (
    ("fall", "fall/pose-bbox56-gru/model.onnx", 0),
    ("stored_pose", "pose/yolo26n-pose.onnx", 1),
    ("bed", "bed/yolo26l-seg.onnx", 1),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(models: Path, fixtures: Path, destination: Path) -> Path:
    if not __debug__:
        raise RuntimeError("reference recording requires enabled assertions")
    if os.environ.get("ORT_DISABLE_TELEMETRY") != "1":
        raise RuntimeError("start the recorder with ORT_DISABLE_TELEMETRY=1")
    # Load the vendor only after checking the full-process opt-out.
    import numpy as np
    import onnxruntime as ort

    if ort.__version__ != "1.29.0":
        raise RuntimeError("reference recording requires the pinned ORT 1.29.0")
    manifest_bytes = MANIFEST.read_bytes()
    if sha(manifest_bytes) != MANIFEST_SHA256:
        raise RuntimeError("restore the pinned fixture manifest before recording CPU references")
    manifest = json.loads(manifest_bytes)
    runtime = Path(ort.__file__).parent / "capi" / "libonnxruntime.so.1.29.0"
    report = {
        "scope": "Python ORT CPU reference; not a CUDA oracle or GPU acceptance",
        "manifest_sha256": MANIFEST_SHA256,
        "runtime_version": ort.__version__,
        "runtime_sha256": sha(runtime.read_bytes()),
        "recorder_sha256": sha(Path(__file__).read_bytes()),
        "cases": [],
    }
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for role, model_path, threads in MODELS:
        model = (models / model_path).read_bytes()
        model_digest = sha(model)
        if model_digest != manifest["models"][role]["onnx_sha256"]:
            raise RuntimeError(f"{role}: model digest differs from the pinned input")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = threads
        session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        assert session.get_providers() == ["CPUExecutionProvider"]
        assert len(session.get_inputs()) == 1
        descriptions = (
            [("windows", manifest["fall"]["windows"])]
            if role == "fall"
            else [(entry["id"], entry["images"]) for entry in manifest[role]]
        )
        for name, description in descriptions:
            raw = (fixtures / description["path"]).read_bytes()
            pin = manifest["fixtures"][description["path"]]
            assert len(raw) == pin["bytes"] and sha(raw) == pin["sha256"]
            data = np.frombuffer(raw, dtype="<f4").reshape(description["shape"])
            inputs = (row[np.newaxis, ...].copy() for row in data) if role == "fall" else [data]
            for index, values in enumerate(inputs):
                outputs = session.run(None, {session.get_inputs()[0].name: values})
                entries = []
                for output_index, output in enumerate(outputs):
                    assert output.dtype == np.float32 and np.isfinite(output).all()
                    content = output.tobytes()
                    filename = f"reference-{len(report['cases']):04d}-{output_index}.f32"
                    with (destination / filename).open("xb") as stream:
                        stream.write(content)
                    entries.append(
                        {
                            "shape": list(output.shape),
                            "python_cpu_sha256": sha(content),
                            "reference_path": filename,
                        }
                    )
                report["cases"].append(
                    {
                        "role": role,
                        "input": name,
                        "window": index if role == "fall" else None,
                        "model_sha256": model_digest,
                        "input_sha256": sha(values.tobytes()),
                        "providers": session.get_providers(),
                        "threads": threads,
                        "outputs": entries,
                    }
                )
        del session
    assert len(report["cases"]) == 132
    path = destination / "receipt.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(record(args.models, args.fixtures, args.output))


if __name__ == "__main__":
    main()
