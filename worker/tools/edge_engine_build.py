"""Build one immutable TensorRT engine after model provisioning."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path


class EngineBuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_for(
    *,
    engine: Path,
    onnx: Path,
    parser_lib: Path,
    infer_config: Path,
    tracker_config: Path,
    image_digest: str,
) -> dict[str, str]:
    if not image_digest:
        raise EngineBuildError("image_digest must not be empty")
    return {
        "engine_sha256": sha256(engine),
        "onnx_sha256": sha256(onnx),
        "parser_lib_sha256": sha256(parser_lib),
        "infer_config_sha256": sha256(infer_config),
        "tracker_config_sha256": sha256(tracker_config),
        "image_digest": image_digest,
    }


def build_engine(
    *,
    onnx: Path,
    engine: Path,
    identity_path: Path,
    parser_lib: Path,
    infer_config: Path,
    tracker_config: Path,
    image_digest: str,
    force: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    required = (onnx, parser_lib, infer_config, tracker_config)
    missing = next((path for path in required if not path.is_file()), None)
    if missing is not None:
        raise EngineBuildError(f"required Flow build artifact is absent: {missing}")
    if engine.exists() and identity_path.exists() and not force:
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and existing == identity_for(
            engine=engine,
            onnx=onnx,
            parser_lib=parser_lib,
            infer_config=infer_config,
            tracker_config=tracker_config,
            image_digest=image_digest,
        ):
            return existing
        raise EngineBuildError("existing engine identity differs; rerun edge-engine-build --force")
    engine.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        ["trtexec", f"--onnx={onnx}", f"--saveEngine={engine}", "--fp16"],
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
        infer_config=infer_config,
        tracker_config=tracker_config,
        image_digest=image_digest,
    )
    identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(prog="edge-engine-build")
    for name in ("onnx", "engine", "identity", "parser-lib", "infer-config", "tracker-config"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_engine(
        onnx=Path(args.onnx),
        engine=Path(args.engine),
        identity_path=Path(args.identity),
        parser_lib=Path(args.parser_lib),
        infer_config=Path(args.infer_config),
        tracker_config=Path(args.tracker_config),
        image_digest=args.image_digest,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
