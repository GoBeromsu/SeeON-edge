"""Content-addressed TensorRT engine cache for the ``nvidia`` cutover.

The baked manifest carries an ``engine_plan`` (weights + exporter + precision +
builder identity, NO digests -- weights never enter the image). The deploy host
runs the explicit build step once with the models mounted and a GPU present;
boot preflight only *verifies* the resulting cache and never builds.

The cache subdirectory is keyed by the plan identity, so the C6 image's empty
cache-root identity (``.identity.json`` at the cache root) stays untouched and
old-image rollback keeps passing its own preflight (C7 rollback hazard).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

PLAN_PREFIX: Final = "c7"
PLAN_IDENTITY_FILENAME: Final = ".identity.json"
REQUIRED_MODELS: Final = frozenset({"bed", "person", "pose"})
_TRTEXEC: Final = "/usr/src/tensorrt/bin/trtexec"
# Dynamic spatial profile covering every letterbox outcome of supported
# geometries (16:9 -> 384x640, 4:3 -> 512x640, square -> 640x640, portrait ->
# 640x384): min covers the smallest padded tensor either axis can take.
_SHAPE_ARGS: Final = (
    "--minShapes=images:1x3x384x384",
    "--optShapes=images:1x3x384x640",
    "--maxShapes=images:1x3x640x640",
)

EngineBuilder = Callable[[Path, Path], None]


@dataclass(frozen=True, slots=True)
class EngineCacheError(RuntimeError):
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class EnginePlan:
    cache_dir: Path
    plan_key: str
    weights: Mapping[str, Path]
    exporter: str
    precision: str
    builder: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_plan(manifest: Mapping[str, Any]) -> EnginePlan:
    """Compute the content-addressed cache directory for this host's weights."""
    plan = manifest.get("engine_plan")
    if not isinstance(plan, dict):
        raise EngineCacheError("engine_plan_missing", "manifest carries no engine_plan")
    models = plan.get("models")
    if not isinstance(models, list) or not models:
        raise EngineCacheError("engine_plan_invalid", "engine_plan.models must be non-empty")
    weights: dict[str, Path] = {}
    for entry in models:
        if not isinstance(entry, dict):
            raise EngineCacheError("engine_plan_invalid", "model entry must be an object")
        name, weight = entry.get("name"), entry.get("weight")
        if not isinstance(name, str) or not isinstance(weight, str):
            raise EngineCacheError("engine_plan_invalid", "model name/weight is invalid")
        weights[name] = Path(weight)
    if set(weights) != set(REQUIRED_MODELS):
        raise EngineCacheError(
            "engine_plan_invalid",
            f"engine_plan must name exactly {sorted(REQUIRED_MODELS)}, got {sorted(weights)}",
        )
    exporter = str(plan.get("exporter", ""))
    precision = str(plan.get("precision", ""))
    builder = str(plan.get("builder", ""))
    if precision != "fp32":
        raise EngineCacheError("engine_plan_invalid", f"precision {precision!r} is rejected")
    fingerprint = hashlib.sha256()
    for name in sorted(weights):
        weight_path = weights[name]
        if not weight_path.is_file():
            raise EngineCacheError("engine_weight_missing", str(weight_path))
        fingerprint.update(f"{name}:{_sha256(weight_path)}\n".encode())
    fingerprint.update(f"{exporter}|{precision}|{builder}\n".encode())
    plan_key = fingerprint.hexdigest()
    cache_root = Path(str(manifest["engine_cache"]["path"]))
    return EnginePlan(
        cache_dir=cache_root / f"{PLAN_PREFIX}-{plan_key[:32]}",
        plan_key=plan_key,
        weights=dict(weights),
        exporter=exporter,
        precision=precision,
        builder=builder,
    )


def verify_plan_cache(manifest: Mapping[str, Any]) -> Path:
    """Boot-time gate: the cache must be complete and digest-exact, never built."""
    plan = resolve_plan(manifest)
    identity_path = plan.cache_dir / PLAN_IDENTITY_FILENAME
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineCacheError(
            "engine_cache_unbuilt",
            f"run the explicit engine build step first ({identity_path}): {exc}",
        ) from exc
    if not isinstance(identity, dict) or set(identity) != set(REQUIRED_MODELS):
        raise EngineCacheError("engine_identity_mismatch", "identity names are stale")
    for name, digest in identity.items():
        engine = plan.cache_dir / f"{name}.engine"
        if not engine.is_file() or _sha256(engine) != digest:
            raise EngineCacheError("engine_identity_mismatch", name)
    return plan.cache_dir


def _trtexec_builder(onnx_path: Path, engine_path: Path) -> None:
    command = (
        _TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        *_SHAPE_ARGS,
    )
    _ = subprocess.run(command, check=True, timeout=3600)  # noqa: S603 - image-owned builder


def build_plan_cache(
    manifest: Mapping[str, Any],
    *,
    engine_builder: EngineBuilder = _trtexec_builder,
    onnx_exporter: Callable[[str, Path, Path], Path] | None = None,
) -> Path:
    """Explicit deploy-host build step (GPU + read-only weights mounted)."""
    plan = resolve_plan(manifest)
    plan.cache_dir.mkdir(parents=True, exist_ok=True)
    identity: dict[str, str] = {}
    with tempfile.TemporaryDirectory(dir=plan.cache_dir.parent) as scratch:
        for name in sorted(plan.weights):
            onnx_path = _export_onnx(name, plan.weights[name], Path(scratch), onnx_exporter)
            engine_path = plan.cache_dir / f"{name}.engine"
            engine_builder(onnx_path, engine_path)
            if not engine_path.is_file():
                raise EngineCacheError("engine_build_failed", name)
            engine_path.chmod(0o444)
            identity[name] = _sha256(engine_path)
    identity_path = plan.cache_dir / PLAN_IDENTITY_FILENAME
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=plan.cache_dir, delete=False, encoding="utf-8"
    ) as out:
        _ = out.write(encoded)
        temporary = Path(out.name)
    _ = temporary.replace(identity_path)
    return plan.cache_dir


def _export_onnx(
    name: str,
    weight: Path,
    scratch: Path,
    onnx_exporter: Callable[[str, Path, Path], Path] | None,
) -> Path:
    if onnx_exporter is not None:
        return onnx_exporter(name, weight, scratch)
    from worker.native.deepstream.export import MODEL_EXPORTS

    spec = MODEL_EXPORTS[name]
    export_root = scratch / name / "repo"
    export_weight = export_root / spec.weight_path
    export_weight.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copyfile(weight, export_weight)
    output_dir = scratch / name / "onnx"
    command = (
        sys.executable,
        "-m",
        "worker.native.deepstream.export",
        "--repo-root",
        str(export_root),
        "--output-dir",
        str(output_dir),
        "--tasks",
        name,
    )
    _ = subprocess.run(command, check=True, timeout=600)  # noqa: S603 - image-owned exporter
    return output_dir / f"{name}.onnx"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "resolve"))
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    loaded = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    if arguments.action == "build":
        print(build_plan_cache(loaded))
    elif arguments.action == "verify":
        print(verify_plan_cache(loaded))
    else:
        print(resolve_plan(loaded).cache_dir)


__all__ = [
    "PLAN_IDENTITY_FILENAME",
    "PLAN_PREFIX",
    "REQUIRED_MODELS",
    "EngineCacheError",
    "EnginePlan",
    "build_plan_cache",
    "resolve_plan",
    "verify_plan_cache",
]
