"""Manifest-driven DeepStream runtime preflight for the public ``nvidia`` profile.

The image sets ``SEEON_DEEPSTREAM_MANIFEST``. Outside that image the existing
profile probes remain usable by unit tests and developer environments; inside it,
this gate runs before the ordinary CUDA/device-residency probe and therefore
before model construction or camera activation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Never

from worker.native.deepstream.engine_cache import EngineCacheError, verify_plan_cache
from worker.native.deepstream.export import artifact_sha256

CommandRunner = Callable[[tuple[str, ...], float], str]
MountProbe = Callable[[Path], bool]

MANIFEST_ENV: Final = "SEEON_DEEPSTREAM_MANIFEST"
FIRST_FAULT_ENV: Final = "SEEON_DEEPSTREAM_FIRST_FAULT"
ENGINE_IDENTITY_FILENAME: Final = ".identity.json"


@dataclass(frozen=True, slots=True)
class DeepStreamPreflightError(RuntimeError):
    """One typed refusal that must stop boot before source activation."""

    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeepStreamPreflightError("manifest_invalid", str(exc)) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise DeepStreamPreflightError("manifest_invalid", "schema_version must equal 1")
    return value


def _engine_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    cache = manifest.get("engine_cache")
    if not isinstance(cache, dict) or not isinstance(cache.get("engines"), list):
        raise DeepStreamPreflightError("manifest_invalid", "engine_cache.engines must be a list")
    identity: dict[str, str] = {}
    for raw in cache["engines"]:
        if not isinstance(raw, dict):
            raise DeepStreamPreflightError("manifest_invalid", "engine entry must be an object")
        name, digest = raw.get("name"), raw.get("sha256")
        if not isinstance(name, str) or not name or not isinstance(digest, str):
            raise DeepStreamPreflightError("manifest_invalid", "engine name/sha256 is invalid")
        identity[name] = digest
    return identity


def prepare_engine_cache(manifest_path: Path | str) -> dict[str, str]:
    """Populate only the writable engine cache in an explicit build/cache step."""
    manifest = _load_manifest(Path(manifest_path))
    cache = manifest["engine_cache"]
    target = Path(cache["path"])
    target.mkdir(parents=True, exist_ok=True)
    identity = _engine_identity(manifest)
    for raw in cache["engines"]:
        source = Path(raw["source"])
        if not source.is_file() or artifact_sha256(source) != raw["sha256"]:
            raise DeepStreamPreflightError(
                "engine_source_mismatch", f"engine source identity failed for {raw['name']}"
            )
        destination = target / f"{raw['name']}.engine"
        with tempfile.NamedTemporaryFile(dir=target, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, temporary)
        temporary_path.chmod(0o444)
        temporary_path.replace(destination)
    _write_json_atomic(target / ENGINE_IDENTITY_FILENAME, identity)
    return identity


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as out:
        out.write(encoded)
        temporary = Path(out.name)
    temporary.replace(path)


def _default_command_runner(command: tuple[str, ...], timeout: float) -> str:
    completed = subprocess.run(  # noqa: S603 - every executable is image-owned manifest policy
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout


def _default_mount_is_read_only(path: Path) -> bool:
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def _version_tuple(value: str) -> tuple[int, ...]:
    pieces = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    return tuple(pieces)


def _probe_gpu(runtime: Mapping[str, Any], run: CommandRunner) -> dict[str, str]:
    try:
        output = run(
            (
                "nvidia-smi",
                "--query-gpu=driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ),
            10.0,
        ).strip()
    except Exception as exc:  # noqa: BLE001 - normalized into a typed boot refusal
        raise DeepStreamPreflightError("gpu_absent", str(exc)) from exc
    first = output.splitlines()[0] if output else ""
    fields = tuple(part.strip() for part in first.replace(",", "|").split("|"))
    if len(fields) != 2:
        raise DeepStreamPreflightError("gpu_absent", "nvidia-smi returned no usable GPU")
    driver, compute = fields
    if _version_tuple(driver) < _version_tuple(str(runtime["minimum_driver"])):
        raise DeepStreamPreflightError(
            "driver_version_mismatch", f"driver {driver} is below {runtime['minimum_driver']}"
        )
    if _version_tuple(compute) < _version_tuple(str(runtime["minimum_compute_capability"])):
        raise DeepStreamPreflightError(
            "compute_capability_mismatch",
            f"compute capability {compute} is below {runtime['minimum_compute_capability']}",
        )
    return {"driver": driver, "compute_capability": compute}


def _expect_command_version(
    run: CommandRunner,
    command: tuple[str, ...],
    expected: str,
    code: str,
) -> str:
    try:
        output = run(command, 10.0)
    except Exception as exc:  # noqa: BLE001 - normalized into a typed boot refusal
        raise DeepStreamPreflightError(code, str(exc)) from exc
    if expected not in output:
        raise DeepStreamPreflightError(code, f"expected {expected!r} in {output.strip()!r}")
    return expected


def _fail(error: DeepStreamPreflightError, first_fault_path: Path | None) -> Never:
    if first_fault_path is not None:
        _write_json_atomic(
            first_fault_path,
            {
                "code": error.code,
                "detail": error.detail,
                "profile": "nvidia",
                "stage": "deepstream_preflight",
                "status": "refused",
            },
        )
    raise error


def _run_deepstream_preflight(
    manifest_path: Path | str,
    command_runner: CommandRunner,
    mount_is_read_only: MountProbe,
) -> dict[str, Any]:
    manifest = _load_manifest(Path(manifest_path))
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise DeepStreamPreflightError("manifest_invalid", "runtime must be an object")
    gpu = _probe_gpu(runtime, command_runner)
    _expect_command_version(
        command_runner,
        ("dpkg-query", "-W", "-f=${Version}\\n", "cuda-cudart-13-2"),
        str(runtime["cuda"]),
        "cuda_version_mismatch",
    )
    _expect_command_version(
        command_runner,
        ("dpkg-query", "-W", "-f=${Version}\\n", "libnvinfer10"),
        str(runtime["tensorrt"]),
        "tensorrt_version_mismatch",
    )
    version_path = runtime.get("deepstream_version_path")
    if version_path is not None:
        text = Path(version_path).read_text(encoding="utf-8")
        if f"Version: {runtime['deepstream']}" not in text:
            raise DeepStreamPreflightError(
                "deepstream_version_mismatch", f"expected DeepStream {runtime['deepstream']}"
            )
    plugins = manifest.get("plugins")
    if not isinstance(plugins, list) or not plugins:
        raise DeepStreamPreflightError("manifest_invalid", "plugins must be non-empty")
    for plugin in plugins:
        element = str(plugin["element"])
        _expect_command_version(
            command_runner,
            ("gst-inspect-1.0", element),
            str(plugin["version"]),
            "plugin_version_mismatch",
        )
        path = Path(plugin["path"])
        if not path.is_file() or artifact_sha256(path) != plugin["sha256"]:
            raise DeepStreamPreflightError(
                "plugin_digest_mismatch", f"plugin identity failed for {element}"
            )
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise DeepStreamPreflightError("manifest_invalid", "models must be an object")
    model_path = Path(models["path"])
    if not model_path.is_dir():
        raise DeepStreamPreflightError("model_target_missing", str(model_path))
    if models.get("require_read_only") is not True or not mount_is_read_only(model_path):
        raise DeepStreamPreflightError("model_target_writable", str(model_path))
    native = manifest.get("native")
    if not isinstance(native, dict):
        raise DeepStreamPreflightError("manifest_invalid", "native must be an object")
    native_path = Path(native["path"])
    if not native_path.is_file() or artifact_sha256(native_path) != native["sha256"]:
        raise DeepStreamPreflightError("native_digest_mismatch", str(native_path))
    cache = manifest["engine_cache"]
    cache_path = Path(cache["path"])
    expected_identity = _engine_identity(manifest)
    identity_path = cache_path / ENGINE_IDENTITY_FILENAME
    try:
        cached_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeepStreamPreflightError("engine_identity_mismatch", str(exc)) from exc
    if cached_identity != expected_identity:
        raise DeepStreamPreflightError("engine_identity_mismatch", "cache manifest is stale")
    for name, digest in expected_identity.items():
        engine = cache_path / f"{name}.engine"
        if not engine.is_file() or artifact_sha256(engine) != digest:
            raise DeepStreamPreflightError("engine_identity_mismatch", name)
    # C7: the three inference engines are content-addressed by the engine_plan
    # (weights + exporter + precision + builder). The deploy host builds them
    # once in an explicit step; boot only verifies and NEVER builds.
    try:
        plan_cache_dir = verify_plan_cache(manifest)
    except EngineCacheError as exc:
        raise DeepStreamPreflightError(exc.code, exc.detail) from exc
    plan_identity = json.loads(
        (plan_cache_dir / ENGINE_IDENTITY_FILENAME).read_text(encoding="utf-8")
    )
    timeout = float(manifest["warmup"]["timeout_seconds"])
    try:
        warmup_text = command_runner(
            (str(native_path), "--warmup", str(plan_cache_dir)), timeout
        )
        # nvstreammux writes its EOS diagnostic to stdout. Only the final,
        # exact native receipt is authoritative; earlier success-like text is not.
        warmup = json.loads(warmup_text.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001 - normalized into a typed boot refusal
        raise DeepStreamPreflightError("warmup_failed", str(exc)) from exc
    if warmup != {
        "status": "ok",
        "frames": 1,
        "source": "loopback",
        "engines": ["bed", "person", "pose"],
        "inference": "ok",
    }:
        raise DeepStreamPreflightError("warmup_failed", "native warmup receipt is invalid")
    return {
        "status": "ok",
        "profile": "nvidia",
        "runtime": dict(runtime),
        "gpu": gpu,
        "engines": plan_identity,
        "engine_cache_dir": str(plan_cache_dir),
        "warmup": warmup,
    }


def run_deepstream_preflight(
    manifest_path: Path | str,
    *,
    command_runner: CommandRunner = _default_command_runner,
    mount_is_read_only: MountProbe = _default_mount_is_read_only,
    first_fault_path: Path | None = None,
) -> dict[str, Any]:
    """Verify immutable runtime identity and perform one bounded NVMM warmup."""
    try:
        return _run_deepstream_preflight(manifest_path, command_runner, mount_is_read_only)
    except DeepStreamPreflightError as error:
        _fail(error, first_fault_path)


def run_configured_deepstream_preflight(
    env: Mapping[str, str] = os.environ,
) -> dict[str, Any] | None:
    """Run only when the pinned image declares its manifest location."""
    manifest = env.get(MANIFEST_ENV)
    if not manifest:
        return None
    fault = Path(env.get(FIRST_FAULT_ENV, "/var/lib/seeon-state/deepstream-first-fault.json"))
    return run_deepstream_preflight(manifest, first_fault_path=fault)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prepare-engine-cache", action="store_true")
    arguments = parser.parse_args()
    result = (
        prepare_engine_cache(arguments.manifest)
        if arguments.prepare_engine_cache
        else run_deepstream_preflight(arguments.manifest)
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
