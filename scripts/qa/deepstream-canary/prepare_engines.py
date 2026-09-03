# /// script
# requires-python = ">=3.11"
# ///
# --- How to run ---
# python prepare_engines.py /opt/seeon/deepstream-manifest.json /receipts/engine-prepare.json

"""Build or reuse the run-local content-addressed TensorRT engine plan."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

ENGINE_CACHE: Final = Path("/var/cache/seeon/tensorrt")


def _run(action: str, manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "worker.native.deepstream.engine_cache",
            action,
            str(manifest),
        ),
        check=False,
        capture_output=True,
        text=True,
    )


def _prepare_bootstrap_cache(manifest: Path) -> int:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "worker.native.deepstream.preflight",
            str(manifest),
            "--prepare-engine-cache",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stdout + completed.stderr, file=sys.stderr)
    return completed.returncode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = Path(sys.argv[1])
    receipt_path = Path(sys.argv[2])
    source = Path(os.environ.get("CANARY_ENGINE_CACHE_SOURCE", "/missing"))
    if source.is_dir():
        shutil.copytree(source, ENGINE_CACHE, dirs_exist_ok=True)
    if _prepare_bootstrap_cache(manifest) != 0:
        return 2
    verified = _run("verify", manifest)
    action = "reused"
    if verified.returncode != 0:
        built = _run("build", manifest)
        if built.returncode != 0:
            print(built.stdout + built.stderr, file=sys.stderr)
            return built.returncode
        action = "built"
    final = _run("verify", manifest)
    if final.returncode != 0:
        print(final.stdout + final.stderr, file=sys.stderr)
        return final.returncode
    plan_dirs = tuple(path for path in ENGINE_CACHE.glob("c7-*") if path.is_dir())
    if len(plan_dirs) != 1:
        print("engine plan directory is not unique", file=sys.stderr)
        return 2
    plan = plan_dirs[0]
    (ENGINE_CACHE / ".identity.json").chmod(0o444)
    (plan / ".identity.json").chmod(0o444)
    engines = {path.name: _sha256(path) for path in sorted(plan.glob("*.engine"))}
    if set(engines) != {"bed.engine", "person.engine", "pose.engine"}:
        print("engine set is incomplete", file=sys.stderr)
        return 2
    receipt = {
        "schema_version": 1,
        "action": action,
        "plan_key": plan.name,
        "manifest_sha256": _sha256(manifest),
        "identity_sha256": _sha256(plan / ".identity.json"),
        "bootstrap_identity_sha256": _sha256(ENGINE_CACHE / ".identity.json"),
        "engines": engines,
    }
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as target:
        _ = target.write(encoded)
        target.flush()
        os.fsync(target.fileno())
    print(encoded.decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
