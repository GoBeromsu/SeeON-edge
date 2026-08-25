"""Corpus generation, GPU parsing, and immutable execution artifact binding."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from worker.tools.deepstream_canary.models import ArtifactBindings
from worker.tools.deepstream_canary.safety import CanarySafetyError, LiveSnapshot
from worker.tools.deepstream_canary.telemetry import (
    NativeWindowSample,
    RuntimeGpuSample,
    emit_rung_receipt,
)


@dataclass(frozen=True, slots=True)
class ExecutionArtifactSources:
    worker_image: str
    support_images: tuple[str, ...]
    gate_policy: str
    compose: str


@dataclass(frozen=True, slots=True)
class ReceiptEmissionRequest:
    evidence_dir: Path
    rungs: tuple[str, ...]
    artifacts: ExecutionArtifactSources


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def generate_corpus(root: Path) -> Path:
    corpus = root / "run" / "scratch" / "loopback.mp4"
    completed = subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=15",
            "-t",
            "60",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-bf",
            "0",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
            str(corpus),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CanarySafetyError("corpus_generation_failed", completed.stderr[-500:])
    _ = corpus.chmod(0o444)
    _ = corpus.parent.chmod(0o755)
    return corpus


def gpu_sample(snapshot: LiveSnapshot) -> RuntimeGpuSample | None:
    child = tuple(
        line for line in snapshot.gpu_processes if "seeon-deep" in line
    )
    if len(child) > 1:
        raise CanarySafetyError("native_child_gpu_process_count", str(len(child)))
    if child:
        fields = tuple(item.strip() for item in child[0].split(","))
        if len(fields) != 3:
            raise CanarySafetyError("native_child_gpu_process_invalid", child[0])
        child_pid = int(fields[0])
        child_memory_mib = float(fields[2])
    else:
        process = subprocess.run(
            (
                "docker",
                "top",
                "seeon-ds-canary-ml-worker-1",
                "-eo",
                "pid,comm",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        matches = tuple(
            line.split()[0]
            for line in process.stdout.splitlines()
            if "seeon-deep" in line
        )
        if process.returncode != 0:
            return None
        if not matches:
            return None
        if len(matches) != 1:
            raise CanarySafetyError("native_child_process_count", str(len(matches)))
        child_pid = int(matches[0])
        child_memory_mib = 0.0
    return RuntimeGpuSample(
        child_pid=child_pid,
        child_memory_mib=child_memory_mib,
        global_used_mib=snapshot.gpu_used_mib,
        total_mib=snapshot.gpu_total_mib,
        utilization=snapshot.gpu_utilization,
    )


def native_windows(path: Path) -> tuple[NativeWindowSample, ...]:
    if not path.is_file():
        return ()
    return tuple(
        NativeWindowSample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def emit_receipts(request: ReceiptEmissionRequest, corpus: Path) -> tuple[Path, ...]:
    run_root = request.evidence_dir / "run"
    bindings = ArtifactBindings(
        worker_image=request.artifacts.worker_image,
        support_images=request.artifacts.support_images,
        models_manifest=_sha256_tree(run_root / "models"),
        engine_manifest=_sha256_tree(run_root / "engine-cache"),
        corpus=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        gate_policy=request.artifacts.gate_policy,
        compose=request.artifacts.compose,
    )
    receipts: list[Path] = []
    for rung in request.rungs:
        telemetry = request.evidence_dir / "raw" / f"telemetry-{rung}.json"
        if not telemetry.is_file():
            raise CanarySafetyError("rung_telemetry_missing", rung)
        receipts.append(emit_rung_receipt(telemetry, request.evidence_dir, bindings))
    return tuple(receipts)


__all__ = [
    "ExecutionArtifactSources",
    "ReceiptEmissionRequest",
    "emit_receipts",
    "generate_corpus",
    "gpu_sample",
    "native_windows",
]
