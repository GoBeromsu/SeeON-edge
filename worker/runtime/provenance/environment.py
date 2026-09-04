from __future__ import annotations

import platform
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from worker.adapters.device.nvml.probe import probe_nvml_gpu_status
from worker.runtime.profile.boot import BootContext
from worker.runtime.provenance.models import RuntimeEnvironmentFacts

_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ZERO_REVISION = "0" * 40
_PACKAGED_IMAGE_REVISION_PATH = Path("/opt/seeon/ml-worker-image-revision")
_GitCommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]


def _is_valid_source_revision(candidate: str) -> bool:
    return candidate != _ZERO_REVISION and _SOURCE_REVISION.fullmatch(candidate) is not None


def _run_git(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=2.0,
    )


def _packaged_image_revision(path: Path) -> tuple[bool, str | None]:
    try:
        raw = path.read_text(encoding="ascii")
    except FileNotFoundError:
        return False, None
    except (OSError, UnicodeError):
        return True, None
    candidate = raw.removesuffix("\n")
    return True, candidate if _is_valid_source_revision(candidate) else None


def resolve_worker_build_revision(
    explicit: str | None,
    *,
    repository_root: Path | None = None,
    packaged_image_revision_path: Path | None = None,
    git_runner: _GitCommandRunner | None = None,
) -> str | None:
    """Resolve a packaged image identity or HEAD from a provably clean checkout."""
    marker_path = (
        _PACKAGED_IMAGE_REVISION_PATH
        if packaged_image_revision_path is None
        else packaged_image_revision_path
    )
    packaged, baked_revision = _packaged_image_revision(marker_path)
    if packaged:
        if (
            explicit is None
            or not _is_valid_source_revision(explicit)
            or explicit != baked_revision
        ):
            return None
        return explicit

    # An environment value alone does not prove that the running bytes came from
    # an immutable image. Local source execution derives identity only from Git.
    if explicit is not None:
        return None

    root = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else repository_root.resolve()
    )
    run_git = _run_git if git_runner is None else git_runner
    try:
        status = run_git(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            )
        )
        if status.returncode != 0 or status.stdout:
            return None
        revision = run_git(("git", "-C", str(root), "rev-parse", "--verify", "HEAD"))
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = revision.stdout.strip()
    if revision.returncode != 0 or not _is_valid_source_revision(candidate):
        return None
    return candidate


def collect_runtime_environment_facts(
    boot: BootContext,
    build_revision: str | None,
) -> RuntimeEnvironmentFacts:
    """Read an allow-listed software/driver identity; never enumerate environment values."""
    profile = boot.runtime_profile.canonical_profile
    accelerator_runtime: str | None = None
    driver_version: str | None = None
    device_name: str | None = None
    if profile == "flow":
        # P1b-AC7: the flow worker never imports Torch. Its model runtime is
        # ONNX Runtime on CPU and its accelerator facts come from NVML, which
        # is what DeepStream's own preflight reads.
        import onnxruntime

        status = probe_nvml_gpu_status()
        driver_version = status.driver_version
        device_name = status.device_name or None
        # The CUDA runtime the media plane links is the pinned DeepStream
        # image's; the driver reports the highest CUDA it supports.
        accelerator_runtime = None if driver_version is None else f"CUDA driver {driver_version}"
        model_runtime = "onnxruntime"
        model_runtime_version = str(onnxruntime.__version__)
    else:
        import torch

        if profile == "nvidia":
            cuda_version = torch.version.cuda
            accelerator_runtime = None if cuda_version is None else f"CUDA {cuda_version}"
            status = probe_nvml_gpu_status()
            driver_version = status.driver_version
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device_name = str(torch.cuda.get_device_name(0))
            elif status.device_name:
                device_name = status.device_name
        model_runtime = "torch"
        model_runtime_version = str(torch.__version__)
    return RuntimeEnvironmentFacts(
        # WorkerRuntime's constructor receives an already-resolved identity from
        # the composition root. Keeping that constructor seam explicit lets tests
        # supply deterministic provenance without impersonating a packaged image.
        worker_build_revision=(
            build_revision
            if build_revision is not None and _is_valid_source_revision(build_revision)
            else ""
        ),
        os_name=platform.system(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        model_runtime=model_runtime,
        model_runtime_version=model_runtime_version,
        accelerator_runtime=accelerator_runtime,
        driver_version=driver_version,
        device_name=device_name,
    )


__all__ = ["collect_runtime_environment_facts", "resolve_worker_build_revision"]
