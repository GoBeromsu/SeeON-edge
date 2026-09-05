"""Download, verify, and place every manifest artifact under a models root.

Idempotent: a destination whose SHA-256 already matches is skipped. Anything
else -- missing, wrong size, wrong hash, a stale ``.part`` from an interrupted
run -- is re-downloaded from scratch into a fresh temp file and renamed into
place only after the hash matches. A mismatch never leaves a file at the
final path, so a later worker boot cannot load a half-written weight.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from worker.tools.fetch_models.http_source import (
    ByteSource,
    RetryableSourceError,
    RetryPolicy,
    SourceError,
)
from worker.tools.fetch_models.manifest import SIDECAR_ROOT, Artifact, Bundle, Manifest

HF_TOKEN_ENV: Final = "HF_TOKEN"
PART_SUFFIX: Final = ".part"
_HASH_CHUNK: Final = 1 << 20
_FALL_BUNDLE_ROOT: Final = "fall/pose-bbox56-gru"
_FALL_PT_PATH: Final = f"{_FALL_BUNDLE_ROOT}/model.pt"
_FALL_ONNX_PATH: Final = f"{_FALL_BUNDLE_ROOT}/model.onnx"

Outcome = Literal["present", "fetched", "sidecar-present", "sidecar-written"]


class VerificationError(RuntimeError):
    """A downloaded body did not match the manifest's size or SHA-256."""


@dataclass(frozen=True)
class FetchResult:
    path: str
    outcome: Outcome
    sha256: str


@dataclass
class FetchReport:
    results: list[FetchResult] = field(default_factory=list)

    @property
    def fetched(self) -> int:
        return sum(1 for result in self.results if result.outcome == "fetched")

    @property
    def is_noop(self) -> bool:
        return all(result.outcome.endswith("present") for result in self.results)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _matches(path: Path, size: int | None, sha256: str) -> bool:
    if not path.is_file():
        return False
    if size is not None and path.stat().st_size != size:
        return False
    return sha256_of(path) == sha256


def _headers_for(artifact: Artifact, env: Mapping[str, str]) -> dict[str, str]:
    headers = {"User-Agent": "seeon-edge-model-fetch/1"}
    token = env.get(HF_TOKEN_ENV, "").strip()
    if token and artifact.source.wants_hf_token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _download_once(
    artifact: Artifact,
    part: Path,
    source: ByteSource,
    headers: Mapping[str, str],
) -> str:
    digest = hashlib.sha256()
    written = 0
    with part.open("wb") as handle:
        for chunk in source.stream(artifact.url, headers):
            written += len(chunk)
            if written > artifact.size:
                raise VerificationError(
                    f"{artifact.path}: body exceeds manifest size {artifact.size} bytes"
                )
            digest.update(chunk)
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    if written != artifact.size:
        raise VerificationError(
            f"{artifact.path}: size mismatch (manifest {artifact.size}, got {written})"
        )
    actual = digest.hexdigest()
    if actual != artifact.sha256:
        raise VerificationError(
            f"{artifact.path}: sha256 mismatch (manifest {artifact.sha256}, got {actual})"
        )
    return actual


def fetch_artifact(
    artifact: Artifact,
    root: Path,
    source: ByteSource,
    *,
    env: Mapping[str, str],
    retry: RetryPolicy,
    force: bool = False,
    log: Callable[[str], None] = lambda _message: None,
) -> FetchResult:
    dest = root / artifact.path
    part = dest.with_name(dest.name + PART_SUFFIX)
    if not force and _matches(dest, artifact.size, artifact.sha256):
        return FetchResult(artifact.path, "present", artifact.sha256)
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = _headers_for(artifact, env)
    attempt = 1
    while True:
        try:
            log(f"downloading {artifact.url} -> {dest}")
            actual = _download_once(artifact, part, source, headers)
            break
        except RetryableSourceError as exc:
            part.unlink(missing_ok=True)
            if attempt >= retry.attempts:
                raise SourceError(
                    f"giving up on {artifact.url} after {attempt} attempts ({exc})"
                ) from exc
            wait = retry.wait_seconds(attempt, exc.retry_after)
            log(f"attempt {attempt} for {artifact.url} failed ({exc}); retrying in {wait:.0f}s")
            retry.sleep(wait)
            attempt += 1
        except (SourceError, VerificationError, OSError):
            part.unlink(missing_ok=True)
            raise
    os.replace(part, dest)
    return FetchResult(artifact.path, "fetched", actual)


def place_sidecar(relative: str, root: Path, sidecar_root: Path = SIDECAR_ROOT) -> FetchResult:
    bundled = sidecar_root / relative
    if not bundled.is_file():
        raise VerificationError(f"bundled sidecar missing from the image: {bundled}")
    expected = sha256_of(bundled)
    dest = root / relative
    if _matches(dest, None, expected):
        return FetchResult(relative, "sidecar-present", expected)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + PART_SUFFIX)
    part.write_bytes(bundled.read_bytes())
    os.replace(part, dest)
    return FetchResult(relative, "sidecar-written", expected)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_bundle_tree(bundle: Bundle, directory: Path) -> None:
    """Reject every shape other than the exact, immutable published tree."""
    info = directory.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise VerificationError(f"bundle {bundle.sha256}: unsafe bundle root")
    artifacts = (*bundle.members, *bundle.receipts)
    expected_files = {artifact.path for artifact in artifacts} | {"manifest.json"}
    expected_directories = {
        parent.as_posix()
        for artifact in artifacts
        for parent in Path(artifact.path).parents
        if parent != Path(".")
    }
    found_files: set[str] = set()
    found_directories: set[str] = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (path.is_dir() or stat.S_ISREG(info.st_mode)):
            raise VerificationError(f"bundle {bundle.sha256}: unsafe member {relative}")
        if stat.S_ISREG(info.st_mode):
            found_files.add(relative)
        else:
            found_directories.add(relative)
    if found_files != expected_files or found_directories != expected_directories:
        expected_tree = (
            f"files {sorted(expected_files)}, directories {sorted(expected_directories)}"
        )
        found_tree = f"files {sorted(found_files)}, directories {sorted(found_directories)}"
        raise VerificationError(
            f"bundle {bundle.sha256}: tree mismatch (expected {expected_tree}; got {found_tree})"
        )
    manifest = directory / "manifest.json"
    if not manifest.is_file() or manifest.read_bytes() != bundle.manifest_bytes:
        raise VerificationError(f"bundle {bundle.sha256}: manifest is not canonical")
    for artifact in artifacts:
        path = directory / artifact.path
        if not _matches(path, artifact.size, artifact.sha256):
            raise VerificationError(f"bundle {bundle.sha256}: member mismatch {artifact.path}")


def fetch_bundle(
    bundle: Bundle,
    root: Path,
    source: ByteSource,
    *,
    env: Mapping[str, str],
    retry: RetryPolicy,
    log: Callable[[str], None] = lambda _message: None,
) -> FetchReport:
    """Fetch a bundle into a private sibling, then atomically publish its tree."""
    bundles_root = root / "bundles"
    destination = bundles_root / bundle.sha256
    if destination.exists() or destination.is_symlink():
        _verify_bundle_tree(bundle, destination)
        return FetchReport(
            [
                FetchResult(artifact.path, "present", artifact.sha256)
                for artifact in (*bundle.members, *bundle.receipts)
            ]
        )

    bundles_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle.sha256}.", dir=bundles_root))
    report = FetchReport()
    try:
        for member in bundle.members:
            result = fetch_artifact(member, staging, source, env=env, retry=retry, log=log)
            report.results.append(result)
        for receipt in bundle.receipts:
            result = fetch_artifact(receipt, staging, source, env=env, retry=retry, log=log)
            report.results.append(result)
        _write_bytes(staging / "manifest.json", bundle.manifest_bytes)
        _verify_bundle_tree(bundle, staging)
        for path in staging.rglob("*"):
            if path.is_dir():
                _fsync_directory(path)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(bundles_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def fetch_all(
    manifest: Manifest,
    root: Path,
    source: ByteSource,
    *,
    env: Mapping[str, str],
    retry: RetryPolicy,
    force: bool = False,
    log: Callable[[str], None] = lambda _message: None,
    sidecar_root: Path = SIDECAR_ROOT,
) -> FetchReport:
    """Fetch every artifact and sidecar; raise on the first failure."""
    report = FetchReport()
    root.mkdir(parents=True, exist_ok=True)
    for artifact in manifest.artifacts:
        result = fetch_artifact(artifact, root, source, env=env, retry=retry, force=force, log=log)
        log(f"{result.outcome:16} {result.sha256}  {result.path}")
        report.results.append(result)
    for relative in manifest.sidecars:
        result = place_sidecar(relative, root, sidecar_root)
        log(f"{result.outcome:16} {result.sha256}  {result.path}")
        report.results.append(result)
    for bundle in manifest.bundles:
        bundle_report = fetch_bundle(bundle, root, source, env=env, retry=retry, log=log)
        for result in bundle_report.results:
            log(f"{result.outcome:16} {result.sha256}  bundles/{bundle.sha256}/{result.path}")
        report.results.extend(bundle_report.results)
    # The Flow worker composes the ORT runner and refuses a fall bundle with no
    # model.onnx. The published manifest may legitimately omit it - the ONNX is
    # a publication-time export the edge image cannot produce, since Torch is
    # excluded under P1b-AC7 - so judge the PROVISIONED bundle, not the manifest:
    # an already-exported bundle on disk is fine, a fresh site without one is
    # refused here with the reason, instead of at worker boot.
    artifact_paths = {artifact.path for artifact in manifest.artifacts}
    if _FALL_PT_PATH in artifact_paths and not (root / _FALL_ONNX_PATH).is_file():
        raise VerificationError(
            "provisioned pose+bbox56 fall bundle is incomplete: missing model.onnx at "
            f"{root / _FALL_ONNX_PATH}; publish model.onnx with the bundle"
        )
    return report
