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
from worker.tools.fetch_models.manifest import SIDECAR_ROOT, Artifact, Manifest

HF_TOKEN_ENV: Final = "HF_TOKEN"
PART_SUFFIX: Final = ".part"
_HASH_CHUNK: Final = 1 << 20

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
    return report
