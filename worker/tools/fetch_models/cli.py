"""``python -m worker.tools.fetch_models`` -- provision the models root.

Exit codes: 0 every artifact present and verified; 1 a download or hash
verification failed (nothing unverified is left at a final path); 2 usage or
manifest error. ``HF_TOKEN`` is read from the environment for Hugging Face
sources only and is never echoed.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

from worker.tools.fetch_models.fetcher import VerificationError, fetch_all
from worker.tools.fetch_models.http_source import (
    RetryPolicy,
    SourceError,
    UrllibSource,
    attempts_from_env,
)
from worker.tools.fetch_models.manifest import MANIFEST_PATH, ManifestError, load_manifest

DEST_ENV: Final = "ML_WORKER_FETCH_MODELS_DEST"
PROG: Final = "fetch_models"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m worker.tools.{PROG}",
        description=(
            "Download the pinned model artifacts listed in manifest.json into a "
            "models root, verifying size and SHA-256 of every file. Idempotent."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=(
            "models root to populate (default: $ML_WORKER_FETCH_MODELS_DEST, "
            "else ./models relative to the current directory)"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="manifest to fetch (default: the committed manifest.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the destination already verifies",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify what is on disk; exit 1 if anything is missing or mismatched",
    )
    return parser


def _resolve_dest(explicit: Path | None, env: Mapping[str, str]) -> Path:
    if explicit is not None:
        return explicit
    from_env = env.get(DEST_ENV, "").strip()
    return Path(from_env) if from_env else Path("models")


class _RefusingSource:
    """``--check`` byte source: any download attempt is a failure."""

    def stream(self, url: str, headers: Mapping[str, str]) -> Iterator[bytes]:
        del headers
        raise SourceError(f"--check: {url} would need downloading")
        yield b""  # pragma: no cover -- makes this a generator like the real source


def main(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None) -> int:
    environment = os.environ if env is None else env
    args = _build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(f"{PROG}: {message}", file=sys.stderr, flush=True)

    try:
        manifest = load_manifest(args.manifest)
        attempts = attempts_from_env(environment)
    except (ManifestError, SourceError) as exc:
        log(str(exc))
        return 2

    dest = _resolve_dest(args.dest, environment)
    source = _RefusingSource() if args.check else UrllibSource()
    log(f"models root: {dest}")
    for name, pinned in manifest.sources.items():
        log(f"source {name}: {pinned.kind} {pinned.source_locator} @ {pinned.ref}")
    if environment.get("HF_TOKEN", "").strip():
        log("HF_TOKEN present; sent to Hugging Face sources only")

    try:
        report = fetch_all(
            manifest,
            dest,
            source,
            env=environment,
            retry=RetryPolicy(attempts=attempts),
            force=args.force,
            log=log,
        )
    except (SourceError, VerificationError, OSError) as exc:
        log(f"FAILED: {exc}")
        return 1
    verb = "verified, nothing to do" if report.is_noop else f"fetched {report.fetched} file(s)"
    log(f"done: {len(report.results)} file(s) {verb}")
    return 0
