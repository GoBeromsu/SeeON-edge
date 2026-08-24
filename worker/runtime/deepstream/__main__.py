"""Explicit isolated C5 dark-runner CLI."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.runner import DarkRunRequest, DarkSource, run_dark_child


class _Arguments(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.child: Path = Path("/usr/local/bin/seeon-deepstream-child")
        self.state_dir: Path = Path()
        self.source: list[str] = []
        self.inject_fatal: str | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m worker.runtime.deepstream")
    _ = parser.add_argument(
        "--child",
        type=Path,
        default=Path("/usr/local/bin/seeon-deepstream-child"),
    )
    _ = parser.add_argument("--state-dir", type=Path, required=True)
    _ = parser.add_argument("--source", action="append", default=[])
    _ = parser.add_argument(
        "--inject-fatal",
        choices=("cuda", "xid", "context", "native_heap", "tensorrt"),
    )
    return parser


def _source(raw: str) -> DarkSource:
    camera, separator, uri = raw.partition("=")
    if separator == "" or camera == "" or uri == "":
        raise argparse.ArgumentTypeError("source must be CAMERA=URI")
    return DarkSource(camera, uri)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv, namespace=_Arguments())
    state_dir: Path = args.state_dir
    request = DarkRunRequest(
        child=ChildConfig(
            executable=args.child,
            worker_boot_id=uuid.uuid4(),
            socket_dir=state_dir / "ipc",
            first_fault_path=state_dir / "deepstream-first-fault.json",
            lease_state_dir=state_dir,
        ),
        sources=tuple(_source(raw) for raw in args.source),
        inject_fatal=args.inject_fatal,
    )
    return run_dark_child(request)


if __name__ == "__main__":
    raise SystemExit(main())
