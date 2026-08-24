#!/usr/bin/env python3
"""Retired: analysis/QA persistence is no longer a service database owner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay persistence is retired; this command does not open SQLite."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--relay-token", required=True)
    parser.add_argument("--module-id", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--requested-by", required=True)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    _ = _parser().parse_args(argv)
    print(
        json.dumps(
            {
                "status": "refused",
                "detail": "qa and runtime-analysis persistence are retired",
            },
            sort_keys=True,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
