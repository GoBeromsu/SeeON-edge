#!/usr/bin/env python3
"""Replay a captured analysis timeline through the worker control surface.

Does not open SQLite. The captured timeline is a caller-supplied JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a captured analysis timeline without SQLite persistence."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--relay-token", required=True)
    parser.add_argument("--module-id", required=True)
    parser.add_argument("--policy-json", required=True, help="Exact EffectivePolicy JSON.")
    parser.add_argument("--requested-by", required=True)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--trace-json",
        type=Path,
        default=None,
        help="Captured analysis timeline JSON. Required; persistence recovery is retired.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    _ = arguments.database, arguments.camera_id, arguments.requested_by
    if arguments.trace_json is None or not arguments.trace_json.is_file():
        print(
            json.dumps(
                {
                    "status": "refused",
                    "detail": "trace JSON is required; analysis persistence is retired",
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        trace = json.loads(arguments.trace_json.read_text(encoding="utf-8"))
        if not isinstance(trace, dict):
            print(
                json.dumps(
                    {"status": "refused", "detail": "trace JSON must be an object"},
                    sort_keys=True,
                )
            )
            return 2
        policy = json.loads(arguments.policy_json)
        if not isinstance(policy, dict):
            print(
                json.dumps(
                    {"status": "refused", "detail": "policy JSON must be an object"},
                    sort_keys=True,
                )
            )
            return 2
        payload = {"trace": trace, "module_id": arguments.module_id, "policy": policy}
        request = Request(
            arguments.worker_url.rstrip("/") + "/replay",
            data=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Edge-Relay-Token": arguments.relay_token,
            },
            method="POST",
        )
        with urlopen(request, timeout=arguments.timeout_sec) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict) or result.get("reproducible") is not True:
            print(
                json.dumps(
                    {"status": "refused", "detail": "worker refused incomplete replay input"},
                    sort_keys=True,
                )
            )
            return 2
    except (OSError, ValueError, HTTPError, URLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "refused", "detail": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "event_count": int(result["event_count"]),
                "reproducible": True,
                "module_qualified_id": result.get("module_qualified_id"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
