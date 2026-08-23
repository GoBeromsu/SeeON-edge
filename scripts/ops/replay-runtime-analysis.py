#!/usr/bin/env python3
"""Replay one complete backend-owned analysis timeline through the worker control surface."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a complete captured analysis timeline.")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--worker-url", required=True)
    parser.add_argument("--relay-token", required=True)
    parser.add_argument("--module-id", required=True)
    parser.add_argument("--policy-json", required=True, help="Exact EffectivePolicy JSON.")
    parser.add_argument("--requested-by", required=True)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository))
    from backend.app.features.qa.runtime_trace_store import (
        ReplayInputUnavailable,
        RuntimeAnalysisStore,
    )
    from backend.app.features.qa.store import QaStore

    arguments = _parser().parse_args(argv)
    try:
        trace = RuntimeAnalysisStore(arguments.database).recover(arguments.camera_id)
        policy = json.loads(arguments.policy_json)
        if not isinstance(policy, dict):
            print(
                json.dumps(
                    {"status": "refused", "detail": "policy JSON must be an object"},
                    sort_keys=True,
                )
            )
            return 2
        payload = {"trace": trace.as_dict(), "module_id": arguments.module_id, "policy": policy}
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
    except (ReplayInputUnavailable, ValueError, HTTPError, URLError) as error:
        print(json.dumps({"status": "refused", "detail": str(error)}, sort_keys=True))
        return 2
    run = QaStore(arguments.database).record_run(
        camera_id=arguments.camera_id,
        module_qualified_id=str(result["module_qualified_id"]),
        policy_qualified_id=str(result["policy_qualified_id"]),
        effective_policy_id=str(result["effective_policy_id"]),
        frame_count=len(result["frames"]),
        event_count=int(result["event_count"]),
        source_kind="captured",
        source_run_id=None,
        requested_by=arguments.requested_by,
        requested_at=datetime.now(UTC).isoformat(),
        result=result,
    )
    print(json.dumps({"run_id": run.run_id, "event_count": run.event_count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
