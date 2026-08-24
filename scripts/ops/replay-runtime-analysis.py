#!/usr/bin/env python3
"""Replay a captured analysis timeline through the worker control surface.

Does not open SQLite. The captured timeline is a caller-supplied JSON file.
Worker responses are parsed at this boundary into a typed success or a typed
JSON refusal. Response reads are bounded by the same replay wire constant the
worker uses for `/replay` request bodies.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import override
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shared.events.replay_wire import MAX_REPLAY_BODY_BYTES  # noqa: E402


@dataclass(frozen=True, slots=True)
class ReplayResponseError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class AcceptedReplay:
    event_count: int
    module_qualified_id: str | None


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


def _read_bounded_body(response: http.client.HTTPResponse) -> bytes:
    declared = response.getheader("Content-Length")
    if declared is None:
        raw = response.read(MAX_REPLAY_BODY_BYTES + 1)
        if len(raw) > MAX_REPLAY_BODY_BYTES:
            raise ReplayResponseError(
                f"replay response exceeds maximum of {MAX_REPLAY_BODY_BYTES} bytes"
            )
        return raw
    try:
        length = int(declared)
    except ValueError as error:
        raise ReplayResponseError("invalid Content-Length") from error
    if length < 0 or length > MAX_REPLAY_BODY_BYTES:
        raise ReplayResponseError(
            f"replay response exceeds maximum of {MAX_REPLAY_BODY_BYTES} bytes"
        )
    raw = response.read(length)
    if len(raw) != length:
        raise ReplayResponseError("truncated replay response")
    return raw


def parse_worker_replay(raw: bytes) -> AcceptedReplay:
    """Parse one worker `/replay` body. Raises ReplayResponseError on any defect."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ReplayResponseError("malformed JSON") from error
    if not isinstance(payload, dict):
        raise ReplayResponseError("replay response must be an object")
    if "reproducible" not in payload:
        raise ReplayResponseError("reproducible is required")
    reproducible = payload["reproducible"]
    if not isinstance(reproducible, bool):
        raise ReplayResponseError("reproducible must be a boolean")
    if not reproducible:
        raise ReplayResponseError("worker refused incomplete replay input")
    if "event_count" not in payload:
        raise ReplayResponseError("event_count is required")
    event_count = payload["event_count"]
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 0:
        raise ReplayResponseError("event_count must be a non-negative integer")
    module_id = payload.get("module_qualified_id")
    if module_id is not None and not isinstance(module_id, str):
        raise ReplayResponseError("module_qualified_id must be text")
    return AcceptedReplay(event_count, module_id)


def _refuse(detail: str) -> int:
    print(json.dumps({"status": "refused", "detail": detail}, sort_keys=True))
    return 2


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    _ = arguments.database, arguments.camera_id, arguments.requested_by
    if arguments.trace_json is None or not arguments.trace_json.is_file():
        return _refuse("trace JSON is required; analysis persistence is retired")
    try:
        trace = json.loads(arguments.trace_json.read_text(encoding="utf-8"))
        if not isinstance(trace, dict):
            return _refuse("trace JSON must be an object")
        policy = json.loads(arguments.policy_json)
        if not isinstance(policy, dict):
            return _refuse("policy JSON must be an object")
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
            accepted = parse_worker_replay(_read_bounded_body(response))
    except ReplayResponseError as error:
        return _refuse(str(error))
    except (http.client.IncompleteRead, http.client.RemoteDisconnected):
        return _refuse("truncated replay response")
    except (OSError, ValueError, HTTPError, URLError, json.JSONDecodeError) as error:
        return _refuse(str(error))
    print(
        json.dumps(
            {
                "event_count": accepted.event_count,
                "module_qualified_id": accepted.module_qualified_id,
                "reproducible": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
