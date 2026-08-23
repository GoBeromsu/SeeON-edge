#!/usr/bin/env python3
"""Drain schema-16 evidence through the running backend relay before migration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drain legacy schema-16 evidence events.")
    _ = parser.add_argument("--database", required=True, type=Path)
    _ = parser.add_argument("--relay-url", required=True)
    _ = parser.add_argument("--relay-token", required=True)
    _ = parser.add_argument("--timeout-sec", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository))
    from backend.app.edge_db.legacy_drain import LegacyEvidenceDrain  # noqa: PLC0415
    from shared.events.evidence_export_client import RelayEvidenceClient  # noqa: PLC0415

    arguments = _parser().parse_args(argv)
    transport = RelayEvidenceClient(
        arguments.relay_url, arguments.relay_token, arguments.timeout_sec
    )
    # The relay projects through the running API, which holds a shared deployment
    # lock while it writes. Holding the migrator's exclusive lock while waiting
    # for that HTTP response deadlocks both processes. Each drain database write
    # is already its own short SQLite transaction in LegacyEvidenceDrain.
    result = LegacyEvidenceDrain(arguments.database, transport).run()
    print(
        json.dumps(
            {
                "delivered": result.delivered,
                "permanent": result.permanent,
                "retryable": result.retryable,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    # Success means every legacy event actually reached the backend. A permanent
    # or compatibility failure leaves undelivered evidence in the database, so
    # exiting 0 here would tell an operator the drain was complete and let the
    # schema-17 migration proceed over evidence that was never delivered.
    return 0 if result.retryable == 0 and result.permanent == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
