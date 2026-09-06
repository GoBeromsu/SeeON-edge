#!/usr/bin/env python
"""Inspect and requeue evidence the backend refused.

A 422 means the backend rejected a payload. The entry is retained rather than
deleted, because deleting refused evidence and reporting it delivered is how 41
real bed-exit events were destroyed on this deployment. Retention is only useful
if an operator can act on it, and a retention area that fills has to be
drainable -- otherwise the bound turns into a second stall.

    # what is being held, and why
    python scripts/ops/review-refused-evidence.py --state-dir /var/lib/seeon-state

    # after the cause is fixed (a schema field, a relay version), put it back
    python scripts/ops/review-refused-evidence.py --state-dir /var/lib/seeon-state --requeue

Exit codes:
  0  nothing retained, or the requeue completed
  1  evidence is retained and needs review (inspection mode)
  2  usage or environment error
  3  requeue could not complete because the live queue is at capacity
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from shared.events.delivery_queue import (  # noqa: E402
    MAX_DEAD_LETTERED_BYTES,
    MAX_DEAD_LETTERED_ENTRIES,
    DeliveryQueue,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="runtime state root the worker was started with (--state-dir)",
    )
    parser.add_argument(
        "--requeue",
        action="store_true",
        help="return retained entries to the live queue for another attempt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    queue_directory = args.state_dir / "delivery-queue"
    if not queue_directory.is_dir():
        print(
            json.dumps(
                {
                    "status": "error",
                    "detail": f"no delivery queue at {queue_directory}",
                },
                sort_keys=True,
            )
        )
        return 2

    queue = DeliveryQueue(queue_directory)
    retention = queue.dead_letter_directory
    retained = (
        sorted(path for path in retention.iterdir() if path.is_file()) if retention.is_dir() else []
    )

    # The refusal status is the leading filename component, so the operator sees
    # why each entry is held without opening any payload.
    reasons = Counter(path.name.split(".", 1)[0] for path in retained)
    total_bytes = sum(path.stat().st_size for path in retained)

    if not args.requeue:
        print(
            json.dumps(
                {
                    "status": "retained" if retained else "clear",
                    "directory": str(retention),
                    "count": len(retained),
                    "bytes": total_bytes,
                    "max_count": MAX_DEAD_LETTERED_ENTRIES,
                    "max_bytes": MAX_DEAD_LETTERED_BYTES,
                    "by_reason": dict(sorted(reasons.items())),
                },
                sort_keys=True,
            )
        )
        return 1 if retained else 0

    requeued = 0
    for path in retained:
        # The queue owns its lock, bounds and atomic publication; re-admitting
        # by writing the file back would bypass all three.
        if not queue.requeue_dead_lettered(path):
            print(
                json.dumps(
                    {
                        "status": "incomplete",
                        "requeued": requeued,
                        "remaining": len(retained) - requeued,
                        "detail": (
                            "live queue would not accept the entry (at capacity, "
                            "or a different entry already holds its identity); "
                            "drain the queue and re-run"
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 3
        requeued += 1

    print(json.dumps({"status": "requeued", "requeued": requeued}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
