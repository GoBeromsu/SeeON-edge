#!/usr/bin/env python3
"""One-shot backend operator command for manifest-authoritative clip repair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair clip/event relations from final manifests."
    )
    _ = parser.add_argument("--clip-store", required=True, type=Path)
    _ = parser.add_argument("--maintenance-root", required=True, type=Path)
    _ = parser.add_argument("--quiescence-receipt", required=True, type=Path)
    _ = parser.add_argument("--apply", action="store_true", help="Mutate after all safety checks.")
    _ = parser.add_argument("--expected-owner-uid", type=int, default=os.getuid())
    return parser


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository))
    from backend.app.features.clips.consistency_ops import (  # noqa: PLC0415
        ClipConsistencyError,
        RepairRequest,
        repair_clip_consistency,
    )

    arguments = _parser().parse_args(argv)
    try:
        receipt = repair_clip_consistency(
            RepairRequest(
                clip_store=arguments.clip_store,
                maintenance_root=arguments.maintenance_root,
                quiescence_receipt=arguments.quiescence_receipt,
                apply=arguments.apply,
                expected_owner_uid=arguments.expected_owner_uid,
            )
        )
    except ClipConsistencyError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
