"""CLI for authoritative clip consistency repair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from worker.pipeline.output.evidence.clip_consistency_repair import repair_clip_consistency
from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    RepairRequest,
)
from worker.pipeline.output.evidence.clip_store_lock import ClipStoreLockedError


class _Arguments(argparse.Namespace):
    state_db: Path = Path()
    clip_store: Path = Path()
    apply: bool = False
    resume: bool = False
    maintenance_root: Path | None = None
    journal: Path | None = None
    quiescence_receipt: Path | None = None
    prebackup_receipt: Path | None = None
    expected_owner_uid: int = os.getuid()
    ffprobe_bin: str = "ffprobe"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair clip relations from authoritative final manifests."
    )
    _ = parser.add_argument("--state-db", type=Path, required=True)
    _ = parser.add_argument("--clip-store", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    _ = mode.add_argument("--apply", action="store_true")
    _ = mode.add_argument("--resume", action="store_true")
    _ = parser.add_argument("--maintenance-root", type=Path)
    _ = parser.add_argument("--journal", type=Path)
    _ = parser.add_argument("--quiescence-receipt", type=Path)
    _ = parser.add_argument("--prebackup-receipt", type=Path)
    _ = parser.add_argument("--expected-owner-uid", type=int, default=os.getuid())
    _ = parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv, namespace=_Arguments())
    try:
        receipt = repair_clip_consistency(
            RepairRequest(
                state_db=arguments.state_db,
                clip_store=arguments.clip_store,
                apply=arguments.apply,
                resume=arguments.resume,
                maintenance_root=arguments.maintenance_root,
                journal_path=arguments.journal,
                quiescence_receipt=arguments.quiescence_receipt,
                prebackup_receipt=arguments.prebackup_receipt,
                expected_owner_uid=arguments.expected_owner_uid,
                ffprobe_bin=arguments.ffprobe_bin,
            )
        )
    except (ClipConsistencyError, ClipStoreLockedError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
