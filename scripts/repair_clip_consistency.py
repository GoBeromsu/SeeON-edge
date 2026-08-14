#!/usr/bin/env python3
"""Launch the schema-9 clip consistency maintenance command."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    _ = runpy.run_module(
        "worker.pipeline.output.evidence.clip_consistency_cli",
        run_name="__main__",
    )
