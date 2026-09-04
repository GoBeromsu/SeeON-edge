from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worker.runtime.flow.cold_start import (
    EngineIdentityError,
    FlowColdStart,
    verify_engine_identity,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cold_start_verifies_before_warmup(tmp_path: Path) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    files = {}
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps({"engine_sha256": _digest(engine), "image_digest": "image", **files})
    )
    warmed = []
    FlowColdStart(engine, identity, files, lambda: warmed.append(True)).run()
    assert warmed == [True]


def test_missing_engine_names_operator_tool(tmp_path: Path) -> None:
    with pytest.raises(EngineIdentityError, match="edge-engine-build"):
        verify_engine_identity(tmp_path / "missing.engine", tmp_path / "identity.json", {})
