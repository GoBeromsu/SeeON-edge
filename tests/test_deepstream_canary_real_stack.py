from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.real_stack
def test_zero_and_loopback_canary_when_real_qa_is_authorized(tmp_path: Path) -> None:
    # Given: an explicit opt-in, digest-pinned image, models, and a fresh evidence root.
    if os.environ.get("SEEON_CANARY_REAL_QA") != "1":
        pytest.skip("set SEEON_CANARY_REAL_QA=1 for the isolated RTX canary")
    worker_image = os.environ.get("CANARY_WORKER_IMAGE")
    expected_revision = os.environ.get("CANARY_EXPECTED_REVISION")
    model_dir = os.environ.get("SEEON_CANARY_MODEL_DIR")
    if worker_image is None or expected_revision is None or model_dir is None:
        pytest.skip(
            "CANARY_WORKER_IMAGE, CANARY_EXPECTED_REVISION, and "
            "SEEON_CANARY_MODEL_DIR are required"
        )
    evidence = tmp_path / "canary"

    # When: the real worker runs the zero-camera and deterministic loopback ladder.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worker.tools.deepstream_canary",
            "run",
            "--rungs",
            "zero,loopback",
            "--evidence-dir",
            str(evidence),
            "--worker-image",
            worker_image,
            "--expected-revision",
            expected_revision,
            "--model-dir",
            model_dir,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=1500,
    )

    # Then: orchestration and the independent raw-receipt verifier both pass.
    assert completed.returncode == 0, completed.stderr
    verified = subprocess.run(
        [
            sys.executable,
            "scripts/qa/verify_deepstream_delivery.py",
            "canary",
            "--evidence-root",
            str(evidence),
            "--output",
            str(evidence / "verified.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert verified.returncode == 0, verified.stderr
