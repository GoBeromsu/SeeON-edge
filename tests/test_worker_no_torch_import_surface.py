from __future__ import annotations

import subprocess
import sys


def test_flow_import_surface_does_not_import_torch() -> None:
    code = """
import sys
import worker.adapters.deepstream
import worker.adapters.model.ort_pose_bbox56
import worker.interfaces
import worker.domains
import worker.pipeline.perception
from worker.runtime.flow import FlowMediaPlane, FlowMediaPlaneConfig
assert "torch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
