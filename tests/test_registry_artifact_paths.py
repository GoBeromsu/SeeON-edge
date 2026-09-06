"""Default YOLO runner ``model_path`` constructor arguments resolve to the
canonical models/<task>/<weights> layout.

tests/test_runners_registry.py (rewritten in commit 622b091) covers
ModelRegistry factory identity and MODELS_DIR resolution for the sklearn fall
detector, but not the ``model_path`` constructor defaults on YoloPoseRunner /
YoloBedSegRunner themselves — a distinct concern (this is what those runner
classes fall back to when constructed with no explicit path, independent of
the registry). Not superseded; ported directly.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from worker.adapters.model.yolo_bed_seg import YoloBedSegRunner
from worker.adapters.model.yolo_pose import YoloPoseRunner


def test_default_yolo_runners_use_canonical_worker_artifact_paths() -> None:
    pose_default = inspect.signature(YoloPoseRunner).parameters["model_path"].default
    bed_default = inspect.signature(YoloBedSegRunner).parameters["model_path"].default

    assert Path(pose_default).parts[-3:] == ("models", "pose", "yolo26n-pose.pt")
    assert Path(bed_default).parts[-3:] == ("models", "bed", "yolo26l-seg.pt")
