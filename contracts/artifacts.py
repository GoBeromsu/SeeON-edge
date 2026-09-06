from __future__ import annotations

from pathlib import Path
from typing import Final

POSE_MODEL_SIZES: Final[tuple[str, ...]] = ("n", "s", "m", "l", "x")
WEIGHTS_DIR: Final = Path(__file__).resolve().parent.parent / "models" / "pose"
BED_WEIGHTS_DIR: Final = Path(__file__).resolve().parent.parent / "models" / "bed"


def pose_weight_filename(size: str) -> str:
    """Map a YOLO26-pose size letter to its weight filename (model-contract swap)."""
    if size not in POSE_MODEL_SIZES:
        raise ValueError(f"Unknown YOLO26-pose size {size!r}; expected one of {POSE_MODEL_SIZES}")
    return f"yolo26{size}-pose.pt"


def pose_weight_path(size: str) -> Path:
    """Resolve a pose-size letter to its weight path under the models/pose/ cache."""
    return WEIGHTS_DIR / pose_weight_filename(size)


def bed_seg_weight_path() -> Path:
    return BED_WEIGHTS_DIR / "yolo26l-seg.pt"
