"""Per-frame affine metadata: the shipped letterbox and its two inverse rules.

The shipped path is ultralytics ``LetterBox(imgsz=640, auto=True, stride=32,
padding_value=114, scaleup=True, center=True)``, driven by ``rect=True``. A 16:9
source therefore lands as 640x360 content inside a 640x384 tensor with 12px of
top/bottom padding.

Boxes and keypoints do NOT share an inverse. ``ultralytics.utils.ops.scale_boxes``
subtracts ``round(pad / 2 - 0.1)`` while ``ops.scale_coords`` subtracts the
unrounded ``pad / 2``. On even padding the two agree; on the odd 100x102 fixture
they diverge (6 vs 6.5). Both rules are reproduced verbatim -- collapsing them
into one is a parity break, not a simplification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

LETTERBOX_SIZE: Final = 640
LETTERBOX_STRIDE: Final = 32
LETTERBOX_PAD_VALUE: Final = 114
BOX_INVERSE_IDENTITY: Final = "letterbox-inverse-box.v1"
KEYPOINT_INVERSE_IDENTITY: Final = "letterbox-inverse-keypoint.v1"


@dataclass(frozen=True, slots=True)
class AffineMetadata:
    """One frame's forward letterbox and both inverse mappings.

    ``gain`` is the single isotropic scale ultralytics computes as
    ``min(new / old)``; ``content_*`` is the resized source inside the tensor.
    """

    source_height: int
    source_width: int
    tensor_height: int
    tensor_width: int
    content_height: int
    content_width: int
    gain: float
    box_pad_x: int
    box_pad_y: int
    keypoint_pad_x: float
    keypoint_pad_y: float
    pad_value: int = LETTERBOX_PAD_VALUE

    @property
    def pad_top(self) -> int:
        return int(self.keypoint_pad_y)

    @property
    def pad_bottom(self) -> int:
        return self.tensor_height - self.content_height - self.pad_top

    @property
    def pad_left(self) -> int:
        return int(self.keypoint_pad_x)

    @property
    def pad_right(self) -> int:
        return self.tensor_width - self.content_width - self.pad_left

    def invert_box(self, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """Map a tensor-space xyxy box back onto source pixels.

        Reproduces ``ops.scale_boxes`` (rounded pad, divide by gain, clamp to the
        source) followed by the shipped adapters' ``int()`` truncation.
        """
        x1 = (box[0] - self.box_pad_x) / self.gain
        y1 = (box[1] - self.box_pad_y) / self.gain
        x2 = (box[2] - self.box_pad_x) / self.gain
        y2 = (box[3] - self.box_pad_y) / self.gain
        return (
            int(_clamp(x1, self.source_width)),
            int(_clamp(y1, self.source_height)),
            int(_clamp(x2, self.source_width)),
            int(_clamp(y2, self.source_height)),
        )

    def invert_keypoint(self, point: tuple[float, float]) -> tuple[int, int]:
        """Map a tensor-space point back onto source pixels.

        Reproduces ``ops.scale_coords``, which uses the UNROUNDED pad. This is
        the rule that separates keypoints and mask contours from boxes.
        """
        x = (point[0] - self.keypoint_pad_x) / self.gain
        y = (point[1] - self.keypoint_pad_y) / self.gain
        return (
            int(_clamp(x, self.source_width)),
            int(_clamp(y, self.source_height)),
        )


def _clamp(value: float, upper: int) -> float:
    if value < 0.0:
        return 0.0
    return float(upper) if value > upper else value


def letterbox_affine(
    source_height: int,
    source_width: int,
    *,
    size: int = LETTERBOX_SIZE,
    stride: int = LETTERBOX_STRIDE,
) -> AffineMetadata:
    """Compute the shipped letterbox geometry for one source resolution.

    Mirrors ``LetterBox.get_params`` with ``auto=True`` (minimum rectangle: the
    padding is reduced modulo the stride) and ``center=True`` (padding split
    evenly across both sides).
    """
    if source_height <= 0 or source_width <= 0:
        raise ValueError(
            f"source geometry must be positive, received {source_height}x{source_width}"
        )
    gain = min(size / source_height, size / source_width)
    content_width = round(source_width * gain)
    content_height = round(source_height * gain)
    # auto=True keeps only the remainder against the stride, which is what turns
    # a 16:9 frame into a 640x384 tensor rather than a square 640x640 one.
    pad_width = (size - content_width) % stride
    pad_height = (size - content_height) % stride
    tensor_width = content_width + pad_width
    tensor_height = content_height + pad_height
    return AffineMetadata(
        source_height=source_height,
        source_width=source_width,
        tensor_height=tensor_height,
        tensor_width=tensor_width,
        content_height=content_height,
        content_width=content_width,
        gain=gain,
        box_pad_x=round(pad_width / 2 - 0.1),
        box_pad_y=round(pad_height / 2 - 0.1),
        keypoint_pad_x=pad_width / 2,
        keypoint_pad_y=pad_height / 2,
    )


__all__ = [
    "BOX_INVERSE_IDENTITY",
    "KEYPOINT_INVERSE_IDENTITY",
    "LETTERBOX_PAD_VALUE",
    "LETTERBOX_SIZE",
    "LETTERBOX_STRIDE",
    "AffineMetadata",
    "letterbox_affine",
]
