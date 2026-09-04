"""Native preprocessing that reproduces the shipped tensor byte-for-byte.

The channel order here is the single most consequential decision in this module,
and it is deliberately "wrong-looking". The shipped chain is:

1. Ingest converts the decoded BGR frame to TRUE RGB
   (``worker/pipeline/ingest/*.py``, ``cv2.COLOR_BGR2RGB``), which is what
   ``worker/types/capabilities.py`` declares as ``HOST_RGB``.
2. ``worker/adapters/model/yolo_api.py:predict_one`` hands that numpy array
   straight to ``model.predict(source=...)``.
3. Ultralytics 8.4.61 assumes a numpy source is BGR, so
   ``ultralytics/engine/predictor.py`` applies ``im = im[..., ::-1]`` a SECOND
   time before ``BHWC -> BCHW``.

The array is therefore flipped twice and the network receives BGR planes. The
``rgb24-to-*`` identity strings in ``worker/domains/registry.py`` name the
ADAPTER INPUT format, not the tensor, so there is no contradiction. Feeding true
RGB to the exported model is "more correct" and silently destroys parity with
every shipped detection, keypoint and event timeline -- so the inversion is
reproduced, not fixed.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from worker.native.deepstream.parity.geometry import (
    LETTERBOX_PAD_VALUE,
    AffineMetadata,
    letterbox_affine,
)

ChannelOrder: TypeAlias = Literal["bgr", "rgb"]

#: The plane order the shipped models actually receive. Proven empirically in
#: ``.omo`` evidence ``task-3-baseline-channel-order.json``: an adapter input of
#: ``[10, 20, 30]`` arrives at the network as planes ``[30, 20, 10]``.
NATIVE_CHANNEL_ORDER: Final[ChannelOrder] = "bgr"
TENSOR_PRECISION: Final = "fp32"
PREPROCESS_IDENTITY: Final = "deepstream-letterbox-bgr-fp32.v1"


def preprocess_tensor(
    frame: NDArray[np.uint8],
    *,
    channel_order: ChannelOrder = NATIVE_CHANNEL_ORDER,
    affine: AffineMetadata | None = None,
) -> NDArray[np.float32]:
    """Build the NCHW FP32 network tensor for one true-RGB host frame.

    ``channel_order`` exists only so the parity suite can construct the
    intentional-inversion RED fixture. Production always uses the default.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 host frame, received shape {frame.shape}")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    metadata = letterbox_affine(height, width) if affine is None else affine
    resized = frame
    if (metadata.content_height, metadata.content_width) != (height, width):
        resized = cv2.resize(
            frame,
            (metadata.content_width, metadata.content_height),
            interpolation=cv2.INTER_LINEAR,
        )
    padded = cv2.copyMakeBorder(
        resized,
        metadata.pad_top,
        metadata.pad_bottom,
        metadata.pad_left,
        metadata.pad_right,
        cv2.BORDER_CONSTANT,
        value=(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE),
    )
    # The shipped second flip. See this module's docstring before "fixing" it.
    ordered = padded[..., ::-1] if channel_order == "bgr" else padded
    tensor = ordered.transpose((2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor[np.newaxis, ...])


def preprocess_batch(
    frames: tuple[NDArray[np.uint8], ...],
    *,
    channel_order: ChannelOrder = NATIVE_CHANNEL_ORDER,
) -> NDArray[np.float32]:
    """Stack one mixed-camera batch, preserving input order.

    Row ``i`` of the returned tensor belongs to ``frames[i]``. Reordering is a
    parity break the comparator is required to catch, so no sorting happens here.
    """
    if not frames:
        return np.zeros((0, 3, 0, 0), dtype=np.float32)
    tensors = tuple(
        preprocess_tensor(frame, channel_order=channel_order) for frame in frames
    )
    shapes = {tensor.shape[1:] for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(
            "mixed-geometry batch requires one tensor shape per batch, "
            f"received {sorted(shapes)}"
        )
    return np.ascontiguousarray(np.concatenate(tensors, axis=0))


__all__ = [
    "NATIVE_CHANNEL_ORDER",
    "PREPROCESS_IDENTITY",
    "TENSOR_PRECISION",
    "ChannelOrder",
    "preprocess_batch",
    "preprocess_tensor",
]
