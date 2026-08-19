from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from worker.adapters.decode.nvdec_cuvid.errors import NvdecReadError
from worker.adapters.decode.nvdec_cuvid.models import StreamMetadata

_RGB_CHANNELS: Final = 3


def raw_frame_size(metadata: StreamMetadata) -> int:
    return metadata.width * metadata.height * _RGB_CHANNELS


def rgb24_image(payload: bytes, metadata: StreamMetadata) -> NDArray[np.uint8]:
    expected_size = raw_frame_size(metadata)
    if len(payload) != expected_size:
        raise NvdecReadError(expected_size, len(payload))
    return (
        np.frombuffer(payload, dtype=np.uint8)
        .reshape(metadata.height, metadata.width, _RGB_CHANNELS)
        .copy()
    )


__all__ = ["raw_frame_size", "rgb24_image"]
