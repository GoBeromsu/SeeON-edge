"""Fail-closed validation for batched model input (ADR-0002 fail-fast).

A batched forward stacks every frame into one tensor, so a single row with
the wrong dtype, rank, channel count, or geometry either crashes deep inside
the backend or -- worse -- is silently letterboxed/coerced into something
whose result no longer matches what the same frame would have produced
alone. Both outcomes are defects, so the seam refuses the batch here with a
typed error naming the offending camera.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from contracts.runner import Image

_EXPECTED_DTYPE = np.uint8
_EXPECTED_NDIM = 3
_EXPECTED_CHANNELS = 3


@dataclass(slots=True)
class BatchInputError(ValueError):
    """A batch the serving seam refuses to run rather than coerce."""

    task: str
    camera_id: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.task} batch input rejected for camera {self.camera_id!r}: {self.detail}"


def validated_batch_images(
    task: str,
    images: Sequence[tuple[str, Image]],
) -> tuple[Image, ...]:
    """Return the batch images, or raise ``BatchInputError`` naming the camera.

    Every row must be an HxWx3 uint8 array, and all rows must share one
    geometry: mixed sizes would be letterboxed differently per row, which
    breaks single/batch parity instead of merely looking untidy.
    """
    validated: list[Image] = []
    expected_geometry: tuple[int, int] | None = None
    for camera_id, image in images:
        array = _as_array(task, camera_id, image)
        if array.dtype != _EXPECTED_DTYPE:
            raise BatchInputError(
                task=task,
                camera_id=camera_id,
                detail=f"dtype must be {_EXPECTED_DTYPE.__name__}, received {array.dtype}",
            )
        if array.ndim != _EXPECTED_NDIM or array.shape[2] != _EXPECTED_CHANNELS:
            raise BatchInputError(
                task=task,
                camera_id=camera_id,
                detail=(
                    f"shape must be (height, width, {_EXPECTED_CHANNELS}), received {array.shape}"
                ),
            )
        geometry = (int(array.shape[0]), int(array.shape[1]))
        if expected_geometry is None:
            expected_geometry = geometry
        elif geometry != expected_geometry:
            raise BatchInputError(
                task=task,
                camera_id=camera_id,
                detail=(
                    f"batch geometry must match {expected_geometry[1]}x{expected_geometry[0]}, "
                    f"received {geometry[1]}x{geometry[0]}"
                ),
            )
        validated.append(array)
    return tuple(validated)


def _as_array(task: str, camera_id: str, image: Image) -> Image:
    if not isinstance(image, np.ndarray):
        raise BatchInputError(
            task=task,
            camera_id=camera_id,
            detail=f"image must be a numpy array, received {type(image).__name__}",
        )
    return image


__all__ = ["BatchInputError", "validated_batch_images"]
