from __future__ import annotations

from typing import TypeAlias

# The fall-window classifier's model input: either a flat engineered-feature
# vector ("features" mode) or a per-frame sequence of rows ("sequence" mode,
# one row per window frame). Plain nested tuples of floats -- never an
# ndarray -- so `worker.domains.fall` stays numeric/hardware-agnostic; the
# model adapter that actually runs inference (`worker.adapters.model`)
# converts this into whatever array type its framework needs.
FallModelInput: TypeAlias = tuple[float, ...] | tuple[tuple[float, ...], ...]

__all__ = ["FallModelInput"]
