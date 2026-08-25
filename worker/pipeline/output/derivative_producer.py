"""Machine-consumed derivative producer identity facts."""

from enum import StrEnum


class DerivativeProducer(StrEnum):
    CPU_REFERENCE = "cpu-reference"
    NATIVE_GPU = "native-gpu"


__all__ = ["DerivativeProducer"]
