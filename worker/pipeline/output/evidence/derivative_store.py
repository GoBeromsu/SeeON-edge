from __future__ import annotations

from pathlib import Path

from worker.pipeline.output.annotated_derivative import AnnotatedDerivativeJob, DerivativeArtifact
from worker.pipeline.output.evidence.derivative_artifact_store import (
    DerivativeArtifactCapacityError as DerivativeCapacityError,
)
from worker.pipeline.output.evidence.derivative_artifact_store import (
    DerivativeArtifactConflictError as DerivativeConflictError,
)
from worker.pipeline.output.evidence.derivative_artifact_store import (
    DerivativeArtifactStore,
    StoredDerivativeArtifact,
)

StoredDerivative = StoredDerivativeArtifact


class DerivativeStore(DerivativeArtifactStore):
    """Filesystem-only derivative publication retained for evidence composition."""

    def __init__(self, store_root: Path, *, max_disk_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        super().__init__(store_root, max_disk_bytes=max_disk_bytes)

    def publish(
        self, job: AnnotatedDerivativeJob, artifact: DerivativeArtifact, *, updated_at: str
    ) -> StoredDerivative:
        del updated_at
        return super().publish(job, artifact)


__all__ = [
    "DerivativeCapacityError",
    "DerivativeConflictError",
    "DerivativeStore",
    "StoredDerivative",
]
