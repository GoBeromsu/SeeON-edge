"""Build WireProvenance from identities already resolved at composition."""

from __future__ import annotations

from shared.events.execution_records import ExecutionRecordContractError, WireProvenance


class ExecutionRecordProvenanceError(RuntimeError):
    """A required execution identity is missing; the worker must refuse to start."""


def build_wire_provenance(
    *,
    worker_build_revision: str | None,
    worker_image_digest: str | None,
    model_digest: str | None,
    calibration_digest: str | None,
    preprocessing_identity: str | None,
    config_digest: str | None,
    policy_identity: str | None,
) -> WireProvenance:
    fields = {
        "worker_build_revision": worker_build_revision,
        "worker_image_digest": worker_image_digest,
        "model_digest": model_digest,
        "calibration_digest": calibration_digest,
        "preprocessing_identity": preprocessing_identity,
        "config_digest": config_digest,
        "policy_identity": policy_identity,
    }
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise ExecutionRecordProvenanceError(
            "execution-record provenance missing: " + ", ".join(missing)
        )
    try:
        return WireProvenance(
            worker_build_revision=str(worker_build_revision),
            worker_image_digest=str(worker_image_digest),
            model_digest=str(model_digest),
            calibration_digest=str(calibration_digest),
            preprocessing_identity=str(preprocessing_identity),
            config_digest=str(config_digest),
            policy_identity=str(policy_identity),
        )
    except ExecutionRecordContractError as error:
        raise ExecutionRecordProvenanceError(str(error)) from error


__all__ = ["ExecutionRecordProvenanceError", "build_wire_provenance"]
