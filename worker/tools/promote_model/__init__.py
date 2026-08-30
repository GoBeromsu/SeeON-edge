"""Out-of-band, stdlib-only promotion contract validation."""

from worker.tools.promote_model.contracts import (
    AppliedModelSelection,
    ContractError,
    ModelSelection,
    canonical_digest,
    canonical_json_bytes,
    compare_desired_to_applied,
    parse_applied_model_selection,
    parse_model_selection,
    validate_applied_against_desired,
    validate_evaluation_receipt_identity,
    validate_field_receipt_identity,
)

__all__ = [
    "AppliedModelSelection",
    "ContractError",
    "ModelSelection",
    "canonical_digest",
    "canonical_json_bytes",
    "compare_desired_to_applied",
    "parse_applied_model_selection",
    "parse_model_selection",
    "validate_applied_against_desired",
    "validate_evaluation_receipt_identity",
    "validate_field_receipt_identity",
]
