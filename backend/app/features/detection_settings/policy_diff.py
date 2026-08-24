"""Non-mutating policy proposal comparison."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db.configuration import open_configuration_database
from backend.app.features.detection_settings.policy_models import PolicyDiff
from backend.app.features.detection_settings.policy_rows import (
    database_camera_id,
    effective_policy,
    policy_record,
    record_token,
)
from shared.detection_policies import (
    PolicyDocumentError,
    make_effective_policy,
    parse_policy_values,
    policy_values_dict,
)


@dataclass(frozen=True, slots=True)
class PolicyProposal:
    facility_id: str
    module_id: str
    module_version: int
    schema_id: str
    schema_version: int
    camera_id: str | None
    values: object


def build_policy_diff(path: Path, proposal: PolicyProposal) -> PolicyDiff:
    if proposal.values is None and proposal.camera_id is None:
        raise PolicyDocumentError("facility default policy values cannot be null")
    parsed = (
        None
        if proposal.values is None
        else parse_policy_values(
            module_id=proposal.module_id,
            module_version=proposal.module_version,
            schema_id=proposal.schema_id,
            schema_version=proposal.schema_version,
            values=proposal.values,
        )
    )
    with closing(open_configuration_database(path)) as connection:
        camera_key = database_camera_id(connection, proposal.camera_id)
        record = policy_record(
            connection,
            proposal.facility_id,
            camera_key,
            proposal.module_id,
            proposal.module_version,
        )
        current = effective_policy(
            connection,
            proposal.facility_id,
            camera_key,
            proposal.module_id,
            proposal.module_version,
        )
        facility = effective_policy(
            connection,
            proposal.facility_id,
            None,
            proposal.module_id,
            proposal.module_version,
        )
    proposed = (
        facility
        if parsed is None
        else make_effective_policy(
            module_id=proposal.module_id,
            module_version=proposal.module_version,
            values=parsed,
            source="facility-default" if proposal.camera_id is None else "camera-override",
            facility_revision_id=(
                0 if proposal.camera_id is None else facility.facility_revision_id
            ),
            camera_revision_id=None if proposal.camera_id is None else 0,
        )
    )
    changed = (
        current.source == "camera-override"
        if parsed is None
        else (current.source, current.values) != (proposed.source, proposed.values)
    )
    compared_payload: dict[str, object] = {
        "module_id": proposal.module_id,
        "module_version": proposal.module_version,
        "schema_id": proposal.schema_id,
        "schema_version": proposal.schema_version,
        "camera_id": proposal.camera_id,
        "values": None if parsed is None else policy_values_dict(parsed),
    }
    return PolicyDiff(changed, current, proposed, compared_payload, record_token(record))
