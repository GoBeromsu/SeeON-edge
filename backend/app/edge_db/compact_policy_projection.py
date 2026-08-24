"""Project current and immediately previous detection policy state."""

from __future__ import annotations

import sqlite3
from typing import TypeAlias

SqliteValue: TypeAlias = None | int | float | str | bytes


def project_policies(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    """Map each current activation with at most its one-step rollback revision."""
    revisions = {
        str(row[0]): row
        for row in source.execute(
            "SELECT revision_id,facility_id,camera_id,module_id,module_version,schema_id,"
            "schema_version,values_json,content_sha256,created_at "
            "FROM control_detection_policy_revisions"
        )
    }
    activations = source.execute(
        "SELECT activation_id,facility_id,camera_id,module_id,module_version,active_revision_id,"
        "previous_revision_id,activation_generation,status,refusal_reason,activated_at,applied_at "
        "FROM control_detection_policy_activations ORDER BY activation_generation,activation_id"
    ).fetchall()
    current: dict[tuple[str, str | None, str], tuple[SqliteValue, ...]] = {}
    for activation in activations:
        key = (str(activation[1]), activation[2], str(activation[3]))
        current[key] = activation
    for policy_id, (_key, activation) in enumerate(sorted(current.items()), start=1):
        active = revisions.get(str(activation[5]))
        if active is None:
            raise sqlite3.DatabaseError("active policy revision is missing")
        previous = None if activation[6] is None else revisions.get(str(activation[6]))
        if activation[6] is not None and previous is None:
            raise sqlite3.DatabaseError("previous policy revision is missing")
        target.execute(
            "INSERT INTO policies (policy_id,facility_id,camera_id,module_id,"
            "module_version,schema_id,"
            "schema_version,active_values_json,active_content_sha256,previous_present,"
            "previous_values_json,previous_content_sha256,activation_generation,status,"
            "refusal_reason,activated_at,applied_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                policy_id,
                activation[1],
                activation[2],
                activation[3],
                activation[4],
                active[5],
                active[6],
                active[7],
                active[8],
                int(previous is not None),
                None if previous is None else previous[7],
                None if previous is None else previous[8],
                activation[7],
                activation[8],
                activation[9],
                activation[10],
                activation[11],
                activation[11] or activation[10],
            ),
        )


__all__ = ["project_policies"]
