from __future__ import annotations

from backend.app.features.audit.catalog import AuditAction
from tests_support.audit_production_owners import (
    AuditOwnerCatalogError,
    assert_owner_catalog_complete,
    production_action_owners,
)


def test_every_catalog_action_has_a_callable_production_owner() -> None:
    owners = production_action_owners()
    assert_owner_catalog_complete(owners)
    assert set(owners) == set(AuditAction)
    assert all(callable(owner) for action_owners in owners.values() for owner in action_owners)


def test_final_seven_owner_bindings_are_mutation_sensitive() -> None:
    owners = production_action_owners()
    governed = (
        AuditAction.CONNECTION_SYNC,
        AuditAction.TOPOLOGY_CONFIRM,
        AuditAction.CLIP_PLAY,
        AuditAction.EVIDENCE_RECEIPT,
        AuditAction.RELAY_SNAPSHOT_ATTACHMENT,
        AuditAction.RELAY_SNAPSHOT_DISPOSITION,
    )
    for action in governed:
        mutated = dict(owners)
        del mutated[action]
        try:
            assert_owner_catalog_complete(mutated)
        except AuditOwnerCatalogError:
            continue
        raise AssertionError(f"owner removal was not detected: {action.value}")
