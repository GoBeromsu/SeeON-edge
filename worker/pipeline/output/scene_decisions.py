from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from worker.pipeline.trace.capture import TraceIdentity, snapshots_for
from worker.pipeline.trace.models import DecisionTrace, content_id

_SCENE_ANALYSIS_TRACE_ID: Final = content_id({"kind": "scene-current-frame"})

@dataclass(frozen=True, slots=True)
class SceneDecisionProvider:
    """Read current hardware-neutral decider snapshots after each update."""

    identities: tuple[TraceIdentity, ...]

    def current_decisions(self) -> tuple[DecisionTrace, ...]:
        decisions: list[DecisionTrace] = []
        for identity_index, identity in enumerate(self.identities):
            for snapshot in snapshots_for(identity):
                body = {
                    "identity_index": identity_index,
                    "module": identity.module_qualified_id,
                    "policy": identity.policy_qualified_id,
                    "effective_policy_id": identity.effective_policy_id,
                    "runtime_manifest_sha256": identity.runtime_manifest_sha256,
                    "reason": snapshot.reason,
                    "previous_state": snapshot.previous_state,
                    "current_state": snapshot.current_state,
                    "triggered": snapshot.triggered,
                    "track_id": snapshot.track_id,
                    "bed_id": snapshot.bed_id,
                    "values": dict(snapshot.values),
                    "missing_values": dict(snapshot.missing_values),
                }
                decisions.append(
                    DecisionTrace(
                        trace_id=content_id(body),
                        analysis_trace_id=_SCENE_ANALYSIS_TRACE_ID,
                        identity_index=identity_index,
                        module_qualified_id=identity.module_qualified_id,
                        policy_qualified_id=identity.policy_qualified_id,
                        effective_policy_id=identity.effective_policy_id,
                        runtime_manifest_sha256=identity.runtime_manifest_sha256,
                        snapshot=snapshot,
                    )
                )
        return tuple(decisions)


__all__ = ["SceneDecisionProvider"]
