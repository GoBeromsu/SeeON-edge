from __future__ import annotations

from worker.pipeline.output.scene_decisions import SceneDecisionProvider
from worker.pipeline.trace.capture import TraceIdentity
from worker.types import DecisionTraceSnapshot


def test_provider_reads_current_trace_snapshots_with_identity_provenance() -> None:
    snapshots = (
        DecisionTraceSnapshot(
            reason="fall-onset",
            previous_state="clear",
            current_state="fall",
            triggered=True,
            track_id=7,
            bed_id=None,
            values={"fall_probability": 0.9, "operating_threshold": 0.7},
        ),
    )
    identity = TraceIdentity(
        module_qualified_id="fall.v1",
        component_qualified_ids=(f"pose.sha256.{'a' * 64}",),
        policy_qualified_id="fall-policy.v1",
        effective_policy_id="b" * 64,
        runtime_manifest_sha256="c" * 64,
        snapshot_provider=lambda: snapshots,
    )

    decisions = SceneDecisionProvider((identity,)).current_decisions()

    assert len(decisions) == 1
    assert decisions[0].module_qualified_id == "fall.v1"
    assert decisions[0].snapshot is snapshots[0]
    assert len(decisions[0].trace_id) == len(decisions[0].analysis_trace_id) == 64
