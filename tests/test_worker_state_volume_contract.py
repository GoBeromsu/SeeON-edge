"""The worker must write its durable queue into the mounted volume.

`resolve_state_dir()` returns `~/.local/state/ml-worker` and deliberately has no
environment override. Inside a container that is `/root/.local/state/ml-worker`
-- the writable layer. The deployment mounted `worker-local-state` and then
never told the worker to use it, so every pending evidence envelope lived in the
container layer and was destroyed by any `--force-recreate`, image update, or
restart that replaced the container.

That voids the guarantee this entire release unit exists to provide: the durable
delivery queue replaced runtime SQLite precisely so evidence survives a backend
outage. It also made the pre-v17 filesystem gate decorative, because the
inventory service scanned a volume nothing had ever written to and always
reported clear.

The three views must agree: what the worker is told to use, what Compose mounts
for it, and what the inventory gate scans.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE = (_ROOT / "compose.edge.yaml").read_text(encoding="utf-8")

#: The subdirectory `DurableEvidenceStager` and the inventory gate both use.
_QUEUE_SUBDIR = "delivery-queue"


def _service_block(name: str) -> str:
    match = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|\Z)", _COMPOSE, re.M | re.S)
    assert match, f"compose.edge.yaml declares no service named {name!r}"
    return match.group(1)


def _mount_target(block: str, volume: str) -> str:
    match = re.search(rf"-\s+{re.escape(volume)}:([^\s:]+)", block)
    assert match, f"service does not mount {volume!r}"
    return match.group(1)


def test_the_worker_is_told_to_use_its_mounted_state_volume() -> None:
    """Without an explicit --state-dir the queue lands in the container layer."""
    block = _service_block("ml-worker")
    mounted = _mount_target(block, "worker-local-state")

    match = re.search(r"-\s+--state-dir\n\s+-\s+(\S+)", block)
    assert match, (
        "ml-worker does not pass --state-dir, so it falls back to the home "
        "default and writes its durable delivery queue into the container's "
        "writable layer, where container replacement destroys pending evidence"
    )
    assert match.group(1) == mounted, (
        f"ml-worker is told to use {match.group(1)!r} but its worker-local-state "
        f"volume is mounted at {mounted!r}; the queue would not be on the volume"
    )


def test_the_inventory_gate_scans_the_same_volume_the_worker_writes() -> None:
    """A gate reading a different volume always reports clear."""
    worker_block = _service_block("ml-worker")
    inventory_block = _service_block("edge-filesystem-inventory")

    # Same volume, mounted in both services. The container paths may differ; the
    # volume identity is what makes the gate see the worker's queue.
    _mount_target(worker_block, "worker-local-state")
    inventory_target = _mount_target(inventory_block, "worker-local-state")

    from backend.app.edge_db.inventory import DEFAULT_RUNTIME_STATE_DIR

    assert str(DEFAULT_RUNTIME_STATE_DIR) == inventory_target, (
        f"the inventory gate scans {DEFAULT_RUNTIME_STATE_DIR} but its "
        f"worker-local-state volume is mounted at {inventory_target}; it would "
        f"inspect an empty directory and wave the cutover through"
    )


def test_both_ends_agree_on_where_the_queue_lives_inside_the_volume() -> None:
    """Same volume is not enough; they must look at the same subdirectory."""
    from backend.app.edge_db import inventory as inventory_module
    from worker.runtime.worker import _delivery_queue_dir

    probe = Path("/probe-state")
    assert _delivery_queue_dir(probe) == probe / _QUEUE_SUBDIR, (
        "the worker's queue directory helper changed shape; the inventory gate "
        "would scan the wrong subdirectory"
    )
    assert _QUEUE_SUBDIR in inventory_module.__doc__ or _QUEUE_SUBDIR in (
        (_ROOT / "backend/app/edge_db/inventory.py").read_text(encoding="utf-8")
    ), (
        f"the inventory gate does not reference {_QUEUE_SUBDIR!r}, so it is not "
        f"scanning the directory the worker writes"
    )


def test_the_worker_state_default_is_unsuitable_for_a_container() -> None:
    """Guard the reasoning: the default really is home-relative.

    If this ever becomes an absolute system path, the explicit --state-dir is
    still correct but this test's rationale would be stale and should be
    revisited rather than silently passing.
    """
    from worker.runtime.state_dir import resolve_state_dir

    resolved = resolve_state_dir()
    assert resolved.is_relative_to(Path.home()), (
        "resolve_state_dir no longer returns a home-relative path; re-examine "
        "whether the container still needs an explicit --state-dir"
    )


def test_the_refused_evidence_command_can_actually_be_run() -> None:
    """A documented command with nowhere to run is not a remedy.

    The worker image carries no `scripts/ops`, and the backend image has no
    writable `worker-local-state` mount, so the requeue step in the runbook
    could not have been executed in any container. That makes the retention
    bound a gate the operator cannot clear.
    """
    block = _service_block("edge-refused-evidence")

    mounted = _mount_target(block, "worker-local-state")
    assert ":ro" not in block.split(mounted)[1].split("\n")[0], (
        "the operator service mounts the queue read-only, so requeue cannot write"
    )
    assert "scripts/ops/review-refused-evidence.py" in block, (
        "the service does not invoke the documented command"
    )
    assert f"- {mounted}" in block, (
        "the command is not pointed at the volume the service mounts"
    )
    assert 'profiles: ["ops"]' in block or "- ops" in block, (
        "a one-shot operator tool must not start with the stack"
    )
