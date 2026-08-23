from __future__ import annotations

from pathlib import Path

from shared.events.delivery_queue import (
    DeliveryQueue,
    SnapshotAttachmentEntry,
    SnapshotDispositionEntry,
)


def test_snapshot_entries_survive_restart_without_duplicate_attachment(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path / "delivery-queue")
    attachment = SnapshotAttachmentEntry(
        "event-a", "snapshot-a", "a" * 64, "snapshots/a.jpg", 7, "image/jpeg"
    )
    disposition = SnapshotDispositionEntry(
        "event-a", "snapshot-missing", "MISSING", "capture failed"
    )
    assert queue.try_admit(attachment).accepted
    assert queue.try_admit(attachment).already_admitted
    assert queue.try_admit(disposition).accepted

    reopened = DeliveryQueue(tmp_path / "delivery-queue")
    entries = tuple(reopened.entries())
    assert [entry["kind"] for entry in entries] == [
        "SNAPSHOT_ATTACHMENT",
        "SNAPSHOT_DISPOSITION",
    ]
