from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pytest

from worker.pipeline.output.evidence import clip_maintenance
from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
)
from worker.pipeline.output.evidence.evidence_retention import (
    EvidenceRetention,
    PurgeCandidate,
    RotationReport,
)

DiskUsage = namedtuple("DiskUsage", "total used free")


def test_clip_maintenance_throttles_automatic_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[None] = []

    def rotate(
        _retention: EvidenceRetention,
        candidates: Iterable[PurgeCandidate],
        *,
        retention_cutoff: datetime,
        disk_high_watermark: float,
    ) -> RotationReport:
        del candidates, retention_cutoff, disk_high_watermark
        calls.append(None)
        return RotationReport((), (), (), False)

    monkeypatch.setattr(EvidenceRetention, "rotate", rotate)
    monkeypatch.setattr(clip_maintenance.time, "monotonic", lambda: 100.0)
    maintenance = clip_maintenance.ClipMaintenance(
        ClipRecorderConfig(store_dir=tmp_path, rotate_min_interval_seconds=30.0),
        ClipRecorderStats(),
        is_clip_held=lambda _clip_id: False,
        disk_usage_provider=lambda _path: DiskUsage(100, 10, 90),
    )

    maintenance.rotate()
    maintenance.rotate()
    maintenance.rotate(force=True)

    assert calls == [None, None]
