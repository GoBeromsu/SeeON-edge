from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

import numpy as np
import pytest
from numpy.typing import NDArray

from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from shared.edge_db.migrator import migrate_database
from worker.domains.fall import FallEventLatch
from worker.pipeline.analytics import CompositeExtractor, NamedExtractor
from worker.pipeline.bus import Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import CoordinatedInference
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.pipeline.trace import (
    BoundedTraceWriter,
    TraceCapture,
    TraceIdentity,
    TraceRetentionPolicy,
)
from worker.replay import cli as replay_cli
from worker.runtime.provenance import AppliedRuntimeManifestStore
from worker.runtime.provenance.models import AppliedRuntimeManifest
from worker.types import FramePacket, ModuleResult

CAMERA_ID = "camera-cli"
FACILITY_ID = "facility-cli"
BOOT_ID = "boot-cli"
FALL_SHA256 = "b" * 64
FIXTURE_OPERATING_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 1
    stride: int = 1
    mode: Literal["features"] = "features"


class _FallModel:
    metadata = _FallMetadata()
    operating_threshold = FIXTURE_OPERATING_THRESHOLD
    artifact_digest = FALL_SHA256

    def predict(self, features: NDArray[np.float32]) -> float:
        del features
        return 0.82


class _PoseRunner:
    def run(self, image: Image) -> RunnerResult:
        del image
        keypoints = tuple(c for i in range(17) for c in (40 + i, 30 + i, 0.9))
        return pose_result((keypoints,), ((20, 10, 120, 115, 0.95),))


@final
class _SinglePacketSubscription:
    def __init__(self, packet: FramePacket) -> None:
        self._packet: FramePacket | None = packet

    def take(self, *, timeout_sec: float | None = None) -> CoordinatedInference | None:
        del timeout_sec
        packet, self._packet = self._packet, None
        if packet is None:
            return None
        return CoordinatedInference(
            packet,
            ModuleResult("pose", _PoseRunner().run(packet.frame.image), 0.0, "pose"),
        )

    def close(self) -> None:
        self._packet = None


class _NullRecorder:
    def on_event(
        self, trigger_packet: object, event: object, *, allow_new_clip: bool = True
    ) -> str | None:
        del trigger_packet, event, allow_new_clip
        return "clip-cli"


def _seed_fall_traces(database: Path, *, frame_count: int = 3) -> None:
    manifest = AppliedRuntimeManifest.from_content(
        {"manifest_schema_version": 1, "cameras": [{"camera_id": CAMERA_ID}]}
    )
    AppliedRuntimeManifestStore(database).persist(
        manifest, boot_instance_id=BOOT_ID, applied_at="2026-08-14T00:00:00Z"
    )
    from shared.detection_policies import FallPolicyV1, make_effective_policy

    policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(FIXTURE_OPERATING_THRESHOLD),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    detector = FallEventLatch(
        _FallModel(),
        camera_id=CAMERA_ID,
        facility_id=FACILITY_ID,
        operating_threshold=FIXTURE_OPERATING_THRESHOLD,
    )
    capture = TraceCapture(
        identities=(
            TraceIdentity(
                module_qualified_id="fall.v1",
                component_qualified_ids=(
                    f"pose.sha256.{'a' * 64}",
                    f"fall-classifier.sha256.{FALL_SHA256}",
                ),
                policy_qualified_id="fall.policy.v1",
                effective_policy_id=policy.effective_policy_id,
                runtime_manifest_sha256=manifest.sha256,
                snapshot_provider=lambda: detector.last_trace_snapshots,
            ),
        )
    )
    writer = BoundedTraceWriter(database, TraceRetentionPolicy.testing())
    sink = EvidenceEventSink(
        stager=DurableEvidenceStager(
            database_path=database,
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            resident_id=None,
            config_version=1,
            clock=lambda: 1.0,
            runtime_manifest_sha256=manifest.sha256,
        ),
        recorder=_NullRecorder(),  # type: ignore[arg-type]
        now=lambda: dt.datetime(2026, 8, 14, tzinfo=dt.UTC),
    )
    writer.start()
    try:
        for seq in range(1, frame_count + 1):
            runner = _PoseRunner()
            extractor = NamedExtractor(
                module_name="pose",
                runner=runner,
                _call=runner.run,
                _clock=lambda: 1.0,
                output_adapter="pose",
            )
            analytics = CompositeExtractor(
                extractors=(extractor,),
                scheduler=Scheduler({"pose": 1}),
                tracker=GreedyIouTracker(),
                scene_state=SceneState(CAMERA_ID),
            )
            packet = FramePacket(
                camera_id=CAMERA_ID,
                frame=Frame(seq, float(seq), np.zeros((120, 180, 3), dtype=np.uint8)),
                pts=float(seq),
                seq=seq,
                width=180,
                height=120,
                decode_time_ms=0.25,
                worker_boot_id=BOOT_ID,
                stream_epoch=1,
            )
            pump = CameraPipelinePump(
                CAMERA_ID,
                _SinglePacketSubscription(packet),
                analytics,
                EventAggregator(
                    (detector,),
                    IncidentManager(cooldown_sec=0.0),
                    monotonic=lambda seq=seq: float(seq),
                ),
                sink,
                max_frames=1,
                evidence_attacher=AlertEvidenceAttacher(
                    domain_audit={}, runtime_manifest_sha256=manifest.sha256
                ),
                trace_capture=capture,
                trace_writer=writer,
            )
            pump.run()
    finally:
        writer.stop()


@pytest.fixture
def fall_model_artifact_dir(tmp_path: Path) -> Path:
    """A minimal, real (non-mocked) local LSTM artifact directory: no network, CPU only."""
    import torch
    import yaml

    artifact_dir = tmp_path / "fall-model"
    artifact_dir.mkdir()
    torch.manual_seed(0)
    from worker.adapters.model.torch_lstm_fall import _LstmNet

    net = _LstmNet(hidden=4, layers=1, dropout=0.0)
    torch.save(net.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "arch.json").write_text(
        json.dumps({"hidden": 4, "layers": 1, "dropout": 0.0}), encoding="utf-8"
    )
    import hashlib

    digest = hashlib.sha256((artifact_dir / "model.pt").read_bytes()).hexdigest()
    (artifact_dir / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "type": "lstm",
                "framework": "pytorch",
                "mode": "sequence",
                "artifact_dir": ".",
                "weights": "model.pt",
                "architecture": "arch.json",
                "metadata": "metadata.yaml",
                "window": 1,
                "stride": 1,
                "input_shape": [1, 51],
                "operating_threshold": FIXTURE_OPERATING_THRESHOLD,
                "schema_version": 1,
                "preprocessing_identity": "legacy-coco17-xyc-frame-normalized-zero-fill-v1",
                "artifact_digest": digest,
            }
        ),
        encoding="utf-8",
    )
    return artifact_dir


def test_help_flag_exits_zero_and_documents_all_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        replay_cli.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for flag in (
        "--edge-db",
        "--camera-id",
        "--module",
        "--operating-threshold",
        "--min-containment",
    ):
        assert flag in output


def test_missing_edge_db_exits_with_config_error(tmp_path: Path) -> None:
    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(tmp_path / "does-not-exist.sqlite3"),
            "--camera-id",
            CAMERA_ID,
            "--module",
            "bed_exit",
            "--min-containment",
            "0.5",
            "--hold-frames",
            "2",
            "--grace-frames",
            "1",
        ]
    )
    assert exit_code == replay_cli.CONFIG_ERROR_EXIT_CODE


def test_corrupt_edge_db_exits_with_config_error_and_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not a real sqlite file")
    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(corrupt),
            "--camera-id",
            CAMERA_ID,
            "--module",
            "bed_exit",
            "--min-containment",
            "0.5",
            "--hold-frames",
            "2",
            "--grace-frames",
            "1",
        ]
    )
    assert exit_code == replay_cli.CONFIG_ERROR_EXIT_CODE
    assert "Traceback" not in capsys.readouterr().err


def test_unknown_camera_exits_with_no_frames_code(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(database),
            "--camera-id",
            "camera-never-seen",
            "--module",
            "bed_exit",
            "--min-containment",
            "0.5",
            "--hold-frames",
            "2",
            "--grace-frames",
            "1",
        ]
    )
    assert exit_code == replay_cli.NO_FRAMES_EXIT_CODE


def test_missing_baseline_policy_flags_exits_with_config_error(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    exit_code = replay_cli.main(
        ["--edge-db", str(database), "--camera-id", CAMERA_ID, "--module", "fall"]
    )
    assert exit_code == replay_cli.CONFIG_ERROR_EXIT_CODE


def test_partial_bed_exit_candidate_flags_exit_with_config_error(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _seed_fall_traces(database)
    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(database),
            "--camera-id",
            CAMERA_ID,
            "--module",
            "bed_exit",
            "--min-containment",
            "0.5",
            "--hold-frames",
            "2",
            "--grace-frames",
            "1",
            "--candidate-min-containment",
            "0.4",
        ]
    )
    assert exit_code == replay_cli.CONFIG_ERROR_EXIT_CODE


def test_fall_replay_happy_path_prints_stable_json(
    tmp_path: Path, fall_model_artifact_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _seed_fall_traces(database)

    argv = [
        "--edge-db",
        str(database),
        "--camera-id",
        CAMERA_ID,
        "--module",
        "fall",
        "--operating-threshold",
        str(FIXTURE_OPERATING_THRESHOLD),
        "--fall-model-artifact-dir",
        str(fall_model_artifact_dir),
    ]
    exit_code = replay_cli.main(argv)
    assert exit_code == replay_cli.CLEAN_EXIT_CODE
    first_output = capsys.readouterr().out
    payload = json.loads(first_output)
    assert payload["camera_id"] == CAMERA_ID
    assert payload["module_qualified_id"] == "fall.v1"
    assert payload["frame_count"] == 3
    assert payload["reproducible"] is True
    assert payload["non_reproducible_reason"] is None
    assert payload["boot_ids"] == [BOOT_ID]

    exit_code_again = replay_cli.main(argv)
    assert exit_code_again == exit_code
    second_output = capsys.readouterr().out
    assert second_output == first_output


def test_fall_replay_comparison_reports_structured_mismatch(
    tmp_path: Path, fall_model_artifact_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _seed_fall_traces(database)

    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(database),
            "--camera-id",
            CAMERA_ID,
            "--module",
            "fall",
            "--operating-threshold",
            str(FIXTURE_OPERATING_THRESHOLD),
            "--candidate-operating-threshold",
            "0.99",
            "--fall-model-artifact-dir",
            str(fall_model_artifact_dir),
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == replay_cli.MISMATCH_EXIT_CODE
    assert payload["comparison"]["identical"] is False
    assert len(payload["comparison"]["mismatches"]) > 0


def test_fall_module_requires_model_artifact_dir(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _seed_fall_traces(database)
    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(database),
            "--camera-id",
            CAMERA_ID,
            "--module",
            "fall",
            "--operating-threshold",
            str(FIXTURE_OPERATING_THRESHOLD),
        ]
    )
    assert exit_code == replay_cli.CONFIG_ERROR_EXIT_CODE


def _insert_analysis_frame(
    database: Path,
    *,
    boot_id: str,
    camera_id: str,
    epoch: int,
    seq: int,
    source_time: float,
    applied_at: str,
) -> None:
    trace_id = hashlib.sha256(
        f"{boot_id}:{camera_id}:{epoch}:{seq}:{source_time}".encode()
    ).hexdigest()
    connection = sqlite3.connect(database)
    try:
        manifest = "e" * 64
        connection.execute(
            "INSERT OR IGNORE INTO runtime_manifest_contents VALUES (?, 1, '{}', ?)",
            (manifest, applied_at),
        )
        connection.execute(
            "INSERT OR IGNORE INTO runtime_manifest_boots VALUES (?, ?, ?)",
            (boot_id, manifest, applied_at),
        )
        connection.execute(
            "INSERT OR IGNORE INTO runtime_manifest_cameras VALUES (?, ?, ?, ?)",
            (boot_id, camera_id, manifest, applied_at),
        )
        connection.execute(
            """
            INSERT INTO runtime_analysis_traces (
                trace_id, trace_schema_version, worker_boot_id, camera_id,
                stream_epoch, frame_seq, pts, pts_missing_reason,
                source_time_sec, source_time_missing_reason, frame_width,
                frame_height, bed_region_provenance, storage_bytes
            ) VALUES (?, 1, ?, ?, ?, ?, NULL, 'not-available', ?, NULL, 180, 120, 'empty', 64)
            """,
            (trace_id, boot_id, camera_id, epoch, seq, source_time),
        )
        connection.commit()
    finally:
        connection.close()


def test_multi_boot_cli_orders_boots_and_marks_reproducible(
    tmp_path: Path, fall_model_artifact_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _seed_fall_traces(database)
    # Later boot with earlier stream-relative source_time must still sort after boot-cli.
    _insert_analysis_frame(
        database,
        boot_id="boot-cli-second",
        camera_id=CAMERA_ID,
        epoch=1,
        seq=1,
        source_time=0.25,
        applied_at="2026-08-14T01:00:00Z",
    )
    _insert_analysis_frame(
        database,
        boot_id="boot-cli-second",
        camera_id=CAMERA_ID,
        epoch=1,
        seq=2,
        source_time=0.5,
        applied_at="2026-08-14T01:00:00Z",
    )

    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(database),
            "--camera-id",
            CAMERA_ID,
            "--module",
            "fall",
            "--operating-threshold",
            str(FIXTURE_OPERATING_THRESHOLD),
            "--fall-model-artifact-dir",
            str(fall_model_artifact_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == replay_cli.CLEAN_EXIT_CODE
    assert payload["reproducible"] is True
    assert payload["boot_ids"] == [BOOT_ID, "boot-cli-second"]
    frame_boots = [frame["frame_key"][0] for frame in payload["frames"]]
    assert frame_boots[:3] == [BOOT_ID, BOOT_ID, BOOT_ID]
    assert frame_boots[3:] == ["boot-cli-second", "boot-cli-second"]


def test_truncated_trace_cli_exits_non_reproducible(
    tmp_path: Path, fall_model_artifact_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _seed_fall_traces(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            INSERT INTO runtime_trace_cursors (
                camera_id, handoff_dropped_frames, pruned_frames,
                oldest_retained_seq, newest_retained_seq, updated_at_source_sec,
                persistence_failed_frames, retention_blocked_frames,
                oldest_retained_boot_id, oldest_retained_stream_epoch,
                oldest_retained_trace_id, newest_retained_boot_id,
                newest_retained_stream_epoch, newest_retained_trace_id
            ) VALUES (?, 3, 2, 1, 3, 3.0, 0, 0, ?, 1, ?, ?, 1, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                handoff_dropped_frames = 3,
                pruned_frames = 2
            """,
            (CAMERA_ID, BOOT_ID, "a" * 64, BOOT_ID, "b" * 64),
        )
        connection.commit()
    finally:
        connection.close()

    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(database),
            "--camera-id",
            CAMERA_ID,
            "--module",
            "fall",
            "--operating-threshold",
            str(FIXTURE_OPERATING_THRESHOLD),
            "--fall-model-artifact-dir",
            str(fall_model_artifact_dir),
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == replay_cli.NON_REPRODUCIBLE_EXIT_CODE
    assert payload["reproducible"] is False
    assert payload["non_reproducible_reason"] is not None
    assert "pruned_frames=2" in payload["non_reproducible_reason"]
    assert "handoff_dropped_frames=3" in payload["non_reproducible_reason"]


def test_truncated_ab_cli_prefers_non_reproducible_exit(
    tmp_path: Path, fall_model_artifact_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    _seed_fall_traces(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            INSERT INTO runtime_trace_cursors (
                camera_id, handoff_dropped_frames, pruned_frames,
                oldest_retained_seq, newest_retained_seq, updated_at_source_sec,
                persistence_failed_frames, retention_blocked_frames
            ) VALUES (?, 1, 0, 1, 3, 3.0, 0, 0)
            ON CONFLICT(camera_id) DO UPDATE SET handoff_dropped_frames = 1
            """,
            (CAMERA_ID,),
        )
        connection.commit()
    finally:
        connection.close()

    exit_code = replay_cli.main(
        [
            "--edge-db",
            str(database),
            "--camera-id",
            CAMERA_ID,
            "--module",
            "fall",
            "--operating-threshold",
            str(FIXTURE_OPERATING_THRESHOLD),
            "--candidate-operating-threshold",
            "0.99",
            "--fall-model-artifact-dir",
            str(fall_model_artifact_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == replay_cli.NON_REPRODUCIBLE_EXIT_CODE
    assert payload["baseline"]["reproducible"] is False
    assert payload["candidate"]["reproducible"] is False
    assert "comparison" in payload
