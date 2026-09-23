"""Hermetic end-to-end: pump -> lanes -> exporter -> Backend query."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from observability_stack_fixtures import serve_backend, wait_until
from test_execution_record_wiring import _metadata, _pump

from shared.events.evidence_export_contract import EventReceipt
from shared.events.execution_records import WireProvenance
from shared.events.execution_records_client import ExecutionRecordsClient
from worker.domains.registry import FALL_MODULE_QUALIFIED_ID
from worker.pipeline.diagnostics.exporter import ExecutionRecordExporter
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublicationError
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
from worker.pipeline.output.evidence.smart_record_actor import SmartRecordActor
from worker.runtime.flow.evidence import FlowEvidenceBinding
from worker.runtime.flow.execution_record_emit import emit_policy_consume
from worker.types.metadata import MetadataCounters, MetadataFrame
from worker.types.trace import DecisionIdentity

_CAMERA = "cam-1"
_RELAY_TOKEN = "obs-relay-token"
_BUDGET_BYTES = 2**20
_PTS_STEP_NS = 66_666_667
_QUERY_FROM_NS = 0
_QUERY_TO_NS = (1 << 62) - 1
_PROVENANCE = WireProvenance(
    worker_build_revision="abc123",
    worker_image_digest="sha256:deadbeef",
    model_digest="model-1",
    calibration_digest="cal-1",
    preprocessing_identity="pose-bbox56/v1",
    config_digest="cfg-1",
    policy_identity="fall.policy:2",
)


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        module_qualified_id=FALL_MODULE_QUALIFIED_ID,
        effective_policy_id="a" * 64,
    )


def _frame(pump: object, seq: int) -> MetadataFrame:
    child = pump._child  # noqa: SLF001
    assert isinstance(child, UUID)
    return _metadata(child=child, seq=seq, pts=100 + seq * _PTS_STEP_NS)


def _drive_frames(
    pump: object,
    count: int,
    *,
    publish: bool,
    consume: bool,
) -> None:
    slot = pump._slot  # noqa: SLF001
    for seq in range(count):
        metadata = _frame(pump, seq)
        if publish:
            assert slot.publish(metadata) is True
        pump._process(metadata)  # noqa: SLF001
        if consume:
            emit_policy_consume(
                pump._execution_records,  # noqa: SLF001
                metadata,
                before=MetadataCounters(),
                after=slot.counters(),
                processed_count=seq + 1,
            )


def _exporter(
    lanes: ExecutionRecordLanes,
    base_url: str,
    relay_token: str,
    *,
    batch_max: int = 16,
    flush_ms: int = 20,
) -> ExecutionRecordExporter:
    return ExecutionRecordExporter(
        lanes=lanes,
        client=ExecutionRecordsClient(base_url, relay_token),
        provenance=_PROVENANCE,
        batch_max=batch_max,
        flush_ms=flush_ms,
    )


def _query(backend: object) -> dict[str, Any]:
    return backend.query(_CAMERA, _QUERY_FROM_NS, _QUERY_TO_NS, limit=500)


def _triggered_decisions(body: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for record in body["records"]:
        if record["record_kind"] != "policy.decision":
            continue
        payload = record["payload"]
        if isinstance(payload, dict) and payload.get("triggered") is True:
            found.append(record)
    return found


def _availability_kind_at(body: dict[str, Any], timestamp_ns: int) -> str | None:
    for row in body["availability"]:
        if row["from_ns"] <= timestamp_ns <= row["to_ns"]:
            return str(row["kind"])
    return None


class _RecordingPlane:
    def start_recording(
        self, camera_id: str, *, lookback_sec: int, duration_sec: int, on_sealed: object
    ) -> int:
        del camera_id, lookback_sec, duration_sec, on_sealed
        return 1

    def stop_recording(self, camera_id: str, session_id: int) -> None:
        del camera_id, session_id


class _NoClipPublisher:
    def publish(self, sealed: object, events: object) -> object:
        del sealed, events
        raise FlowClipPublicationError("clip publication unused in hermetic e2e")


def test_alert_joins_record_with_five_kinds_provenance_and_availability(tmp_path) -> None:
    """Five kinds plus decision -> delivery -> acceptance.

    decision->delivery is by frame identity (camera, boot, epoch, frame_seq)
    plus edge_event_id (delivery.causal_unit_id). decision_trace_id joins the
    triggered policy.decision to the alert audit, not to delivery/acceptance.
    """
    lanes = ExecutionRecordLanes(lane_capacity=64)
    emitted: list[object] = []
    queue_dir = tmp_path / "delivery-queue"
    stager = DurableEvidenceStager(queue_dir, camera_id=_CAMERA, facility_id="facility-a")
    plane = _RecordingPlane()
    sealed: list[FlowEvidenceBinding] = []
    actor = SmartRecordActor(
        camera_id=_CAMERA,
        media_plane=plane,
        clock=lambda: 0.0,
        sink=lambda clip: sealed[0].on_sealed(clip),
        lookback_sec=10,
        clip_id_factory=lambda: "primary-clip",
    )
    binding = FlowEvidenceBinding(
        actor=actor,
        stager=stager,
        publisher=_NoClipPublisher(),
        sidecars=FlowSealedSidecars(tmp_path / "sidecars"),
        camera_id=_CAMERA,
        execution_records=lanes,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    sealed.append(binding)
    pump = _pump(
        lanes,
        identity=_identity(),
        emitted=emitted,
        fall_transition=0.9,
        event_sink=binding,
    )
    exporter: ExecutionRecordExporter | None = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            _drive_frames(pump, 3, publish=True, consume=True)

            def _five_kinds() -> bool:
                kinds = {row["record_kind"] for row in _query(backend)["records"]}
                return {
                    "sdk.frame",
                    "model.score",
                    "policy.decision",
                    "policy.consume",
                    "event.delivery",
                }.issubset(kinds)

            wait_until(_five_kinds, timeout=5.0, what="five execution-record kinds on Backend")
            body = _query(backend)
            triggered = _triggered_decisions(body)
            assert triggered, "the immediate classifier must trigger a fall in this fixture"
            (decision,) = triggered
            assert emitted, "a triggered decision must emit an alert"
            (event,) = emitted
            audit = event.audit  # type: ignore[attr-defined]
            assert audit is not None
            assert audit["decision_trace_id"] == decision["payload"]["decision_trace_id"]

            kinds = {row["record_kind"] for row in body["records"]}
            assert "sdk.frame" in kinds
            assert "model.score" in kinds
            assert "policy.decision" in kinds
            assert "policy.consume" in kinds
            assert "event.delivery" in kinds

            deliveries = [
                row
                for row in body["records"]
                if row["record_kind"] == "event.delivery" and row["outcome"] == "admitted"
            ]
            assert deliveries, "staging must emit a stream-scoped event.delivery"
            (delivery,) = deliveries
            assert delivery["frame_seq"] == decision["frame_seq"]
            assert delivery["frame_seq"] is not None
            # Query is camera-scoped; worker_boot_id / stream_epoch are stored
            # but not projected on ExecutionRecordView.
            edge_event_id = str(event.identity)  # type: ignore[attr-defined]
            assert delivery["causal_unit_id"] == edge_event_id

            class _Accepting:
                def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt:
                    del payload_json
                    return EventReceipt("accepted_local", edge_event_id, "")

                def send_snapshot_attachment(self, payload: dict[str, object]) -> None:
                    del payload

                def send_snapshot_disposition(self, payload: dict[str, object]) -> None:
                    del payload

                def send_clip(self, claim: object) -> None:
                    del claim
                    raise AssertionError("no clip delivery expected")

            sender = EvidenceSender(
                queue_dir,
                SenderConfig("http://relay.test", "token", _CAMERA),
                transport=_Accepting(),
                execution_records=lanes,
                observing_boot_id="boot-observer",
            )
            sender.run_once()

            def _acceptance() -> bool:
                return any(
                    row["record_kind"] == "backend.acceptance" for row in _query(backend)["records"]
                )

            wait_until(_acceptance, timeout=5.0, what="backend.acceptance on Backend")
            joined = _query(backend)
            acceptances = [
                row for row in joined["records"] if row["record_kind"] == "backend.acceptance"
            ]
            assert acceptances
            (acceptance,) = acceptances
            assert acceptance["causal_unit_id"] == edge_event_id
            assert acceptance["causal_unit_id"] == delivery["causal_unit_id"]
            assert acceptance["outcome"] == "accepted_local"

            provenance_ids = {row["provenance_id"] for row in joined["records"]}
            assert len(provenance_ids) == 1
            (provenance_id,) = provenance_ids
            assert provenance_id

            observed = [int(row["observed_at_ns"]) for row in joined["records"]]
            first_observed = min(observed)
            last_observed = max(observed)
            assert _availability_kind_at(joined, first_observed) == "AVAILABLE"
            assert _availability_kind_at(joined, last_observed) == "AVAILABLE"
            tail = [
                row
                for row in joined["availability"]
                if int(row["from_ns"]) > last_observed and row["kind"] == "UNKNOWN"
            ]
            assert tail, "the tail after last_observed must be UNKNOWN"

            queryable = joined["queryable_range"]
            assert queryable["min_observed_at_ns"] == first_observed
            assert queryable["max_observed_at_ns"] == last_observed
        finally:
            if exporter is not None:
                exporter.stop()


def test_lane_overflow_is_reported_as_missing_not_recorded(tmp_path) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=2)
    pump = _pump(lanes, identity=_identity(), emitted=[], fall_transition=0.9)
    # Mixed producers on purpose: sdk.frame carries PTS-derived ns while
    # policy.decision / model.score carry process-monotonic ns, so dropped
    # items arrive with non-monotonic observed_at_ns. The lane must still
    # synthesize a valid per-producer gap (min/max ns) rather than kill the
    # exporter thread on a contract error.
    _drive_frames(pump, 6, publish=True, consume=True)
    exporter: ExecutionRecordExporter | None = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token, batch_max=32)
            exporter.start()

            def _overflow_row() -> dict[str, Any] | None:
                body = _query(backend)
                for row in body["coverage"]:
                    if (
                        row["coverage_kind"] == "MISSING_NOT_RECORDED"
                        and row["cause"] == "lane-overflow"
                        and row["from_sequence"] is not None
                        and row["to_sequence"] is not None
                    ):
                        return row
                return None

            wait_until(
                lambda: _overflow_row() is not None,
                timeout=5.0,
                what="lane-overflow MISSING_NOT_RECORDED coverage",
            )
            body = _query(backend)
            row = _overflow_row()
            assert row is not None
            assert row["exact"] is True
            assert int(row["to_sequence"]) >= int(row["from_sequence"])
            missing = [
                item
                for item in body["availability"]
                if item["kind"] == "MISSING_NOT_RECORDED"
                and not (
                    int(item["to_ns"]) < int(row["from_ns"])
                    or int(item["from_ns"]) > int(row["to_ns"])
                )
            ]
            assert missing, "loss must appear as MISSING_NOT_RECORDED availability, never hidden"
        finally:
            if exporter is not None:
                exporter.stop()


def test_restart_reopens_the_same_execution_records(tmp_path) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=64)
    pump = _pump(lanes, identity=_identity(), emitted=[], fall_transition=0.9)
    exporter: ExecutionRecordExporter | None = None
    record_ids: list[str] = []
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            _drive_frames(pump, 3, publish=True, consume=True)
            wait_until(
                lambda: len(_query(backend)["records"]) >= 1,
                timeout=5.0,
                what="exported execution records before restart",
            )
            record_ids = [row["record_id"] for row in _query(backend)["records"]]
            assert record_ids
        finally:
            if exporter is not None:
                exporter.stop()

    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as restarted:
        body = _query(restarted)
        assert [row["record_id"] for row in body["records"]] == record_ids
