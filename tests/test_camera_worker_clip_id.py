"""Clip-id semantics for the composition root, replacing the deleted
``edge.runtime.camera_worker.CameraWorker``-embedded clip-recorder wiring.

``CameraWorker`` no longer exists: the new architecture splits its
responsibilities across ``WorkerRuntime`` composition (``worker/runtime/
worker.py``) and ``EvidenceEventSink.emit_for_frame()`` (``worker/pipeline/
output/event_sink.py``). Every edge test in the original
``tests/test_camera_worker_clip_id.py`` is disposed of below with a citation
to what now proves the same property; only ``_NullClipRecorder`` and the
``_CameraClipRecorderView``/``_default_clip_recorder`` fallback wiring are
genuinely new composition surface without existing coverage, so this file
ports those, not edge's ``CameraWorker`` test bodies.

Superseded (same property, different call site, cited by file:line):

* ``test_camera_worker_persists_canonical_event_identity_before_side_effects``
  and ``test_camera_worker_reuses_persisted_identity_after_restart`` --
  canonical-identity persistence-before-reuse is now a synchronous,
  structural property of ``EventIdentityStore.resolve()``
  (``worker/pipeline/decision/event_identity.py:41-54``), whose
  ``_append()`` fsyncs the journal (``event_identity.py:81-93``) before
  ``resolve()`` returns, and ``IncidentManager.admit()``
  (``worker/pipeline/decision/incident_manager.py:66-83``) only ever returns
  an event carrying the resolved identity -- there is no code path that
  reaches ``EvidenceEventSink.emit_for_frame()`` with an unpersisted identity. Proven
  end-to-end, including the restart case, by
  ``tests/test_worker_incident_manager.py::
  test_persisted_source_identity_reuses_the_edge_event_id_after_restart``.

* ``test_camera_worker_propagates_recorder_clip_id_on_event`` --
  ``EvidenceEventSink.emit_for_frame()`` binds the recorder's returned clip id
  via ``stager.complete()``; proven by
  ``tests/test_worker_event_sink.py::
  test_event_sink_stages_then_binds_the_admitted_business_event`` (asserts
  ``recorder.calls`` and ``stager.completions`` bind the same clip id).

* ``test_camera_worker_stages_before_recorder_and_skips_immediate_network``
  -- the stage-before-bind ordering is now unconditional statement order in
  ``EvidenceEventSink.emit_for_frame()`` (``stager.stage()``, then
  ``recorder.on_event()``, then ``stager.complete()``); there is no
  "immediate network" branch to skip since ``EvidenceEventSink`` only ever
  writes to the durable outbox, never delivers over the network itself
  (delivery is a separate ``EvidenceSender`` pulling from the outbox). Proven
  by the same ``test_event_sink_stages_then_binds_the_admitted_business_event``.

* ``test_camera_worker_sets_null_clip_id_without_recorder`` -- "no recorder"
  no longer exists as a state: ``EvidenceEventSink.recorder`` is a mandatory
  frozen-dataclass field (``event_sink.py:34-38``), so every sink always has
  *some* ``EventClipRecorder``. The observable behavior (event completes with
  ``clip_id=None``) is proven both by
  ``tests/test_worker_event_sink.py::
  test_event_sink_completes_without_clip_when_recording_is_unavailable`` and,
  at the composition layer, by this file's
  ``test_null_clip_recorder_always_reports_no_bound_clip`` below, since a
  never-started shared recorder is the composition-level path that actually
  produces this state (see ``_default_clip_recorder`` below).

Obsolete by design:

* ``test_camera_worker_emits_identityless_events_within_incident_cooldown``
  -- ``BusinessEvent.identity: str | int`` (``worker/types/business_event.py:
  10``) has no default and is validated at construction; an identity-less
  event is no longer constructible, so there is nothing left to admit.

Genuine production gap found, reported per task instructions, NOT fixed and
NOT asserted as passing behavior by any test here:

* ``test_camera_worker_emits_distinct_events_while_throttling_clip_recording``
  and ``test_camera_worker_throttle_does_not_open_clip_after_active_clip_ends``
  exercised edge's ``clip_recording_min_interval_sec`` throttle, which
  suppressed opening a *new* clip for events arriving within an interval of
  an already-active one (while still emitting the event itself). No
  equivalent exists in the new composition:
  ``EvidenceEventSink.emit_for_frame()`` always calls
  ``self.recorder.on_event(trigger_packet, edge_event_id, event.event_type)``
  with the default ``allow_new_clip=True`` -- nothing in ``WorkerRuntime``,
  ``EventAggregator``, or ``IncidentManager`` computes an elapsed-time-based
  decision and threads an ``allow_new_clip=False`` override through. The
  underlying mechanism the throttle would need still exists one layer down
  (``ClipAdmission.accept_event``/``ClipActor.handle_event`` both accept an
  ``allow_new_clip`` parameter), but nothing in the composition root wires a
  time-based policy into it. This is a real behavior regression versus edge
  and should be tracked, not silently dropped by omission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import worker.runtime.worker as worker_module
from contracts.frame import Frame
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder
from worker.runtime.config import WorkerConfig
from worker.runtime.worker import WorkerRuntime
from worker.types import BusinessEvent, FramePacket


class _FakeServingClient:
    def create(self, task: str, **_options: str | int | float | bool | None) -> object:
        raise AssertionError("composition tests must not create real models")


def _config(*camera_ids: str) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 1,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": f"facility-{camera_id.removeprefix('camera-')}",
                    "rtsp_url": f"rtsp://example.test/{camera_id}",
                    "heartbeat_interval_sec": 30.0,
                }
                for camera_id in camera_ids
            ],
        }
    )


def _runtime(*camera_ids: str) -> WorkerRuntime:
    return WorkerRuntime(_config(*camera_ids), serving_client=_FakeServingClient())


def _packet(camera_id: str = "camera-a") -> FramePacket:
    return FramePacket(
        camera_id=camera_id,
        frame=Frame(1, 1.0, np.zeros((2, 2, 3), dtype=np.uint8)),
        pts=1.0,
        seq=1,
        width=2,
        height=2,
        decode_time_ms=0.1,
        worker_boot_id="boot-1",
        stream_epoch=1,
    )


def _event(identity: str = "event-1") -> BusinessEvent:
    return BusinessEvent(
        "fall",
        "fall_detected",
        identity,
        "camera-a",
        "facility-a",
        1.0,
        0.9,
    )


def test_null_clip_recorder_always_reports_no_bound_clip() -> None:
    # Given: the interim/degraded recorder composition falls back to when the
    # shared ClipRecorder never started (see _compose_evidence_export's
    # docstring in worker/runtime/worker.py:434-447: "misconfiguration or
    # unavailability degrades to _NullClipRecorder").
    recorder = worker_module._NullClipRecorder()  # noqa: SLF001

    # When/Then: no event, regardless of the allow_new_clip request, ever
    # gets a bound clip id.
    packet = _packet()
    try:
        assert recorder.on_event(packet, _event()) is None
        assert recorder.on_event(packet, _event(), allow_new_clip=False) is None
    finally:
        packet.release()


def test_default_clip_recorder_degrades_to_null_when_the_shared_recorder_never_started() -> None:
    # Given: a runtime whose shared ClipRecorder has not (yet, or ever)
    # started -- the same state _compose_evidence_export leaves it in when
    # ClipRecorder.start() raises (worker.py:471-475).
    runtime = _runtime("camera-a")
    camera = runtime.config.cameras[0]
    assert runtime._clip_recorder is None  # noqa: SLF001

    # When: composition asks for this camera's clip recorder.
    recorder = runtime._default_clip_recorder(camera)  # noqa: SLF001

    # Then: it degrades to the null recorder rather than failing camera
    # activation, matching the branch EvidenceEventSink already exercises.
    assert isinstance(recorder, worker_module._NullClipRecorder)  # noqa: SLF001
    packet = _packet()
    try:
        assert recorder.on_event(packet, _event()) is None
    finally:
        packet.release()


def test_default_clip_recorder_view_forwards_the_trigger_packet() -> None:
    # Given: a runtime whose shared ClipRecorder has started (simulated here
    # by a fake standing in for the real worker.ClipRecorder actor, since
    # test_worker_evidence_export_composition.py already proves the real
    # startup path composes distinct per-camera views over one shared
    # ClipRecorder instance).
    @dataclass(slots=True)
    class _FakeSharedRecorder:
        calls: list[tuple[FramePacket, BusinessEvent, bool]] = field(default_factory=list)

        def on_event(
            self,
            trigger_packet: FramePacket,
            event: BusinessEvent,
            *,
            allow_new_clip: bool = True,
            detected_at: object | None = None,
        ) -> str | None:
            del detected_at
            self.calls.append((trigger_packet, event, allow_new_clip))
            return "clip-xyz"

    runtime = _runtime("camera-a")
    camera = runtime.config.cameras[0]
    fake_recorder = _FakeSharedRecorder()
    runtime._clip_recorder = fake_recorder  # type: ignore[assignment]  # noqa: SLF001

    # When: composition asks for this camera's view and an event is emitted
    # through it.
    view = runtime._default_clip_recorder(camera)  # noqa: SLF001
    assert isinstance(view, worker_module._CameraClipRecorderView)  # noqa: SLF001
    assert view.camera_id == "camera-a"
    packet = _packet()
    try:
        event = _event()
        result = view.on_event(packet, event)

        # Then: the view delegates the authoritative triggering packet without
        # reducing its boot/stream/frame identity back to a camera string.
        assert result == "clip-xyz"
        assert fake_recorder.calls == [(packet, event, True)]
    finally:
        packet.release()


def test_default_clip_recorder_wraps_the_real_clip_recorder_type() -> None:
    # Given: the ClipRecorderFactory type alias in worker.py promises callers
    # an EventClipRecorder; assert the composed view actually wraps the real
    # worker ClipRecorder type once started, not just any object with the
    # right shape.
    runtime = _runtime("camera-a")
    camera = runtime.config.cameras[0]
    recorder = ClipRecorder.__new__(ClipRecorder)
    runtime._clip_recorder = recorder  # noqa: SLF001

    view = runtime._default_clip_recorder(camera)  # noqa: SLF001

    assert isinstance(view, worker_module._CameraClipRecorderView)  # noqa: SLF001
    assert view.recorder is recorder
