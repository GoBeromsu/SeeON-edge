"""serving seam batch-input evolution contract (ADR-0002, acceptance 5).

The seam defines a typed batched-inference swap point (BatchServingClient) for
50-camera scale; the in-process client stays single-frame (ServingClient) and
does NOT implement the batch contract yet (batching backend deferred).

The edge original's other three tests are superseded, not ported — all three
protocol-satisfaction/subset checks are covered by
tests/test_worker_model_serving.py:99-102
(test_batch_serving_client_remains_a_pure_deferred_protocol), which asserts
issubclass(BatchServingClient, ServingClient), that BatchServingClient is
still a pure Protocol (``_is_protocol`` is True), and that "infer_batch" is
its own attribute (a stricter check than the edge original's
``hasattr(BatchServingClient, "infer_batch")``); and by
tests/test_worker_model_serving.py:56-61
(test_in_process_client_satisfies_only_the_single_item_protocol), which
asserts isinstance(InProcessServingClient(...), ServingClient) and
not isinstance(InProcessServingClient(...), BatchServingClient).

Only the fourth edge test survives: it exercises an independent, non-registry
fake client, proving BatchServingClient's runtime_checkable Protocol accepts
any structurally-matching object rather than only worker's own classes — a
distinct guarantee neither superseding test makes.

Added for the multi-stream serving work: the ``infer_batch`` RESULT-ORDER
contract (results[i] belongs to frames[i]). The protocol's return type
(``tuple[RunnerResult, ...]``) carries no frame identity, so nothing but this
contract stops a batching backend from handing one camera's skeleton to
another. See the ``_FakeBatchedPoseClient`` docstring for the shuffle proof.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np

from contracts.frame import Frame
from contracts.runner import PoseRunnerResult, RunnerProtocol, RunnerResult, pose_result
from worker.interfaces.serving import BatchServingClient
from worker.types import FramePacket

_ServingOption = str | int | float | bool | None


def test_batch_contract_shape_is_provisionable_by_a_future_client() -> None:
    class _FakeBatched:
        def create(self, task, **kwargs):  # noqa: ANN001, ARG002
            raise NotImplementedError

        def infer_batch(self, task, frames, **kwargs):  # noqa: ANN001, ARG002
            return [None for _ in frames]

    client = _FakeBatched()
    assert isinstance(client, BatchServingClient)
    assert client.infer_batch("pose", [1, 2, 3]) == [None, None, None]


# --- result-order contract (nvidia-multistream-serving todo 2c) ------------
#
# The Wave-3 capability coordinator drains one latest-only slot per camera
# into ONE batched forward and then hands result[i] back to frames[i]'s
# camera. Nothing in the type system ties a result row to its source frame,
# so a reordering backend (or a coordinator that zips a sorted batch against
# the unsorted frame list) would silently deliver camera A's skeleton to
# camera B. These tests pin the positional contract by making every fake
# result carry its own frame's identity, so a mismatch is detectable rather
# than plausible-looking.


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((4, 4, 3), seq % 251, dtype=np.uint8)
    return FramePacket(
        camera_id=camera_id,
        frame=Frame(index=seq, time_sec=float(seq), image=image),
        pts=float(seq),
        seq=seq,
        width=4,
        height=4,
        decode_time_ms=0.0,
    )


def _identity_box(packet: FramePacket) -> tuple[float, float, float, float, float]:
    """Encode the frame's identity into the result payload itself.

    ``camera_id`` is "camera-<n>"; the box carries (n, seq) so a caller can
    recover which frame a result row was computed from without trusting the
    row's position -- which is exactly what the order contract is about.
    """
    camera_index = int(packet.camera_id.rsplit("-", 1)[1])
    return (camera_index, packet.seq, camera_index + 1, packet.seq + 1, 0.9)


def _decode_identity(result: RunnerResult) -> tuple[str, int]:
    assert isinstance(result, PoseRunnerResult)
    box = tuple(result.boxes[0])
    return f"camera-{int(box[0])}", int(box[1])


@final
class _FakeBatchedPoseClient:
    """Structural ``BatchServingClient`` whose results carry frame identity.

    SHUFFLE PROOF: uncommenting the marked line below makes ``infer_batch``
    return a correctly-sized but positionally-wrong tuple. The order tests in
    this module must FAIL when it is uncommented -- that is how we know they
    can catch a real reordering backend rather than merely counting rows.
    """

    def __init__(self) -> None:
        self.batches: list[tuple[tuple[str, int], ...]] = []

    def create(self, task: str, **options: _ServingOption) -> RunnerProtocol:
        raise NotImplementedError(f"batched fake serves {task} only via infer_batch")

    def infer_batch(
        self,
        task: str,
        frames: Sequence[FramePacket],
        **options: _ServingOption,
    ) -> tuple[RunnerResult, ...]:
        assert task == "pose"
        ordered = list(frames)
        # ordered = ordered[1:] + ordered[:1]  # SHUFFLE PROOF -- see docstring
        self.batches.append(tuple((packet.camera_id, packet.seq) for packet in ordered))
        return tuple(
            pose_result((tuple(float(value) for value in range(51)),), (_identity_box(packet),))
            for packet in ordered
        )


def test_infer_batch_result_order_round_trips_camera_id_and_seq() -> None:
    client = _FakeBatchedPoseClient()
    frames = tuple(
        _packet(f"camera-{index}", seq=100 + index * 7) for index in range(1, 14)
    )

    results = client.infer_batch("pose", frames)

    assert isinstance(client, BatchServingClient)
    assert len(results) == len(frames)
    assert tuple(_decode_identity(result) for result in results) == tuple(
        (packet.camera_id, packet.seq) for packet in frames
    )


def test_infer_batch_result_order_is_positional_not_sorted_by_camera_or_seq() -> None:
    """A batch whose input order differs from every natural sort order.

    If a backend (or coordinator) sorted the batch by camera id or seq and
    zipped the sorted results against the unsorted frames, positional
    round-trip would break -- so feed an order that is neither.
    """
    client = _FakeBatchedPoseClient()
    frames = (
        _packet("camera-7", seq=3),
        _packet("camera-2", seq=99),
        _packet("camera-11", seq=1),
        _packet("camera-2", seq=100),
    )
    identities = tuple((packet.camera_id, packet.seq) for packet in frames)
    assert identities != tuple(sorted(identities))

    results = client.infer_batch("pose", frames)

    assert tuple(_decode_identity(result) for result in results) == identities
    assert client.batches == [identities]


def test_infer_batch_order_contract_detects_a_shuffled_backend() -> None:
    """The detection power of the two tests above, asserted in-band.

    Rather than relying on a human to uncomment the shuffle line, reproduce
    the same corruption here (rotate the returned tuple) and assert the
    positional round-trip check rejects it. Without this, a fake that always
    returned frames in sorted order would keep the order tests green.
    """
    client = _FakeBatchedPoseClient()
    frames = tuple(_packet(f"camera-{index}", seq=index * 5) for index in range(1, 6))

    honest = client.infer_batch("pose", frames)
    shuffled = honest[1:] + honest[:1]

    identities = tuple((packet.camera_id, packet.seq) for packet in frames)
    assert tuple(_decode_identity(result) for result in honest) == identities
    assert tuple(_decode_identity(result) for result in shuffled) != identities
    assert len(shuffled) == len(frames)
