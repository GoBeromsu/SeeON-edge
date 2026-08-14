"""Characterization: the in-process serving seam is a pure pass-through.

Worker obtains runners through a ``ServingClient`` so GPU serving can later be
extracted into its own service without touching orchestration/perception/
domain code (worker/interfaces/serving.py). ``InProcessServingClient`` is the
identity implementation (worker/adapters/model/in_process.py); this test pins
that it delegates to the underlying registry identically (same task, same
kwargs, same returned object) across the full production provisioning
sequence.

The edge original's other two tests are superseded, not ported:
- default-registry binding: tests/test_worker_model_serving.py:64-77
  (test_in_process_client_defaults_to_the_process_registry) proves the same
  default-binding behavior via monkeypatch plus an actually-provisioned
  object, a stronger check than reading the private ``_registry`` attribute
  directly (as the edge original did).
- protocol satisfaction: tests/test_worker_model_serving.py:56-61
  (test_in_process_client_satisfies_only_the_single_item_protocol) proves
  isinstance(ServingClient), not isinstance(BatchServingClient), and the
  absence of infer_batch, which subsumes the edge original's bare
  ``callable(client.create)`` check.
"""

from __future__ import annotations

from contracts.runner import Image, RunnerProtocol, RunnerResult, pose_result
from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import ModelOption, ModelRegistry


def test_in_process_serving_client_passes_through_to_registry() -> None:
    calls: list[tuple[str, dict[str, ModelOption]]] = []

    class _SentinelRunner:
        def __call__(self, _image: Image) -> RunnerResult:
            return pose_result((), ())

    sentinel = _SentinelRunner()

    class StubRegistry(ModelRegistry):
        def create(self, task: str, **kwargs: ModelOption) -> RunnerProtocol:
            calls.append((task, kwargs))
            return sentinel

    client = InProcessServingClient(StubRegistry())

    # Provision the same task sequence as the registry-based production
    # composition path in worker/runtime/model_composition.py; fall is created
    # separately by worker/runtime/worker.py. "person" is new relative to the
    # edge original because production wiring provisions it alongside pose/bed.
    assert client.create("pose", device="cpu") is sentinel
    assert client.create("person", device="cpu") is sentinel
    assert client.create("bed", device="cpu") is sentinel
    assert client.create("fall") is sentinel

    # Equivalence: identical delegation to the underlying registry across the
    # full production task sequence.
    assert calls == [
        ("pose", {"device": "cpu"}),
        ("person", {"device": "cpu"}),
        ("bed", {"device": "cpu"}),
        ("fall", {}),
    ]
