"""Bounded diagnostics for captured YOLO arrays, not an inference qualification.

``analyze`` performs decoder-only contract replay through the real Python runners.
It executes no inference, provider, CUDA, or GPU code. The session factory's CPU
provider argument is asserted only because it is the source runners' contract.
The caller supplies the captured RGB/input/outputs and the real model with its
valid digest sidecar. Digest verification is not replaced or bypassed.

Whole-row matching does not establish a TopK tie cause and never replaces the
original, ordered raw allclose gate. There is deliberately no overall PASS.

``record`` is separate: it is the GPU oracle recorder for the Rust parity
tests. It runs the same source runners, unchanged, on the onnxruntime CUDA
provider with TF32 off, and writes their inputs, raw outputs, fall windows and
decider replay as fixtures with a manifest. It never uses the CPU provider as an
oracle; nodes ORT places on CPU are refused unless ``--allow-cpu-shape-nodes``
is given and the ONNX graph proves none of them reads image-derived data.
Rust output is never recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import struct
import subprocess
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from worker.adapters.model import seg_postprocess
from worker.adapters.model.ort_bed_seg import OrtBedSegRunner
from worker.adapters.model.ort_clip_pose import OrtClipPoseRunner

Role = Literal["bed", "stored_pose"]
_RTOL = _ATOL = 1e-4
_ROWS = 300
_MAX_MODEL_SIDE = 1280
_MAX_PROTO_SIDE = 320
_MAX_RGB_SIDE = 1920
_MAX_RGB_PIXELS = 1920 * 1080
_OUTPUT_NAMES = {"bed": ("output0", "output1"), "stored_pose": ("output0",)}
# WorkerRuntime's ClipAnalysisProfile owns stored-clip person admission.
# The live DeepStream parser's separate strict 0.05 rule is not this lane.
_THRESHOLDS = {"bed": 0.25, "stored_pose": 0.25}
# Production reads each decoded bed instance's box, score and polygon:
# nvidia_bed_zone_recognizer.bed_zone_response (score, 16-point polygon) and
# pipeline/analytics/merge._bed_box_from_output (box, score, polygon). The
# runner's own default is 48 points.
_BED_POLYGON_POINTS = (48, 16)


def _float_array(value: object, name: str) -> NDArray[np.float32]:
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ValueError(f"{name} must be a float32 array")
    return value


def _validate(
    role: Role,
    rgb: NDArray[np.uint8],
    input_tensor: NDArray[np.float32],
    candidate: Mapping[str, NDArray[np.float32]],
    baseline: Mapping[str, NDArray[np.float32]],
) -> None:
    if role not in _OUTPUT_NAMES:
        raise ValueError("role must be bed or stored_pose")
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3:
        raise ValueError("rgb must be an HxWx3 uint8 array")
    height, width, channels = rgb.shape
    if (
        channels != 3
        or min(height, width) <= 0
        or max(height, width) > _MAX_RGB_SIDE
        or height * width > _MAX_RGB_PIXELS
    ):
        raise ValueError("rgb geometry exceeds diagnostic limits")
    tensor = _float_array(input_tensor, "input_tensor")
    if (
        tensor.ndim != 4
        or tensor.shape[:2] != (1, 3)
        or not 0 < tensor.shape[2] == tensor.shape[3] <= _MAX_MODEL_SIDE
        or (role == "stored_pose" and tensor.shape != (1, 3, 640, 640))
    ):
        raise ValueError("input_tensor must have bounded square NCHW geometry (pose: 640)")
    scale = tensor.shape[2] / max(height, width)
    if min(round(height * scale), round(width * scale)) <= 0:
        raise ValueError("rgb geometry would resize to an empty dimension")
    if not np.isfinite(tensor).all():
        raise ValueError("input_tensor must be finite")
    names = _OUTPUT_NAMES[role]
    columns = 38 if role == "bed" else 57
    for label, outputs in (("candidate", candidate), ("baseline", baseline)):
        if (
            not isinstance(outputs, Mapping)
            or len(outputs) != len(names)
            or set(outputs) != set(names)
        ):
            raise ValueError(f"{label} outputs must contain exactly {names}")
        for name in names:
            values = _float_array(outputs[name], f"{label}/{name}")
            if name == "output0":
                valid_shape = values.shape == (1, _ROWS, columns)
            else:
                valid_shape = (
                    values.ndim == 4
                    and values.shape[:2] == (1, 32)
                    and 0 < values.shape[2] <= _MAX_PROTO_SIDE
                    and 0 < values.shape[3] <= _MAX_PROTO_SIDE
                )
            if not valid_shape:
                raise ValueError(f"{label}/{name} has invalid or out-of-bounds shape")
            if not np.isfinite(values).all():
                raise ValueError(f"{label}/{name} must be finite")
    for name in names:
        if candidate[name].shape != baseline[name].shape:
            raise ValueError(f"{name} candidate and baseline shapes must be identical")


def _raw_summary(
    candidate: NDArray[np.float32], baseline: NDArray[np.float32]
) -> dict[str, object]:
    close = np.isclose(candidate, baseline, rtol=_RTOL, atol=_ATOL)
    return {
        "shape": list(candidate.shape),
        "element_count": int(candidate.size),
        "allclose": bool(close.all()),
        "max_abs": float(np.abs(np.subtract(candidate, baseline, dtype=np.float64)).max()),
        "outside_tolerance_count": int(candidate.size - np.count_nonzero(close)),
        "unequal_count": int(np.count_nonzero(candidate != baseline)),
        "comparison": "original element order, including unpermuted prototype channels",
    }


def _match_rows(candidate: NDArray[np.float32], baseline: NDArray[np.float32]) -> dict[str, object]:
    """Maximum bipartite matching; ascending-index BFS augmenting paths.

    Call only after validation. Allocate a 300x300 boolean graph, not a
    300x300xcolumns comparison tensor. Every one of the 57/38 columns participates.
    Ambiguity means graph degree > 1, not proof of multiple perfect matchings.
    """
    adjacency = np.empty((_ROWS, _ROWS), dtype=bool)
    for index, row in enumerate(candidate):
        adjacency[index] = (row[5] == baseline[:, 5]) & np.isclose(
            row, baseline, rtol=_RTOL, atol=_ATOL
        ).all(axis=1)
    candidate_match = [-1] * _ROWS
    baseline_match = [-1] * _ROWS
    for start in range(_ROWS):
        pending = deque([start])
        predecessor = [-1] * _ROWS
        end = -1
        while pending and end == -1:
            current = pending.popleft()
            for neighbor_value in np.flatnonzero(adjacency[current]):
                neighbor = int(neighbor_value)
                if predecessor[neighbor] != -1:
                    continue
                predecessor[neighbor] = current
                owner = baseline_match[neighbor]
                if owner == -1:
                    end = neighbor
                    break
                pending.append(owner)
        while end != -1:
            current = predecessor[end]
            previous = candidate_match[current]
            candidate_match[current] = end
            baseline_match[end] = current
            end = previous

    def unmatched(rows: NDArray[np.float32], matches: list[int]) -> list[dict[str, object]]:
        return [
            {"index": index, "class": float(rows[index, 5]), "score": float(rows[index, 4])}
            for index, match in enumerate(matches)
            if match == -1
        ]

    matched = sum(index != -1 for index in candidate_match)
    return {
        "row_count": _ROWS,
        "column_count": int(candidate.shape[1]),
        "matched_count": matched,
        "perfect_match": matched == _ROWS,
        "candidate_to_baseline": [None if index == -1 else index for index in candidate_match],
        "ambiguous_nodes": {
            "candidate": np.flatnonzero(adjacency.sum(axis=1) > 1).tolist(),
            "baseline": np.flatnonzero(adjacency.sum(axis=0) > 1).tolist(),
        },
        "ambiguity_definition": "nodes with more than one admissible whole-row neighbor",
        "unmatched_candidate": unmatched(candidate, candidate_match),
        "unmatched_baseline": unmatched(baseline, baseline_match),
        "comparison": "all columns allclose, with additional exact class-value equality",
        "scope": "matching only; does not prove TopK ties or waive the ordered raw gate",
    }


@dataclass(frozen=True)
class _RecordedInput:
    name: str
    shape: tuple[int, ...]


class RecordedSession:
    """Single-use decoder-only contract replay; never an inference session.

    Internal seam for analyze's validated arrays. ``factory`` checks the source
    runner's CPU provider argument; no provider is created or executed.
    """

    def __init__(
        self,
        role: Role,
        input_tensor: NDArray[np.float32],
        outputs: Mapping[str, NDArray[np.float32]],
        model_path: Path,
    ) -> None:
        self.role = role
        self.input_tensor = input_tensor
        self.outputs = outputs
        self.model_path = model_path.expanduser().resolve()
        self.factory_calls = 0
        self.run_calls = 0

    def factory(self, model_path: str, providers: list[str]) -> RecordedSession:
        assert model_path == str(self.model_path), "replay model path changed"
        assert providers == ["CPUExecutionProvider"], "source CPU provider contract changed"
        assert self.factory_calls == 0, "replay factory is single-use"
        self.factory_calls += 1
        return self

    def get_inputs(self) -> list[_RecordedInput]:
        return [_RecordedInput("images", tuple(self.input_tensor.shape))]

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        assert self.factory_calls == 1 and self.run_calls == 0, "replay is single-use"
        if self.role == "bed":
            assert output_names is None, "bed source must request all outputs"
        else:
            assert output_names == ["output0"], "pose source output request changed"
        assert tuple(input_feed) == ("images",), "captured input name must be images"
        actual_feed = input_feed["images"]
        assert actual_feed.dtype == np.float32, "source input dtype changed"
        assert np.array_equal(self.input_tensor, actual_feed), (
            "captured input differs from source feed"
        )
        self.run_calls += 1
        return [self.outputs[name] for name in _OUTPUT_NAMES[self.role]]


def _replay(
    role: Role,
    rgb: NDArray[np.uint8],
    input_tensor: NDArray[np.float32],
    outputs: Mapping[str, NDArray[np.float32]],
    model_path: Path,
) -> tuple[list[dict[str, object]], str]:
    session = RecordedSession(role, input_tensor, outputs, model_path)
    instances: list[dict[str, object]] = []
    if role == "stored_pose":
        runner = OrtClipPoseRunner(
            model_path, threshold=_THRESHOLDS[role], session_factory=session.factory
        )
        for box in runner.detect_persons(rgb):
            instances.append({"box": [float(value) for value in box[:4]], "score": float(box[4])})
    else:
        bed_runner = OrtBedSegRunner(
            str(model_path), confidence=_THRESHOLDS[role], session_factory=session.factory
        )
        for box in bed_runner.detect_beds(rgb).boxes:
            instances.append(
                {
                    "box": [int(value) for value in box[:4]],
                    "score": float(box[4]),
                    "polygon": [[int(x), int(y)] for x, y in box[5]],
                }
            )
        runner = bed_runner
    assert session.run_calls == 1, "source runner did not consume the captured output"
    return instances, runner.artifact_digest


def _compare_consumers(
    role: Role, candidate: list[dict[str, object]], baseline: list[dict[str, object]]
) -> dict[str, object]:
    count_equal = len(candidate) == len(baseline)
    boxes_equal = scores_equal = polygons_equal = count_equal
    differences: list[dict[str, object]] = []
    for index in range(max(len(candidate), len(baseline))):
        actual = candidate[index] if index < len(candidate) else None
        expected = baseline[index] if index < len(baseline) else None
        fields: list[str] = []
        if actual is None or expected is None:
            fields.append("presence")
        else:
            box_equal = (
                actual["box"] == expected["box"]
                if role == "bed"
                else bool(np.allclose(actual["box"], expected["box"], rtol=_RTOL, atol=_ATOL))
            )
            score_equal = bool(
                np.isclose(actual["score"], expected["score"], rtol=_RTOL, atol=_ATOL)
            )
            polygon_equal = role != "bed" or actual["polygon"] == expected["polygon"]
            boxes_equal &= box_equal
            scores_equal &= score_equal
            polygons_equal &= polygon_equal
            if not box_equal:
                fields.append("box")
            if not score_equal:
                fields.append("score")
            if not polygon_equal:
                fields.append("polygon")
        if fields:
            differences.append(
                {"index": index, "fields": fields, "candidate": actual, "baseline": expected}
            )
    return {
        "candidate_count": len(candidate),
        "baseline_count": len(baseline),
        "presence_equal": bool(candidate) == bool(baseline),
        "count_equal": count_equal,
        "ordered_boxes_equal": boxes_equal,
        "ordered_scores_allclose": scores_equal,
        "ordered_polygons_equal": polygons_equal if role == "bed" else None,
        "order_equal": count_equal and boxes_equal and scores_equal and polygons_equal,
        "box_comparison": "exact integers" if role == "bed" else "allclose at unchanged tolerance",
        "polygon_comparison": "exact ordered vertices" if role == "bed" else "not applicable",
        "candidate": candidate,
        "baseline": baseline,
        "differences": differences,
    }


def analyze(
    role: Role,
    rgb: NDArray[np.uint8],
    input_tensor: NDArray[np.float32],
    candidate_outputs: Mapping[str, NDArray[np.float32]],
    baseline_outputs: Mapping[str, NDArray[np.float32]],
    model_path: str | Path,
) -> dict[str, object]:
    """Return a JSON-compatible report, without inference or an overall PASS.

    Exactly 300 rows and canonical output names are required: output0 (57 pose
    or 38 bed columns), plus bed output1 (1,32,H,W), H/W <= 320. Input is square
    (1,3,N,N), N <= 1280 (pose exactly 640). RGB sides are <= 1920 and area <=
    1920*1080. Nonfinite or invalid arrays are rejected before comparison/replay.
    The source runners own preprocessing, admission, boxes, masks and polygons.
    Their assertions/errors, including real digest and exact-feed failures,
    propagate; there is no fallback, sorting, IoU waiver, or tolerance adjustment.
    """
    _validate(role, rgb, input_tensor, candidate_outputs, baseline_outputs)
    raw = {
        name: _raw_summary(candidate_outputs[name], baseline_outputs[name])
        for name in _OUTPUT_NAMES[role]
    }
    matching = _match_rows(candidate_outputs["output0"][0], baseline_outputs["output0"][0])
    path = Path(model_path)
    candidate, candidate_digest = _replay(role, rgb, input_tensor, candidate_outputs, path)
    baseline, baseline_digest = _replay(role, rgb, input_tensor, baseline_outputs, path)
    if candidate_digest != baseline_digest:
        raise ValueError("model digest changed between decoder-only replays")
    consumer = _compare_consumers(role, candidate, baseline)
    consumer.update(
        {
            "scope": "decoder-only contract replay; no inference/provider/GPU execution",
            "runner": "OrtBedSegRunner" if role == "bed" else "OrtClipPoseRunner",
            "threshold": _THRESHOLDS[role],
            "session_factory_provider_argument": ["CPUExecutionProvider"],
            "input_name": "images",
            "input_shape": list(input_tensor.shape),
            "input_array_equal": True,
            "requested_output_names": None if role == "bed" else ["output0"],
            "model_digest_verified": True,
        }
    )
    positive_bed = role == "bed" and bool(candidate) and bool(baseline)
    if role != "bed":
        positive_reason = "stored_pose does not exercise the bed consumer"
    elif not positive_bed:
        positive_reason = (
            "At least one source replay returned no beds; empty-result agreement is not "
            "positive-bed coverage or an exemption from the raw gate."
        )
    else:
        positive_reason = (
            "Both source replays returned bed instances; decoder coverage only, not "
            "positive-bed GPU qualification or proof of a real bed in the image."
        )
    return {
        "role": role,
        "scope": "captured-array diagnostics and decoder-only contract replay",
        "inference_executed": False,
        "provider_execution_qualified": False,
        "gpu_execution_qualified": False,
        "raw_order_root_cause": "not established; whole-row matching does not prove TopK ties",
        "model_sha256": candidate_digest,
        "rtol": _RTOL,
        "atol": _ATOL,
        "raw_gate": all(output["allclose"] for output in raw.values()),
        "raw_outputs": raw,
        "row_matching": matching,
        "consumer_replay": consumer,
        "positive_bed_covered": positive_bed,
        "positive_bed_reason": positive_reason,
        "positive_bed_gpu_qualified": False,
        "limits": {
            "rows": _ROWS,
            "named_outputs": 2,
            "model_side": _MAX_MODEL_SIDE,
            "prototype_side": _MAX_PROTO_SIDE,
            "rgb_side": _MAX_RGB_SIDE,
            "rgb_pixels": _MAX_RGB_PIXELS,
        },
    }


_ONNX_SHA256 = {
    "bed": "de1c081df29936d6cf42329ec53b4d22f1e02f13bf1ea3b769907dc1d95a3d86",
    "stored_pose": "724ae1b1b4420cea85f98ac7c5ccabf3d56229f7c25390475e09b7056e3b9526",
    "fall": "258ae9d9460534e659bf97af4bc55a083c830fa190ff6c8ed347db6bdbf32163",
}
# The static profile each TensorRT engine is built for; the placement probe
# resolves the ONNX's dynamic axes to it.
_ENGINE_INPUT_SHAPES = {
    "bed": (1, 3, 1280, 1280),
    "stored_pose": (1, 3, 640, 640),
    "fall": (1, 30, 56),
}
_CUDA_OPTIONS = {"device_id": "0", "use_tf32": "0", "cudnn_conv_algo_search": "HEURISTIC"}
_STRICT_PLACEMENT = ("session.disable_cpu_ep_fallback", "1")
_RERUN_ATOL = 1e-6
_CORPUS_SIZE = (640, 360)
_CORPUS_FPS = 30
_SCENES = ("a1-fall", "a1-normal", "a4-fall", "a4-normal")
_SCENE_FRAMES = 360
_PARITY_FRAMES = (("a1-fall", 0), ("a1-fall", 180), ("a4-fall", 0), ("a4-fall", 180))
_CLIP_SCENE = "a1-fall"
_CLIP_FRAMES = 60
_RGB_MAGIC = b"SPRGB001"
_REPLAY_TRACK = 1


class RecorderError(RuntimeError):
    """The oracle cannot be recorded faithfully; no partial fixture is trusted."""


class _CapturingSession:
    """Runs each call twice, refuses rerun drift and keeps copies of the call."""

    def __init__(self, session: Any, oracle: _CudaOracle) -> None:
        self._session = session
        self._oracle = oracle
        self.captures: list[tuple[dict[str, np.ndarray], list[np.ndarray]]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def run(self, output_names: Any, input_feed: Mapping[str, Any], *args: Any, **kwargs: Any):
        first = self._session.run(output_names, input_feed, *args, **kwargs)
        second = self._session.run(output_names, input_feed, *args, **kwargs)
        for left, right in zip(first, second, strict=True):
            left, right = np.asarray(left), np.asarray(right)
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RecorderError("ORT CUDA rerun changed an output shape or dtype")
            diff = (
                float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                if left.size
                else 0.0
            )
            if not diff <= _RERUN_ATOL:
                raise RecorderError(f"ORT CUDA rerun differs by {diff:.3g}")
            self._oracle.max_rerun_diff = max(self._oracle.max_rerun_diff, diff)
        self.captures.append(
            (
                {name: np.array(value, copy=True) for name, value in input_feed.items()},
                [np.array(value, copy=True) for value in first],
            )
        )
        return first

    def take(self) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        if len(self.captures) != 1:
            raise RecorderError(f"expected one ORT call, captured {len(self.captures)}")
        return self.captures.pop()


class _CudaOracle:
    """Session factory handed to the source runners in place of their CPU one."""

    def __init__(self, *, allow_cpu_shape_nodes: bool, profile_dir: Path) -> None:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RecorderError("onnxruntime has no CUDA provider; refusing a CPU oracle")
        self._ort = ort
        self._allow_cpu_shape_nodes = allow_cpu_shape_nodes
        self._profile_dir = profile_dir
        self.sessions: dict[str, _CapturingSession] = {}
        self.placement: dict[str, dict[str, Any]] = {}
        self.max_rerun_diff = 0.0

    def _providers(self) -> list[tuple[str, dict[str, str]]]:
        return [("CUDAExecutionProvider", dict(_CUDA_OPTIONS))]

    def factory(self, model_path: str, providers: Sequence[str]) -> _CapturingSession:
        if list(providers) != ["CPUExecutionProvider"]:
            raise RecorderError("source runners must request the CPU provider this oracle replaces")
        path = Path(model_path)
        digest = _sha256_file(path)
        role = next((name for name, sha in _ONNX_SHA256.items() if sha == digest), None)
        if role is None:
            raise RecorderError(f"{path.name}: sha256 {digest} is not an approved model")
        if role in self.sessions:
            raise RecorderError(f"{role}: session opened twice")
        strict = self._ort.SessionOptions()
        strict.add_session_config_entry(*_STRICT_PLACEMENT)
        placement: dict[str, Any] = {"strict": True, "cpu_nodes": None}
        try:
            session = self._ort.InferenceSession(
                str(path), sess_options=strict, providers=self._providers()
            )
        except Exception as exc:
            text = str(exc).strip()
            reason = text.splitlines()[0] if text else type(exc).__name__
            if not self._allow_cpu_shape_nodes:
                raise RecorderError(
                    f"{role}: strict GPU placement failed ({reason}); rerun with "
                    "--allow-cpu-shape-nodes to probe and admit shape-only CPU nodes"
                ) from exc
            placement = {
                "strict": False,
                "strict_error": reason,
                "cpu_nodes": self._probe_cpu_nodes(path, role),
            }
            session = self._ort.InferenceSession(
                str(path), sess_options=self._ort.SessionOptions(), providers=self._providers()
            )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RecorderError(f"{role}: ORT did not register the CUDA provider")
        placement["providers"] = list(session.get_providers())
        self.placement[role] = placement
        captured = _CapturingSession(session, self)
        self.sessions[role] = captured
        return captured

    def _probe_cpu_nodes(self, path: Path, role: str) -> dict[str, Any]:
        """Profile one run and admit CPU nodes only if none reads image-derived data.

        Dataflow starts at the graph inputs; a ``Shape`` output depends only on
        the static engine shape, so it ends the image dependency.
        """
        options = self._ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(self._profile_dir / f"{role}-placement")
        session = self._ort.InferenceSession(
            str(path), sess_options=options, providers=self._providers()
        )
        shape = _ENGINE_INPUT_SHAPES[role]
        specs = session.get_inputs()
        if (
            len(specs) != 1
            or specs[0].type != "tensor(float)"
            or len(specs[0].shape) != len(shape)
            or any(
                isinstance(dim, int) and dim != want
                for dim, want in zip(specs[0].shape, shape, strict=True)
            )
        ):
            raise RecorderError(f"{role}: ONNX input does not admit engine shape {shape}")
        feed = {specs[0].name: np.zeros(shape, dtype=np.float32)}
        session.run(None, feed)
        events = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        nodes: dict[str, dict[str, Any]] = {}
        for event in events:
            args = event.get("args", {})
            name = str(event.get("name", ""))
            if (
                event.get("cat") != "Node"
                or args.get("provider") != "CPUExecutionProvider"
                or not name.endswith("_kernel_time")
            ):
                continue
            output_types = sorted(
                {dtype for shape in args.get("output_type_shape", []) for dtype in shape}
            )
            nodes[name.removesuffix("_kernel_time")] = {
                "op": args.get("op_name"),
                "outputs": output_types,
            }
        if not nodes:
            raise RecorderError(f"{role}: strict placement failed but the probe saw no CPU node")
        refused = sorted(set(nodes) - _shape_only_nodes(path))
        if refused:
            raise RecorderError(f"{role}: ORT placed image-derived nodes on CPU: {refused[:5]}")
        return {
            "count": len(nodes),
            "ops": dict(sorted(Counter(str(node["op"]) for node in nodes.values()).items())),
            "output_types": sorted({dtype for node in nodes.values() for dtype in node["outputs"]}),
            "image_derived_inputs": 0,
            "nodes": sorted(nodes),
        }


def _shape_only_nodes(path: Path) -> set[str]:
    """Names of top-level nodes whose inputs never derive from graph input values."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    initializers = {tensor.name for tensor in graph.initializer}
    derived = {value.name for value in graph.input if value.name not in initializers}
    shape_only: set[str] = set()
    for node in graph.node:  # ONNX requires topological order.
        if any(attribute.type in (5, 10) for attribute in node.attribute):
            continue  # A subgraph (GRAPH/GRAPHS) is never admitted.
        reads_image = any(name in derived for name in node.input if name)
        if not reads_image:
            shape_only.add(node.name)
        if reads_image and node.op_type != "Shape":
            derived.update(node.output)
    return shape_only


class _FixtureWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, dict[str, Any]] = {}

    def data(self, relative: str, payload: bytes) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return self.register(relative)

    def register(self, relative: str) -> str:
        payload = (self.root / relative).read_bytes()
        self.files[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        return relative

    def tensor(self, relative: str, value: np.ndarray) -> dict[str, Any]:
        array = np.ascontiguousarray(value, dtype="<f4")
        self.data(relative, array.tobytes())
        return {"path": relative, "shape": list(array.shape)}

    def rgb(self, relative: str, rgb: np.ndarray) -> str:
        height, width = rgb.shape[:2]
        return self.data(relative, _RGB_MAGIC + struct.pack("<II", width, height) + rgb.tobytes())


class _FallOracle:
    """The classifier's model: the real runner, recording each window and logit."""

    def __init__(self, runner: Any, session: _CapturingSession) -> None:
        self._runner = runner
        self._session = session
        self.frame = -1
        self.windows: list[np.ndarray] = []
        self.logits: list[np.float32] = []
        self.predictions: list[dict[str, Any]] = []

    def predict(self, features: Any) -> Any:
        self._session.captures.clear()
        result = self._runner.predict(features)
        feed, outputs = self._session.take()
        window = np.ascontiguousarray(feed["window"][0], dtype="<f4")
        logit = np.float32(np.asarray(outputs[0]).reshape(-1)[0])
        if window.shape != (30, 56) or float(result.model_evidence.raw_logit) != float(logit):
            raise RecorderError("fall runner input or logit does not match its ORT call")
        self.windows.append(window)
        self.logits.append(logit)
        self.predictions.append(
            {
                "frame": self.frame,
                "window_sha256": hashlib.sha256(window.tobytes()).hexdigest(),
                "logit": _bits32(logit),
                "temperature": _bits64(result.model_evidence.applied_temperature),
                "fall_transition": _bits64(result.fall_transition),
            }
        )
        return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bits64(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def _bits32(value: np.float32) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def _decode_bed(
    outputs: Sequence[np.ndarray],
    letterbox: seg_postprocess.Letterbox,
    *,
    model_size: int,
    max_points: int,
) -> tuple[tuple[seg_postprocess.BedInstance, ...], list[NDArray[np.bool_]]]:
    """Runs the production bed decode and keeps each instance's binary mask."""
    masks: list[NDArray[np.bool_]] = []
    contour = seg_postprocess.largest_external_contour

    def capture(mask: NDArray[np.bool_]) -> Any:
        masks.append(np.array(mask, dtype=bool))
        return contour(mask)

    seg_postprocess.largest_external_contour = capture
    try:
        instances = seg_postprocess.decode_end_to_end_segmentation(
            outputs[0],
            outputs[1],
            letterbox,
            model_size=model_size,
            confidence=_THRESHOLDS["bed"],
            max_points=max_points,
        )
    finally:
        seg_postprocess.largest_external_contour = contour
    if len(masks) != len(instances):
        raise RecorderError("bed decode did not trace one mask per instance")
    return instances, masks


def _bed_decisions(
    writer: _FixtureWriter,
    name: str,
    rgb: NDArray[np.uint8],
    images: np.ndarray,
    outputs: Sequence[np.ndarray],
    detected: tuple[seg_postprocess.BedInstance, ...],
) -> dict[str, Any]:
    """Records what production consumes from the ORT bed outputs."""
    model_size = int(images.shape[-1])
    tensor, letterbox = seg_postprocess.letterbox_rgb(rgb, model_size)
    if not np.array_equal(tensor, images):
        raise RecorderError("bed letterbox differs from the runner's input")
    decoded = {
        points: _decode_bed(outputs, letterbox, model_size=model_size, max_points=points)
        for points in _BED_POLYGON_POINTS
    }
    runner_points = _BED_POLYGON_POINTS[0]
    instances, masks = decoded[runner_points]
    if tuple(instances) != tuple(detected):
        raise RecorderError("bed decode differs from the runner's result")
    for points, (other, other_masks) in decoded.items():
        same = [a[:5] == b[:5] for a, b in zip(instances, other, strict=True)]
        if not all(same) or not all(map(np.array_equal, masks, other_masks)):
            raise RecorderError(f"bed decode at {points} points changed an instance")
    height, width = rgb.shape[:2]
    stacked = np.asarray(masks, dtype=np.uint8).reshape(len(masks), height, width)
    return {
        "letterbox": {
            "source_height": letterbox.source_height,
            "source_width": letterbox.source_width,
            "scale": _bits64(letterbox.scale),
            "resized_height": letterbox.resized_height,
            "resized_width": letterbox.resized_width,
            "pad_top": letterbox.pad_top,
            "pad_left": letterbox.pad_left,
        },
        "masks": {
            "path": writer.data(f"bed/{name}.masks.u8", stacked.tobytes()),
            "shape": list(stacked.shape),
        },
        "instances": [
            {
                "box": list(instance[:4]),
                "score": _bits32(np.float32(instance[4])),
                **{
                    f"polygon{points}": [list(point) for point in decoded[points][0][row][5]]
                    for points in _BED_POLYGON_POINTS
                },
            }
            for row, instance in enumerate(instances)
        ],
    }


def _field(value: object) -> str:
    if value is None:
        return "-"
    text = str(getattr(value, "value", value))
    if "\t" in text or "\n" in text:
        raise RecorderError("replay text fields must not contain tabs or newlines")
    return text


def _replay_lines(
    frame_index: int, evaluated: bool, events: Sequence[Any], snapshots: Sequence[Any]
) -> list[str]:
    """Canonical lines shared with the Rust replay; see gpu_parity.rs."""
    lines = [f"F\t{frame_index}\t{int(evaluated)}"]
    for event in events:
        probability = "-" if event.probability is None else _bits64(event.probability)
        text = (event.domain, event.event_type, event.identity, event.camera_id, event.facility_id)
        lines.append(
            "\t".join(
                (
                    "E",
                    *(_field(value) for value in text),
                    _bits64(event.time_sec),
                    probability,
                    _field(event.person_id),
                    _field(event.bed_id),
                )
            )
        )
    for snapshot in snapshots:
        values = sorted((_field(name), value) for name, value in snapshot.values.items())
        missing = sorted(
            (_field(name), _field(reason)) for name, reason in snapshot.missing_values.items()
        )
        lines.append(
            "\t".join(
                (
                    "T",
                    _field(snapshot.reason),
                    _field(snapshot.previous_state),
                    _field(snapshot.current_state),
                    str(int(snapshot.triggered)),
                    _field(snapshot.track_id),
                    _field(snapshot.bed_id),
                    str(len(values)),
                    str(len(missing)),
                )
            )
        )
        for name, value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise RecorderError(f"trace value {name} is neither int nor float")
            kind = f"I\t{value}" if isinstance(value, int) else f"F\t{_bits64(value)}"
            lines.append(f"V\t{name}\t{kind}")
        lines.extend(f"M\t{name}\t{reason}" for name, reason in missing)
    return lines


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if (rgb.shape[1], rgb.shape[0]) != _CORPUS_SIZE:
        raise RecorderError(f"{path.name}: corpus frames must be {_CORPUS_SIZE}")
    return rgb


def _ensure_sidecar(path: Path, expected: str) -> None:
    """Writes the runner digest sidecar for an approved model copy, never a new one."""
    if _sha256_file(path) != expected:
        raise RecorderError(f"{path.name}: sha256 differs from the approved model")
    sidecar = path.with_name(f"{path.name}.sha256")
    if sidecar.exists():
        if sidecar.read_text(encoding="ascii") != f"{expected}\n":
            raise RecorderError(f"{sidecar.name}: existing sidecar disagrees")
        return
    sidecar.write_text(f"{expected}\n", encoding="ascii")


def _pose_observation(
    rows: np.ndarray, detections: Sequence[tuple[float, ...]], width: int, height: int
) -> tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]] | None:
    """Rebuilds the runner's admitted boxes from its raw rows and returns the first
    with its unclipped keypoints; the runner itself returns boxes only."""
    scale = min(640 / width, 640 / height)
    kept: list[tuple[tuple[float, ...], np.ndarray]] = []
    for row in rows:
        if row[5] != 0 or row[4] < _THRESHOLDS["stored_pose"]:
            continue
        box = (
            max(0.0, min(float(width), float(row[0] / scale))),
            max(0.0, min(float(height), float(row[1] / scale))),
            max(0.0, min(float(width), float(row[2] / scale))),
            max(0.0, min(float(height), float(row[3] / scale))),
        )
        if box[0] < box[2] and box[1] < box[3]:
            kept.append(((*box, float(row[4])), row))
    if tuple(box for box, _ in kept) != tuple(detections):
        raise RecorderError("raw pose rows do not reproduce the runner's detections")
    if not kept:
        return None
    box, row = kept[0]
    keypoints = tuple(
        (float(row[6 + 3 * k] / scale), float(row[7 + 3 * k] / scale), float(row[8 + 3 * k]))
        for k in range(17)
    )
    return box[:4], keypoints


def _record_replay(
    scene_dir: Path,
    scene: str,
    pose: Any,
    pose_session: _CapturingSession,
    fall: _FallOracle,
    writer: _FixtureWriter,
) -> dict[str, Any]:
    from shared.detection_policies import FallPolicyV2
    from worker.domains.fall.classifier import FallWindowClassifier
    from worker.domains.fall.policy import FallDomainDecider, FallPolicyDecider

    decider = FallDomainDecider(
        classifier=FallWindowClassifier(model=fall),
        policy=FallPolicyDecider("cam", "fac", "boot", "epoch", 0, policy=FallPolicyV2()),
    )
    first_prediction = len(fall.predictions)
    frames: list[dict[str, Any]] = []
    lines: list[str] = []
    events_total = 0
    for index in range(_SCENE_FRAMES):
        rgb = _load_rgb(scene_dir / f"frame_{index:04d}.png")
        height, width = rgb.shape[:2]
        pose_session.captures.clear()
        detections = pose.detect_persons(rgb)
        _, outputs = pose_session.take()
        observed = _pose_observation(np.asarray(outputs[0])[0], detections, width, height)
        time_sec = index / _CORPUS_FPS
        if observed is None:
            live: tuple[int, ...] = ()
            observation = SimpleNamespace(track_ids=(), boxes=(), keypoints=())
        else:
            (x1, y1, x2, y2), keypoints = observed
            live = (_REPLAY_TRACK,)
            observation = SimpleNamespace(
                track_ids=live,
                boxes=(SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2),),
                keypoints=(keypoints,),
            )
        fall.frame = index
        events = decider.update(
            SimpleNamespace(
                observation=observation,
                frame_width=width,
                frame_height=height,
                time_sec=time_sec,
                frame_index=index,
                live_track_ids=live,
            )
        )
        events_total += len(events)
        lines.extend(
            _replay_lines(
                index, decider.last_update_evaluated, events, decider.last_trace_snapshots
            )
        )
        frames.append(
            {
                "frame": index,
                "time_sec": _bits64(time_sec),
                "pts_ns": int(time_sec * 1_000_000_000),
                "live": list(live),
                "bbox": None if observed is None else [_bits64(v) for v in observed[0]],
                "keypoints": None
                if observed is None
                else [[_bits64(v) for v in point] for point in observed[1]],
            }
        )
    packets = {
        "scene": scene,
        "frame_width": _CORPUS_SIZE[0],
        "frame_height": _CORPUS_SIZE[1],
        "track_id": _REPLAY_TRACK,
        "frames": frames,
        "predictions": fall.predictions[first_prediction:],
    }
    return {
        "scene": scene,
        "frames": _SCENE_FRAMES,
        "packets": writer.data(
            f"replay/{scene}.packets.json", json.dumps(packets, indent=1).encode()
        ),
        "expected": writer.data(f"replay/{scene}.expected.txt", ("\n".join(lines) + "\n").encode()),
        "predictions": len(fall.predictions) - first_prediction,
        "events": events_total,
    }


def _record_clip(corpus: Path, writer: _FixtureWriter) -> dict[str, Any]:
    import av

    if "libx264" not in av.codecs_available:
        raise RecorderError("PyAV lacks libx264")
    relative = f"clip/{_CLIP_SCENE}.mp4"
    path = writer.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=_CORPUS_FPS)
        stream.width, stream.height = _CORPUS_SIZE
        stream.pix_fmt = "yuv420p"
        for index in range(_CLIP_FRAMES):
            rgb = _load_rgb(corpus / _CLIP_SCENE / f"frame_{index:04d}.png")
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    writer.register(relative)
    decoded = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            decoded.append(
                {"pts": frame.pts, "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
            )
    if len(decoded) != _CLIP_FRAMES:
        raise RecorderError(f"clip decoded {len(decoded)} frames, encoded {_CLIP_FRAMES}")
    return {
        "path": relative,
        "width": _CORPUS_SIZE[0],
        "height": _CORPUS_SIZE[1],
        "encoder": "libx264 yuv420p",
        "pyav": av.__version__,
        "conversion": "PyAV VideoFrame.to_ndarray(format='rgb24'), default SWS_BILINEAR",
        "frames": decoded,
    }


def _gpu_identity() -> dict[str, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
            "-i",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, driver, compute = (part.strip() for part in output.split(","))
    return {"name": name, "driver": driver, "compute_capability": compute}


def _corpus_receipts(corpus: Path) -> dict[str, str]:
    receipts = {
        str(path.relative_to(corpus)): _sha256_file(path)
        for path in [corpus / "render-receipt.json"]
        + [corpus / scene / "render-receipt.json" for scene in _SCENES]
        if path.is_file()
    }
    if not receipts:
        raise RecorderError("the synthetic corpus has no render-receipt.json")
    return receipts


def record(
    *,
    corpus: Path,
    bed_onnx: Path,
    pose_onnx: Path,
    fall_bundle: Path,
    out: Path,
    allow_cpu_shape_nodes: bool = False,
) -> dict[str, Any]:
    """Records ORT-CUDA oracle fixtures for the Rust GPU parity tests."""
    import onnxruntime as ort

    from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner

    if out.exists() and any(out.iterdir()):
        raise RecorderError("fixture directory must be absent or empty")
    _ensure_sidecar(bed_onnx, _ONNX_SHA256["bed"])
    _ensure_sidecar(pose_onnx, _ONNX_SHA256["stored_pose"])
    if _sha256_file(fall_bundle / "model.onnx") != _ONNX_SHA256["fall"]:
        raise RecorderError("fall bundle model.onnx differs from the approved model")
    receipts = _corpus_receipts(corpus)
    profile_dir = out / "placement"
    profile_dir.mkdir(parents=True)
    writer = _FixtureWriter(out)
    oracle = _CudaOracle(allow_cpu_shape_nodes=allow_cpu_shape_nodes, profile_dir=profile_dir)

    bed = OrtBedSegRunner(
        str(bed_onnx), confidence=_THRESHOLDS["bed"], session_factory=oracle.factory
    )
    pose = OrtClipPoseRunner(
        pose_onnx, threshold=_THRESHOLDS["stored_pose"], session_factory=oracle.factory
    )
    fall_runner = OrtPoseBbox56Runner.from_artifact_dir(fall_bundle, session_factory=oracle.factory)
    for session in oracle.sessions.values():
        session.captures.clear()

    bed_fixtures, pose_fixtures = [], []
    for scene, index in _PARITY_FRAMES:
        name = f"{scene}-{index:04d}"
        rgb = _load_rgb(corpus / scene / f"frame_{index:04d}.png")
        frame = writer.rgb(f"frames/{name}.rgb", rgb)
        detected_beds = bed.detect_beds(rgb).boxes
        feed, outputs = oracle.sessions["bed"].take()
        if [np.shape(output) for output in outputs] != [(1, 300, 38), (1, 32, 320, 320)]:
            raise RecorderError("bed outputs do not have the approved shapes")
        bed_fixtures.append(
            {
                "id": name,
                "frame": frame,
                "images": writer.tensor(f"bed/{name}.images.f32", feed["images"]),
                "output0": writer.tensor(f"bed/{name}.output0.f32", outputs[0]),
                "output1": writer.tensor(f"bed/{name}.output1.f32", outputs[1]),
                "detections": len(detected_beds),
                **_bed_decisions(writer, name, rgb, feed["images"], outputs, tuple(detected_beds)),
            }
        )
        detected_persons = pose.detect_persons(rgb)
        feed, outputs = oracle.sessions["stored_pose"].take()
        if [np.shape(output) for output in outputs] != [(1, 300, 57)]:
            raise RecorderError("pose output does not have the approved shape")
        pose_fixtures.append(
            {
                "id": name,
                "frame": frame,
                "images": writer.tensor(f"pose/{name}.images.f32", feed["images"]),
                "output0": writer.tensor(f"pose/{name}.output0.f32", outputs[0]),
                "detections": len(detected_persons),
            }
        )

    fall = _FallOracle(fall_runner, oracle.sessions["fall"])
    replay = [
        _record_replay(corpus / scene, scene, pose, oracle.sessions["stored_pose"], fall, writer)
        for scene in _SCENES
    ]
    if not fall.windows:
        raise RecorderError("the replay produced no fall predictions")
    fall_fixture = {
        "windows": writer.tensor("fall/windows.f32", np.stack(fall.windows)),
        "logits": writer.tensor("fall/logits.f32", np.asarray(fall.logits, dtype=np.float32)),
    }
    clip = _record_clip(corpus, writer)

    manifest = {
        "schema": "seeon-gpu-oracle-fixtures/v2",
        "recorder": {
            "path": "tests_support/native_yolo_parity.py",
            "sha256": _sha256_file(Path(__file__)),
            "commit": "working-tree",
        },
        "oracle": {
            "onnxruntime": ort.__version__,
            "build_info": ort.get_build_info(),
            "onnx_placement_analysis": importlib.metadata.version("onnx"),
            "provider": "CUDAExecutionProvider",
            "provider_options": dict(_CUDA_OPTIONS),
            "strict_session_config": dict([_STRICT_PLACEMENT]),
            "rerun_atol": _RERUN_ATOL,
            "max_rerun_abs_diff": oracle.max_rerun_diff,
        },
        "gpu": _gpu_identity(),
        "corpus": {"receipts": receipts, "status": "unverified"},
        "models": {
            role: {"onnx_sha256": _ONNX_SHA256[role], "placement": oracle.placement[role]}
            for role in _ONNX_SHA256
        },
        "tolerance": {"max_abs": _ATOL, "ulp": 2, "rows_from_score": _THRESHOLDS["bed"]},
        "bed": bed_fixtures,
        "stored_pose": pose_fixtures,
        "fall": fall_fixture,
        "replay": replay,
        "clip": clip,
        "fixtures": dict(sorted(writer.files.items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record ORT-CUDA oracle fixtures for the Rust GPU parity tests."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bed-onnx", type=Path, required=True)
    parser.add_argument("--pose-onnx", type=Path, required=True)
    parser.add_argument("--fall-bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-cpu-shape-nodes", action="store_true")
    args = parser.parse_args(argv)
    manifest = record(
        corpus=args.corpus,
        bed_onnx=args.bed_onnx,
        pose_onnx=args.pose_onnx,
        fall_bundle=args.fall_bundle,
        out=args.out,
        allow_cpu_shape_nodes=args.allow_cpu_shape_nodes,
    )
    summary = {
        "fixtures": len(manifest["fixtures"]),
        "max_rerun_abs_diff": manifest["oracle"]["max_rerun_abs_diff"],
        "placement": {
            role: model["placement"]["strict"] for role, model in manifest["models"].items()
        },
        "replay": [
            (item["scene"], item["predictions"], item["events"]) for item in manifest["replay"]
        ],
    }
    print(json.dumps(summary))
    return 0


__all__ = ["analyze", "record"]


if __name__ == "__main__":
    raise SystemExit(_main())
