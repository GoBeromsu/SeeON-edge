"""Synthetic decoder-seam tests only: these fixtures are NOT GPU evidence.

Temporary model bytes/sidecars exercise real digest verification, not ONNX
loading or a claim about a real model. No inference/provider/GPU runs here.
The native C++ parser's strict 0.05 boundary is outside this Python-consumer scope.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests_support.native_yolo_parity import RecordedSession, analyze
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.seg_postprocess import letterbox_rgb

_UNIT_MODEL_BYTES = b"unit decoder seam only; not an ONNX model or GPU evidence"
_ROLES = ("bed", "stored_pose")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "decoder-seam-only.onnx"
    path.write_bytes(_UNIT_MODEL_BYTES)
    path.with_suffix(".onnx.sha256").write_text(
        hashlib.sha256(_UNIT_MODEL_BYTES).hexdigest() + "\n", encoding="ascii"
    )
    return path


def _case(role: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    if role == "bed":
        tensor, _ = letterbox_rgb(rgb, 8)
        outputs = {
            "output0": np.zeros((1, 300, 38), dtype=np.float32),
            "output1": np.zeros((1, 32, 8, 8), dtype=np.float32),
        }
    else:
        # An all-black square has an all-zero source pose tensor at any scale.
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        outputs = {"output0": np.zeros((1, 300, 57), dtype=np.float32)}
    return rgb, tensor, outputs


def _copy(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: values.copy() for name, values in outputs.items()}


def _put(
    outputs: dict[str, np.ndarray],
    role: str,
    index: int,
    *,
    box: tuple[float, float, float, float] = (1, 1, 3, 3),
    score: float = 0.9,
    class_value: float | None = None,
) -> None:
    """Construct fixture rows with known coordinates, not a parallel decoder."""
    row = outputs["output0"][0, index]
    row[:] = 0
    row[:4] = np.asarray(box) * (80 if role == "stored_pose" else 1)
    row[4] = score
    row[5] = (59 if role == "bed" else 0) if class_value is None else class_value
    if role == "bed":
        row[6] = 1


def _unique_rejected(outputs: dict[str, np.ndarray]) -> None:
    rows = outputs["output0"][0]
    rows[:, 5] = 1
    rows[:, -1] = np.arange(300, dtype=np.float32) + 10


@pytest.mark.parametrize("role", _ROLES)
def test_full_300_row_permutation_matches_whole_rows_and_preserves_duplicates(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _unique_rejected(baseline)
    baseline["output0"][0, 299] = baseline["output0"][0, 298]
    permutation = np.roll(np.arange(300)[::-1], 7)
    candidate = _copy(baseline)
    candidate["output0"] = baseline["output0"][:, permutation].copy()

    report = analyze(role, rgb, tensor, candidate, baseline, model_path)
    matching = report["row_matching"]
    assert matching["row_count"] == matching["matched_count"] == 300
    assert matching["column_count"] == (38 if role == "bed" else 57)
    assert matching["perfect_match"] is True
    assert matching["unmatched_candidate"] == matching["unmatched_baseline"] == []
    assert matching["ambiguous_nodes"] == {
        "candidate": np.flatnonzero(permutation >= 298).tolist(),
        "baseline": [298, 299],
    }
    pairs = matching["candidate_to_baseline"]
    assert len(pairs) == len(set(pairs)) == 300
    for index, neighbor in enumerate(pairs):
        assert candidate["output0"][0, index, 5] == baseline["output0"][0, neighbor, 5]
        np.testing.assert_allclose(
            candidate["output0"][0, index],
            baseline["output0"][0, neighbor],
            rtol=1e-4,
            atol=1e-4,
        )
    assert analyze(role, rgb, tensor, candidate, baseline, model_path)["row_matching"] == matching
    assert report["raw_gate"] is False
    assert report["consumer_replay"]["order_equal"] is True
    assert "does not prove TopK" in report["raw_order_root_cause"]


@pytest.mark.parametrize("role", _ROLES)
def test_augmenting_path_rematches_instead_of_greedy_neighbor_loss(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _unique_rejected(baseline)
    baseline["output0"][0, :2, -1] = (0, 0.00015)
    candidate = _copy(baseline)
    # Row 0 can use either neighbor. Row 1 can use only neighbor 0.
    candidate["output0"][0, :2, -1] = (0.000075, -0.00005)
    report = analyze(role, rgb, tensor, candidate, baseline, model_path)
    matching = report["row_matching"]
    assert matching["matched_count"] == 300
    assert matching["candidate_to_baseline"][:2] == [1, 0]
    assert matching["ambiguous_nodes"] == {"candidate": [0], "baseline": [0]}


@pytest.mark.parametrize("role", _ROLES)
def test_duplicate_neighbors_cannot_be_reused_or_missing_duplicates_discarded(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _unique_rejected(baseline)
    baseline["output0"][0, 1:3] = baseline["output0"][0, 0]
    candidate = _copy(baseline)
    candidate["output0"][0, 2] = baseline["output0"][0, 3]
    matching = analyze(role, rgb, tensor, candidate, baseline, model_path)["row_matching"]
    assert matching["matched_count"] == 299
    assert matching["perfect_match"] is False
    assert [row["index"] for row in matching["unmatched_candidate"]] == [3]
    assert [row["index"] for row in matching["unmatched_baseline"]] == [2]
    neighbors = [index for index in matching["candidate_to_baseline"] if index is not None]
    assert len(neighbors) == len(set(neighbors)) == 299


@pytest.mark.parametrize("role", _ROLES)
def test_independent_column_permutations_are_not_whole_row_matches(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _unique_rejected(baseline)
    baseline["output0"][0, :2, -2:] = ((1, 2), (2, 1))
    candidate = _copy(baseline)
    candidate["output0"][0, :2, -2:] = ((1, 1), (2, 2))
    matching = analyze(role, rgb, tensor, candidate, baseline, model_path)["row_matching"]
    assert matching["matched_count"] == 298
    assert [row["index"] for row in matching["unmatched_candidate"]] == [0, 1]
    assert [row["index"] for row in matching["unmatched_baseline"]] == [0, 1]


@pytest.mark.parametrize("role", _ROLES)
def test_unmatched_selected_rows_keep_original_indices_finite_classes_and_scores(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _unique_rejected(baseline)
    _put(baseline, role, 17, score=0.5)
    _put(baseline, role, 219, box=(4, 4, 6, 6), score=0.75)
    candidate = _copy(baseline)
    candidate["output0"][0, 17, -1] += 1
    candidate["output0"][0, 219, 5] = np.nextafter(
        baseline["output0"][0, 219, 5], np.float32(np.inf)
    )
    matching = analyze(role, rgb, tensor, candidate, baseline, model_path)["row_matching"]
    assert matching["matched_count"] == 298
    for side, outputs in (("candidate", candidate), ("baseline", baseline)):
        assert matching[f"unmatched_{side}"] == [
            {
                "index": index,
                "class": float(outputs["output0"][0, index, 5]),
                "score": float(outputs["output0"][0, index, 4]),
            }
            for index in (17, 219)
        ]
    json.dumps(matching, allow_nan=False)


@pytest.mark.parametrize("role", _ROLES)
def test_matching_requires_exact_class_even_when_raw_allclose_accepts_it(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _unique_rejected(baseline)
    baseline["output0"][0, :, 5] = 59 if role == "bed" else 0
    candidate = _copy(baseline)
    candidate["output0"][0, 17, 5] = np.nextafter(baseline["output0"][0, 17, 5], np.float32(np.inf))
    report = analyze(role, rgb, tensor, candidate, baseline, model_path)
    assert report["raw_gate"] is True
    assert report["row_matching"]["matched_count"] == 299
    assert [row["index"] for row in report["row_matching"]["unmatched_candidate"]] == [17]


@pytest.mark.parametrize("role", _ROLES)
def test_source_threshold_has_representable_below_equal_above_cases(
    role: str, model_path: Path
) -> None:
    rgb, tensor, outputs = _case(role)
    threshold = np.float32(0.25)
    scores = (
        np.nextafter(threshold, np.float32(-np.inf)),
        threshold,
        np.nextafter(threshold, np.float32(np.inf)),
    )
    for index, score in enumerate(scores):
        _put(outputs, role, index, box=(index, index, index + 2, index + 2), score=score)
    consumer = analyze(role, rgb, tensor, outputs, outputs, model_path)["consumer_replay"]
    assert consumer["threshold"] == 0.25
    assert consumer["candidate_count"] == consumer["baseline_count"] == 2
    assert [item["box"] for item in consumer["candidate"]] == [[1, 1, 3, 3], [2, 2, 4, 4]]
    assert [item["score"] for item in consumer["candidate"]] == [
        float(score) for score in scores[1:]
    ]
    assert consumer["order_equal"] is True


@pytest.mark.parametrize("role", _ROLES)
def test_source_class_predicate_including_finite_fractional_bed_classes(
    role: str, model_path: Path
) -> None:
    rgb, tensor, outputs = _case(role)
    if role == "bed":
        classes = (59, 59.75, np.nextafter(np.float32(59), np.float32(0)), 60, -59.75)
        expected_count = 2
    else:
        classes = (0, np.nextafter(np.float32(0), np.float32(1)), 0.75, -0.75, 1)
        expected_count = 1
    for index, class_value in enumerate(classes):
        _put(outputs, role, index, class_value=class_value)
    consumer = analyze(role, rgb, tensor, outputs, outputs, model_path)["consumer_replay"]
    assert consumer["candidate_count"] == expected_count
    assert [item["box"] for item in consumer["candidate"]] == [[1, 1, 3, 3]] * expected_count


@pytest.mark.parametrize("role", _ROLES)
def test_source_invalid_zero_and_clipped_degenerate_box_behavior(
    role: str, model_path: Path
) -> None:
    rgb, tensor, outputs = _case(role)
    boxes = (
        (3, 1, 2, 3),  # Reversed x.
        (1, 3, 3, 2),  # Reversed y.
        (1, 1, 1, 3),  # Zero width before clipping.
        (1, 1, 3, 1),  # Zero height before clipping.
        (9, 1, 10, 2),  # Becomes zero width after clipping.
        (-3, 1, -1, 2),  # Becomes zero width after clipping.
        (-2, -2, 3, 3),  # Remains nondegenerate after clipping.
    )
    for index, box in enumerate(boxes):
        _put(outputs, role, index, box=box)
    consumer = analyze(role, rgb, tensor, outputs, outputs, model_path)["consumer_replay"]
    # The bed source admits before integer clipping; do not silently repair it.
    expected = [[8, 1, 8, 2], [0, 1, 0, 2], [0, 0, 3, 3]] if role == "bed" else [[0, 0, 3, 3]]
    assert [item["box"] for item in consumer["candidate"]] == expected
    assert consumer["candidate_count"] == len(expected)


@pytest.mark.parametrize("role", _ROLES)
def test_rejected_row_swap_keeps_consumer_equal_but_raw_gate_false(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _put(baseline, role, 0, class_value=58 if role == "bed" else 1)
    _put(baseline, role, 1, box=(4, 4, 6, 6), class_value=60 if role == "bed" else 2)
    candidate = _copy(baseline)
    candidate["output0"][0, :2] = baseline["output0"][0, [1, 0]]
    report = analyze(role, rgb, tensor, candidate, baseline, model_path)
    assert report["raw_gate"] is False
    assert report["row_matching"]["matched_count"] == 300
    assert report["consumer_replay"]["order_equal"] is True
    assert report["consumer_replay"]["candidate_count"] == 0
    assert report["positive_bed_covered"] is False
    assert report["positive_bed_reason"]
    assert report["gpu_execution_qualified"] is False
    assert "overall_pass" not in report and "pass" not in report


@pytest.mark.parametrize("role", _ROLES)
def test_admitted_same_score_row_swap_is_an_order_difference_not_a_waiver(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _put(baseline, role, 0, box=(1, 1, 3, 3), score=0.75)
    _put(baseline, role, 1, box=(4, 4, 6, 6), score=0.75)
    if role == "bed":
        baseline["output1"][0, 0] = 1
    candidate = _copy(baseline)
    candidate["output0"][0, :2] = baseline["output0"][0, [1, 0]]
    report = analyze(role, rgb, tensor, candidate, baseline, model_path)
    consumer = report["consumer_replay"]
    assert report["raw_gate"] is False
    assert report["row_matching"]["perfect_match"] is True
    assert consumer["presence_equal"] is consumer["count_equal"] is True
    assert consumer["ordered_boxes_equal"] is False
    assert consumer["ordered_scores_allclose"] is True
    assert consumer["order_equal"] is False
    assert [item["index"] for item in consumer["differences"]] == [0, 1]
    assert all("box" in item["fields"] for item in consumer["differences"])
    if role == "bed":
        assert consumer["ordered_polygons_equal"] is False


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("removed", [1, 2])
def test_duplicate_consumer_count_and_presence_are_not_deduplicated(
    role: str, removed: int, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _put(baseline, role, 0)
    _put(baseline, role, 1)
    candidate = _copy(baseline)
    candidate["output0"][0, 2 - removed : 2] = 0
    report = analyze(role, rgb, tensor, candidate, baseline, model_path)
    consumer = report["consumer_replay"]
    assert consumer["baseline_count"] == 2
    assert consumer["candidate_count"] == 2 - removed
    assert consumer["presence_equal"] is (removed == 1)
    assert consumer["count_equal"] is consumer["order_equal"] is False
    assert [item["index"] for item in consumer["differences"]] == list(range(2 - removed, 2))
    assert all(item["fields"] == ["presence"] for item in consumer["differences"])
    assert report["row_matching"]["matched_count"] == 300 - removed


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize(("delta", "close"), [(0.00001, True), (0.001, False)])
def test_scores_use_unchanged_tolerance_separately_from_geometry(
    role: str, delta: float, close: bool, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    _put(baseline, role, 0, score=0.5)
    candidate = _copy(baseline)
    candidate["output0"][0, 0, 4] += delta
    report = analyze(role, rgb, tensor, candidate, baseline, model_path)
    consumer = report["consumer_replay"]
    assert report["rtol"] == report["atol"] == 1e-4
    assert report["raw_gate"] is close
    assert consumer["ordered_boxes_equal"] is True
    assert consumer["ordered_scores_allclose"] is close
    assert consumer["order_equal"] is close
    if not close:
        assert consumer["differences"][0]["fields"] == ["score"]


@pytest.mark.parametrize(("delta", "close"), [(0.001, True), (0.1, False)])
def test_stored_pose_source_boxes_use_unchanged_tolerance(
    delta: float, close: bool, model_path: Path
) -> None:
    rgb, tensor, baseline = _case("stored_pose")
    _put(baseline, "stored_pose", 0)
    candidate = _copy(baseline)
    candidate["output0"][0, 0, 0] += delta
    consumer = analyze("stored_pose", rgb, tensor, candidate, baseline, model_path)[
        "consumer_replay"
    ]
    assert consumer["ordered_boxes_equal"] is close
    assert consumer["ordered_scores_allclose"] is True
    assert consumer["order_equal"] is close


def test_bed_integer_boxes_do_not_get_a_one_pixel_tolerance(model_path: Path) -> None:
    rgb, tensor, baseline = _case("bed")
    _put(baseline, "bed", 0, box=(1.99999, 1, 5, 4))
    candidate = _copy(baseline)
    candidate["output0"][0, 0, 0] = 2.00001
    report = analyze("bed", rgb, tensor, candidate, baseline, model_path)
    consumer = report["consumer_replay"]
    assert report["raw_gate"] is True
    assert consumer["baseline"][0]["box"] == [1, 1, 5, 4]
    assert consumer["candidate"][0]["box"] == [2, 1, 5, 4]
    assert consumer["ordered_boxes_equal"] is consumer["order_equal"] is False
    assert consumer["ordered_scores_allclose"] is True


_RECTANGLE = [
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 1],
    [5, 2],
    [5, 3],
    [5, 4],
    [4, 4],
    [3, 4],
    [2, 4],
    [2, 3],
    [2, 2],
]


@pytest.mark.parametrize(("logit", "polygon"), [(-0.00001, []), (0.0, []), (0.00001, _RECTANGLE)])
def test_source_masks_use_strict_half_threshold_crop_and_ordered_contour(
    logit: float, polygon: list[list[int]], model_path: Path
) -> None:
    rgb, tensor, outputs = _case("bed")
    _put(outputs, "bed", 0, box=(2, 1, 5, 4))
    outputs["output1"][0, 0] = logit
    consumer = analyze("bed", rgb, tensor, outputs, outputs, model_path)["consumer_replay"]
    assert consumer["candidate_count"] == 1
    # Integer crop is [2,5) x [1,4); vertices come from the actual source tracer.
    assert consumer["candidate"][0]["polygon"] == polygon
    assert consumer["ordered_polygons_equal"] is True


def test_tiny_prototype_delta_can_change_polygon_without_a_pixel_waiver(model_path: Path) -> None:
    rgb, tensor, baseline = _case("bed")
    _put(baseline, "bed", 0, box=(2, 1, 5, 4))
    candidate = _copy(baseline)
    candidate["output1"][0, 0] = 0.00001
    report = analyze("bed", rgb, tensor, candidate, baseline, model_path)
    consumer = report["consumer_replay"]
    assert report["raw_gate"] is True
    assert report["row_matching"]["perfect_match"] is True
    assert consumer["ordered_boxes_equal"] is consumer["ordered_scores_allclose"] is True
    assert consumer["ordered_polygons_equal"] is consumer["order_equal"] is False
    assert consumer["baseline"][0]["polygon"] == []
    assert consumer["candidate"][0]["polygon"] == _RECTANGLE
    assert consumer["differences"][0]["fields"] == ["polygon"]


def test_every_output_is_reported_and_prototype_channels_are_never_permuted(
    model_path: Path,
) -> None:
    rgb, tensor, baseline = _case("bed")
    _put(baseline, "bed", 0, box=(2, 1, 5, 4))
    baseline["output1"][0, 0] = 1
    baseline["output1"][0, 1] = -1
    candidate = _copy(baseline)
    candidate["output1"][0, :2] = baseline["output1"][0, [1, 0]]
    report = analyze("bed", rgb, tensor, candidate, baseline, model_path)
    assert set(report["raw_outputs"]) == {"output0", "output1"}
    rows, protos = report["raw_outputs"]["output0"], report["raw_outputs"]["output1"]
    assert rows["element_count"] == 300 * 38
    assert rows["max_abs"] == rows["outside_tolerance_count"] == 0
    assert rows["allclose"] is True
    assert protos["element_count"] == 32 * 8 * 8
    assert protos["max_abs"] == 2
    assert protos["unequal_count"] == protos["outside_tolerance_count"] == 128
    assert protos["allclose"] is report["raw_gate"] is False
    assert report["row_matching"]["matched_count"] == 300
    assert report["consumer_replay"]["ordered_polygons_equal"] is False


@pytest.mark.parametrize("role", _ROLES)
def test_all_300_identical_admitted_rows_retain_every_consumer_instance(
    role: str, model_path: Path
) -> None:
    rgb, tensor, outputs = _case(role)
    _put(outputs, role, 0)
    outputs["output0"][0, 1:] = outputs["output0"][0, 0]
    report = analyze(role, rgb, tensor, outputs, outputs, model_path)
    assert report["row_matching"]["matched_count"] == 300
    assert report["row_matching"]["candidate_to_baseline"] == list(range(300))
    assert report["row_matching"]["ambiguous_nodes"] == {
        "candidate": list(range(300)),
        "baseline": list(range(300)),
    }
    consumer = report["consumer_replay"]
    assert consumer["candidate_count"] == consumer["baseline_count"] == 300
    assert len(consumer["candidate"]) == 300
    assert consumer["order_equal"] is True
    assert report["positive_bed_gpu_qualified"] is False


@pytest.mark.parametrize("role", _ROLES)
def test_raw_counts_include_below_tolerance_changes_without_mutating_captures(
    role: str, model_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    candidate = _copy(baseline)
    candidate["output0"][0, 0, 0] = 2
    candidate["output0"][0, 299, -1] = 0.00005
    original_candidate, original_baseline = _copy(candidate), _copy(baseline)
    original_tensor, original_rgb = tensor.copy(), rgb.copy()
    report = analyze(role, rgb, tensor, candidate, baseline, model_path)
    raw = report["raw_outputs"]["output0"]
    assert raw["max_abs"] == 2
    assert raw["outside_tolerance_count"] == 1
    assert raw["unequal_count"] == 2
    assert raw["allclose"] is report["raw_gate"] is False
    for name in baseline:
        np.testing.assert_array_equal(candidate[name], original_candidate[name])
        np.testing.assert_array_equal(baseline[name], original_baseline[name])
    np.testing.assert_array_equal(tensor, original_tensor)
    np.testing.assert_array_equal(rgb, original_rgb)


@pytest.mark.parametrize("role", _ROLES)
def test_report_is_plain_json_and_empty_agreement_is_not_qualification(
    role: str, model_path: Path
) -> None:
    rgb, tensor, outputs = _case(role)
    report = analyze(role, rgb, tensor, outputs, outputs, model_path)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["inference_executed"] is False
    assert report["provider_execution_qualified"] is False
    assert report["gpu_execution_qualified"] is False
    assert report["positive_bed_covered"] is report["positive_bed_gpu_qualified"] is False
    assert report["positive_bed_reason"]
    assert report["model_sha256"] == hashlib.sha256(_UNIT_MODEL_BYTES).hexdigest()
    assert report["consumer_replay"]["scope"].startswith("decoder-only contract replay")
    assert report["consumer_replay"]["session_factory_provider_argument"] == [
        "CPUExecutionProvider"
    ]
    assert report["consumer_replay"]["model_digest_verified"] is True
    assert report["raw_gate"] is True
    assert "overall_pass" not in report


@pytest.mark.parametrize("role", _ROLES)
def test_nonblack_nonsquare_source_preprocessing_must_match_captured_feed(
    role: str, model_path: Path
) -> None:
    _, _, outputs = _case(role)
    rgb = np.full((4, 8, 3), (17, 80, 231), dtype=np.uint8)
    if role == "bed":
        tensor, _ = letterbox_rgb(rgb, 8)
    else:
        # Explicit fixture: uniform 2:1 RGB fills the top 320 rows, zero below.
        tensor = np.zeros((1, 3, 640, 640), dtype=np.float32)
        tensor[0, :, :320, :] = np.asarray((17, 80, 231), dtype=np.float32)[:, None, None] / 255.0
    report = analyze(role, rgb, tensor, outputs, outputs, model_path)
    assert report["consumer_replay"]["input_array_equal"] is True
    assert report["consumer_replay"]["input_shape"] == list(tensor.shape)
    changed = tensor.copy()
    changed[0, 0, 0, 0] = np.nextafter(changed[0, 0, 0, 0], np.float32(np.inf))
    assert np.allclose(tensor, changed, rtol=1e-4, atol=1e-4)
    with pytest.raises(ModelLoadError, match="cannot run") as error:
        analyze(role, rgb, changed, outputs, outputs, model_path)
    assert isinstance(error.value.__cause__, AssertionError)
    assert "captured input differs" in str(error.value.__cause__)


@pytest.mark.parametrize("role", _ROLES)
def test_recorded_session_checks_metadata_provider_names_feed_and_single_use(
    role: str, model_path: Path
) -> None:
    _, tensor, outputs = _case(role)
    # Mapping insertion order cannot swap output0/output1 returned to the bed source.
    session = RecordedSession(role, tensor, dict(reversed(list(outputs.items()))), model_path)
    (spec,) = session.get_inputs()
    assert spec.name == "images" and spec.shape == tensor.shape
    with pytest.raises(AssertionError, match="provider"):
        session.factory(str(model_path.resolve()), ["CUDAExecutionProvider"])
    with pytest.raises(AssertionError, match="model path"):
        session.factory(str(model_path.with_name("other.onnx")), ["CPUExecutionProvider"])
    assert session.factory(str(model_path.resolve()), ["CPUExecutionProvider"]) is session
    requested = None if role == "bed" else ["output0"]
    with pytest.raises(AssertionError, match="outputs|output request"):
        session.run(["wrong"], {"images": tensor})
    with pytest.raises(AssertionError, match="input name"):
        session.run(requested, {"wrong": tensor})
    with pytest.raises(AssertionError, match="dtype"):
        session.run(requested, {"images": tensor.astype(np.float64)})
    changed = tensor.copy()
    changed.flat[0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(AssertionError, match="captured input differs"):
        session.run(requested, {"images": changed})
    replayed = session.run(requested, {"images": tensor})
    assert replayed[0] is outputs["output0"]
    if role == "bed":
        assert replayed[1] is outputs["output1"]
    assert session.run_calls == 1
    with pytest.raises(AssertionError, match="single-use"):
        session.run(requested, {"images": tensor})
    with pytest.raises(AssertionError, match="single-use"):
        session.factory(str(model_path.resolve()), ["CPUExecutionProvider"])


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("defect", ["model_missing", "sidecar_missing", "tampered", "bad_digest"])
def test_real_model_digest_verification_is_not_bypassed(
    role: str, defect: str, model_path: Path
) -> None:
    rgb, tensor, outputs = _case(role)
    if defect == "model_missing":
        model_path = model_path.with_name("absent.onnx")
    elif defect == "sidecar_missing":
        model_path.with_suffix(".onnx.sha256").unlink()
    elif defect == "tampered":
        model_path.write_bytes(b"different unit decoder bytes")
    else:
        model_path.with_suffix(".onnx.sha256").write_text("not-a-digest\n", encoding="ascii")
    with pytest.raises(ModelLoadError):
        analyze(role, rgb, tensor, outputs, outputs, model_path)


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_in_any_row_column_is_rejected_before_analysis_or_model_access(
    role: str, side: str, nonfinite: float, tmp_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    candidate = _copy(baseline)
    outputs = candidate if side == "candidate" else baseline
    outputs["output0"][0, 299, -1] = nonfinite
    with pytest.raises(ValueError, match=f"{side}/output0 must be finite"):
        analyze(role, rgb, tensor, candidate, baseline, tmp_path / "no-model.onnx")


@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_prototype_channels_are_not_ignored(
    side: str, nonfinite: float, tmp_path: Path
) -> None:
    rgb, tensor, baseline = _case("bed")
    candidate = _copy(baseline)
    outputs = candidate if side == "candidate" else baseline
    outputs["output1"][0, 31, -1, -1] = nonfinite
    with pytest.raises(ValueError, match=f"{side}/output1 must be finite"):
        analyze("bed", rgb, tensor, candidate, baseline, tmp_path / "no-model.onnx")


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize(
    "defect",
    [
        "row_count",
        "too_many_rows",
        "column_count",
        "dtype",
        "extra_output",
        "missing_output",
        "name",
    ],
)
def test_invalid_output_contract_is_rejected_before_model_access(
    role: str, defect: str, tmp_path: Path
) -> None:
    rgb, tensor, baseline = _case(role)
    candidate = _copy(baseline)
    if defect == "row_count":
        candidate["output0"] = candidate["output0"][:, :299]
    elif defect == "too_many_rows":
        candidate["output0"] = np.zeros((1, 301, candidate["output0"].shape[2]), dtype=np.float32)
    elif defect == "column_count":
        candidate["output0"] = candidate["output0"][:, :, :-1]
    elif defect == "dtype":
        candidate["output0"] = candidate["output0"].astype(np.float64)
    elif defect == "extra_output":
        candidate["extra"] = np.zeros(1, dtype=np.float32)
    elif defect == "missing_output":
        del candidate["output0"]
    else:
        candidate["detections"] = candidate.pop("output0")
    with pytest.raises(ValueError):
        analyze(role, rgb, tensor, candidate, baseline, tmp_path / "no-model.onnx")


@pytest.mark.parametrize("shape", [(1, 31, 8, 8), (1, 32, 0, 8), (1, 32, 321, 8), (1, 32, 7, 8)])
def test_invalid_or_mismatched_prototype_geometry_is_rejected(
    shape: tuple[int, ...], tmp_path: Path
) -> None:
    rgb, tensor, baseline = _case("bed")
    candidate = _copy(baseline)
    candidate["output1"] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        analyze("bed", rgb, tensor, candidate, baseline, tmp_path / "no-model.onnx")


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("defect", ["dtype", "nonfinite", "batch", "channels", "nonsquare", "size"])
def test_invalid_input_tensor_is_rejected_before_analysis(
    role: str, defect: str, tmp_path: Path
) -> None:
    rgb, tensor, outputs = _case(role)
    if defect == "dtype":
        tensor = tensor.astype(np.float64)
    elif defect == "nonfinite":
        tensor.flat[-1] = np.nan
    else:
        size = tensor.shape[2]
        shape = {
            "batch": (2, 3, size, size),
            "channels": (1, 4, size, size),
            "nonsquare": (1, 3, size, size - 1),
            "size": (1, 3, 1281, 1281) if role == "bed" else (1, 3, 320, 320),
        }[defect]
        tensor = np.broadcast_to(np.float32(0), shape)
    with pytest.raises(ValueError, match="input_tensor"):
        analyze(role, rgb, tensor, outputs, outputs, tmp_path / "no-model.onnx")


@pytest.mark.parametrize(
    "shape", [(0, 8, 3), (8, 8, 4), (1921, 1, 3), (1920, 1920, 3), (1, 1920, 3)]
)
def test_rgb_geometry_is_bounded_before_any_decoder_allocation(
    shape: tuple[int, ...], tmp_path: Path
) -> None:
    _, tensor, outputs = _case("bed")
    rgb = np.broadcast_to(np.uint8(0), shape)
    with pytest.raises(ValueError, match="rgb geometry"):
        analyze("bed", rgb, tensor, outputs, outputs, tmp_path / "no-model.onnx")


def test_role_is_explicit_and_rgb_dtype_is_not_coerced(tmp_path: Path) -> None:
    rgb, tensor, outputs = _case("bed")
    with pytest.raises(ValueError, match="role must be"):
        analyze("pose", rgb, tensor, outputs, outputs, tmp_path / "no-model.onnx")
    with pytest.raises(ValueError, match="uint8"):
        analyze("bed", rgb.astype(np.float32), tensor, outputs, outputs, tmp_path / "no-model.onnx")
