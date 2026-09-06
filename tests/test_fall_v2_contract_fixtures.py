"""Frozen, runtime-independent contracts for fall v2 fixture artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from pathlib import Path

from worker.domains.fall.pose_bbox56 import PoseBbox56Track, pose_bbox56_tracks

_EDGE_ROOT = Path(__file__).resolve().parents[1]
_POLICY_FIXTURE = _EDGE_ROOT / "tests" / "fixtures_fall_policy_v2.json"
_POLICY_SHA256 = "9234acebd07f7494bc107d0471eff52ae08cb46b75ecfa47d9c709f4e16ea1b7"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _load(path: Path) -> tuple[bytes, dict[str, object]]:
    payload = path.read_bytes()
    return payload, json.loads(payload)


def _bundle_pose_fixture() -> tuple[bytes, dict[str, object]]:
    root = Path(
        os.environ.get(
            "FALL_MODEL_BUNDLE_DIR",
            _EDGE_ROOT / "models/fall/pose-bbox56-gru",
        )
    )
    manifest = json.loads((root / "bundle-manifest.json").read_bytes())
    candidates = [
        item
        for item in manifest["files"]
        if Path(item["relative_path"]).parent == Path("conformance")
    ]
    assert len(candidates) == 1
    member = candidates[0]
    payload = (root / member["relative_path"]).read_bytes()
    assert len(payload) == member["size"]
    assert hashlib.sha256(payload).hexdigest() == member["sha256"]
    return payload, json.loads(payload)


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def test_pose_fixture_is_canonical() -> None:
    payload, fixture = _bundle_pose_fixture()

    assert payload == _canonical_bytes(fixture)


def test_pose_fixture_raw_cases_match_executable_transform() -> None:
    _, fixture = _bundle_pose_fixture()
    tolerance = float(fixture["comparison"]["absolute_tolerance"])
    for case in fixture["raw_cases"]:
        frame = case["frame"]
        tracks = tuple(
            PoseBbox56Track(
                track_id=person["track_id"],
                keypoints=tuple(
                    tuple(float(value) for value in point) for point in person["keypoints"]
                ),
                bbox=(
                    None
                    if person["bbox"] is None
                    else tuple(float(value) for value in person["bbox"])
                ),
            )
            for person in case["persons"]
        )
        actual = pose_bbox56_tracks(tracks, frame["width"], frame["height"])
        expected_rows = case.get("expected_rows")
        if expected_rows is not None:
            assert [track_id for track_id, _ in actual] == [
                expected["track_id"] for expected in expected_rows
            ]
            for (_, row), expected in zip(actual, expected_rows, strict=True):
                assert len(row) == 56
                assert all(
                    abs(value - expected_value) <= tolerance
                    for value, expected_value in zip(row, expected["row"], strict=True)
                )
        elif not tracks:
            assert actual == ()
        else:
            expected_windows = case["expected_windows"]
            for (_, row), expected in zip(actual, expected_windows, strict=True):
                assert all(
                    abs(value - expected_value) <= tolerance
                    for value, expected_value in zip(row, expected["rows"][-1], strict=True)
                )


def test_pose_vector_contract_and_negative_cases() -> None:
    _, fixture = _bundle_pose_fixture()

    assert fixture["preprocessing_identity"] == "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"
    assert fixture["temporal"] == {"fps": 15, "stride_frames": 5, "window_frames": 30}
    assert fixture["class_order"] == ["background", "fall_transition", "fallen"]
    assert fixture["vector"]["length"] == 56
    assert fixture["vector"]["tail_indices"] == {
        "valid": 55,
        "x1": 51,
        "x2": 53,
        "y1": 52,
        "y2": 54,
    }
    assert fixture["confidence"]["rule"].startswith("keep when confidence >= gate")

    cases = {case["case_id"]: case for case in fixture["raw_cases"]}
    assert {
        "missing-bbox",
        "degenerate-bbox",
        "low-confidence-keypoint",
        "clipped-finite-coordinates",
        "nan-rejection",
        "infinity-rejection",
        "multi-person-ordering",
        "missing-live-coast",
        "reconnect-after-eviction",
    } <= cases.keys()
    for case in cases.values():
        for expected in case.get("expected_rows", ()):
            row = expected["row"]
            assert len(row) == 56
            assert all(math.isfinite(value) for value in row)
            assert all(
                abs(value - _float32(value)) <= fixture["comparison"]["absolute_tolerance"]
                for value in row
            )
        for window in case.get("expected_windows", ()):
            for row in window["rows"]:
                assert len(row) == 56
                assert all(math.isfinite(value) for value in row)

    assert (
        cases["threshold-equality"]["expected_rows"][0]["row"][2] == fixture["confidence"]["gate"]
    )
    assert cases["low-confidence-keypoint"]["expected_rows"][0]["row"][12:15] == [
        0.0,
        0.0,
        0.0,
    ]
    assert cases["clipped-finite-coordinates"]["expected_rows"][0]["row"][51:] == [
        0.0,
        0.0,
        0.995,
        0.99,
        1.0,
    ]
    for case_id in (
        "missing-bbox",
        "degenerate-bbox",
        "nan-rejection",
        "infinity-rejection",
    ):
        assert cases[case_id]["expected_rows"][0]["row"] == [0.0] * 56
    reconnect = cases["reconnect-after-eviction"]
    reconnect_window = reconnect["expected_windows"][0]
    reconnect_rows = reconnect_window["rows"]
    assert "zero-fills prior temporal positions" in reconnect["window_rule"]
    assert all(row == [0.0] * 56 for row in reconnect_rows[:-1])
    assert len(reconnect_rows[-1]) == 56


def test_policy_fixture_freezes_fall_v2_surface_and_state_transitions() -> None:
    payload, fixture = _load(_POLICY_FIXTURE)

    assert payload == _canonical_bytes(fixture)
    assert hashlib.sha256(payload).hexdigest() == _POLICY_SHA256
    assert fixture["schema_id"] == "fall.policy.v2"
    assert fixture["class_order"] == ["background", "fall_transition", "fallen"]
    assert fixture["canonical_digest_inputs"] == [
        "schema_id",
        "class_order",
        "class_thresholds",
        "transition",
        "fallen_confirmation",
        "hysteresis_recovery",
        "stale",
        "dedup",
        "startup",
        "track_lifecycle",
        "camera_aggregation",
        "alert_semantics",
    ]
    assert set(fixture) == set(fixture["canonical_digest_inputs"]) | {"canonical_digest_inputs"}
    assert fixture["class_thresholds"] == {
        "background_max": 0.5,
        "fall_transition_min": 0.7,
        "fallen_min": 0.8,
    }
    assert fixture["transition"] == {
        "m": 3,
        "n": 5,
        "probability_field": "fall_transition",
        "threshold": 0.7,
    }
    assert fixture["fallen_confirmation"] == {
        "minimum_consecutive_frames": 3,
        "probability_field": "fallen",
        "threshold": 0.8,
    }
    assert fixture["hysteresis_recovery"]["minimum_consecutive_frames"] == 5
    assert fixture["stale"]["track_ttl_frames"] == 45
    assert fixture["dedup"]["cooldown_frames"] == 90
    assert fixture["startup"]["already_fallen"] == (
        "initialize internal fallen state without emitting an alert"
    )
    assert {"reconnect", "eviction", "reuse", "generation"} <= fixture["track_lifecycle"].keys()
    assert fixture["camera_aggregation"] == {
        "mode": "per-camera OR across live tracks",
        "require_live_track": True,
        "window": "same decision tick",
    }
    assert fixture["alert_semantics"] == {
        "existing_alert": "fires on confirmed fall_transition",
        "fallen": ("internal state only; it does not independently fire the existing alert"),
    }


def test_fixtures_exclude_grayscale_and_bed_vocabulary() -> None:
    pose_payload, _ = _bundle_pose_fixture()
    text = pose_payload.decode() + _POLICY_FIXTURE.read_text()

    assert "grayscale" not in text.lower()
    assert "bed" not in text.lower()


def test_resampled_gap_rows_are_zero_rows_at_the_exact_cadence() -> None:
    """P1a-AC4/AC5: a dropped 15 fps bucket becomes one ``valid=0`` zero row.

    The resampler is the single owner of fall input cadence, so a gap must
    produce the same 56-wide zero row the classifier would see for a missing
    observation -- never a repeated or interpolated pose.
    """
    from worker.domains.fall.pose_bbox56 import pose_bbox56_row
    from worker.pipeline.perception.pts_resample import CADENCE_NS, PtsResampler

    zero_row = pose_bbox56_row((), None, 640, 360)
    assert zero_row == (0.0,) * 56

    resampler: PtsResampler[str] = PtsResampler()
    assert [row.valid for row in resampler.push(0, "a")] == [1]
    assert [row.valid for row in resampler.push(CADENCE_NS, "b")] == [1]
    # Two buckets skipped: exactly two invalid rows, then the observed row.
    produced = resampler.push(4 * CADENCE_NS, "c")
    assert [(row.pts_ns, row.valid, row.value) for row in produced] == [
        (2 * CADENCE_NS, 0, None),
        (3 * CADENCE_NS, 0, None),
        (4 * CADENCE_NS, 1, "c"),
    ]
    # A second row inside an already-emitted bucket is dropped, not duplicated.
    assert resampler.push(4 * CADENCE_NS + 1, "d") == ()
