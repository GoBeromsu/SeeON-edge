//! Port of tests/test_rust_bed_polygon_parity.py onto fixtures recorded from
//! `seg_postprocess.simplify_polygon`. Comparison is exact ordered vertices.

#[path = "fixtures/support.rs"]
mod support;

use seeon_worker::bed_polygon::{BedPolygonError, simplify_polygon};
use serde_json::Value;
use support::{array, int, points_of, points_sha256, text, usize_of};

const FIXTURE: &str = "bed_polygon/bed_polygon.json";
const ORACLE_SOURCE: &str = "worker/adapters/model/seg_postprocess.py";
const ORACLE_SHA256: &str = "1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a";
const MAX_POINTS: i64 = 4096;
const CAPACITY_ERROR: &str = "max_points must be positive";

type Points = Vec<[i64; 2]>;

fn cases(name: &str) -> Vec<Value> {
    let mut fixture = support::load(FIXTURE, ORACLE_SOURCE, ORACLE_SHA256);
    assert_eq!(
        text(&fixture["oracle"]),
        "worker.adapters.model.seg_postprocess.simplify_polygon"
    );
    array(&fixture["tests"][name].take()).to_vec()
}

/// The recorded input after its own sha256 gate.
fn recorded_input(case: &Value) -> (Points, i64) {
    let points = points_of(&case["points"]);
    assert_eq!(
        points_sha256(&points),
        text(&case["input_sha256"]),
        "recorded points"
    );
    (points, int(&case["max_points"]))
}

/// Gates the recorded call against the Python test's own input, then compares.
fn check(case: &Value, points: &[[i64; 2]], capacity: i64) {
    let (recorded, recorded_capacity) = recorded_input(case);
    assert_eq!(recorded, points, "input points");
    assert_eq!(recorded_capacity, capacity, "input capacity");
    compare(case, points, capacity);
}

fn compare(case: &Value, points: &[[i64; 2]], capacity: i64) {
    let actual = simplify_polygon(points, capacity);
    match case.get("expected") {
        Some(expected) => assert_eq!(
            actual,
            Ok(points_of(expected)),
            "{} points at capacity {capacity}",
            points.len()
        ),
        None => {
            assert_eq!(text(&case["expected_error"]), CAPACITY_ERROR);
            assert_eq!(
                actual,
                Err(BedPolygonError::Capacity),
                "capacity {capacity}"
            );
        }
    }
}

#[test]
fn short_duplicate_closed_and_tied_polygons() {
    let sets: [&[[i64; 2]]; 7] = [
        &[],
        &[[4, 7]],
        &[[4, 7], [9, 12]],
        &[[3, 5]; 9],
        &[[0, 0], [10, 0], [10, 10], [0, 10]],
        &[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        &[[0, 0], [0, 0], [5, 7], [10, 0], [10, 0], [5, -7]],
    ];
    let calls: Vec<(&[[i64; 2]], i64)> = sets
        .iter()
        .flat_map(|&points| {
            let length = points.len() as i64;
            [1, 2, 3, 48, length.max(1), length + 1].map(|capacity| (points, capacity))
        })
        .collect();
    let cases = cases("short_duplicate_closed_and_tied_polygons");
    assert_eq!(cases.len(), calls.len());
    for (case, (points, capacity)) in cases.iter().zip(calls) {
        check(case, points, capacity);
    }
}

#[test]
fn capacity_one_fallback_keeps_original_first_not_furthest_anchor() {
    let points = [[1, 1], [0, 0], [10, 0], [10, 10], [0, 10]];
    let cases = cases("capacity_one_fallback_keeps_original_first_not_furthest_anchor");
    assert_eq!(cases.len(), 1);
    assert_eq!(points_of(&cases[0]["expected"]), [points[0]]);
    check(&cases[0], &points, 1);
}

#[test]
fn identical_slow_path_can_return_empty_without_a_minimum_polygon() {
    let cases = cases("identical_slow_path_can_return_empty_without_a_minimum_polygon");
    assert_eq!(cases.len(), 1);
    assert!(points_of(&cases[0]["expected"]).is_empty());
    check(&cases[0], &[[7, 4]; 8], 1);
}

#[test]
fn imported_contours_preserve_simplification_order() {
    // Inputs are the oracle's largest_external_contour output, gated by sha256.
    let kinds = [
        "rectangle",
        "concave",
        "staircase",
        "shared-corners",
        "hole",
    ];
    let cases = cases("imported_contours_preserve_simplification_order");
    assert_eq!(cases.len(), kinds.len() * 4);
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(text(&case["contour"]), kinds[index / 4]);
        let (points, capacity) = recorded_input(case);
        assert!(points.len() > 48, "{} contour length", kinds[index / 4]);
        assert_eq!(capacity, [2, 3, 12, 48][index % 4]);
        compare(case, &points, capacity);
    }
}

#[test]
fn deterministic_integer_polygons_and_reverse_order() {
    // Inputs are default_rng(seed).integers(-2000, 2001, (129, 2)), gated by sha256.
    let cases = cases("deterministic_integer_polygons_and_reverse_order");
    assert_eq!(cases.len(), 12);
    for (seed, group) in [607, 608, 1_048_576].into_iter().zip(cases.chunks(4)) {
        let (points, _) = recorded_input(&group[0]);
        assert_eq!(points.len(), 129);
        assert!(
            points
                .iter()
                .flatten()
                .all(|coordinate| (-2000..=2000).contains(coordinate))
        );
        let reversed: Points = points.iter().rev().copied().collect();
        let calls: [(&[[i64; 2]], i64); 4] =
            [(&points, 2), (&points, 48), (&reversed, 48), (&points, 128)];
        for (case, (ordered, capacity)) in group.iter().zip(calls) {
            assert_eq!(support::uint(&case["seed"]), seed);
            check(case, ordered, capacity);
        }
    }
}

#[test]
fn inclusive_slow_coordinate_domain_and_unrestricted_fast_copy() {
    let limit = 1_i64 << 53;
    let admitted = [[-limit, 0], [0, limit], [limit, 0], [0, -limit], [1, 3]];
    let extremes = [[i64::MIN, i64::MAX], [i64::MAX, i64::MIN]];
    let cases = cases("inclusive_slow_coordinate_domain_and_unrestricted_fast_copy");
    let calls: [(&[[i64; 2]], i64); 4] = [
        (&admitted, 3),
        (&admitted, 1),
        (&extremes, 2),
        (&extremes, i64::MAX),
    ];
    assert_eq!(cases.len(), calls.len());
    for (case, (points, capacity)) in cases.iter().zip(calls) {
        check(case, points, capacity);
    }
    // Outside the exact f64 domain Python has no defined answer; the Python
    // test pins the Rust refusal itself, so these are literal admission checks.
    assert_eq!(
        simplify_polygon(&extremes, 1),
        Err(BedPolygonError::CoordinateDomain)
    );
    for outside in [-limit - 1, limit + 1] {
        assert_eq!(
            simplify_polygon(&[[outside, 0], [0, 1]], 1),
            Err(BedPolygonError::CoordinateDomain),
            "{outside}"
        );
    }
}

#[test]
fn capacity_validation_precedes_even_an_empty_fast_path() {
    let cases = cases("capacity_validation_precedes_even_an_empty_fast_path");
    let calls: [(&[[i64; 2]], i64); 4] =
        [(&[], 0), (&[], -1), (&[[0, 0]], i64::MIN), (&[[0, 0]], 1)];
    assert_eq!(cases.len(), calls.len());
    for (index, (case, (points, capacity))) in cases.iter().zip(calls).enumerate() {
        assert_eq!(
            case.get("expected_error").is_some(),
            index < 3,
            "case {index}"
        );
        check(case, points, capacity);
    }
}

#[test]
fn repeated_calls_and_full_point_transport_capacity() {
    let full: Points = (0..MAX_POINTS)
        .map(|i| [i - 2048, (i * 17) % 331])
        .collect();
    let short = [[4, 4], [1, 0], [9, 0], [9, 8], [1, 8]];
    let cases = cases("repeated_calls_and_full_point_transport_capacity");
    assert_eq!(cases.len(), 3);
    assert_eq!(usize_of(&cases[0]["max_points"]), full.len());
    check(&cases[0], &full, MAX_POINTS);
    check(&cases[1], &short, 1);
    check(&cases[2], &short, 3);
    // The Python differential sends each call more than once in one request;
    // a repeated call must reproduce the same oracle result.
    for _ in 0..3 {
        compare(&cases[0], &full, MAX_POINTS);
    }
    for case in [&cases[1], &cases[2]] {
        compare(case, &short, int(&case["max_points"]));
    }
}
