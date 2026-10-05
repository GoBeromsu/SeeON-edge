//! Port of tests/test_rust_bed_contour_parity.py onto fixtures recorded from
//! `seg_postprocess.largest_external_contour`. Comparison is exact ordered
//! integer vertices; every regenerated mask is gated by its recorded sha256.

#[path = "fixtures/support.rs"]
mod support;

use seeon_worker::bed_contour::{BedContourError, largest_external_contour};
use serde_json::Value;
use std::ops::Range;
use support::{Pcg64, array, int, points_of, points_sha256, text, uint, usize_of};

const FIXTURE: &str = "bed_contour/bed_contour.json";
const ORACLE_SOURCE: &str = "worker/adapters/model/seg_postprocess.py";
const ORACLE_SHA256: &str = "1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a";
const MAX_OUTPUT: i64 = 1024 * 1024;

fn test_value(name: &str) -> Value {
    let mut fixture = support::load(FIXTURE, ORACLE_SOURCE, ORACLE_SHA256);
    assert_eq!(
        text(&fixture["oracle"]),
        "worker.adapters.model.seg_postprocess.largest_external_contour"
    );
    fixture["tests"][name].take()
}

fn cases(name: &str) -> Vec<Value> {
    array(&test_value(name)).to_vec()
}

/// A boolean ndarray as packed row-major 0/1 bytes.
#[derive(Clone)]
struct Mask {
    height: usize,
    width: usize,
    cells: Vec<u8>,
}

impl Mask {
    fn zeros(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            cells: vec![0; height * width],
        }
    }

    fn ones(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            cells: vec![1; height * width],
        }
    }

    fn set(&mut self, y: usize, x: usize) {
        self.cells[y * self.width + x] = 1;
    }

    /// `mask[rows, columns] = value`.
    fn fill(&mut self, rows: Range<usize>, columns: Range<usize>, value: u8) {
        for y in rows {
            for x in columns.clone() {
                self.cells[y * self.width + x] = value;
            }
        }
    }

    fn with(mut self, points: &[(usize, usize)]) -> Self {
        for &(y, x) in points {
            self.set(y, x);
        }
        self
    }

    fn filled(mut self, rows: Range<usize>, columns: Range<usize>, value: u8) -> Self {
        self.fill(rows, columns, value);
        self
    }

    fn contour(&self) -> Result<Vec<[i64; 2]>, BedContourError> {
        largest_external_contour(&self.cells, self.width as i64, self.height as i64)
    }
}

fn border(height: usize, width: usize) -> Mask {
    Mask::ones(height, width).filled(1..height - 1, 1..width - 1, 0)
}

fn ring(height: usize, width: usize, margin: usize) -> Mask {
    Mask::ones(height, width).filled(margin..height - margin, margin..width - margin, 0)
}

fn diagonal(size: usize, anti: bool) -> Mask {
    let mut mask = Mask::zeros(size, size);
    for index in 0..size {
        mask.set(index, if anti { size - 1 - index } else { index });
    }
    mask
}

fn both_diagonals(size: usize) -> Mask {
    let mut mask = diagonal(size, false);
    for (cell, other) in mask.cells.iter_mut().zip(diagonal(size, true).cells) {
        *cell |= other;
    }
    mask
}

fn staircase(steps: usize) -> Mask {
    let mut mask = Mask::zeros(steps, steps + 1);
    for index in 0..steps {
        mask.set(index, index);
        mask.set(index, index + 1);
    }
    mask
}

fn separated(second_width: usize) -> [Mask; 3] {
    let canvas = Mask::zeros(6, 6 + second_width);
    let first = canvas.clone().filled(1..4, 1..4, 1);
    let second = canvas.filled(1..4, 6..6 + second_width, 1);
    let full = first.clone().filled(1..4, 6..6 + second_width, 1);
    [full, first, second]
}

/// The recorded oracle vertices for exactly this mask.
fn expected(case: &Value, mask: &Mask) -> Vec<[i64; 2]> {
    assert_eq!(
        (usize_of(&case["height"]), usize_of(&case["width"])),
        (mask.height, mask.width),
        "recorded shape"
    );
    assert_eq!(
        support::sha256_u8(&mask.cells),
        text(&case["mask_sha256"]),
        "recorded {}x{} mask",
        mask.height,
        mask.width
    );
    points_of(&case["expected"])
}

fn check(case: &Value, mask: &Mask) {
    let expected = expected(case, mask);
    assert_eq!(
        mask.contour(),
        Ok(expected),
        "{}x{} mask",
        mask.height,
        mask.width
    );
}

fn check_all(name: &str, masks: &[Mask]) {
    let cases = cases(name);
    assert_eq!(cases.len(), masks.len(), "{name} cases");
    for (case, mask) in cases.iter().zip(masks) {
        check(case, mask);
    }
}

#[test]
fn empty_and_one_pixel() {
    let empty = Mask::zeros(4, 5);
    check_all(
        "empty_and_one_pixel",
        &[
            empty.clone(),
            Mask::zeros(0, 6),
            Mask::zeros(6, 0),
            Mask::zeros(0, 0),
            Mask::zeros(4, 5).with(&[(2, 3)]),
            Mask::ones(1, 1),
            Mask::zeros(1, 2).with(&[(0, 0)]),
            empty,
        ],
    );
}

#[test]
fn strips_and_rectangles() {
    check_all(
        "strips_and_rectangles",
        &[
            Mask::zeros(5, 12).filled(2..3, 0..12, 1),
            Mask::zeros(12, 4).filled(0..12, 1..2, 1),
            Mask::ones(4, 7),
            Mask::zeros(8, 9).filled(2..6, 1..8, 1),
        ],
    );
}

#[test]
fn all_border_rings() {
    let masks = [(3, 3), (3, 9), (8, 3), (6, 7)].map(|(height, width)| border(height, width));
    check_all("all_border_rings", &masks);
}

#[test]
fn both_diagonals_and_single_diagonals() {
    let mut masks: Vec<Mask> = [1, 2, 7, 8].into_iter().map(both_diagonals).collect();
    masks.extend(
        [(6, false), (6, true), (5, false), (5, true)].map(|(size, anti)| diagonal(size, anti)),
    );
    check_all("both_diagonals", &masks);
}

#[test]
fn holes() {
    check_all(
        "holes",
        &[
            ring(5, 5, 1),
            ring(4, 9, 1),
            ring(9, 4, 1),
            ring(7, 8, 2),
            Mask::zeros(10, 12)
                .filled(1..9, 1..11, 1)
                .filled(3..7, 4..9, 0),
            border(5, 8),
            ring(6, 6, 1),
            ring(8, 5, 2),
        ],
    );
}

/// The oracle picks the winner; Rust must return the full mask's oracle result.
fn check_separated(name: &str, second_width: usize, winner: usize) {
    let masks = separated(second_width);
    let cases = cases(name);
    assert_eq!(cases.len(), 3);
    let recorded: Vec<Vec<[i64; 2]>> = cases
        .iter()
        .zip(&masks)
        .map(|(case, mask)| expected(case, mask))
        .collect();
    assert_eq!(recorded[0], recorded[winner], "oracle winner");
    assert_ne!(recorded[0], recorded[3 - winner], "oracle loser");
    assert_eq!(masks[0].contour(), Ok(recorded[0].clone()));
}

#[test]
fn equal_area_disconnected_components_keep_the_first_maximum() {
    check_separated(
        "equal_area_disconnected_components_keep_the_first_maximum",
        3,
        1,
    );
}

#[test]
fn larger_disconnected_component_replaces_the_earlier_loop() {
    check_separated(
        "larger_disconnected_component_replaces_the_earlier_loop",
        4,
        2,
    );
}

#[test]
fn corner_touch_branches() {
    check_all(
        "corner_touch_branches",
        &[
            Mask::zeros(7, 7)
                .filled(0..3, 0..3, 1)
                .filled(3..6, 3..6, 1),
            Mask::zeros(5, 5).with(&[(0, 0), (1, 1), (1, 2), (2, 1)]),
            Mask::zeros(2, 2).with(&[(0, 0), (1, 1)]),
            both_diagonals(4),
        ],
    );
}

#[test]
fn fixed_landscape_and_portrait_masks() {
    let shapes = [
        (1, 8),
        (8, 1),
        (2, 17),
        (17, 2),
        (5, 9),
        (9, 5),
        (3, 14),
        (14, 3),
    ];
    let masks: Vec<Mask> = shapes
        .into_iter()
        .flat_map(|(height, width)| {
            let mut sparse = Mask::zeros(height, width);
            for y in (0..height).step_by(2) {
                for x in (0..width).step_by(2) {
                    sparse.set(y, x);
                }
            }
            [sparse, Mask::ones(height, width)]
        })
        .collect();
    check_all("fixed_landscape_and_portrait_masks", &masks);
}

#[test]
fn seeded_landscape_and_portrait_masks() {
    let parameters: [(u64, usize, usize, f64); 9] = [
        (731, 9, 14, 0.45),
        (8128, 15, 6, 0.35),
        (65537, 7, 7, 0.5),
        (19, 2, 21, 0.55),
        (23, 21, 2, 0.55),
        (29, 11, 4, 0.4),
        (31, 4, 11, 0.4),
        (37, 3, 16, 0.5),
        (41, 16, 3, 0.5),
    ];
    let cases = cases("seeded_landscape_and_portrait_masks");
    assert_eq!(cases.len(), parameters.len());
    for (case, (seed, height, width, threshold)) in cases.iter().zip(parameters) {
        assert_eq!(uint(&case["seed"]), seed);
        assert_eq!(support::f64_bits(&case["threshold_bits"]), threshold);
        // default_rng(seed).random((height, width)) < threshold, row-major.
        let mut generator = Pcg64::recorded(case);
        let mut mask = Mask::zeros(height, width);
        for cell in &mut mask.cells {
            *cell = u8::from(generator.next_f64() < threshold);
        }
        check(case, &mask);
    }
}

#[test]
fn contours_above_48_points_are_unchanged() {
    let cases = cases("contours_above_48_points_are_unchanged");
    for case in &cases {
        assert!(array(&case["expected"]).len() > 48, "oracle contour length");
    }
    let masks = [Mask::ones(2, 30), Mask::ones(30, 2), staircase(24)];
    assert_eq!(cases.len(), masks.len());
    for (case, mask) in cases.iter().zip(&masks) {
        check(case, mask);
    }
}

#[test]
fn repeat_calls_are_stateless() {
    // one, block, empty, hole
    let masks = [
        Mask::zeros(3, 4).with(&[(1, 2)]),
        Mask::ones(3, 5),
        Mask::zeros(2, 2),
        ring(5, 6, 1),
    ];
    let cases = cases("repeat_calls_are_stateless");
    assert_eq!(cases.len(), masks.len());
    let expected: Vec<Vec<[i64; 2]>> = cases
        .iter()
        .zip(&masks)
        .map(|(case, mask)| expected(case, mask))
        .collect();
    // The Python probe calls [one, block, empty, one] then [hole, one, block, empty].
    let order = [0, 1, 2, 0, 3, 0, 1, 2];
    for (call, &index) in order.iter().enumerate() {
        assert_eq!(
            masks[index].contour(),
            Ok(expected[index].clone()),
            "call {call}"
        );
    }
}

#[test]
fn strided_mask_matches_logical_row_major_values() {
    let base = Mask::zeros(8, 9)
        .filled(1..7, 1..8, 1)
        .filled(3..5, 3..6, 0);
    // base[::2, ::2] as its logical row-major values.
    let mut view = Mask::zeros(4, 5);
    for y in 0..4 {
        for x in 0..5 {
            view.cells[y * 5 + x] = base.cells[2 * y * base.width + 2 * x];
        }
    }
    check_all("strided_mask_matches_logical_row_major_values", &[view]);
}

/// Smallest pixel count whose `4 * p * p` shoelace bound exceeds 2**53.
fn smallest_unproven_pixels() -> i64 {
    let bound = 1_i128 << 53;
    let mut proven = 0_i128;
    let mut step = 1_i128 << 26;
    while step != 0 {
        let candidate = proven + step;
        if 4 * candidate * candidate <= bound {
            proven = candidate;
        }
        step >>= 1;
    }
    let unproven = proven + 1;
    assert!(4 * proven * proven <= bound && 4 * unproven * unproven > bound);
    i64::try_from(unproven).expect("pixel count fits i64")
}

#[test]
fn empty_mask_outside_shoelace_domain() {
    // No shoelace is evaluated; this is not area equivalence outside the domain.
    let case = test_value("empty_mask_outside_shoelace_domain");
    let pixels = smallest_unproven_pixels();
    assert_eq!((int(&case["height"]), int(&case["width"])), (1, pixels));
    let expected = points_of(&case["expected"]);
    assert!(expected.is_empty(), "oracle traces nothing");
    let mask = vec![0_u8; usize::try_from(pixels).expect("pixels fit usize")];
    assert_eq!(largest_external_contour(&mask, pixels, 1), Ok(expected));
}

#[test]
fn rust_only_nonempty_outside_shoelace_domain_is_refused() {
    // A finite ndarray would still trace this; that result is not the oracle.
    let pixels = smallest_unproven_pixels();
    let mut mask = vec![0_u8; usize::try_from(pixels).expect("pixels fit usize")];
    mask[0] = 1;
    assert_eq!(
        largest_external_contour(&mask, pixels, 1),
        Err(BedContourError::AreaDomain)
    );
}

#[test]
fn rust_only_semantic_bounds_are_not_python_ndarray_cases() {
    use BedContourError::{Dimensions, DimensionsOverflow, MaskLength, MaskValue};
    let table: [(i64, i64, &[u8], BedContourError); 10] = [
        (-1, 1, b"", Dimensions),
        (1, -1, b"\x00", Dimensions),
        (i64::MIN, 0, b"", Dimensions),
        (i64::MAX, i64::MAX, b"", DimensionsOverflow),
        (i64::MAX, 3, b"", DimensionsOverflow),
        (2, 2, b"\x01\x00\x01", MaskLength),
        (1, 1, b"", MaskLength),
        (0, 0, b"\x00", MaskLength),
        (1, 1, b"\x02", MaskValue),
        (2, 1, b"\x01\xff", MaskValue),
    ];
    for (width, height, mask, error) in table {
        assert_eq!(
            largest_external_contour(mask, width, height),
            Err(error),
            "width {width} height {height} mask {mask:?}"
        );
    }
}

#[test]
fn semantic_error_keeps_earlier_exact_vertices() {
    let strip = Mask::ones(2, 3);
    let cases = cases("semantic_error_keeps_earlier_exact_vertices");
    assert_eq!(cases.len(), 1);
    let expected = expected(&cases[0], &strip);
    assert_eq!(strip.contour(), Ok(expected.clone()), "before the error");
    assert_eq!(
        largest_external_contour(b"\x01\x00\x01", 2, 2),
        Err(BedContourError::MaskLength)
    );
    assert_eq!(strip.contour(), Ok(expected), "after the error");
}

#[test]
fn long_contour_has_no_vertex_quota() {
    // The core has no vertex quota: a contour past the probe's print buffer is
    // returned whole, never shortened.
    let case = test_value("long_contour_has_no_vertex_quota");
    let width = MAX_OUTPUT / 16 + 8;
    assert_eq!((int(&case["height"]), int(&case["width"])), (1, width));
    let mask = vec![1_u8; usize::try_from(width).expect("width fits usize")];
    let actual = largest_external_contour(&mask, width, 1).expect("in-domain strip traces");
    let count = usize_of(&case["expected_count"]);
    assert!(12 + 1 + 4 + 16 * count > MAX_OUTPUT as usize);
    assert_eq!(actual.len(), count, "vertex count");
    for sample in array(&case["expected_samples"]) {
        let index = usize_of(&sample[0]);
        let point = support::i64s(&sample[1]);
        assert_eq!(actual[index][..], point[..], "vertex {index}");
    }
    assert_eq!(
        points_sha256(&actual),
        text(&case["expected_sha256"]),
        "vertex sha256"
    );
}
