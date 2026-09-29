//! Port of tests/test_rust_bed_sigmoid_parity.py onto fixtures recorded from
//! `seg_postprocess._sigmoid` under NumPy 2.4.6 with float32 exp X86_V3.
//! Comparison is exact float32 bits, as in the Python differential.

#[path = "fixtures/support.rs"]
mod support;

use seeon_worker::bed_sigmoid::bed_mask_sigmoid;
use serde_json::Value;
use support::{Pcg64, array, assert_bits, assert_bulk_u32, text, u32s, usize_of};

const FIXTURE: &str = "bed_sigmoid/bed_sigmoid.json";
const ORACLE_SOURCE: &str = "worker/adapters/model/seg_postprocess.py";
const ORACLE_SHA256: &str = "1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a";
const MAX_COUNT: usize = 1_048_576;

fn case(name: &str) -> Value {
    let mut fixture = support::load(FIXTURE, ORACLE_SOURCE, ORACLE_SHA256);
    assert_eq!(
        text(&fixture["oracle"]),
        "worker.adapters.model.seg_postprocess._sigmoid"
    );
    fixture["tests"][name].take()
}

fn sigmoid_bits(inputs: &[u32]) -> Vec<u32> {
    inputs
        .iter()
        .map(|&bits| bed_mask_sigmoid(f32::from_bits(bits)).to_bits())
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn signed_zero_subnormals_extrema_and_nonfinite_intermediates() {
    let case = case("signed_zero_subnormals_extrema_and_nonfinite_intermediates");
    let inputs: [u32; 22] = [
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
        0x80800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
        0x7FC00001, 0xFFC12345, 0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001, 0xFF800001, 0x7FBFFFFF,
        0xFFBFFFFF,
    ];
    assert_eq!(u32s(&case["input_bits"]), inputs, "recorded inputs");
    assert_bits(
        "special",
        &inputs,
        &sigmoid_bits(&inputs),
        &u32s(&case["expected_bits"]),
    );
}

/// `np.nextafter` walks toward -inf, then restarts at the center toward +inf.
fn neighbours(centers: &[f32], steps: usize) -> Vec<f32> {
    let mut points = Vec::with_capacity(centers.len() * (2 * steps + 1));
    for &center in centers {
        points.push(center);
        for step in [f32::next_down, f32::next_up] {
            let mut value = center;
            for _ in 0..steps {
                value = step(value);
                points.push(value);
            }
        }
    }
    points
}

#[test]
fn zero_rounding_and_saturation_neighbours() {
    let case = case("zero_rounding_and_saturation_neighbours");
    let centers = [
        0.0_f32,
        -0.0,
        2.980_232_2e-8,
        -2.980_232_2e-8,
        88.722_84,
        -88.722_84,
        103.972_084,
        -103.972_084,
    ];
    assert_eq!(u32s(&case["center_bits"]), bits(&centers), "centers");
    assert_eq!(usize_of(&case["steps"]), 256);
    let inputs = bits(&neighbours(&centers, 256));
    assert_eq!(
        support::sha256_u32(&inputs),
        text(&case["input_sha256_le_u32"]),
        "regenerated neighbours"
    );
    assert_bits(
        "neighbours",
        &inputs,
        &sigmoid_bits(&inputs),
        &u32s(&case["expected_bits"]),
    );
}

#[test]
fn branch_populations_and_vector_tails() {
    let case = case("branch_populations_and_vector_tails");
    let base = [
        -100.0_f32, -5.0, -1.0, -0.000_001, -0.0, 0.0, 0.000_001, 1.0, 5.0, 100.0,
    ];
    assert_eq!(u32s(&case["base_bits"]), bits(&base), "base values");
    let lengths: Vec<usize> = array(&case["cases"])
        .iter()
        .map(|entry| usize_of(&entry["length"]))
        .collect();
    assert_eq!(
        lengths,
        [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 128]
    );
    for entry in array(&case["cases"]) {
        let length = usize_of(&entry["length"]);
        // np.resize repeats the flattened base cyclically.
        let resized: Vec<u32> = bits(&base).into_iter().cycle().take(length).collect();
        assert_bits(
            &format!("resized {length}"),
            &resized,
            &sigmoid_bits(&resized),
            &u32s(&entry["resized_expected_bits"]),
        );
        // np.sort leaves the order of equal signed zeros to NumPy; the
        // recorded order is gated as a sorted permutation of the resized input.
        let sorted = u32s(&entry["sorted_input_bits"]);
        let mut recorded = sorted.clone();
        let mut expected_multiset = resized.clone();
        recorded.sort_unstable();
        expected_multiset.sort_unstable();
        assert_eq!(recorded, expected_multiset, "sorted {length} permutation");
        assert!(
            sorted
                .windows(2)
                .all(|pair| f32::from_bits(pair[0]) <= f32::from_bits(pair[1])),
            "sorted {length} order"
        );
        assert_bits(
            &format!("sorted {length}"),
            &sorted,
            &sigmoid_bits(&sorted),
            &u32s(&entry["sorted_expected_bits"]),
        );
    }
}

/// `np.linspace(-104, 104, 200001, dtype=float32)`: float64 `i * step + start`
/// with the endpoint replaced by `stop`, then one cast to float32.
fn linspace_f32(start: f64, stop: f64, count: usize) -> Vec<f32> {
    let step = (stop - start) / (count - 1) as f64;
    let mut values: Vec<f32> = (0..count)
        .map(|index| (index as f64 * step + start) as f32)
        .collect();
    values[count - 1] = stop as f32;
    values
}

#[test]
fn dense_finite_range() {
    let case = case("dense_finite_range");
    assert_eq!(
        (
            support::int(&case["start"]),
            support::int(&case["stop"]),
            usize_of(&case["num"])
        ),
        (-104, 104, 200_001)
    );
    let inputs = bits(&linspace_f32(-104.0, 104.0, 200_001));
    assert_eq!(
        support::sha256_u32(&inputs),
        text(&case["input_sha256_le_u32"]),
        "regenerated linspace"
    );
    assert_bulk_u32("dense", &inputs, &sigmoid_bits(&inputs), &case["expected"]);
}

#[test]
fn seeded_finite_bit_patterns() {
    let case = case("seeded_finite_bit_patterns");
    assert_eq!(support::uint(&case["seed"]), 607_591);
    assert_eq!(usize_of(&case["draws"]), MAX_COUNT);
    let mut generator = Pcg64::recorded(&case);
    // integers(0, 2**32, dtype=uint32) spans the full range: raw next_uint32.
    let draws: Vec<u32> = (0..MAX_COUNT).map(|_| generator.next_u32()).collect();
    assert_eq!(draws[..4], u32s(&case["first_draws"])[..], "first draws");
    assert_eq!(
        support::sha256_u32(&draws),
        text(&case["raw_sha256_le_u32"]),
        "regenerated draws"
    );
    let inputs: Vec<u32> = draws
        .into_iter()
        .filter(|&bits| f32::from_bits(bits).is_finite())
        .collect();
    assert_eq!(inputs.len(), usize_of(&case["finite_count"]));
    assert_eq!(
        support::sha256_u32(&inputs),
        text(&case["input_sha256_le_u32"]),
        "finite inputs"
    );
    assert_bulk_u32("seeded", &inputs, &sigmoid_bits(&inputs), &case["expected"]);
}

#[test]
fn full_transport_capacity_preserves_all_results() {
    let case = case("full_transport_capacity_preserves_all_results");
    assert_eq!(usize_of(&case["count"]), MAX_COUNT);
    let inputs = vec![0_u32; MAX_COUNT];
    assert_bulk_u32("zeros", &inputs, &sigmoid_bits(&inputs), &case["expected"]);
}
