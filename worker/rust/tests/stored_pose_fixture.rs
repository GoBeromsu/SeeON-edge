//! Port of tests/test_rust_stored_pose_parity.py onto fixtures recorded from
//! `OrtClipPoseRunner.detect_persons` with the test's recording session.
//! Comparison is exact float32 tensor bits and ordered float64 person-box bits.
//! Each call composes `person_boxes` then `preprocess`, as the stored-pose probe does.

#[path = "fixtures/support.rs"]
mod support;

use seeon_worker::stored_pose::{
    OUTPUT_SHAPE, OUTPUT_VALUES, PersonBox, StoredPoseError, StoredPoseTensor, TENSOR_VALUES,
    person_boxes,
};
use serde_json::Value;
use support::{array, int, text, u32_of, uint, usize_of};

const FIXTURE: &str = "stored_pose/stored_pose.json";
const ORACLE_SOURCE: &str = "worker/adapters/model/ort_clip_pose.py";
const ORACLE_SHA256: &str = "310ceb4276980c9adcc4a9b3ea58e8b0a41c0eafb54ce8d3d05819073fbf88e1";
const THRESHOLD: f64 = 0.25;
const NET: usize = 640;

/// One Python test's recorded calls and the `_rows` keypoint columns.
struct Recorded {
    cases: Vec<Value>,
    keypoints: Vec<f32>,
}

fn recorded(name: &str) -> Recorded {
    let mut fixture = support::load(FIXTURE, ORACLE_SOURCE, ORACLE_SHA256);
    assert_eq!(
        text(&fixture["oracle"]),
        "worker.adapters.model.ort_clip_pose.OrtClipPoseRunner.detect_persons recording session"
    );
    let keypoints = support::f32s(&fixture["keypoint_columns_bits"]);
    assert_eq!(keypoints.len(), OUTPUT_SHAPE[2] - 6);
    Recorded {
        cases: array(&fixture["tests"][name].take()).to_vec(),
        keypoints,
    }
}

/// Packed row-major RGB8, as `image.tobytes(order="C")` of the Python image.
struct Image {
    width: i64,
    height: i64,
    rgb: Vec<u8>,
}

impl Image {
    fn from_pixels(width: usize, height: usize, pixel: impl Fn(usize, usize) -> [u8; 3]) -> Self {
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.extend(pixel(x, y));
            }
        }
        Self {
            width: width as i64,
            height: height as i64,
            rgb,
        }
    }

    /// The Python test's `_image(width, height, seed)` over a uint32 arange.
    fn pattern(width: usize, height: usize, seed: u32) -> Self {
        Self::from_pixels(width, height, |x, y| {
            let pixel = (y * width + x) as u32;
            [
                pixel + seed,
                (pixel >> 8) + seed + 37,
                (pixel >> 16) + seed + 251,
            ]
            .map(|value| (value % 256) as u8)
        })
    }

    fn filled(width: usize, height: usize, value: u8) -> Self {
        Self::from_pixels(width, height, |_, _| [value; 3])
    }

    /// `_image(17, 13)[::-1, ::2, ::-1]` packed as its logical pixels.
    fn strided() -> Self {
        let base = Self::pattern(17, 13, 0);
        Self::from_pixels(9, 13, |x, y| {
            let offset = ((12 - y) * 17 + 2 * x) * 3;
            let [r, g, b] = [0, 1, 2].map(|channel| base.rgb[offset + channel]);
            [b, g, r]
        })
    }
}

/// A raw output0 with its declared shape.
struct Output {
    values: Vec<f32>,
    shape: Vec<usize>,
}

impl Output {
    /// The Python test's `_rows(*heads)`: uninterpreted finite keypoint columns
    /// on every row and each head written into the leading rows.
    fn rows(keypoints: &[f32], heads: &[[f32; 6]]) -> Self {
        let mut values = vec![0.0; OUTPUT_VALUES];
        for row in values.chunks_exact_mut(OUTPUT_SHAPE[2]) {
            row[6..].copy_from_slice(keypoints);
        }
        for (row, head) in values.chunks_exact_mut(OUTPUT_SHAPE[2]).zip(heads) {
            row[..6].copy_from_slice(head);
        }
        Self {
            values,
            shape: OUTPUT_SHAPE.to_vec(),
        }
    }

    fn zeros(shape: &[usize]) -> Self {
        Self {
            values: vec![0.0; shape.iter().product()],
            shape: shape.to_vec(),
        }
    }
}

fn head(values: [f32; 6]) -> [f32; 6] {
    values
}

fn detect(
    tensor: &mut StoredPoseTensor,
    image: &Image,
    output: &Output,
    threshold: f64,
) -> Result<Vec<PersonBox>, StoredPoseError> {
    person_boxes(
        &output.values,
        &output.shape,
        image.width,
        image.height,
        threshold,
    )
    .and_then(|boxes| {
        tensor
            .preprocess(&image.rgb, image.width, image.height)
            .map(|_| boxes)
    })
}

/// Gates the recorded call against the Python test's own inputs.
fn gate(context: &str, case: &Value, image: &Image, output: &Output, threshold: f64) {
    assert_eq!(
        (int(&case["width"]), int(&case["height"])),
        (image.width, image.height),
        "{context}: recorded dimensions"
    );
    assert_eq!(
        support::sha256_u8(&image.rgb),
        text(&case["rgb_sha256"]),
        "{context}: recorded rgb"
    );
    assert_eq!(
        uint(&case["threshold_bits"]),
        threshold.to_bits(),
        "{context}: recorded threshold"
    );
    let shape: Vec<usize> = array(&case["output_shape"]).iter().map(usize_of).collect();
    assert_eq!(shape, output.shape, "{context}: recorded output shape");
    assert_eq!(
        support::sha256_f32(&output.values),
        text(&case["output_sha256"]),
        "{context}: recorded output"
    );
}

/// Every float32 bit of the reused tensor against one oracle tensor record.
fn assert_tensor(context: &str, tensor: &StoredPoseTensor, expected: &Value) {
    let actual: Vec<u32> = tensor
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(actual.len(), TENSOR_VALUES, "{context}: tensor count");
    assert_eq!(usize_of(&expected["count"]), TENSOR_VALUES);
    for key in ["samples", "content_samples"] {
        for sample in array(&expected[key]) {
            let index = usize_of(&sample[0]);
            assert_eq!(
                actual[index],
                u32_of(&sample[1]),
                "{context}: {key} index {index}"
            );
        }
    }
    assert_eq!(
        support::sha256_u32(&actual),
        text(&expected["sha256_le_u32"]),
        "{context}: tensor sha256"
    );
}

fn box_bits(boxes: &[PersonBox]) -> Vec<[u64; 5]> {
    boxes
        .iter()
        .map(|person| person.map(f64::to_bits))
        .collect()
}

fn expected_boxes(case: &Value) -> Vec<[u64; 5]> {
    array(&case["boxes"])
        .iter()
        .map(|person| {
            let components = array(person);
            assert_eq!(components.len(), 5, "box arity");
            std::array::from_fn(|index| uint(&components[index]))
        })
        .collect()
}

/// One admitted call on `tensor`; returns the oracle's box bits.
fn check(
    tensor: &mut StoredPoseTensor,
    case: &Value,
    image: &Image,
    output: &Output,
    threshold: f64,
) -> Vec<[u64; 5]> {
    let context = format!("{}x{} threshold {threshold:?}", image.width, image.height);
    gate(&context, case, image, output, threshold);
    let boxes = detect(tensor, image, output, threshold)
        .unwrap_or_else(|error| panic!("{context}: {error}"));
    assert_tensor(&context, tensor, &case["tensor"]);
    let expected = expected_boxes(case);
    assert_eq!(box_bits(&boxes), expected, "{context}: person boxes");
    expected
}

/// One call the Python oracle refuses; `tensor` must keep `previous` exactly.
fn refuse(
    tensor: &mut StoredPoseTensor,
    case: &Value,
    call: (&Image, &Output, f64),
    refusal: (&str, StoredPoseError),
    previous: &Value,
) {
    let (image, output, threshold) = call;
    let (python, error) = refusal;
    let context = format!("{}x{} threshold {threshold:?}", image.width, image.height);
    gate(&context, case, image, output, threshold);
    assert!(
        text(&case["expected_error"]).contains(python),
        "{context}: oracle error {}",
        case["expected_error"]
    );
    assert_eq!(
        detect(tensor, image, output, threshold),
        Err(error),
        "{context}"
    );
    assert_tensor(&context, tensor, previous);
}

/// A call outside Python's representable inputs: a literal Rust refusal that
/// must leave the oracle-verified `previous` tensor untouched.
fn refuse_literal(
    tensor: &mut StoredPoseTensor,
    call: (&Image, &Output, f64),
    error: StoredPoseError,
    previous: &Value,
) {
    let (image, output, threshold) = call;
    let context = format!(
        "{}x{} rgb {} output {} {:?}",
        image.width,
        image.height,
        image.rgb.len(),
        output.values.len(),
        output.values.iter().find(|value| !value.is_finite())
    );
    assert_eq!(
        detect(tensor, image, output, threshold),
        Err(error),
        "{context}"
    );
    assert_tensor(&context, tensor, previous);
}

fn full_head(score: f32) -> [f32; 6] {
    head([0.0, 0.0, 640.0, 640.0, score, 0.0])
}

#[test]
fn finite_tensor_bits_cover_indices_channels_ties_and_zero_axes() {
    let sizes = [
        (1, 1),
        (5, 3),
        (640, 640),
        (997, 541),
        (541, 997),
        (853, 479),
        (1280, 1),
        (1, 1280),
        (1280, 3),
        (3, 1280),
        (1280, 5),
        (5, 1280),
        (1280, 7),
        (7, 1280),
        (65537, 1),
        (1, 65537),
    ];
    let recorded = recorded("finite_tensor_bits_cover_indices_channels_ties_and_zero_axes");
    assert_eq!(recorded.cases.len(), sizes.len());
    let output = Output::rows(&recorded.keypoints, &[full_head(0.8)]);
    for (case, (width, height)) in recorded.cases.iter().zip(sizes) {
        let mut tensor = StoredPoseTensor::default();
        check(
            &mut tensor,
            case,
            &Image::pattern(width, height, 0),
            &output,
            THRESHOLD,
        );
    }
}

#[test]
fn padding_clears_across_consecutive_shapes_and_zero_axis() {
    let recorded = recorded("padding_clears_across_consecutive_shapes_and_zero_axis");
    let output = Output::rows(&recorded.keypoints, &[full_head(0.75)]);
    let mut images = vec![Image::filled(NET, NET, 255)];
    images.extend(
        [(997, 173), (173, 997), (1280, 1), (3, 1280), (8, 8)]
            .into_iter()
            .zip(1..)
            .map(|((width, height), seed)| Image::pattern(width, height, seed)),
    );
    assert_eq!(recorded.cases.len(), images.len());
    // One reused tensor, as the probe reuses it across one request.
    let mut tensor = StoredPoseTensor::default();
    for (case, image) in recorded.cases.iter().zip(&images) {
        check(&mut tensor, case, image, &output, THRESHOLD);
    }
}

#[test]
fn packed_rgb_represents_logical_pixels_of_a_strided_python_image() {
    let recorded = recorded("packed_rgb_represents_logical_pixels_of_a_strided_python_image");
    assert_eq!(recorded.cases.len(), 1);
    let output = Output::rows(&recorded.keypoints, &[full_head(0.5)]);
    let mut tensor = StoredPoseTensor::default();
    check(
        &mut tensor,
        &recorded.cases[0],
        &Image::strided(),
        &output,
        THRESHOLD,
    );
}

#[test]
fn inclusive_threshold_uses_float32_weak_scalar_promotion() {
    let thresholds = [
        0.25,
        0.25_f64.next_up(),
        0.25_f64.next_down(),
        0.1,
        0.1_f64.next_up(),
        0.50000001,
        0.9999999999999999,
        0.0,
        -0.0,
        1.0,
        2.0_f64.powi(-150),
        2.0_f64.powi(-150).next_up(),
    ];
    let recorded = recorded("inclusive_threshold_uses_float32_weak_scalar_promotion");
    assert_eq!(recorded.cases.len(), thresholds.len());
    let image = Image::pattern(997, 541, 0);
    for (case, threshold) in recorded.cases.iter().zip(thresholds) {
        // np.float32(threshold) rounds to nearest, as `as f32` does.
        let center = threshold as f32;
        let scores = [
            center,
            center.next_down(),
            center.next_up(),
            -0.0,
            0.0,
            0.25,
            1.0,
            f32::MAX,
        ];
        let heads: Vec<[f32; 6]> = scores
            .iter()
            .map(|&score| head([1.0, 2.0, 30.0, 40.0, score, 0.0]))
            .collect();
        let output = Output::rows(&recorded.keypoints, &heads);
        let mut tensor = StoredPoseTensor::default();
        let boxes = check(&mut tensor, case, &image, &output, threshold);
        if threshold == 0.25_f64.next_up() {
            assert!(
                boxes.iter().any(|person| person[4] == 0.25_f64.to_bits()),
                "oracle keeps a 0.25 score at a threshold just above 0.25"
            );
        }
    }
}

#[test]
fn order_duplicate_boxes_scale_precision_clipping_and_signed_zero() {
    let recorded = recorded("order_duplicate_boxes_scale_precision_clipping_and_signed_zero");
    assert_eq!(recorded.cases.len(), 1);
    let tiny = f32::from_bits(1);
    let first = head([321.25, 7.25, 500.75, 123.75, 0.4, 0.0]);
    let heads = [
        first,
        head([1.0, 2.0, 30.0, 40.0, 0.99, 1.0]),
        head([0.0, 0.0, 100.0, 100.0, 0.9, -0.0]),
        first,
        head([-100.0, -200.0, 2000.0, 2000.0, 0.6, 0.0]),
        head([-0.0, -0.0, 1.0, 2.0, -0.0, -0.0]),
        head([800.0, 0.0, 900.0, 10.0, 1.0, 0.0]),
        head([100.0, 0.0, 90.0, 1.0, 1.0, 0.0]),
        head([0.0, -4.0, 1.0, -1.0, 1.0, 0.0]),
        head([1.0, 1.0, 2.0, 2.0, 1.0, tiny]),
        head([0.0, 0.0, 1.0, 1.0, 1.0, -1.0]),
        head([0.0, 0.0, 1.0, 1.0, 1.0, 0.5]),
        head([1.0, 1.0, 1.0, 2.0, 1.0, 0.0]),
    ];
    let output = Output::rows(&recorded.keypoints, &heads);
    let mut tensor = StoredPoseTensor::default();
    let boxes = check(
        &mut tensor,
        &recorded.cases[0],
        &Image::pattern(997, 541, 0),
        &output,
        0.0,
    );
    // The Python test's own assertions, on the oracle's boxes.
    assert_eq!(boxes.len(), 5);
    assert_eq!(boxes[0], boxes[2]);
    assert_eq!(boxes[0][0], 500.4472961425781_f64.to_bits());
    assert_eq!(boxes[3][..4], [0.0, 0.0, 997.0, 541.0].map(f64::to_bits));
    assert_eq!(boxes[4][..2], [0, 0]);
    assert_eq!(boxes[4][4], 1 << 63);
}

#[test]
fn all_300_rows_remain_in_source_order_without_nms() {
    let recorded = recorded("all_300_rows_remain_in_source_order_without_nms");
    assert_eq!(recorded.cases.len(), 1);
    let heads: Vec<[f32; 6]> = (0..300)
        .map(|i| head([299.0 - i as f32, 1.0, 300.0 - i as f32, 2.0, 0.5, 0.0]))
        .collect();
    let output = Output::rows(&recorded.keypoints, &heads);
    let mut tensor = StoredPoseTensor::default();
    let boxes = check(
        &mut tensor,
        &recorded.cases[0],
        &Image::pattern(640, 640, 0),
        &output,
        0.5,
    );
    assert_eq!(boxes.len(), 300);
    assert_eq!(boxes[0][0], 299.0_f64.to_bits());
    assert_eq!(boxes[299][0], 0.0_f64.to_bits());
}

#[test]
fn finite_coordinates_whose_division_overflows_still_clip_canonically() {
    let recorded = recorded("finite_coordinates_whose_division_overflows_still_clip_canonically");
    assert_eq!(recorded.cases.len(), 1);
    let maximum = f32::MAX;
    let output = Output::rows(
        &recorded.keypoints,
        &[head([-maximum, -maximum, maximum, maximum, 0.75, 0.0])],
    );
    let mut tensor = StoredPoseTensor::default();
    let boxes = check(
        &mut tensor,
        &recorded.cases[0],
        &Image::pattern(997, 541, 0),
        &output,
        THRESHOLD,
    );
    assert_eq!(boxes, [[0.0, 0.0, 997.0, 541.0, 0.75].map(f64::to_bits)]);
}

#[test]
fn float32_division_underflow_precedes_strict_positive_box_admission() {
    let recorded = recorded("float32_division_underflow_precedes_strict_positive_box_admission");
    assert_eq!(recorded.cases.len(), 1);
    let tiny = f32::from_bits(1);
    let output = Output::rows(
        &recorded.keypoints,
        &[
            head([0.0, 0.0, tiny, 640.0, 0.5, 0.0]),
            head([tiny, -tiny, 1.0, 1.0, -0.0, 0.0]),
        ],
    );
    let mut tensor = StoredPoseTensor::default();
    let boxes = check(
        &mut tensor,
        &recorded.cases[0],
        &Image::pattern(1, 1, 0),
        &output,
        0.0,
    );
    assert_eq!(boxes.len(), 1);
    assert_eq!(boxes[0][..2], [0, 0]);
    assert_eq!(boxes[0][4], 1 << 63);
}

#[test]
fn bad_rgb_admission_does_not_partially_mutate_reused_tensor() {
    let recorded = recorded("bad_rgb_admission_does_not_partially_mutate_reused_tensor");
    assert_eq!(recorded.cases.len(), 4);
    let output = Output::rows(&recorded.keypoints, &[full_head(0.5)]);
    let initial = Image::pattern(3, 2, 0);
    let mut tensor = StoredPoseTensor::default();
    check(
        &mut tensor,
        &recorded.cases[0],
        &initial,
        &output,
        THRESHOLD,
    );
    let previous = &recorded.cases[0]["tensor"];
    let empty = |width: i64, height: i64| Image {
        width,
        height,
        rgb: Vec::new(),
    };
    let short = Image {
        rgb: initial.rgb[..initial.rgb.len() - 1].to_vec(),
        ..empty(3, 2)
    };
    let long = Image {
        rgb: [&initial.rgb[..], &[0]].concat(),
        ..empty(3, 2)
    };
    // The probe's transport carries dimensions Python cannot build as arrays.
    for (image, error) in [
        (empty(0, 2), StoredPoseError::Dimensions),
        (empty(-1, 2), StoredPoseError::Dimensions),
        (empty(3, 0), StoredPoseError::Dimensions),
        (empty(i64::MAX, 2), StoredPoseError::DimensionsOverflow),
        (short, StoredPoseError::ImageLength),
        (long, StoredPoseError::ImageLength),
    ] {
        refuse_literal(&mut tensor, (&image, &output, THRESHOLD), error, previous);
    }
    let last = Image::pattern(2, 7, 31);
    check(&mut tensor, &recorded.cases[1], &last, &output, THRESHOLD);
    let previous = &recorded.cases[1]["tensor"];
    for (case, (width, height)) in recorded.cases[2..].iter().zip([(3, 0), (0, 2)]) {
        refuse(
            &mut tensor,
            case,
            (&empty(width, height), &output, THRESHOLD),
            ("dimensions must be positive", StoredPoseError::Dimensions),
            previous,
        );
    }
}

#[test]
fn malformed_output_shapes_and_lengths_fail_closed_without_tensor_changes() {
    let recorded =
        recorded("malformed_output_shapes_and_lengths_fail_closed_without_tensor_changes");
    let shapes: [&[usize]; 5] = [&[], &[300, 57], &[1, 299, 57], &[1, 300, 56], &[2, 300, 57]];
    assert_eq!(recorded.cases.len(), 1 + shapes.len());
    let image = Image::pattern(11, 7, 0);
    let output = Output::rows(&recorded.keypoints, &[full_head(0.5)]);
    let mut tensor = StoredPoseTensor::default();
    check(&mut tensor, &recorded.cases[0], &image, &output, THRESHOLD);
    let previous = &recorded.cases[0]["tensor"];
    for (case, shape) in recorded.cases[1..].iter().zip(shapes) {
        refuse(
            &mut tensor,
            case,
            (&image, &Output::zeros(shape), THRESHOLD),
            ("output0 must have shape", StoredPoseError::OutputShape),
            previous,
        );
    }
    // A declared shape/length mismatch cannot be represented by a NumPy array.
    for size in [OUTPUT_VALUES - 1, OUTPUT_VALUES + 1] {
        let mismatched = Output {
            values: vec![0.0; size],
            shape: OUTPUT_SHAPE.to_vec(),
        };
        refuse_literal(
            &mut tensor,
            (&image, &mismatched, THRESHOLD),
            StoredPoseError::OutputShape,
            previous,
        );
    }
}

#[test]
fn nonfinite_output_is_separate_fail_closed_admission_even_in_ignored_fields() {
    let recorded =
        recorded("nonfinite_output_is_separate_fail_closed_admission_even_in_ignored_fields");
    assert_eq!(recorded.cases.len(), 1);
    let initial = Output::rows(&recorded.keypoints, &[full_head(0.5)]);
    let image = Image::pattern(5, 19, 99);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut tensor = StoredPoseTensor::default();
        check(
            &mut tensor,
            &recorded.cases[0],
            &Image::pattern(19, 5, 0),
            &initial,
            THRESHOLD,
        );
        // Low-score rows, nonperson rows, and unused keypoints must all be scanned.
        // Nonfinite admission is the Rust fail-closed contract, not Python parity.
        for (row, component) in [
            (0, 0),
            (0, 4),
            (0, 5),
            (0, 6),
            (298, 2),
            (299, 0),
            (299, 56),
        ] {
            let mut output = Output::rows(&recorded.keypoints, &[full_head(0.5)]);
            let width = OUTPUT_SHAPE[2];
            output.values[299 * width + 4..299 * width + 6].copy_from_slice(&[1.0, 1.0]);
            output.values[row * width + component] = bad;
            refuse_literal(
                &mut tensor,
                (&image, &output, THRESHOLD),
                StoredPoseError::NonFiniteOutput,
                &recorded.cases[0]["tensor"],
            );
        }
    }
}

#[test]
fn invalid_thresholds_reject_before_mutating_previous_tensor() {
    let thresholds = [
        -0.1,
        0.0_f64.next_down(),
        1.0_f64.next_up(),
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let recorded = recorded("invalid_thresholds_reject_before_mutating_previous_tensor");
    assert_eq!(recorded.cases.len(), 1 + thresholds.len());
    let output = Output::rows(&recorded.keypoints, &[full_head(0.5)]);
    let mut tensor = StoredPoseTensor::default();
    check(
        &mut tensor,
        &recorded.cases[0],
        &Image::pattern(9, 2, 0),
        &output,
        THRESHOLD,
    );
    let image = Image::pattern(2, 9, 42);
    for (case, threshold) in recorded.cases[1..].iter().zip(thresholds) {
        refuse(
            &mut tensor,
            case,
            (&image, &output, threshold),
            ("threshold must be in", StoredPoseError::Threshold),
            &recorded.cases[0]["tensor"],
        );
    }
}
