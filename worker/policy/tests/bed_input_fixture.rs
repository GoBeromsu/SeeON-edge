//! Port of tests/test_rust_bed_input_parity.py onto fixtures recorded from
//! `OrtBedSegRunner.detect_beds` preprocessing (`seg_postprocess.letterbox_rgb`)
//! under NumPy 2.4.6 and OpenCV 4.13.0. Comparison is exact float32 tensor bits
//! and letterbox metadata including the float64 scale bits.

#[path = "fixtures/support.rs"]
mod support;

use seeon_worker::bed_input::{BedInputError, BedInputTensor, Letterbox, TENSOR_VALUES};
use serde_json::Value;
use support::{Pcg64, array, int, text, u32_of, uint, usize_of};

/// Width, height, image bytes and the typed error the Python refusal maps to.
type Refusal<'a> = (i64, i64, &'a [u8], BedInputError);

const FIXTURE: &str = "bed_input/bed_input.json";
const ORACLE_SOURCES: [(&str, &str); 2] = [
    (
        "worker/adapters/model/ort_bed_seg.py",
        "bded38203d399c7acdae5686ee4b6e0a16d2d9bfa26f361ff731130bfa2b76b9",
    ),
    (
        "worker/adapters/model/seg_postprocess.py",
        "1564c2c1cb68bece11dadcc84b305b3ad0e1b6bb937ab979d023cef3afe93a8a",
    ),
];
const MAX_INPUT: usize = 16 * 1024 * 1024;

fn cases(name: &str) -> Vec<Value> {
    let mut fixture = support::load_sources(FIXTURE, &ORACLE_SOURCES);
    assert_eq!(
        text(&fixture["oracle"]),
        "worker.adapters.model.ort_bed_seg.OrtBedSegRunner.detect_beds recording session"
    );
    array(&fixture["tests"][name].take()).to_vec()
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

    /// The Python test's `_image(width, height, seed)`; every term reduces mod 256.
    fn pattern(width: usize, height: usize, seed: u64) -> Self {
        Self::from_pixels(width, height, |x, y| {
            let pixel = (y * width + x) as u64;
            [
                pixel * 17 + seed,
                (pixel / width as u64) * 29 + pixel * 3 + seed + 37,
                (pixel ^ (pixel >> 8)) + seed + 251,
            ]
            .map(|value| (value % 256) as u8)
        })
    }

    fn filled(width: usize, height: usize, value: u8) -> Self {
        Self::from_pixels(width, height, |_, _| [value; 3])
    }

    /// `np.tile(corners, (1, repeats, 1))` of the 2x2 corner image.
    fn corners(repeats: usize) -> Self {
        const CORNERS: [[[u8; 3]; 2]; 2] =
            [[[0, 1, 254], [255, 254, 1]], [[1, 255, 0], [254, 0, 255]]];
        Self::from_pixels(2 * repeats, 2, |x, y| CORNERS[y][x % 2])
    }

    /// `_image(19, 13)[::-1, ::2, ::-1]` packed as its logical pixels.
    fn strided() -> Self {
        let base = Self::pattern(19, 13, 0);
        Self::from_pixels(10, 13, |x, y| {
            let offset = ((12 - y) * 19 + 2 * x) * 3;
            let [r, g, b] = [0, 1, 2].map(|channel| base.rgb[offset + channel]);
            [b, g, r]
        })
    }

    /// `default_rng(seed)`: `integers(2, 1601)`, `integers(2, 1101)`, then
    /// `integers(0, 256, (height, width, 3), dtype=uint8)`.
    fn seeded(case: &Value) -> Self {
        let mut generator = Pcg64::recorded(case);
        let width = 2 + lemire_u32(&mut generator, 1598) as usize;
        let height = 2 + lemire_u32(&mut generator, 1098) as usize;
        let mut rgb = Vec::with_capacity(width * height * 3);
        while rgb.len() < width * height * 3 {
            let draw = generator.next_u32().to_le_bytes();
            let take = (width * height * 3 - rgb.len()).min(4);
            rgb.extend_from_slice(&draw[..take]);
        }
        Self {
            width: width as i64,
            height: height as i64,
            rgb,
        }
    }
}

/// NumPy `buffered_bounded_lemire_uint32` for an inclusive range `0..=range`.
fn lemire_u32(generator: &mut Pcg64, range: u32) -> u32 {
    let exclusive = u64::from(range) + 1;
    let mut product = u64::from(generator.next_u32()) * exclusive;
    if (product as u32 as u64) < exclusive {
        let threshold = (u32::MAX - range) % (range + 1);
        while (product as u32) < threshold {
            product = u64::from(generator.next_u32()) * exclusive;
        }
    }
    (product >> 32) as u32
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn letterbox_bits(letterbox: &Letterbox) -> [u64; 7] {
    [
        letterbox.source_height as u64,
        letterbox.source_width as u64,
        letterbox.scale.to_bits(),
        letterbox.resized_height as u64,
        letterbox.resized_width as u64,
        letterbox.pad_top as u64,
        letterbox.pad_left as u64,
    ]
}

fn expected_letterbox(value: &Value) -> [u64; 7] {
    [
        int(&value["source_height"]) as u64,
        int(&value["source_width"]) as u64,
        uint(&value["scale_bits"]),
        uint(&value["resized_height"]),
        uint(&value["resized_width"]),
        uint(&value["pad_top"]),
        uint(&value["pad_left"]),
    ]
}

/// Gates the recorded image against the Python test's own image, runs one
/// call on `tensor`, and compares every oracle bit. Returns the tensor bits.
fn check(tensor: &mut BedInputTensor, image: &Image, case: &Value) -> Vec<u32> {
    let context = format!("{}x{}", image.width, image.height);
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
    let (values, letterbox) = tensor
        .preprocess(&image.rgb, image.width, image.height)
        .unwrap_or_else(|error| panic!("{context}: {error}"));
    assert_eq!(
        letterbox_bits(&letterbox),
        expected_letterbox(&case["letterbox"]),
        "{context}: letterbox"
    );
    let actual = bits(values);
    let expected = &case["tensor"];
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
    actual
}

fn exercise(images: &[Image], cases: &[Value]) {
    assert_eq!(cases.len(), images.len());
    let mut tensor = BedInputTensor::default();
    for (image, case) in images.iter().zip(cases) {
        check(&mut tensor, image, case);
    }
}

#[test]
fn all_tensor_and_metadata_bits_for_up_down_same_size_and_odd_aspects() {
    let sizes = [
        (1, 1),
        (5, 3),
        (17, 11),
        (319, 181),
        (997, 541),
        (541, 997),
        (1280, 1280),
        (1280, 719),
        (719, 1280),
        (1537, 887),
        (887, 1537),
        (2048, 1025),
        (65537, 1),
        (1, 65537),
    ];
    let cases = cases("all_tensor_and_metadata_bits_for_up_down_same_size_and_odd_aspects");
    assert_eq!(cases.len(), sizes.len());
    for (case, (width, height)) in cases.iter().zip(sizes) {
        exercise(
            &[Image::pattern(width, height, 0)],
            std::slice::from_ref(case),
        );
    }
}

#[test]
fn python_ties_even_including_valid_zero_resized_axes() {
    let cases = cases("python_ties_even_including_valid_zero_resized_axes");
    assert_eq!(cases.len(), 8);
    let calls = [false, true]
        .into_iter()
        .flat_map(|transpose| [(1, 0), (3, 2), (5, 2), (7, 4)].map(|pair| (transpose, pair)));
    for (case, (transpose, (short, resized))) in cases.iter().zip(calls) {
        assert_eq!(case["transpose"].as_bool(), Some(transpose));
        assert_eq!(
            (usize_of(&case["short"]), usize_of(&case["resized"])),
            (short, resized)
        );
        let (width, height) = if transpose {
            (short, 2560)
        } else {
            (2560, short)
        };
        let mut tensor = BedInputTensor::default();
        check(&mut tensor, &Image::pattern(width, height, 19), case);
        let letterbox = &case["letterbox"];
        let axis = if transpose {
            "resized_width"
        } else {
            "resized_height"
        };
        assert_eq!(usize_of(&letterbox[axis]), resized, "oracle {axis}");
        assert_eq!(uint(&letterbox["scale_bits"]), 0.5_f64.to_bits());
    }
}

#[test]
fn half_coordinate_and_uint8_truncation_edges_keep_channel_order() {
    let cases = cases("half_coordinate_and_uint8_truncation_edges_keep_channel_order");
    exercise(&[Image::corners(1), Image::corners(1280)], &cases);
}

#[test]
fn seeded_pixels_and_geometry_are_selected_independently_of_results() {
    let cases = cases("seeded_pixels_and_geometry_are_selected_independently_of_results");
    assert_eq!(cases.len(), 3);
    for (case, seed) in cases.iter().zip([731, 8128, 65537]) {
        assert_eq!(uint(&case["seed"]), seed);
        exercise(&[Image::seeded(case)], std::slice::from_ref(case));
    }
}

#[test]
fn packed_rgb_preserves_logical_pixels_of_a_strided_python_image() {
    let cases = cases("packed_rgb_preserves_logical_pixels_of_a_strided_python_image");
    exercise(&[Image::strided()], &cases);
}

#[test]
fn four_call_reuse_clears_previous_content_and_padding_within_output_budget() {
    let cases = cases("four_call_reuse_clears_previous_content_and_padding_within_output_budget");
    exercise(
        &[
            Image::filled(1280, 1280, 255),
            Image::pattern(1031, 173, 7),
            Image::pattern(173, 1031, 13),
            Image::pattern(2560, 1, 97),
        ],
        &cases,
    );
}

fn assert_refused(
    tensor: &mut BedInputTensor,
    before: &[u32],
    (width, height, rgb): (i64, i64, &[u8]),
    status: BedInputError,
) {
    assert_eq!(
        tensor
            .preprocess(rgb, width, height)
            .map(|(_, letterbox)| letterbox),
        Err(status),
        "{width}x{height} with {} bytes",
        rgb.len()
    );
    assert!(
        bits(tensor.as_slice()) == before,
        "{width}x{height}: a refusal changed the tensor"
    );
}

#[test]
fn typed_dimension_length_and_overflow_failures_preserve_every_previous_bit() {
    use BedInputError::{Dimensions, DimensionsOverflow, ImageLength};
    let groups: [[Refusal<'_>; 3]; 5] = [
        [
            (0, 1, b"", Dimensions),
            (1, -1, b"", Dimensions),
            (i64::MIN, 1, b"", Dimensions),
        ],
        [
            (1, 0, b"", Dimensions),
            (-1, 1, b"", Dimensions),
            (0, i64::MAX, b"", Dimensions),
        ],
        [
            (i64::MAX, 1, b"", DimensionsOverflow),
            (1 << 32, 1 << 32, b"", DimensionsOverflow),
            (i64::MAX / 3 + 1, 1, b"", DimensionsOverflow),
        ],
        [
            (1, 1, b"", ImageLength),
            (1, 1, b"\x01\x02", ImageLength),
            (1, 1, b"\x01\x02\x03\x04", ImageLength),
        ],
        [
            (1280, 1280, b"", ImageLength),
            (i64::MAX / 3, 1, b"", ImageLength),
            (2, 1, b"abc", ImageLength),
        ],
    ];
    let cases = cases("typed_dimension_length_and_overflow_failures_preserve_every_previous_bit");
    assert_eq!(cases.len(), 1);
    let image = Image::pattern(11, 7, 0);
    for group in groups {
        let mut tensor = BedInputTensor::default();
        let before = check(&mut tensor, &image, &cases[0]);
        for (width, height, rgb, status) in group {
            assert_refused(&mut tensor, &before, (width, height, rgb), status);
        }
    }
}

#[test]
fn success_after_error_replaces_content_without_stale_padding() {
    let cases = cases("success_after_error_replaces_content_without_stale_padding");
    assert_eq!(cases.len(), 2);
    let mut tensor = BedInputTensor::default();
    let initial = check(&mut tensor, &Image::pattern(1280, 1280, 3), &cases[0]);
    assert_refused(
        &mut tensor,
        &initial,
        (0, 1, b"private-invalid-rgb"),
        BedInputError::Dimensions,
    );
    check(&mut tensor, &Image::pattern(9, 71, 99), &cases[1]);
}

#[test]
fn exact_input_budget_is_a_length_refusal_that_leaves_zeros() {
    // The Python test pins the probe's own refusal of the largest transport
    // record: no oracle value exists, so this is a literal admission check.
    let mut tensor = BedInputTensor::default();
    let zeros = vec![0_u32; TENSOR_VALUES];
    let rgb = vec![b'x'; MAX_INPUT - 12 - 20];
    assert_refused(
        &mut tensor,
        &zeros,
        (1, 1, &rgb),
        BedInputError::ImageLength,
    );
}
