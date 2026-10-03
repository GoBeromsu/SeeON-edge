//! GPU parity against the recorded ORT-CUDA FP32 oracle (TF32 off). Every golden
//! comes from `tests_support/native_yolo_parity.py`; Rust output is never a
//! golden. A missing input fails with its variable name instead of skipping.
//!
//! Recorded decoded-decision checks use score >= 0.25. Independent raw-output
//! checks compare every element in recorded order, including lower-score rows.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::CStr;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use seeon_deepstream_native::{ClipDecoder, GpuModel, TensorInput, TensorOutput, build_engine};
use seeon_worker::bed_contour::largest_external_contour;
use seeon_worker::bed_input::Letterbox;
use seeon_worker::bed_polygon::simplify_polygon;
use seeon_worker::bed_sigmoid::bed_mask_sigmoid;
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider, FallProbabilities};
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, PoseBbox56Row, ZERO_ROW, pose_bbox56_row};
use seeon_worker::stored_pose::{OUTPUT_VALUES, StoredPoseTensor};
use seeon_worker::temporal::PtsResampler;
use seeon_worker::trace::{DecisionTraceMissingReason, DecisionTraceSnapshot, NumericTraceValue};
use seeon_worker_runtime::bed_gpu::{BedGpu, BedGpuError, DETECTIONS_VALUES, PROTOS_VALUES};
use seeon_worker_runtime::evidence::{
    AcceleratorEvidence, EngineDigest, EvidenceError, PROVIDER, Precision,
};
use seeon_worker_runtime::fall_gpu::{FallGpu, FallGpuError, WINDOW_VALUES};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

#[path = "gpu_parity/ordered.rs"]
mod ordered;

const MAX_ABS: f64 = 1e-4;
/// Bed outputs admit this many float32 ULP of the oracle value where that is
/// wider than `MAX_ABS`; stored pose and fall retain the absolute limit.
const ORACLE_ULP: u32 = 2;
const DEVICE: i32 = 0;
const MANIFEST: &str = include_str!("fixtures/gpu/manifest.json");
const IDENTITY_FILE: &str = "engine-identity.json";
const F32_BYTES: u64 = 4;
const POSE_DIMENSIONS: [i32; 4] = [1, 3, 640, 640];
const SCHEMA: &str = "seeon-gpu-oracle-fixtures/v2";
const ROWS_FROM_SCORE: f64 = 0.25;
const SCORE_COLUMN: usize = 4;
const CLASS_COLUMN: usize = 5;
const BED_ROW: usize = 38;
const POSE_ROW: usize = 57;
const BED_CLASS: i64 = 59;
const MODEL_SIZE: usize = 1280;
const PROTO_CHANNELS: usize = 32;
const PROTO_SIDE: usize = 320;
const MASK_THRESHOLD: f32 = 0.5;
const BED_POLYGON_POINTS: [i64; 2] = [48, 16];
/// The bed recognizer's configured confidence. This band is counted separately;
/// raw comparisons include it, but decoded goldens still use the 0.25 gate.
const RECOGNIZER_CONFIDENCE: f64 = 0.05;
const DIAGNOSTIC_MAX_ABS: f64 = 1e-2;

// GPU tests share one device and its per-model counters; run them one at a time.
static GPU: Mutex<()> = Mutex::new(());

struct Role {
    key: &'static str,
    engine_var: &'static str,
    onnx_var: &'static str,
    engine_file: &'static str,
    input: &'static CStr,
    dimensions: &'static [i32],
}

const BED: Role = Role {
    key: "bed",
    engine_var: "SEEON_TEST_BED_ENGINE",
    onnx_var: "SEEON_TEST_BED_ONNX",
    engine_file: "yolo26l-seg-fp32.engine",
    input: c"images",
    dimensions: &[1, 3, 1280, 1280],
};
const STORED_POSE: Role = Role {
    key: "stored_pose",
    engine_var: "SEEON_TEST_STORED_POSE_ENGINE",
    onnx_var: "SEEON_TEST_STORED_POSE_ONNX",
    engine_file: "yolo26n-pose-fp32.engine",
    input: c"images",
    dimensions: &POSE_DIMENSIONS,
};
const FALL: Role = Role {
    key: "fall",
    engine_var: "SEEON_TEST_FALL_ENGINE",
    onnx_var: "SEEON_TEST_FALL_ONNX",
    engine_file: "pose-bbox56-gru-fp32.engine",
    input: c"window",
    dimensions: &[1, 30, 56],
};
const ROLES: [Role; 3] = [BED, STORED_POSE, FALL];

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn env_path(var: &str) -> PathBuf {
    let value = std::env::var_os(var).unwrap_or_else(|| panic!("{var} is required"));
    let text = value
        .to_str()
        .unwrap_or_else(|| panic!("{var} must be UTF-8"));
    assert!(
        !text.trim().is_empty() && !text.contains('\0'),
        "{var} must be a nonblank path"
    );
    PathBuf::from(value)
}

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).expect("manifest.json is JSON")
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("field {key} must be a string"))
}

fn count(value: &Value, key: &str) -> usize {
    let number = value[key]
        .as_u64()
        .unwrap_or_else(|| panic!("field {key} must be an unsigned integer"));
    usize::try_from(number).expect("count fits usize")
}

fn array<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("field {key} must be an array"))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn read(path: &PathBuf, label: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| panic!("read {label}: {error}"))
}

fn max_abs(label: &str, actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            assert!(actual.is_finite(), "{label} produced a non-finite value");
            (f64::from(*actual) - f64::from(*expected)).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        worst <= MAX_ABS,
        "{label} max_abs {worst:e} exceeds {MAX_ABS:e}"
    );
    worst
}

/// Out-of-repo oracle fixtures, admitted only by the in-repo manifest digests.
struct Fixtures {
    root: PathBuf,
    manifest: Value,
}

impl Fixtures {
    fn open() -> Self {
        let manifest = manifest();
        assert_eq!(manifest["schema"].as_str(), Some(SCHEMA), "manifest schema");
        assert_eq!(
            manifest["tolerance"]["max_abs"].as_f64(),
            Some(MAX_ABS),
            "manifest tolerance"
        );
        assert_eq!(
            manifest["tolerance"]["ulp"].as_u64(),
            Some(u64::from(ORACLE_ULP)),
            "manifest ulp"
        );
        assert_eq!(
            manifest["tolerance"]["rows_from_score"].as_f64(),
            Some(ROWS_FROM_SCORE),
            "manifest gated rows"
        );
        Self {
            root: env_path("SEEON_TEST_GPU_FIXTURES"),
            manifest,
        }
    }

    fn bytes(&self, relative: &str) -> Vec<u8> {
        let expected = &self.manifest["fixtures"][relative];
        assert!(
            expected.is_object(),
            "fixture {relative} is not in the manifest"
        );
        let bytes = read(&self.root.join(relative), relative);
        assert_eq!(
            bytes.len(),
            count(expected, "bytes"),
            "fixture {relative} size"
        );
        assert_eq!(
            hex(&sha256(&bytes)),
            text(expected, "sha256"),
            "fixture {relative} sha256"
        );
        bytes
    }

    fn f32s(&self, spec: &Value) -> Vec<f32> {
        let values: usize = array(spec, "shape")
            .iter()
            .map(|dimension| dimension.as_u64().expect("shape dimension") as usize)
            .product();
        let relative = text(spec, "path");
        let bytes = self.bytes(relative);
        assert_eq!(
            bytes.len(),
            values * 4,
            "{relative} does not match its shape"
        );
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect()
    }

    /// `SPRGB001`, u32 LE width and height, then packed RGB24 rows.
    fn rgb(&self, relative: &str) -> (i64, i64, Vec<u8>) {
        let bytes = self.bytes(relative);
        assert!(
            bytes.len() >= 16 && &bytes[..8] == b"SPRGB001",
            "{relative} header"
        );
        let width = u32::from_le_bytes(bytes[8..12].try_into().expect("width"));
        let height = u32::from_le_bytes(bytes[12..16].try_into().expect("height"));
        let pixels = bytes[16..].to_vec();
        assert_eq!(
            pixels.len(),
            width as usize * height as usize * 3,
            "{relative} pixels"
        );
        (i64::from(width), i64::from(height), pixels)
    }
}

/// Opens the engine named by the role's variable only when `engine-identity.json`
/// beside it binds that exact file to the manifest's ONNX digest on this GPU.
fn admitted_engine(role: &Role, manifest: &Value) -> (GpuModel, EngineDigest) {
    let path = env_path(role.engine_var);
    let digest = sha256(&read(&path, role.engine_var));
    let identity_path = path
        .parent()
        .unwrap_or_else(|| panic!("{} has no parent directory", role.engine_var))
        .join(IDENTITY_FILE);
    let identity: Value = serde_json::from_slice(&read(&identity_path, IDENTITY_FILE))
        .unwrap_or_else(|error| panic!("{IDENTITY_FILE} is not JSON: {error}"));
    let entry = &identity["engines"][role.key];
    assert_eq!(
        Some(text(entry, "engine")),
        path.file_name().and_then(|name| name.to_str()),
        "{} is not the engine recorded for {}",
        role.engine_var,
        role.key
    );
    assert_eq!(
        text(entry, "engine_sha256"),
        hex(&digest),
        "{} digest",
        role.key
    );
    assert_eq!(
        text(entry, "onnx_sha256"),
        text(&manifest["models"][role.key], "onnx_sha256"),
        "{} engine was built from another ONNX",
        role.key
    );
    assert_eq!(entry["tf32_enabled"], json!(false), "{} TF32", role.key);
    assert_eq!(
        text(entry, "compute_capability"),
        text(&manifest["gpu"], "compute_capability"),
        "{} compute capability",
        role.key
    );
    let model = GpuModel::open(&path, DEVICE)
        .unwrap_or_else(|error| panic!("open {} engine: {error:?}", role.key));
    (model, EngineDigest::new(digest))
}

/// One call's receipt: TensorRT FP32 on the configured device, exactly one
/// successful execution with the expected copies, and a gapless sequence.
fn check_receipt(
    evidence: &AcceleratorEvidence,
    digest: EngineDigest,
    bytes: (u64, u64),
    previous: &mut Option<u64>,
) {
    assert_eq!(evidence.provider(), PROVIDER);
    assert_eq!(evidence.provider(), "tensorrt");
    assert_eq!(evidence.precision(), Precision::Fp32);
    assert_eq!(evidence.device_ordinal(), DEVICE);
    assert!(evidence.engine_sha256() == digest, "receipt engine digest");
    assert_eq!(
        (
            evidence.attempted(),
            evidence.succeeded(),
            evidence.failed()
        ),
        (1, 1, 0)
    );
    assert!(evidence.h2d_bytes() > 0 && evidence.d2h_bytes() > 0);
    assert_eq!((evidence.h2d_bytes(), evidence.d2h_bytes()), bytes);
    if let Some(previous) = *previous {
        assert_eq!(
            evidence.call_seq(),
            previous + 1,
            "call_seq must advance by one"
        );
    }
    *previous = Some(evidence.call_seq());
}

fn window_rows(window: &[f32]) -> Vec<PoseBbox56Row> {
    window
        .chunks_exact(ZERO_ROW.len())
        .map(|row| PoseBbox56Row::try_from(row).expect("56 values"))
        .collect()
}

/// Rows the decider consumes (score >= 0.25), strongest first; ties keep row order.
fn gated_rows(values: &[f32], width: usize) -> Vec<&[f32]> {
    let mut rows: Vec<&[f32]> = values
        .chunks_exact(width)
        .filter(|row| f64::from(row[SCORE_COLUMN]) >= ROWS_FROM_SCORE)
        .collect();
    rows.sort_by(|a, b| b[SCORE_COLUMN].total_cmp(&a[SCORE_COLUMN]));
    rows
}

/// One float32 ULP at the value's magnitude, as `np.spacing(np.float32(|value|))`.
fn spacing(value: f32) -> f64 {
    let magnitude = value.abs();
    f64::from(f32::from_bits(magnitude.to_bits() + 1) - magnitude)
}

/// Gated rows must agree in count and, paired in score order, per value within
/// 1e-4 or, with `ulp`, `ORACLE_ULP` ULP of the oracle value where that is wider.
/// Returns the gated row count and the worst difference.
fn compare_rows(
    actual: &[f32],
    expected: &[f32],
    width: usize,
    ulp: bool,
) -> Result<(usize, f64), String> {
    if actual.len() != expected.len() || !expected.len().is_multiple_of(width) {
        return Err(format!(
            "lengths {} and {} for rows of {width}",
            actual.len(),
            expected.len()
        ));
    }
    for (side, values) in [("actual", actual), ("oracle", expected)] {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(format!("{side} value {index} is not finite"));
        }
    }
    let (actual, expected) = (gated_rows(actual, width), gated_rows(expected, width));
    if actual.len() != expected.len() {
        return Err(format!(
            "{} gated rows, oracle has {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut worst = 0.0_f64;
    for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        for (column, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (f64::from(*a) - f64::from(*e)).abs();
            let limit = if ulp {
                MAX_ABS.max(f64::from(ORACLE_ULP) * spacing(*e))
            } else {
                MAX_ABS
            };
            if diff > limit {
                return Err(format!(
                    "gated row {row} column {column}: {a} vs {e}, diff {diff:e} > {limit:e}"
                ));
            }
            worst = worst.max(diff);
        }
    }
    Ok((actual.len(), worst))
}

/// Disclosure only: raw max_abs and the count of values beyond 1e-4.
fn spread(actual: &[f32], expected: &[f32]) -> (f64, usize) {
    assert_eq!(actual.len(), expected.len(), "spread length");
    actual
        .iter()
        .zip(expected)
        .fold((0.0_f64, 0), |(worst, over), (a, e)| {
            let diff = (f64::from(*a) - f64::from(*e)).abs();
            (worst.max(diff), over + usize::from(diff > MAX_ABS))
        })
}

/// Disclosure only: greedy one-to-one pairing of every row by its worst column
/// difference. Returns the pairs within 1e-2 and the best unpaired oracle score.
fn top_rows_matched(actual: &[f32], expected: &[f32], width: usize) -> (usize, Option<f32>) {
    let actual: Vec<&[f32]> = actual.chunks_exact(width).collect();
    let expected: Vec<&[f32]> = expected.chunks_exact(width).collect();
    let mut costs = Vec::with_capacity(actual.len() * expected.len());
    for (i, a) in actual.iter().enumerate() {
        for (j, e) in expected.iter().enumerate() {
            let cost = a
                .iter()
                .zip(e.iter())
                .map(|(a, e)| (f64::from(*a) - f64::from(*e)).abs())
                .fold(0.0_f64, f64::max);
            costs.push((cost, i, j));
        }
    }
    costs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut paired = (vec![false; actual.len()], vec![false; expected.len()]);
    let mut matched = 0;
    for (cost, i, j) in costs {
        if cost > DIAGNOSTIC_MAX_ABS {
            break;
        }
        if !paired.0[i] && !paired.1[j] {
            (paired.0[i], paired.1[j]) = (true, true);
            matched += 1;
        }
    }
    let unmatched = expected
        .iter()
        .zip(&paired.1)
        .filter(|(_, paired)| !**paired)
        .map(|(row, _)| row[SCORE_COLUMN])
        .reduce(f32::max);
    (matched, unmatched)
}

/// Bed rows the recognizer's 0.05 decode also admits below the gate.
fn ungated_bed_rows(values: &[f32]) -> usize {
    values
        .chunks_exact(BED_ROW)
        .filter(|row| {
            row[CLASS_COLUMN] as i64 == BED_CLASS
                && (RECOGNIZER_CONFIDENCE..ROWS_FROM_SCORE).contains(&f64::from(row[SCORE_COLUMN]))
        })
        .count()
}

struct Sample {
    low: usize,
    high: usize,
    weight: f64,
}

/// `_resize_bilinear`'s float32 grid and float64 weights (seg_postprocess.py:227-250).
fn axis_samples(source: usize, destination: usize) -> Vec<Sample> {
    (0..destination)
        .map(|i| {
            let center = i as f32 + 0.5_f32;
            let scaled = center * source as f32;
            let divided = scaled / destination as f32;
            let coordinate = (divided - 0.5_f32).clamp(0.0, (source - 1) as f32);
            let low = coordinate.floor() as usize;
            Sample {
                low,
                high: (low + 1).min(source - 1),
                weight: f64::from(coordinate) - low as f64,
            }
        })
        .collect()
}

fn resize(values: &[f32], (height, width): (usize, usize), shape: (usize, usize)) -> Vec<f32> {
    if (height, width) == shape {
        return values.to_vec();
    }
    let (ys, xs) = (axis_samples(height, shape.0), axis_samples(width, shape.1));
    let at = |y: usize, x: usize| f64::from(values[y * width + x]);
    let mut resized = Vec::with_capacity(shape.0 * shape.1);
    for y in &ys {
        for x in &xs {
            let top = at(y.low, x.low) * (1.0 - x.weight) + at(y.low, x.high) * x.weight;
            let bottom = at(y.high, x.low) * (1.0 - x.weight) + at(y.high, x.high) * x.weight;
            resized.push((top * (1.0 - y.weight) + bottom * y.weight) as f32);
        }
    }
    resized
}

/// What the bed recognizer consumes from one decoded row.
#[derive(Clone, Debug)]
struct BedInstance {
    bbox: [i64; 4],
    score: f32,
    mask: Vec<u8>,
    /// One polygon per entry of `BED_POLYGON_POINTS`.
    polygons: Vec<Vec<[i64; 2]>>,
    /// Closest mask value to the 0.5 threshold; NaN for recorded goldens.
    margin: f64,
}

/// Test transcription of `decode_end_to_end_segmentation`
/// (worker/adapters/model/seg_postprocess.py:65-108) through production's Rust
/// sigmoid, contour and polygon; no production Rust bed decode exists yet. The
/// prototype dot product accumulates in f64 where NumPy uses float32 BLAS.
fn decode_bed(
    detections: &[f32],
    protos: &[f32],
    letterbox: &Letterbox,
    confidence: f64,
) -> Result<Vec<BedInstance>, String> {
    let plane = PROTO_SIDE * PROTO_SIDE;
    if !detections.len().is_multiple_of(BED_ROW) || protos.len() != PROTO_CHANNELS * plane {
        return Err(format!(
            "bed output lengths {} and {}",
            detections.len(),
            protos.len()
        ));
    }
    if detections
        .iter()
        .chain(protos)
        .any(|value| !value.is_finite())
    {
        return Err("bed outputs must be finite".to_owned());
    }
    let (top, left) = (letterbox.pad_top, letterbox.pad_left);
    let content = (letterbox.resized_height, letterbox.resized_width);
    if top + content.0 > MODEL_SIZE || left + content.1 > MODEL_SIZE {
        return Err(format!("letterbox {letterbox:?} exceeds the model input"));
    }
    let source = (
        usize::try_from(letterbox.source_height).map_err(|_| "source height".to_owned())?,
        usize::try_from(letterbox.source_width).map_err(|_| "source width".to_owned())?,
    );
    let inverse = |value: f32, pad: usize, extent: i64| {
        ((f64::from(value) - pad as f64) / letterbox.scale).clamp(0.0, extent as f64) as i64
    };
    let mut instances = Vec::new();
    for row in detections.chunks_exact(BED_ROW) {
        let [x1, y1, x2, y2, score, class] =
            [0, 1, 2, 3, SCORE_COLUMN, CLASS_COLUMN].map(|c| row[c]);
        if class as i64 != BED_CLASS || f64::from(score) < confidence || x2 <= x1 || y2 <= y1 {
            continue;
        }
        let bound = |value: f32| (f64::from(value) / MODEL_SIZE as f64 * PROTO_SIDE as f64) as f32;
        let (x_low, y_low, x_high, y_high) = (bound(x1), bound(y1), bound(x2), bound(y2));
        let mut logits = vec![0.0_f32; plane];
        for (index, value) in logits.iter_mut().enumerate() {
            let (y, x) = ((index / PROTO_SIDE) as f32, (index % PROTO_SIDE) as f32);
            if x >= x_low && x < x_high && y >= y_low && y < y_high {
                let logit: f64 = row[6..]
                    .iter()
                    .enumerate()
                    .map(|(channel, weight)| {
                        f64::from(*weight) * f64::from(protos[channel * plane + index])
                    })
                    .sum();
                *value = bed_mask_sigmoid(logit as f32);
            }
        }
        let model = resize(&logits, (PROTO_SIDE, PROTO_SIDE), (MODEL_SIZE, MODEL_SIZE));
        let cropped: Vec<f32> = model[top * MODEL_SIZE..(top + content.0) * MODEL_SIZE]
            .chunks_exact(MODEL_SIZE)
            .flat_map(|line| &line[left..left + content.1])
            .copied()
            .collect();
        let mask = resize(&cropped, content, source);
        let binary: Vec<u8> = mask
            .iter()
            .map(|value| u8::from(*value > MASK_THRESHOLD))
            .collect();
        let margin = mask
            .iter()
            .map(|value| (f64::from(*value) - f64::from(MASK_THRESHOLD)).abs())
            .fold(f64::INFINITY, f64::min);
        let contour =
            largest_external_contour(&binary, letterbox.source_width, letterbox.source_height)
                .map_err(|error| format!("bed contour: {error:?}"))?;
        let polygons = BED_POLYGON_POINTS
            .iter()
            .map(|points| {
                simplify_polygon(&contour, *points)
                    .map_err(|error| format!("bed polygon: {error:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        instances.push(BedInstance {
            bbox: [
                inverse(x1, left, letterbox.source_width),
                inverse(y1, top, letterbox.source_height),
                inverse(x2, left, letterbox.source_width),
                inverse(y2, top, letterbox.source_height),
            ],
            score,
            mask: binary,
            polygons,
            margin,
        });
    }
    Ok(instances)
}

fn letterbox_of(value: &Value) -> Letterbox {
    let signed = |key: &str| {
        value[key]
            .as_i64()
            .unwrap_or_else(|| panic!("letterbox {key} must be an integer"))
    };
    Letterbox {
        source_height: signed("source_height"),
        source_width: signed("source_width"),
        scale: hex_f64(&value["scale"]),
        resized_height: count(value, "resized_height"),
        resized_width: count(value, "resized_width"),
        pad_top: count(value, "pad_top"),
        pad_left: count(value, "pad_left"),
    }
}

/// The recorder's row-order bed instances with their u8 masks.
fn golden_beds(fixtures: &Fixtures, entry: &Value) -> Vec<BedInstance> {
    let instances = array(entry, "instances");
    let spec = &entry["masks"];
    let shape: Vec<usize> = array(spec, "shape")
        .iter()
        .map(|dimension| dimension.as_u64().expect("mask dimension") as usize)
        .collect();
    let letterbox = &entry["letterbox"];
    assert_eq!(
        shape,
        [
            instances.len(),
            count(letterbox, "source_height"),
            count(letterbox, "source_width")
        ],
        "bed mask shape"
    );
    let pixels = shape[1] * shape[2];
    let masks = fixtures.bytes(text(spec, "path"));
    assert_eq!(masks.len(), instances.len() * pixels, "bed masks size");
    assert!(
        masks.iter().all(|value| *value <= 1),
        "bed masks are 0 or 1"
    );
    let integer = |value: &Value| value.as_i64().expect("integer");
    let polygon = |value: &Value| -> Vec<[i64; 2]> {
        value
            .as_array()
            .expect("polygon")
            .iter()
            .map(|point| {
                let pair = point.as_array().expect("point");
                assert_eq!(pair.len(), 2, "polygon point");
                [integer(&pair[0]), integer(&pair[1])]
            })
            .collect()
    };
    instances
        .iter()
        .zip(masks.chunks_exact(pixels))
        .map(|(instance, mask)| {
            let bbox: Vec<i64> = array(instance, "box").iter().map(integer).collect();
            let score = u32::from_str_radix(text(instance, "score"), 16).expect("score bits");
            BedInstance {
                bbox: bbox.try_into().expect("four box values"),
                score: f32::from_bits(score),
                mask: mask.to_vec(),
                polygons: BED_POLYGON_POINTS
                    .iter()
                    .map(|points| polygon(&instance[format!("polygon{points}")]))
                    .collect(),
                margin: f64::NAN,
            }
        })
        .collect()
}

/// Decision inputs must be identical, paired in the recognizer's score order
/// (worker/runtime/nvidia_bed_zone_recognizer.py:111). With `scores`, score bits too.
fn compare_beds(
    actual: &[BedInstance],
    expected: &[BedInstance],
    scores: bool,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{} bed instances, oracle has {}",
            actual.len(),
            expected.len()
        ));
    }
    let ranked = |instances: &[BedInstance]| {
        let mut ranked: Vec<BedInstance> = instances.to_vec();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
        ranked
    };
    for (rank, (a, e)) in ranked(actual).iter().zip(&ranked(expected)).enumerate() {
        if a.bbox != e.bbox {
            return Err(format!("instance {rank} box {:?} vs {:?}", a.bbox, e.bbox));
        }
        if scores && a.score.to_bits() != e.score.to_bits() {
            return Err(format!(
                "instance {rank} score bits {} vs {}",
                a.score, e.score
            ));
        }
        if a.mask.len() != e.mask.len() {
            return Err(format!(
                "instance {rank} mask sizes {} and {}",
                a.mask.len(),
                e.mask.len()
            ));
        }
        let flipped = a.mask.iter().zip(&e.mask).filter(|(a, e)| a != e).count();
        if flipped > 0 {
            return Err(format!("instance {rank} mask: {flipped} pixels differ"));
        }
        if a.polygons != e.polygons {
            return Err(format!("instance {rank} polygons differ"));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an sm_120 GPU, SEEON_TEST_ENGINE_BUILD_DIR and SEEON_TEST_{BED,STORED_POSE,FALL}_ONNX"]
fn fp32_engines_build_without_tf32_on_sm_120() {
    let manifest = manifest();
    let base = env_path("SEEON_TEST_ENGINE_BUILD_DIR");
    let onnx: Vec<PathBuf> = ROLES.iter().map(|role| env_path(role.onnx_var)).collect();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_nanos();
    let run = base.join(format!("run-{nanos}"));
    fs::create_dir_all(&base).expect("create SEEON_TEST_ENGINE_BUILD_DIR");
    fs::create_dir(&run).expect("create a fresh engine run directory");
    let gpu = &manifest["gpu"];
    let mut engines = Map::new();
    let _gpu = gpu_lock();
    for (role, onnx) in ROLES.iter().zip(&onnx) {
        let onnx_bytes = read(onnx, role.onnx_var);
        let onnx_sha256 = hex(&sha256(&onnx_bytes));
        assert_eq!(
            onnx_sha256,
            text(&manifest["models"][role.key], "onnx_sha256"),
            "{} is not the recorded model",
            role.onnx_var
        );
        let engine = run.join(role.engine_file);
        let identity = build_engine(&onnx_bytes, &engine, DEVICE, role.input, role.dimensions)
            .unwrap_or_else(|error| panic!("build {} engine: {error:?}", role.key));
        assert!(!identity.tf32_enabled, "{} engine allows TF32", role.key);
        assert_eq!(
            (identity.compute_major, identity.compute_minor),
            (12, 0),
            "{} engine is not sm_120",
            role.key
        );
        let compute_capability = format!("{}.{}", identity.compute_major, identity.compute_minor);
        assert_eq!(compute_capability, text(gpu, "compute_capability"));
        assert!(
            (100_000..110_000).contains(&identity.trt_version),
            "TensorRT 10 required, got {}",
            identity.trt_version
        );
        assert_eq!(identity.device_name, text(gpu, "name"));
        let engine_sha256 = hex(&sha256(&read(&engine, role.engine_file)));
        println!(
            "{}: {} sha256={engine_sha256} trt={} device={}",
            role.key, role.engine_file, identity.trt_version, identity.device_name
        );
        engines.insert(
            role.key.to_owned(),
            json!({
                "engine": role.engine_file,
                "engine_sha256": engine_sha256,
                "onnx_sha256": onnx_sha256,
                "input": role.input.to_str().expect("ASCII input name"),
                "dimensions": role.dimensions,
                "trt_version": identity.trt_version,
                "compute_capability": compute_capability,
                "tf32_enabled": identity.tf32_enabled,
                "device_name": identity.device_name,
            }),
        );
    }
    let identity = json!({"schema": "seeon-engine-identity/v1", "engines": engines});
    let identity = serde_json::to_vec_pretty(&identity).expect("identity JSON");
    fs::write(run.join(IDENTITY_FILE), identity).expect("write engine identity");
    println!("engines: {}", run.display());
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_GPU_FIXTURES and SEEON_TEST_BED_ENGINE"]
fn bed_raw_outputs_match_the_ort_cuda_oracle() {
    let fixtures = Fixtures::open();
    let entries = array(&fixtures.manifest, "bed");
    assert!(!entries.is_empty(), "no bed fixtures");
    let _gpu = gpu_lock();
    let (model, digest) = admitted_engine(&BED, &fixtures.manifest);
    let mut bed = BedGpu::new(model, DEVICE, digest);
    let bytes = (
        BED.dimensions.iter().product::<i32>() as u64 * F32_BYTES,
        (DETECTIONS_VALUES + PROTOS_VALUES) as u64 * F32_BYTES,
    );
    let mut sequence = None;
    let mut gated = 0;
    let mut ordered = Vec::with_capacity(entries.len() * 2);
    for entry in entries {
        let id = text(entry, "id");
        let (width, height, rgb) = fixtures.rgb(text(entry, "frame"));
        let images = fixtures.f32s(&entry["images"]);
        let output0 = fixtures.f32s(&entry["output0"]);
        let output1 = fixtures.f32s(&entry["output1"]);
        let letterbox = letterbox_of(&entry["letterbox"]);
        let raw = bed
            .infer(&rgb, width, height)
            .unwrap_or_else(|error| panic!("bed {id}: {error:?}"));
        check_receipt(&raw.evidence, digest, bytes, &mut sequence);
        assert_eq!(raw.letterbox, letterbox, "bed {id} letterbox");
        let (detections, protos) = (raw.detections.to_vec(), raw.protos.to_vec());
        let report = ordered::compare(
            "bed output0",
            id,
            &detections,
            &output0,
            MAX_ABS,
            Some(f64::from(ORACLE_ULP)),
        );
        println!("{report}");
        ordered.push(report);
        let report = ordered::compare(
            "bed output1",
            id,
            &protos,
            &output1,
            MAX_ABS,
            Some(f64::from(ORACLE_ULP)),
        );
        println!("{report}");
        ordered.push(report);
        let (rows, worst) = compare_rows(&detections, &output0, BED_ROW, true)
            .unwrap_or_else(|error| panic!("bed {id} output0: {error}"));
        let decoded = decode_bed(&detections, &protos, &letterbox, ROWS_FROM_SCORE)
            .unwrap_or_else(|error| panic!("bed {id} decode: {error}"));
        assert_eq!(
            decoded.len(),
            count(entry, "detections"),
            "bed {id} instances"
        );
        compare_beds(&decoded, &golden_beds(&fixtures, entry), false)
            .unwrap_or_else(|error| panic!("bed {id} decision inputs: {error}"));
        let input = max_abs(&format!("bed {id} images"), bed.input(), &images);
        let (proto_worst, proto_over) = spread(&protos, &output1);
        let (matched, unmatched) = top_rows_matched(&detections, &output0, BED_ROW);
        let margin = decoded
            .iter()
            .map(|instance| instance.margin)
            .fold(f64::INFINITY, f64::min);
        println!(
            "bed {id}: images={input:e} gated_rows={rows} gated_max_abs={worst:e} \
             instances={} mask_margin={margin:e} protos_max_abs={proto_worst:e} \
             protos_over_1e-4={proto_over}/{} top_rows_within_1e-2={matched}/{} \
             unmatched_max_score={unmatched:?} recognizer_rows_below_gate={}/{}",
            decoded.len(),
            protos.len(),
            detections.len() / BED_ROW,
            ungated_bed_rows(&detections),
            ungated_bed_rows(&output0),
        );
        gated += rows;
    }
    assert!(gated > 0, "no bed row reaches the gate");
    assert!(
        ordered.iter().all(ordered::Report::passes),
        "ordered raw bed outputs exceeded tolerance:\n{}",
        ordered
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_GPU_FIXTURES and SEEON_TEST_STORED_POSE_ENGINE"]
fn stored_pose_raw_output_matches_the_ort_cuda_oracle() {
    let fixtures = Fixtures::open();
    let entries = array(&fixtures.manifest, "stored_pose");
    assert!(!entries.is_empty(), "no stored pose fixtures");
    let _gpu = gpu_lock();
    let (mut model, digest) = admitted_engine(&STORED_POSE, &fixtures.manifest);
    let mut tensor = StoredPoseTensor::default();
    let mut output = vec![0.0_f32; OUTPUT_VALUES];
    let bytes = (
        POSE_DIMENSIONS.iter().product::<i32>() as u64 * F32_BYTES,
        OUTPUT_VALUES as u64 * F32_BYTES,
    );
    let mut sequence = None;
    let mut gated = 0;
    let mut ordered = Vec::with_capacity(entries.len());
    for entry in entries {
        let id = text(entry, "id");
        let (width, height, rgb) = fixtures.rgb(text(entry, "frame"));
        let images = fixtures.f32s(&entry["images"]);
        let output0 = fixtures.f32s(&entry["output0"]);
        let input = tensor
            .preprocess(&rgb, width, height)
            .unwrap_or_else(|error| panic!("stored pose {id} preprocess: {error:?}"))
            .to_vec();
        let input_diff = max_abs(&format!("stored pose {id} images"), &input, &images);
        let before = model.metrics().expect("pose metrics before");
        let shapes = model
            .run(
                TensorInput {
                    name: c"images",
                    values: &input,
                    dimensions: &POSE_DIMENSIONS,
                },
                &mut [TensorOutput {
                    name: c"output0",
                    values: &mut output,
                }],
            )
            .unwrap_or_else(|error| panic!("stored pose {id}: {error:?}"));
        let after = model.metrics().expect("pose metrics after");
        assert_eq!(shapes.len(), 1, "one pose output");
        assert_eq!(shapes[0].rank, 3, "pose output rank");
        assert_eq!(shapes[0].dimensions[..3], [1, 300, 57], "pose output shape");
        let evidence =
            AcceleratorEvidence::from_delta(&before, &after, DEVICE, digest, Precision::Fp32)
                .unwrap_or_else(|error| panic!("stored pose {id} receipt: {error:?}"));
        check_receipt(&evidence, digest, bytes, &mut sequence);
        let report = ordered::compare("stored pose output0", id, &output, &output0, MAX_ABS, None);
        println!("{report}");
        ordered.push(report);
        let (rows, worst) = compare_rows(&output, &output0, POSE_ROW, false)
            .unwrap_or_else(|error| panic!("stored pose {id} output0: {error}"));
        assert_eq!(
            rows,
            count(entry, "detections"),
            "stored pose {id} gated rows"
        );
        let (matched, unmatched) = top_rows_matched(&output, &output0, POSE_ROW);
        println!(
            "stored pose {id}: images={input_diff:e} gated_rows={rows} gated_max_abs={worst:e} \
             top_rows_within_1e-2={matched}/{} unmatched_max_score={unmatched:?}",
            output.len() / POSE_ROW
        );
        gated += rows;
    }
    assert!(gated > 0, "no pose row reaches the gate");
    assert!(
        ordered.iter().all(ordered::Report::passes),
        "ordered raw stored pose outputs exceeded tolerance:\n{}",
        ordered
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
#[ignore = "requires SEEON_TEST_GPU_FIXTURES"]
fn bed_decode_transcription_reproduces_the_recorded_python_decode() {
    let fixtures = Fixtures::open();
    let entries = array(&fixtures.manifest, "bed");
    assert!(!entries.is_empty(), "no bed fixtures");
    let (mut instances, mut margin) = (0, f64::INFINITY);
    for entry in entries {
        let id = text(entry, "id");
        let decoded = decode_bed(
            &fixtures.f32s(&entry["output0"]),
            &fixtures.f32s(&entry["output1"]),
            &letterbox_of(&entry["letterbox"]),
            ROWS_FROM_SCORE,
        )
        .unwrap_or_else(|error| panic!("bed {id} decode: {error}"));
        compare_beds(&decoded, &golden_beds(&fixtures, entry), true)
            .unwrap_or_else(|error| panic!("bed {id} transcription: {error}"));
        instances += decoded.len();
        margin = decoded
            .iter()
            .map(|instance| instance.margin)
            .fold(margin, f64::min);
    }
    assert!(instances > 0, "no recorded bed instance");
    println!("bed transcription: instances={instances} mask_margin={margin:e}");
}

/// Two gated bed rows and one below the gate; at x1 = 700 and x2 = 1100.5 two
/// float32 ULP are wider than 1e-4, and at the coefficient 0.3 the 1e-4 floor is.
fn gate_rows() -> Vec<f32> {
    [0.9_f32, 0.5, 0.01]
        .iter()
        .flat_map(|score| {
            let mut row = vec![0.0_f32; BED_ROW];
            row[..4].copy_from_slice(&[700.0, 100.0, 1100.5, 400.0]);
            row[SCORE_COLUMN] = *score;
            row[CLASS_COLUMN] = BED_CLASS as f32;
            row[6] = 0.3;
            row
        })
        .collect()
}

fn with(values: &[f32], row: usize, column: usize, change: impl Fn(f32) -> f32) -> Vec<f32> {
    let mut changed = values.to_vec();
    let index = row * BED_ROW + column;
    changed[index] = change(changed[index]);
    changed
}

fn ulps(value: f32, steps: u32) -> f32 {
    f32::from_bits(value.to_bits() + steps)
}

fn rejects<T: std::fmt::Debug>(result: Result<T, String>, reason: &str, case: &str) {
    match result {
        Err(error) => assert!(error.contains(reason), "{case}: {error}"),
        Ok(value) => panic!("{case} was accepted: {value:?}"),
    }
}

#[test]
fn gated_rows_admit_two_oracle_ulp_and_nothing_wider() {
    let oracle = gate_rows();
    assert_eq!(compare_rows(&oracle, &oracle, BED_ROW, true), Ok((2, 0.0)));
    for (row, column, change, case) in [
        (1, 6, 0.3 + 9.9e-5, "9.9e-5 at 0.3"),
        (0, 0, ulps(700.0, 2), "2 ULP at 700"),
        (0, 2, ulps(1100.5, 2), "2 ULP at 1100.5"),
    ] {
        let actual = with(&oracle, row, column, |_| change);
        assert!(
            compare_rows(&actual, &oracle, BED_ROW, true).is_ok(),
            "{case}"
        );
    }
    let below_gate = with(&oracle, 2, 0, |value| value + 1.0);
    assert!(compare_rows(&below_gate, &oracle, BED_ROW, false).is_ok());
    // Each rejection reports its limit: the floor below 512, two ULP above it,
    // and the plain floor at any magnitude for pose rows.
    for (row, column, change, ulp, limit, case) in [
        (1, 6, 0.3 + 1.01e-4, true, "1e-4", "1.01e-4 at 0.3"),
        (1, 6, 0.3 + 1e-3, true, "1e-4", "1e-3 at 0.3"),
        (0, 0, ulps(700.0, 3), true, "1.220703125e-4", "3 ULP at 700"),
        (
            0,
            2,
            ulps(1100.5, 3),
            true,
            "2.44140625e-4",
            "3 ULP at 1100.5",
        ),
        (0, 0, ulps(700.0, 2), false, "1e-4", "pose 2 ULP at 700"),
        (0, 2, ulps(1100.5, 1), false, "1e-4", "pose 1 ULP at 1100.5"),
    ] {
        let actual = with(&oracle, row, column, |_| change);
        match compare_rows(&actual, &oracle, BED_ROW, ulp) {
            Err(error) => assert!(
                error.starts_with(&format!("gated row {row} column {column}: "))
                    && error.ends_with(&format!(" > {limit}")),
                "{case}: {error}"
            ),
            Ok(value) => panic!("{case} was accepted: {value:?}"),
        }
    }
    rejects(
        compare_rows(&with(&oracle, 1, 6, |_| f32::NAN), &oracle, BED_ROW, true),
        "not finite",
        "NaN",
    );
}

#[test]
fn gated_rows_must_agree_in_count() {
    let oracle = gate_rows();
    for (row, score, case) in [(1, 0.01, "dropped gated row"), (2, 0.3, "added gated row")] {
        let actual = with(&oracle, row, SCORE_COLUMN, |_| score);
        rejects(
            compare_rows(&actual, &oracle, BED_ROW, true),
            "gated rows",
            case,
        );
    }
}

#[test]
fn gated_rows_pair_in_score_order() {
    let oracle = gate_rows();
    let swapped = [
        &oracle[BED_ROW..2 * BED_ROW],
        &oracle[..BED_ROW],
        &oracle[2 * BED_ROW..],
    ]
    .concat();
    assert_eq!(compare_rows(&swapped, &oracle, BED_ROW, true), Ok((2, 0.0)));
}

#[test]
fn a_score_crossing_the_gate_on_one_side_fails() {
    let at_gate = with(&gate_rows(), 1, SCORE_COLUMN, |_| 0.25);
    let below = with(&at_gate, 1, SCORE_COLUMN, |value| {
        f32::from_bits(value.to_bits() - 1)
    });
    rejects(
        compare_rows(&below, &at_gate, BED_ROW, true),
        "gated rows",
        "actual below",
    );
    rejects(
        compare_rows(&at_gate, &below, BED_ROW, true),
        "gated rows",
        "oracle below",
    );
}

#[test]
fn bed_decision_inputs_must_match_exactly() {
    let instance = |score: f32, bbox: [i64; 4]| BedInstance {
        bbox,
        score,
        mask: vec![0, 1, 1, 0, 1, 1],
        polygons: vec![vec![[0, 0], [2, 0], [2, 1]], vec![[0, 0], [2, 1]]],
        margin: f64::NAN,
    };
    let golden = vec![instance(0.6, [1, 2, 3, 4]), instance(0.9, [5, 6, 7, 8])];
    let reordered: Vec<BedInstance> = golden.iter().rev().cloned().collect();
    assert_eq!(compare_beds(&reordered, &golden, true), Ok(()));
    let changed = |change: &dyn Fn(&mut BedInstance)| {
        let mut actual = golden.clone();
        change(&mut actual[1]);
        actual
    };
    let flipped = changed(&|instance| instance.mask[2] ^= 1);
    rejects(
        compare_beds(&flipped, &golden, false),
        "1 pixels differ",
        "flipped mask pixel",
    );
    let moved = changed(&|instance| instance.bbox[2] += 1);
    rejects(compare_beds(&moved, &golden, false), "box", "moved box");
    let polygon = changed(&|instance| instance.polygons[1][1] = [2, 0]);
    rejects(
        compare_beds(&polygon, &golden, false),
        "polygons",
        "moved vertex",
    );
    let score = changed(&|instance| instance.score = ulps(0.9, 1));
    assert_eq!(compare_beds(&score, &golden, false), Ok(()));
    rejects(
        compare_beds(&score, &golden, true),
        "score bits",
        "score bits",
    );
    rejects(
        compare_beds(&golden[..1], &golden, false),
        "bed instances",
        "dropped instance",
    );
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_GPU_FIXTURES and SEEON_TEST_FALL_ENGINE"]
fn fall_logits_match_the_ort_cuda_oracle() {
    let fixtures = Fixtures::open();
    let windows = fixtures.f32s(&fixtures.manifest["fall"]["windows"]);
    let logits = fixtures.f32s(&fixtures.manifest["fall"]["logits"]);
    assert!(!logits.is_empty(), "no fall windows");
    assert_eq!(windows.len(), logits.len() * WINDOW_VALUES);
    let _gpu = gpu_lock();
    let (model, digest) = admitted_engine(&FALL, &fixtures.manifest);
    let mut fall = FallGpu::new(model, DEVICE, digest);
    let bytes = (WINDOW_VALUES as u64 * F32_BYTES, F32_BYTES);
    let mut sequence = None;
    let mut worst = 0.0_f64;
    for (index, (window, expected)) in windows.chunks_exact(WINDOW_VALUES).zip(&logits).enumerate()
    {
        let score = fall
            .score(&window_rows(window))
            .unwrap_or_else(|error| panic!("fall window {index}: {error:?}"));
        let label = format!("fall window {index} logit");
        worst = worst.max(max_abs(&label, &[score.logit], &[*expected]));
        check_receipt(&score.evidence, digest, bytes, &mut sequence);
    }
    println!("fall: windows={} logit max_abs={worst:e}", logits.len());
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_GPU_FIXTURES, SEEON_TEST_BED_ENGINE and SEEON_TEST_FALL_ENGINE"]
fn owners_latch_after_a_receipt_for_another_device() {
    // Each model runs on DEVICE while its owner is bound to another ordinal, so
    // the call succeeds but its receipt must be refused and the owner latched.
    let fixtures = Fixtures::open();
    let expected = EvidenceError::DeviceMismatch;
    let _gpu = gpu_lock();

    let (model, digest) = admitted_engine(&FALL, &fixtures.manifest);
    let windows = fixtures.f32s(&fixtures.manifest["fall"]["windows"]);
    let rows = window_rows(&windows[..WINDOW_VALUES]);
    let mut fall = FallGpu::new(model, DEVICE + 1, digest);
    let before = fall.metrics().expect("fall metrics are readable");
    assert_eq!(
        fall.score(&rows).err(),
        Some(FallGpuError::Evidence(expected))
    );
    let latched = fall.metrics().expect("fall metrics stay readable");
    let ran = (
        latched.attempted - before.attempted,
        latched.succeeded - before.succeeded,
    );
    assert_eq!(ran, (1, 1), "fall ran exactly once");
    assert_eq!(fall.score(&rows).err(), Some(FallGpuError::Poisoned));
    assert_eq!(
        fall.metrics().unwrap(),
        latched,
        "poisoned fall owner ran again"
    );

    let (model, digest) = admitted_engine(&BED, &fixtures.manifest);
    let entry = &array(&fixtures.manifest, "bed")[0];
    let (width, height, rgb) = fixtures.rgb(text(entry, "frame"));
    let mut bed = BedGpu::new(model, DEVICE + 1, digest);
    let before = bed.metrics().expect("bed metrics are readable");
    assert_eq!(
        bed.infer(&rgb, width, height).err(),
        Some(BedGpuError::Evidence(expected))
    );
    let latched = bed.metrics().expect("bed metrics stay readable");
    let ran = (
        latched.attempted - before.attempted,
        latched.succeeded - before.succeeded,
    );
    assert_eq!(ran, (1, 1), "bed ran exactly once");
    assert_eq!(
        bed.infer(&rgb, width, height).err(),
        Some(BedGpuError::Poisoned)
    );
    assert_eq!(
        bed.metrics().unwrap(),
        latched,
        "poisoned bed owner ran again"
    );
}

#[test]
#[ignore = "requires SEEON_TEST_GPU_FIXTURES and the native libav clip decoder"]
fn clip_frames_match_the_pyav_oracle() {
    let fixtures = Fixtures::open();
    let clip = &fixtures.manifest["clip"];
    let relative = text(clip, "path");
    fixtures.bytes(relative);
    let mut decoder = ClipDecoder::open(&fixtures.root.join(relative))
        .unwrap_or_else(|error| panic!("open {relative}: {error:?}"));
    assert_eq!(
        (decoder.width() as usize, decoder.height() as usize),
        (count(clip, "width"), count(clip, "height"))
    );
    let frames = array(clip, "frames");
    assert!(!frames.is_empty(), "no clip frames");
    for (index, expected) in frames.iter().enumerate() {
        let frame = decoder
            .next_frame()
            .unwrap_or_else(|error| panic!("frame {index}: {error:?}"))
            .unwrap_or_else(|| panic!("clip ended before frame {index}"));
        assert_eq!(frame.pts, expected["pts"].as_i64(), "frame {index} pts");
        assert_eq!(
            hex(&sha256(&frame.rgb)),
            text(expected, "rgb_sha256"),
            "frame {index} rgb"
        );
    }
    assert_eq!(decoder.next_frame().expect("end of clip"), None);
    println!(
        "clip: frames={} decoder={}",
        frames.len(),
        decoder.identity()
    );
}

const STRIDE: u64 = 5;
const TTL: u64 = 45;

/// Mirror of `worker/domains/fall/classifier.py::FallWindowClassifier`. The Rust
/// core has no window classifier yet, so the replay reproduces its buffering
/// and hands each due window to the GPU instead of a model runner.
#[derive(Default)]
struct WindowClassifier {
    counter: u64,
    buffers: BTreeMap<u64, VecDeque<PoseBbox56Row>>,
    last_rows: BTreeMap<u64, PoseBbox56Row>,
    last_seen: BTreeMap<u64, u64>,
    reconnect: BTreeSet<u64>,
    missing: BTreeMap<u64, DecisionTraceMissingReason>,
}

impl WindowClassifier {
    /// Buffers this observation and returns the windows due now, in track order.
    fn update(
        &mut self,
        rows: &BTreeMap<u64, PoseBbox56Row>,
        live: &[u64],
    ) -> Vec<(u64, Vec<PoseBbox56Row>)> {
        self.missing.clear();
        self.counter += 1;
        let live_ids: BTreeSet<u64> = live.iter().copied().collect();
        for &id in live {
            let row = match rows
                .get(&id)
                .filter(|row| row.iter().all(|v| v.is_finite()))
            {
                Some(row) => {
                    self.last_rows.insert(id, *row);
                    *row
                }
                None => self.last_rows.get(&id).copied().unwrap_or(ZERO_ROW),
            };
            let reconnect = &mut self.reconnect;
            let buffer = self.buffers.entry(id).or_insert_with(|| {
                let mut buffer = VecDeque::with_capacity(FALL_WINDOW_FRAMES);
                if reconnect.remove(&id) {
                    buffer.extend(std::iter::repeat_n(ZERO_ROW, FALL_WINDOW_FRAMES - 1));
                }
                buffer
            });
            push_bounded(buffer, row);
            self.last_seen.insert(id, self.counter);
        }
        let buffered: Vec<u64> = self.buffers.keys().copied().collect();
        for id in buffered.into_iter().filter(|id| !live_ids.contains(id)) {
            if self.counter - self.last_seen[&id] >= TTL {
                self.buffers.remove(&id);
                self.last_rows.remove(&id);
                self.last_seen.remove(&id);
                self.reconnect.insert(id);
            } else {
                let row = self.last_rows.get(&id).copied().unwrap_or(ZERO_ROW);
                push_bounded(self.buffers.get_mut(&id).expect("buffered"), row);
            }
        }
        if !self.counter.is_multiple_of(STRIDE) {
            for id in live_ids {
                self.missing
                    .insert(id, DecisionTraceMissingReason::ClassifierStrideNotDue);
            }
            return Vec::new();
        }
        let mut due = Vec::new();
        for id in live_ids {
            match self.buffers.get(&id) {
                Some(buffer) if buffer.len() == FALL_WINDOW_FRAMES => {
                    due.push((id, buffer.iter().copied().collect()));
                }
                _ => {
                    self.missing
                        .insert(id, DecisionTraceMissingReason::ClassifierWarmup);
                }
            }
        }
        due
    }
}

fn push_bounded(buffer: &mut VecDeque<PoseBbox56Row>, row: PoseBbox56Row) {
    if buffer.len() == FALL_WINDOW_FRAMES {
        buffer.pop_front();
    }
    buffer.push_back(row);
}

fn hex_u64(value: &Value) -> u64 {
    u64::from_str_radix(value.as_str().expect("hex string"), 16).expect("u64 bits")
}

fn hex_f64(value: &Value) -> f64 {
    f64::from_bits(hex_u64(value))
}

fn hex_f64s(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(hex_f64)
        .collect()
}

struct Packet {
    frame: i64,
    time_sec: f64,
    pts_ns: i64,
    live: Vec<u64>,
    bbox: Option<Vec<f64>>,
    keypoints: Option<Vec<Vec<f64>>>,
}

struct Prediction {
    frame: i64,
    window_sha256: String,
    logit: f32,
    temperature: f64,
    fall_transition: f64,
}

struct Scene {
    width: i64,
    height: i64,
    track_id: u64,
    packets: Vec<Packet>,
    predictions: Vec<Prediction>,
}

impl Scene {
    fn parse(packets: &Value) -> Self {
        let frames = array(packets, "frames").iter().map(|frame| Packet {
            frame: frame["frame"].as_i64().expect("frame"),
            time_sec: hex_f64(&frame["time_sec"]),
            pts_ns: frame["pts_ns"].as_i64().expect("pts_ns"),
            live: array(frame, "live")
                .iter()
                .map(|id| id.as_u64().expect("track id"))
                .collect(),
            bbox: (!frame["bbox"].is_null()).then(|| hex_f64s(&frame["bbox"])),
            keypoints: (!frame["keypoints"].is_null())
                .then(|| array(frame, "keypoints").iter().map(hex_f64s).collect()),
        });
        let predictions = array(packets, "predictions").iter().map(|prediction| {
            let logit = u32::try_from(hex_u64(&prediction["logit"])).expect("f32 bits");
            Prediction {
                frame: prediction["frame"].as_i64().expect("prediction frame"),
                window_sha256: text(prediction, "window_sha256").to_owned(),
                logit: f32::from_bits(logit),
                temperature: hex_f64(&prediction["temperature"]),
                fall_transition: hex_f64(&prediction["fall_transition"]),
            }
        });
        Self {
            width: packets["frame_width"].as_i64().expect("frame_width"),
            height: packets["frame_height"].as_i64().expect("frame_height"),
            track_id: packets["track_id"].as_u64().expect("track_id"),
            packets: frames.collect(),
            predictions: predictions.collect(),
        }
    }
}

/// Scores due windows on the GPU, checking each against the next recorded
/// Python prediction, and converts logits as the Python fall runner does.
struct Scorer<'a> {
    fall: &'a mut FallGpu,
    digest: EngineDigest,
    sequence: &'a mut Option<u64>,
    predictions: std::slice::Iter<'a, Prediction>,
    used: usize,
}

impl Scorer<'_> {
    fn score(
        &mut self,
        frame: i64,
        due: Vec<(u64, Vec<PoseBbox56Row>)>,
    ) -> BTreeMap<u64, FallProbabilities> {
        let mut probabilities = BTreeMap::new();
        for (id, window) in due {
            let recorded = self
                .predictions
                .next()
                .unwrap_or_else(|| panic!("unrecorded prediction at frame {frame}"));
            self.used += 1;
            assert_eq!(recorded.frame, frame, "prediction frame");
            let bytes: Vec<u8> = window
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(
                hex(&sha256(&bytes)),
                recorded.window_sha256,
                "frame {frame} window"
            );
            let score = self
                .fall
                .score(&window)
                .unwrap_or_else(|error| panic!("frame {frame} fall score: {error:?}"));
            check_receipt(&score.evidence, self.digest, (6720, 4), self.sequence);
            let label = format!("frame {frame} logit");
            max_abs(&label, &[score.logit], &[recorded.logit]);
            let recorded_transition = sigmoid(recorded.logit, recorded.temperature);
            assert!(
                (f64::from(recorded_transition) - recorded.fall_transition).abs() <= MAX_ABS,
                "frame {frame} recorded fall_transition"
            );
            let transition = f64::from(sigmoid(score.logit, recorded.temperature));
            let value = FallProbabilities::new(1.0 - transition, transition, 0.0)
                .unwrap_or_else(|error| panic!("frame {frame} probabilities: {error:?}"));
            probabilities.insert(id, value);
        }
        probabilities
    }
}

/// NumPy's NEP 50 keeps the float32 logit's dtype through the Python scaling.
fn sigmoid(logit: f32, temperature: f64) -> f32 {
    1.0_f32 / (1.0_f32 + (-logit / temperature as f32).exp())
}

fn render(
    lines: &mut Vec<String>,
    frame: i64,
    evaluated: bool,
    events: &[BusinessEvent],
    snapshots: &[DecisionTraceSnapshot],
) {
    let optional = |value: Option<u64>| value.map_or_else(|| "-".to_owned(), |v| v.to_string());
    lines.push(format!("F\t{frame}\t{}", u8::from(evaluated)));
    for event in events {
        lines.push(format!(
            "E\t{}\t{}\t{}\t{}\t{}\t{:016x}\t{}\t{}\t{}",
            event.domain,
            event.event_type,
            event.identity,
            event.camera_id,
            event.facility_id,
            event.time_sec.to_bits(),
            event
                .probability
                .map_or_else(|| "-".to_owned(), |p| format!("{:016x}", p.to_bits())),
            optional(event.person_id),
            optional(event.bed_id),
        ));
    }
    for snapshot in snapshots {
        let mut values: Vec<_> = snapshot.values().iter().collect();
        values.sort_by_key(|(name, _)| name.as_str());
        let mut missing: Vec<_> = snapshot.missing_values().iter().collect();
        missing.sort_by_key(|(name, _)| name.as_str());
        lines.push(format!(
            "T\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            snapshot.reason,
            snapshot.previous_state,
            snapshot.current_state,
            u8::from(snapshot.triggered),
            optional(snapshot.track_id),
            optional(snapshot.bed_id),
            values.len(),
            missing.len(),
        ));
        for (name, value) in values {
            lines.push(match value {
                NumericTraceValue::Integer(value) => format!("V\t{name}\tI\t{value}"),
                NumericTraceValue::Float(value) => {
                    format!("V\t{name}\tF\t{:016x}", value.get().to_bits())
                }
            });
        }
        for (name, reason) in missing {
            lines.push(format!("M\t{name}\t{reason}"));
        }
    }
}

/// Replays one scene through the mirrored decider glue of
/// `worker/domains/fall/policy.py` and returns the canonical lines, the number
/// of predictions used and the number of business events.
fn replay(scene: &Scene, scorer: &mut Scorer<'_>) -> (Vec<String>, usize) {
    let mut decider = FallPolicyDecider::new(
        "cam",
        "fac",
        "boot",
        "epoch",
        0,
        FallPolicy::default(),
        FallCapacities {
            retained_tracks: 64,
            generation_identities: 64,
            episodes: 64,
            vote_window: 5,
        },
    )
    .expect("fall decider");
    let mut resampler = PtsResampler::default();
    let mut classifier = WindowClassifier::default();
    let mut last_pts = None;
    let mut lines = Vec::new();
    let mut event_count = 0;
    for packet in &scene.packets {
        let mut rows = BTreeMap::new();
        if let (Some(bbox), Some(keypoints)) = (&packet.bbox, &packet.keypoints) {
            let row = pose_bbox56_row(keypoints, Some(bbox), scene.width, scene.height);
            rows.insert(scene.track_id, row);
        }
        if last_pts.is_some_and(|last| packet.pts_ns < last) {
            resampler = PtsResampler::default();
            classifier = WindowClassifier::default();
        }
        last_pts = Some(packet.pts_ns);
        let resampled = resampler
            .push(packet.pts_ns, rows)
            .unwrap_or_else(|error| panic!("frame {} resample: {error:?}", packet.frame));
        let events = if resampled.is_empty() {
            decider.coast()
        } else {
            let mut probabilities = BTreeMap::new();
            let mut missing = BTreeMap::new();
            for row in resampled {
                if row.valid != 0 {
                    let value = row.value.expect("a valid row carries a value");
                    let due = classifier.update(&value, &packet.live);
                    probabilities = scorer.score(packet.frame, due);
                    missing = classifier.missing.clone();
                } else {
                    let zero_rows = packet.live.iter().map(|id| (*id, ZERO_ROW)).collect();
                    let due = classifier.update(&zero_rows, &packet.live);
                    scorer.score(packet.frame, due);
                }
            }
            decider.update(
                packet.frame,
                packet.time_sec,
                &probabilities,
                packet.live.iter().copied(),
                Some(&missing),
            )
        }
        .unwrap_or_else(|error| panic!("frame {} decide: {error:?}", packet.frame));
        event_count += events.len();
        render(
            &mut lines,
            packet.frame,
            decider.last_update_evaluated(),
            &events,
            decider.last_trace_snapshots(),
        );
    }
    (lines, event_count)
}

/// Exact, except float trace values and event probabilities within `MAX_ABS`.
fn lines_match(actual: &str, expected: &str) -> bool {
    if actual == expected {
        return true;
    }
    let actual: Vec<&str> = actual.split('\t').collect();
    let expected: Vec<&str> = expected.split('\t').collect();
    let float_field = match (expected[0], expected.len()) {
        ("V", 4) if expected[2] == "F" => 3,
        ("E", 10) => 7,
        _ => return false,
    };
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(&expected)
            .enumerate()
            .all(|(index, (a, e))| {
                if index != float_field || a == e {
                    return a == e;
                }
                match (u64::from_str_radix(a, 16), u64::from_str_radix(e, 16)) {
                    (Ok(a), Ok(e)) => (f64::from_bits(a) - f64::from_bits(e)).abs() <= MAX_ABS,
                    _ => false,
                }
            })
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_GPU_FIXTURES and SEEON_TEST_FALL_ENGINE"]
fn fall_decider_replay_matches_the_python_projection() {
    let fixtures = Fixtures::open();
    let scenes = array(&fixtures.manifest, "replay");
    assert!(!scenes.is_empty(), "no replay scenes");
    let _gpu = gpu_lock();
    let (model, digest) = admitted_engine(&FALL, &fixtures.manifest);
    let mut fall = FallGpu::new(model, DEVICE, digest);
    let mut sequence = None;
    for entry in scenes {
        let name = text(entry, "scene");
        let packets: Value = serde_json::from_slice(&fixtures.bytes(text(entry, "packets")))
            .unwrap_or_else(|error| panic!("{name} packets: {error}"));
        let expected = String::from_utf8(fixtures.bytes(text(entry, "expected")))
            .unwrap_or_else(|error| panic!("{name} expected lines: {error}"));
        let scene = Scene::parse(&packets);
        assert_eq!(scene.packets.len(), count(entry, "frames"), "{name} frames");
        assert_eq!(
            scene.predictions.len(),
            count(entry, "predictions"),
            "{name} predictions"
        );
        let mut scorer = Scorer {
            fall: &mut fall,
            digest,
            sequence: &mut sequence,
            predictions: scene.predictions.iter(),
            used: 0,
        };
        let (lines, events) = replay(&scene, &mut scorer);
        let used = scorer.used;
        let expected: Vec<&str> = expected.lines().collect();
        let mut mismatches = lines.len().abs_diff(expected.len());
        for (index, (actual, expected)) in lines.iter().zip(&expected).enumerate() {
            if !lines_match(actual, expected) {
                if mismatches < 10 {
                    eprintln!(
                        "{name} line {}: expected {expected:?}, got {actual:?}",
                        index + 1
                    );
                }
                mismatches += 1;
            }
        }
        println!(
            "replay {name}: frames={} lines={} predictions={used} events={events} mismatches={mismatches}",
            scene.packets.len(),
            lines.len()
        );
        assert_eq!(mismatches, 0, "{name} replay mismatches");
        assert_eq!(used, scene.predictions.len(), "{name} unused predictions");
        assert_eq!(events, count(entry, "events"), "{name} events");
    }
}
