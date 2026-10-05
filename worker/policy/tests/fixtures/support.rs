//! Fixture loading for `*_fixture.rs`. Fixtures are recorded from the frozen
//! Python oracle; nothing here derives an expected value from Rust output.
//! Gate order: manifest authority, fixture sha256, then the caller's inputs.
#![allow(dead_code)]

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// The oracle environment every fixture must have been recorded under.
const ENVIRONMENT: [(&str, &str); 4] = [
    ("python", "3.12.3"),
    ("numpy", "2.4.6"),
    ("opencv", "4.13.0"),
    ("numpy_exp_f32_dispatch", "X86_V3"),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub fn sha256_u32(values: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex(&hasher.finalize())
}

pub fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex(&hasher.finalize())
}

pub fn sha256_u8(values: &[u8]) -> String {
    sha256(values)
}

fn manifest() -> Value {
    let path = root().join("manifest.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("read manifest: {error}"));
    let manifest: Value = serde_json::from_slice(&bytes).expect("manifest is JSON");
    for (key, expected) in ENVIRONMENT {
        assert_eq!(
            manifest["environment"][key].as_str(),
            Some(expected),
            "manifest environment {key} is not the frozen oracle"
        );
    }
    let recorder = text(&manifest["recorder_sha256"]);
    assert_eq!(recorder.len(), 64, "manifest recorder sha256");
    assert_eq!(
        text(&manifest["recorder_commit"]),
        format!("working-tree:{recorder}"),
        "manifest recorder commit"
    );
    manifest
}

/// Loads `relative` only after its bytes match the manifest and the named
/// oracle source matches the frozen sha256 the Python parity test pinned.
pub fn load(relative: &str, oracle_source: &str, oracle_sha256: &str) -> Value {
    load_sources(relative, &[(oracle_source, oracle_sha256)])
}

/// `load` for a fixture whose oracle spans several frozen source files.
pub fn load_sources(relative: &str, sources: &[(&str, &str)]) -> Value {
    let manifest = manifest();
    for &(oracle_source, oracle_sha256) in sources {
        assert_eq!(
            manifest["oracle_sources_sha256"][oracle_source].as_str(),
            Some(oracle_sha256),
            "manifest oracle source {oracle_source} is not the frozen authority"
        );
    }
    let expected = manifest["files_sha256"][relative]
        .as_str()
        .unwrap_or_else(|| panic!("{relative} is absent from the manifest"));
    let bytes = std::fs::read(root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"));
    assert_eq!(sha256(&bytes), expected, "{relative} differs from manifest");
    serde_json::from_slice(&bytes).unwrap_or_else(|error| panic!("{relative} JSON: {error}"))
}

pub fn text(value: &Value) -> &str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("expected a string, got {value}"))
}

pub fn array(value: &Value) -> &[Value] {
    value
        .as_array()
        .unwrap_or_else(|| panic!("expected an array, got {value}"))
}

pub fn int(value: &Value) -> i64 {
    value
        .as_i64()
        .unwrap_or_else(|| panic!("expected an i64, got {value}"))
}

pub fn uint(value: &Value) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("expected a u64, got {value}"))
}

pub fn usize_of(value: &Value) -> usize {
    usize::try_from(uint(value)).expect("count fits usize")
}

pub fn u32_of(value: &Value) -> u32 {
    u32::try_from(uint(value)).unwrap_or_else(|_| panic!("expected a u32, got {value}"))
}

pub fn u32s(value: &Value) -> Vec<u32> {
    array(value).iter().map(u32_of).collect()
}

pub fn i64s(value: &Value) -> Vec<i64> {
    array(value).iter().map(int).collect()
}

/// A float32 stored as its IEEE bit pattern.
pub fn f32_bits(value: &Value) -> f32 {
    f32::from_bits(u32_of(value))
}

/// A float64 stored as its IEEE bit pattern.
pub fn f64_bits(value: &Value) -> f64 {
    f64::from_bits(uint(value))
}

pub fn f32s(value: &Value) -> Vec<f32> {
    array(value).iter().map(f32_bits).collect()
}

pub fn f64s(value: &Value) -> Vec<f64> {
    array(value).iter().map(f64_bits).collect()
}

/// `[[x, y], ...]` integer vertices.
pub fn points_of(value: &Value) -> Vec<[i64; 2]> {
    array(value)
        .iter()
        .map(|point| {
            let pair = array(point);
            assert_eq!(pair.len(), 2, "point arity");
            [int(&pair[0]), int(&pair[1])]
        })
        .collect()
}

/// sha256 of `struct.pack("<qq", x, y)` over every point.
pub fn points_sha256(points: &[[i64; 2]]) -> String {
    let mut hasher = Sha256::new();
    for [x, y] in points {
        hasher.update(x.to_le_bytes());
        hasher.update(y.to_le_bytes());
    }
    hex(&hasher.finalize())
}

/// NumPy PCG64 (XSL-RR 128/64) from a recorded `bit_generator.state`.
pub struct Pcg64 {
    state: u128,
    increment: u128,
    spare: Option<u32>,
}

impl Pcg64 {
    const MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

    /// `pcg64_state` and `pcg64_inc` are decimal strings with no buffered uint32.
    pub fn recorded(case: &Value) -> Self {
        let parse = |key: &str| -> u128 { text(&case[key]).parse().expect("u128 decimal") };
        Self {
            state: parse("pcg64_state"),
            increment: parse("pcg64_inc"),
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.increment);
        let folded = ((self.state >> 64) as u64) ^ (self.state as u64);
        folded.rotate_right((self.state >> 122) as u32)
    }

    /// The buffered 32-bit output: the low half of a 64-bit draw, then the high half.
    pub fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let value = self.next_u64();
        self.spare = Some((value >> 32) as u32);
        value as u32
    }

    /// `Generator.random()`: the top 53 bits of one 64-bit draw, scaled by 2^-53.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// Every element's exact bits; the first difference names its index and input.
pub fn assert_bits(context: &str, inputs: &[u32], actual: &[u32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len(), "{context}: result count");
    if let Some(index) = (0..actual.len()).find(|&index| actual[index] != expected[index]) {
        panic!(
            "{context}: index {index} input {:#010x} actual {:#010x} expected {:#010x}",
            inputs[index], actual[index], expected[index]
        );
    }
}

/// A large result as the oracle's count, deterministic samples and sha256.
pub fn assert_bulk_u32(context: &str, inputs: &[u32], actual: &[u32], expected: &Value) {
    assert_eq!(
        actual.len(),
        usize_of(&expected["count"]),
        "{context}: result count"
    );
    for sample in array(&expected["samples"]) {
        let index = usize_of(&sample[0]);
        let bits = u32_of(&sample[1]);
        assert_eq!(
            actual[index], bits,
            "{context}: sample {index} input {:#010x}",
            inputs[index]
        );
    }
    assert_eq!(
        sha256_u32(actual),
        text(&expected["sha256_le_u32"]),
        "{context}: result sha256"
    );
}
