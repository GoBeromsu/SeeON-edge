//! T1: `json.rs` against `d/codec-edges.json`, recorded from the four Python
//! canonical serialisers at 030aaf1. Inputs are rebuilt from the golden's own
//! description (hex-float bits, code points, insertion-ordered text), never
//! from its expected output.

use std::fmt;
use std::path::PathBuf;

use seeon_ml_worker::json::{Json, Serialiser, model_selection_digest};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SERIALISERS: [(&str, Serialiser); 4] = [
    ("execution_records", Serialiser::ExecutionRecords),
    ("fetch_models_manifest", Serialiser::FetchModelsManifest),
    ("model_selection", Serialiser::ModelSelection),
    ("provenance", Serialiser::Provenance),
];

fn golden() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire/d/codec-edges.json");
    let text = std::fs::read_to_string(&path).expect("codec-edges golden is readable");
    serde_json::from_str(&text).expect("codec-edges golden is JSON")
}

/// Python `float.hex()` text to bits: `[-]0x1.<hex>p<exp>` (normal) or
/// `[-]0x0.<hex>p-1022` (subnormal or zero), plus `nan` and `[-]inf`.
fn float_from_hex(hex: &str) -> f64 {
    match hex {
        "nan" => return f64::NAN,
        "inf" => return f64::INFINITY,
        "-inf" => return f64::NEG_INFINITY,
        _ => {}
    }
    let (sign, rest) = match hex.strip_prefix('-') {
        Some(rest) => (1u64 << 63, rest),
        None => (0, hex),
    };
    let rest = rest.strip_prefix("0x").expect("hex float has 0x");
    let (mantissa, exponent) = rest.split_once('p').expect("hex float has p");
    let exponent: i64 = exponent.parse().expect("hex float exponent");
    let (lead, fraction) = mantissa.split_once('.').expect("hex float has a point");
    assert!(fraction.len() <= 13, "{hex}: fraction wider than 52 bits");
    let fraction_bits =
        u64::from_str_radix(fraction, 16).expect("hex fraction") << (52 - 4 * fraction.len());
    let bits = match lead {
        "1" => {
            let biased = u64::try_from(exponent + 1023).expect("normal exponent in range");
            (biased << 52) | fraction_bits
        }
        "0" => {
            assert!(
                fraction_bits == 0 || exponent == -1022,
                "{hex}: subnormal exponent"
            );
            fraction_bits
        }
        _ => panic!("{hex}: lead digit"),
    };
    f64::from_bits(sign | bits)
}

fn string_from_code_points(points: &Value) -> String {
    points
        .as_array()
        .expect("code_points is an array")
        .iter()
        .map(|point| {
            let point = point.as_str().expect("code point is text");
            let scalar = u32::from_str_radix(point.strip_prefix("U+").expect("U+ prefix"), 16)
                .expect("code point hex");
            char::from_u32(scalar).expect("code point is a scalar value")
        })
        .collect()
}

/// Parses JSON text keeping object members in document order, as the
/// golden's `input_text_note` asks.
struct Ordered(Json);

impl<'de> Deserialize<'de> for Ordered {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(OrderedVisitor)
    }
}

struct OrderedVisitor;

impl<'de> Visitor<'de> for OrderedVisitor {
    type Value = Ordered;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }
    fn visit_unit<E>(self) -> Result<Ordered, E> {
        Ok(Ordered(Json::Null))
    }
    fn visit_bool<E>(self, value: bool) -> Result<Ordered, E> {
        Ok(Ordered(Json::Bool(value)))
    }
    fn visit_i64<E>(self, value: i64) -> Result<Ordered, E> {
        Ok(Ordered(Json::Int(i128::from(value))))
    }
    fn visit_u64<E>(self, value: u64) -> Result<Ordered, E> {
        Ok(Ordered(Json::Int(i128::from(value))))
    }
    fn visit_f64<E>(self, value: f64) -> Result<Ordered, E> {
        Ok(Ordered(Json::Float(value)))
    }
    fn visit_str<E>(self, value: &str) -> Result<Ordered, E> {
        Ok(Ordered(Json::Str(value.to_owned())))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Ordered, A::Error> {
        let mut items = Vec::new();
        while let Some(Ordered(item)) = seq.next_element()? {
            items.push(item);
        }
        Ok(Ordered(Json::Array(items)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Ordered, A::Error> {
        let mut members = Vec::new();
        while let Some((key, Ordered(member))) = map.next_entry::<String, Ordered>()? {
            members.push((key, member));
        }
        Ok(Ordered(Json::Object(members)))
    }
}

fn case_input(case: &Value) -> Json {
    let input = &case["input"];
    match case["group"].as_str().expect("case group") {
        "float" => Json::Float(float_from_hex(input["hex"].as_str().expect("input.hex"))),
        "string" => Json::Str(string_from_code_points(&input["code_points"])),
        "key-order" => {
            let text = input["input_text"].as_str().expect("input.input_text");
            serde_json::from_str::<Ordered>(text)
                .expect("key-order input text")
                .0
        }
        group => panic!("unknown case group {group}"),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Compares one serialiser's result with the golden; returns the mismatches.
fn check(
    id: &str,
    name: &str,
    serialiser: Serialiser,
    value: &Json,
    expected: &Value,
) -> Vec<String> {
    let mut failures = Vec::new();
    let actual = serialiser.canonical(value);
    match (expected["verdict"].as_str(), &actual) {
        (Some("ok"), Ok(text)) => {
            if Some(text.as_str()) != expected["text"].as_str() {
                failures.push(format!(
                    "{id} {name}: text {text:?} != {}",
                    expected["text"]
                ));
            }
            if Some(text.len() as u64) != expected["utf8_len"].as_u64() {
                failures.push(format!("{id} {name}: utf8_len {}", text.len()));
            }
            if Some(sha256_hex(text.as_bytes()).as_str()) != expected["utf8_sha256"].as_str() {
                failures.push(format!("{id} {name}: utf8_sha256"));
            }
            if let Some(digest) = expected.get("canonical_digest")
                && model_selection_digest(value).ok().as_deref() != digest.as_str()
            {
                failures.push(format!("{id} {name}: canonical_digest"));
            }
        }
        (Some("refused"), Err(_)) => {}
        (verdict, _) => failures.push(format!("{id} {name}: golden {verdict:?}, rust {actual:?}")),
    }
    failures
}

#[test]
fn canonical_json_matches_python_codec_edges() {
    let golden = golden();
    let mut failures = Vec::new();
    let mut counts = [("float", 0usize), ("string", 0), ("key-order", 0)];
    for case in golden["cases"].as_array().expect("cases") {
        let id = case["id"].as_str().expect("case id");
        let group = case["group"].as_str().expect("case group");
        let value = case_input(case);
        for (name, serialiser) in SERIALISERS {
            failures.extend(check(id, name, serialiser, &value, &case["results"][name]));
            let count = counts
                .iter_mut()
                .find(|(known, _)| *known == group)
                .expect("known group");
            count.1 += 1;
        }
    }
    println!("codec-edges checks: {counts:?}");
    assert_eq!(counts, [("float", 60), ("string", 80), ("key-order", 8)]);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
