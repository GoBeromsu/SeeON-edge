//! Oracles: ort_pose_bbox56's calibration contract and Python stdlib hashlib/math.

use std::process::Command;
use std::sync::OnceLock;

use seeon_ml_worker::json::Json;
use seeon_ml_worker::run::calibration::{Calibration, CalibrationError, parse};

const IDENTITY: &str = "pose-bbox56/낙상/é";

fn python_stdout(script: &str, arguments: &[&str]) -> String {
    let output = Command::new("python3")
        .args(["-c", script])
        .args(arguments)
        .output()
        .expect("Python stdlib oracle is required");
    assert!(output.status.success());
    String::from_utf8(output.stdout).expect("UTF-8 oracle output")
}

fn valid() -> Json {
    static DIGEST: OnceLock<String> = OnceLock::new();
    let digest = DIGEST.get_or_init(|| {
        python_stdout(
            "import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode('utf-8')).hexdigest())",
            &[IDENTITY],
        )
        .trim()
        .to_owned()
    });
    Json::from(&serde_json::json!({
        "class_order": ["non_fall", "fall_transition_proxy"],
        "preprocessing_identity_digest": digest,
        "temperature": 1.25,
        "temporal_rule": {"m": 2, "n": 3},
        "threshold": 0.7,
        "promotion_eligible": true
    }))
}

fn replace(document: &mut Json, key: &str, value: Option<Json>) {
    let Json::Object(members) = document else {
        panic!("fixture must be an object");
    };
    members.retain(|(name, _)| name != key);
    if let Some(value) = value {
        members.push((key.to_owned(), value));
    }
}

fn changed(key: &str, value: Option<Json>) -> Json {
    let mut document = valid();
    replace(&mut document, key, value);
    document
}

fn temporal(votes: Json, window: Json) -> Json {
    Json::Object(vec![("m".to_owned(), votes), ("n".to_owned(), window)])
}

#[test]
fn parses_integer_and_float_temperatures_without_f32_narrowing() {
    for (input, expected) in [
        (Json::Int(2), 2.0),
        (Json::Float(1.25), 1.25),
        (Json::Float(f64::MAX), f64::MAX),
        (Json::Float(f64::from_bits(1)), f64::from_bits(1)),
    ] {
        assert_eq!(
            parse(&changed("temperature", Some(input)), IDENTITY, None),
            Ok(Calibration {
                temperature: expected,
                receipt_threshold: Some(0.7),
                transition_votes: 2,
                transition_window: 3,
                promotion_eligible: true,
            })
        );
    }
}

#[test]
fn refuses_nonobject_documents() {
    for document in [
        Json::Null,
        Json::Bool(true),
        Json::Array(vec![]),
        Json::Int(1),
    ] {
        assert_eq!(
            parse(&document, IDENTITY, None),
            Err(CalibrationError::InvalidDocument)
        );
    }
}

#[test]
fn class_order_is_required_and_exact() {
    for order in [
        None,
        Some(Json::Null),
        Some(Json::Str("non_fall,fall_transition_proxy".to_owned())),
        Some(Json::from(&serde_json::json!([]))),
        Some(Json::from(&serde_json::json!(["non_fall"]))),
        Some(Json::from(&serde_json::json!([
            "fall_transition_proxy",
            "non_fall"
        ]))),
        Some(Json::from(&serde_json::json!(["non_fall", "fall"]))),
        Some(Json::from(&serde_json::json!(["non_fall", false]))),
        Some(Json::from(&serde_json::json!([
            "non_fall",
            "fall_transition_proxy",
            "extra"
        ]))),
    ] {
        assert_eq!(
            parse(&changed("class_order", order), IDENTITY, None),
            Err(CalibrationError::ClassOrder)
        );
    }
}

#[test]
fn binds_exact_utf8_identity_to_python_sha256() {
    assert!(parse(&valid(), IDENTITY, None).is_ok());
    // Same human-readable accent, different UTF-8: identities are not normalized.
    assert_eq!(
        parse(&valid(), "pose-bbox56/낙상/e\u{301}", None),
        Err(CalibrationError::PreprocessingIdentity)
    );
    for digest in [None, Some(Json::Null), Some(Json::Str("0".repeat(64)))] {
        assert_eq!(
            parse(
                &changed("preprocessing_identity_digest", digest),
                IDENTITY,
                None
            ),
            Err(CalibrationError::PreprocessingIdentity)
        );
    }
}

#[test]
fn temperature_rejects_missing_nonnumeric_nonfinite_and_nonpositive_values() {
    for value in [
        None,
        Some(Json::Null),
        Some(Json::Bool(true)),
        Some(Json::Bool(false)),
        Some(Json::Str("1.0".to_owned())),
        Some(Json::Array(vec![])),
    ] {
        assert_eq!(
            parse(&changed("temperature", value), IDENTITY, None),
            Err(CalibrationError::TemperatureType)
        );
    }
    for value in [
        Json::Float(f64::NAN),
        Json::Float(f64::INFINITY),
        Json::Float(f64::NEG_INFINITY),
        Json::Float(0.0),
        Json::Float(-0.0),
        Json::Int(0),
        Json::Int(-1),
        Json::Float(-0.5),
    ] {
        assert_eq!(
            parse(&changed("temperature", Some(value)), IDENTITY, None),
            Err(CalibrationError::TemperatureValue)
        );
    }
}

#[test]
fn temporal_rule_requires_both_integers_and_ordered_positive_bounds() {
    for rule in [
        None,
        Some(Json::Null),
        Some(Json::Array(vec![Json::Int(1), Json::Int(2)])),
        Some(Json::Object(vec![])),
        Some(Json::Object(vec![("m".to_owned(), Json::Int(1))])),
        Some(Json::Object(vec![("n".to_owned(), Json::Int(2))])),
        Some(temporal(Json::Bool(true), Json::Int(3))),
        Some(temporal(Json::Int(1), Json::Bool(true))),
        Some(temporal(Json::Float(1.0), Json::Int(3))),
        Some(temporal(Json::Int(1), Json::Float(3.0))),
        Some(temporal(Json::Str("1".to_owned()), Json::Int(3))),
        Some(temporal(Json::Int(1), Json::Null)),
        Some(temporal(Json::Int(0), Json::Int(3))),
        Some(temporal(Json::Int(-1), Json::Int(3))),
        Some(temporal(Json::Int(1), Json::Int(0))),
        Some(temporal(Json::Int(3), Json::Int(2))),
    ] {
        assert_eq!(
            parse(&changed("temporal_rule", rule), IDENTITY, None),
            Err(CalibrationError::TemporalRule)
        );
    }
    for (votes, window) in [(1, 1), (1, i128::MAX), (i128::MAX, i128::MAX)] {
        let parsed = parse(
            &changed(
                "temporal_rule",
                Some(temporal(Json::Int(votes), Json::Int(window))),
            ),
            IDENTITY,
            None,
        )
        .expect("valid integer bounds");
        assert_eq!(
            (parsed.transition_votes, parsed.transition_window),
            (votes, window)
        );
    }
}

#[test]
fn promotion_flag_is_required_boolean_not_truthiness() {
    for flag in [
        None,
        Some(Json::Null),
        Some(Json::Int(1)),
        Some(Json::Int(0)),
        Some(Json::Str("true".to_owned())),
        Some(Json::Float(1.0)),
    ] {
        assert_eq!(
            parse(&changed("promotion_eligible", flag), IDENTITY, None),
            Err(CalibrationError::PromotionEligibility)
        );
    }
    let parsed = parse(
        &changed("promotion_eligible", Some(Json::Bool(false))),
        IDENTITY,
        None,
    )
    .expect("ineligible calibration is usable without receipt selection");
    assert!(!parsed.promotion_eligible);
    assert_eq!(parsed.receipt_threshold, Some(0.7));
}

#[test]
fn absent_or_invalid_optional_threshold_does_not_become_a_default() {
    for threshold in [
        None,
        Some(Json::Null),
        Some(Json::Bool(true)),
        Some(Json::Bool(false)),
        Some(Json::Str("0.7".to_owned())),
        Some(Json::Array(vec![])),
        Some(Json::Object(vec![])),
        Some(Json::Float(f64::NAN)),
        Some(Json::Float(f64::INFINITY)),
        Some(Json::Float(f64::NEG_INFINITY)),
        Some(Json::Float(-0.01)),
        Some(Json::Float(1.01)),
        Some(Json::Int(2)),
    ] {
        let parsed = parse(&changed("threshold", threshold), IDENTITY, None)
            .expect("optional threshold is not required");
        assert_eq!(parsed.receipt_threshold, None);
    }
    for (threshold, expected) in [
        (Json::Int(0), 0.0),
        (Json::Int(1), 1.0),
        (Json::Float(0.0), 0.0),
        (Json::Float(1.0), 1.0),
        (Json::Float(0.25), 0.25),
    ] {
        let parsed = parse(
            &changed("threshold", Some(threshold)),
            IDENTITY,
            Some(expected),
        )
        .expect("numeric endpoint or interior grant");
        assert_eq!(parsed.receipt_threshold, Some(expected));
    }
}

#[test]
fn receipt_selection_requires_publisher_eligibility_and_numeric_matching_grant() {
    assert!(parse(&valid(), IDENTITY, Some(0.7)).is_ok());
    assert_eq!(
        parse(
            &changed("promotion_eligible", Some(Json::Bool(false))),
            IDENTITY,
            Some(0.7)
        ),
        Err(CalibrationError::ReceiptIneligible)
    );
    for (grant, declared) in [
        (None, 0.7),
        (Some(Json::Null), 0.7),
        (Some(Json::Bool(true)), 1.0),
        (Some(Json::Bool(false)), 0.0),
        (Some(Json::Str("0.7".to_owned())), 0.7),
        (Some(Json::Float(0.6)), 0.7),
        (Some(Json::Float(f64::NAN)), 0.7),
        (Some(Json::Float(0.7)), f64::NAN),
        (Some(Json::Float(f64::INFINITY)), 0.7),
        (Some(Json::Float(0.7)), f64::INFINITY),
        (Some(Json::Float(f64::INFINITY)), f64::NEG_INFINITY),
    ] {
        assert_eq!(
            parse(&changed("threshold", grant), IDENTITY, Some(declared)),
            Err(CalibrationError::ReceiptGrant)
        );
    }
}

#[test]
fn receipt_closeness_matches_python_at_relative_and_zero_boundaries() {
    // Python supplies cases and answers; IEEE bits avoid decimal decoder rounding.
    let output = python_stdout(
        r#"
import json, math, struct
boundary = 1.0 - 1e-9
pairs = [
    (0.7, 0.7), (0.7, 0.7 + 6e-10), (0.7, 0.7 + 8e-10),
    (1.0, math.nextafter(boundary, 0.0)), (1.0, boundary),
    (1.0, math.nextafter(boundary, 1.0)),
    (0.0, 0.0), (-0.0, 0.0), (0.0, 1e-300), (0.0, 5e-324),
    (1e-12, 1e-12 + 0.5e-21), (1e-12, 1e-12 + 2e-21),
    (5e-324, 1e-323),
]
pairs += [(b, a) for a, b in pairs]
def bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]
print(json.dumps([(bits(a), bits(b), math.isclose(a, b)) for a, b in pairs]))
"#,
        &[],
    );
    let cases: Vec<(u64, u64, bool)> = serde_json::from_str(&output).expect("Python cases");
    assert!(cases.iter().any(|case| case.2));
    assert!(cases.iter().any(|case| !case.2));
    for (grant_bits, declared_bits, close) in cases {
        let granted = f64::from_bits(grant_bits);
        let declared = f64::from_bits(declared_bits);
        let result = parse(
            &changed("threshold", Some(Json::Float(granted))),
            IDENTITY,
            Some(declared),
        );
        if close {
            assert_eq!(
                result.expect("Python matching grant").receipt_threshold,
                Some(granted)
            );
        } else {
            assert_eq!(result, Err(CalibrationError::ReceiptGrant));
        }
    }
}

#[test]
fn raw_grant_check_does_not_add_selection_admission_bounds() {
    // Python's receipt helper uses the raw number, independently of the optional
    // [0, 1] projection. Real selection admission belongs to the caller.
    for grant in [-1.0, 2.0, f64::INFINITY, f64::NEG_INFINITY] {
        let expected = python_stdout(
            "import math, sys; v = float(sys.argv[1]); print(math.isclose(v, v))",
            &[&grant.to_string()],
        );
        assert_eq!(expected.trim(), "True");
        let parsed = parse(
            &changed("threshold", Some(Json::Float(grant))),
            IDENTITY,
            Some(grant),
        )
        .expect("equal numeric raw grant follows Python");
        assert_eq!(parsed.receipt_threshold, None);
    }
}
