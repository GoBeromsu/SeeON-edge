//! ADR0009 effective-config identity: Python canonical JSON is the independent oracle.

use std::process::Command;

use seeon_ml_worker::json::{Json, JsonError};
use seeon_ml_worker::run::config_digest::config_digest;

fn configuration(secret: &str, threshold: f64) -> Json {
    Json::from(&serde_json::json!({
        "module": {"threshold": threshold, "name": "낙상"},
        "relay": {"TOKEN": secret, "relay_token": secret, "access_token": secret,
            "refresh_token": secret, "password": secret, "client_secret": secret,
            "authorization": secret},
        "cameras": [{"camera_id": "fixture-camera", "rtsp_url": secret}]
    }))
}

#[test]
fn digest_matches_independent_python_canonical_oracle() {
    // The redacted fixture is declared from the ADR, not produced by Rust.
    let python = r#"
import hashlib, json
value = {
    'module': {'threshold': 0.7, 'name': '낙상'},
    'relay': {'TOKEN': '[redacted]', 'relay_token': '[redacted]',
              'access_token': '[redacted]', 'refresh_token': '[redacted]',
              'password': '[redacted]', 'client_secret': '[redacted]',
              'authorization': '[redacted]'},
    'cameras': [{'camera_id': 'fixture-camera', 'rtsp_url': '[redacted]'}],
}
wire = json.dumps(value, sort_keys=True, separators=(',', ':'),
                  ensure_ascii=False, allow_nan=False).encode('utf-8')
print(hashlib.sha256(wire).hexdigest())
"#;
    let output = Command::new("python3")
        .args(["-c", python])
        .output()
        .expect("Python stdlib oracle is required");
    assert!(output.status.success());
    let expected = String::from_utf8(output.stdout).expect("ASCII SHA256");
    assert_eq!(
        config_digest(&configuration("synthetic-sensitive-value", 0.7)).expect("valid config"),
        expected.trim()
    );
}

#[test]
fn identity_tracks_policy_not_secret_values_or_key_order() {
    let first = config_digest(&configuration("synthetic-A", 0.7)).expect("valid config");
    let mut changed_secret = configuration("synthetic-B", 0.7);
    if let Json::Object(members) = &mut changed_secret {
        members.reverse();
    }
    assert_eq!(
        first,
        config_digest(&changed_secret).expect("valid reordered config")
    );
    assert_ne!(
        first,
        config_digest(&configuration("synthetic-A", 0.8)).expect("changed effective policy")
    );
}

#[test]
fn ambiguous_or_nonfinite_configuration_is_refused() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let config = Json::Object(vec![("threshold".to_owned(), Json::Float(value))]);
        assert_eq!(config_digest(&config), Err(JsonError::NonFinite));
    }
    let duplicate = Json::Object(vec![
        ("threshold".to_owned(), Json::Float(0.7)),
        ("threshold".to_owned(), Json::Float(0.8)),
    ]);
    assert_eq!(config_digest(&duplicate), Err(JsonError::DuplicateKey));
}
