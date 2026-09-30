//! Accelerator receipt as the `payload.accelerator` object of design §2.6.

use seeon_worker_runtime::evidence::AcceleratorEvidence;

use crate::json::Json;

/// The eleven §2.6 members, each read from its accessor.
pub fn to_json(evidence: &AcceleratorEvidence) -> Json {
    let text = |value: &str| Json::Str(value.to_owned());
    let count = |value: u64| Json::Int(i128::from(value));
    Json::Object(vec![
        ("provider".to_owned(), text(evidence.provider())),
        ("precision".to_owned(), text(evidence.precision().as_str())),
        (
            "device_ordinal".to_owned(),
            Json::Int(i128::from(evidence.device_ordinal())),
        ),
        (
            "engine_sha256".to_owned(),
            Json::Str(evidence.engine_sha256().to_string()),
        ),
        ("call_seq".to_owned(), count(evidence.call_seq())),
        ("attempted".to_owned(), count(evidence.attempted())),
        ("succeeded".to_owned(), count(evidence.succeeded())),
        ("failed".to_owned(), count(evidence.failed())),
        ("h2d_bytes".to_owned(), count(evidence.h2d_bytes())),
        ("d2h_bytes".to_owned(), count(evidence.d2h_bytes())),
        ("elapsed_ns".to_owned(), count(evidence.elapsed_ns())),
    ])
}

#[cfg(test)]
mod tests {
    use seeon_deepstream_native::GpuMetrics;
    use seeon_worker_runtime::evidence::{EngineDigest, Precision};

    use super::*;

    #[test]
    fn the_receipt_has_exactly_the_eleven_section_2_6_members() {
        let before = GpuMetrics {
            attempted: 6,
            succeeded: 5,
            failed: 1,
            host_to_device_bytes: 1000,
            device_to_host_bytes: 300,
            elapsed_ns: 4000,
            device: 2,
        };
        let after = GpuMetrics {
            attempted: 7,
            succeeded: 6,
            failed: 1,
            host_to_device_bytes: 1123,
            device_to_host_bytes: 345,
            elapsed_ns: 4789,
            device: 2,
        };
        let digest = EngineDigest::new(std::array::from_fn(|index| index as u8));
        let evidence = AcceleratorEvidence::from_delta(&before, &after, 2, digest, Precision::Fp32)
            .expect("one complete call");

        let Json::Object(members) = to_json(&evidence) else {
            panic!("the receipt is a JSON object");
        };
        let mut keys: Vec<&str> = members.iter().map(|(key, _)| key.as_str()).collect();
        keys.sort_unstable();
        let expected = [
            ("attempted", Json::Int(1)),
            ("call_seq", Json::Int(7)),
            ("d2h_bytes", Json::Int(45)),
            ("device_ordinal", Json::Int(2)),
            ("elapsed_ns", Json::Int(789)),
            (
                "engine_sha256",
                Json::Str(
                    "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f".to_owned(),
                ),
            ),
            ("failed", Json::Int(0)),
            ("h2d_bytes", Json::Int(123)),
            ("precision", Json::Str("fp32".to_owned())),
            ("provider", Json::Str("tensorrt".to_owned())),
            ("succeeded", Json::Int(1)),
        ];
        let names: Vec<&str> = expected.iter().map(|(name, _)| *name).collect();
        assert_eq!(keys, names);
        for (name, value) in &expected {
            let member = members.iter().find(|(key, _)| key == name);
            assert_eq!(member.map(|(_, got)| got), Some(value), "member {name}");
        }
    }
}
