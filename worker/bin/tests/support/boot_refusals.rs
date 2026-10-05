//! Stage 4 A1 row 10: actual process exits before lease, model readiness or sources.
//! CPU auxiliaries still require the supplied real schema-2 identity and models.
use std::fs;
use std::time::Duration;

use seeon_ml_worker::config::model_bundle::identity::AuxiliaryRuntime;
use seeon_ml_worker::config::pull::{EDGE_DATABASE_FORMAT_IDENTITY, EDGE_DATABASE_SCHEMA_VERSION};
use seeon_ml_worker::run::{BootPolicy, Settings};
use serde_json::{Value, json};

use super::fixture::{CONFIG_PATH, Fixture, IDENTITY_PATH, Server};
use super::{relay_guard, run};

#[test]
#[ignore = "requires genuine schema-2 identity via SEEON_TEST_ENGINE_IDENTITY with only live_pose and CPU ONNX hashes, real supplied model/engine inputs and isolated ml-api loopback alias"]
fn wrong_release_format_exits_3_before_config_with_cpu_auxiliaries() {
    let _guard = relay_guard();
    let fixture = Fixture::with_provider("release-format-cpu", AuxiliaryRuntime::OnnxRuntimeCpu);
    let identity = json!({
        "edge_database_schema_version": EDGE_DATABASE_SCHEMA_VERSION,
        "format": "not-the-seeon-edge-format",
    });
    let server = Server::start_with_identity(identity, Some(fixture.config(true)), None);
    assert_refusal(
        &fixture,
        server,
        3,
        "ml-worker: release identity refused: FormatMismatch\n",
        &[IDENTITY_PATH],
    );
}

#[test]
#[ignore = "requires genuine schema-2 identity via SEEON_TEST_ENGINE_IDENTITY with only live_pose and CPU ONNX hashes, real supplied model/engine inputs and isolated ml-api loopback alias"]
fn malformed_successful_release_identity_exits_1_before_config_with_cpu_auxiliaries() {
    let _guard = relay_guard();
    for (label, identity) in [
        (
            "release-schema-cpu",
            json!({
                "edge_database_schema_version": "not-a-schema-version",
                "format": EDGE_DATABASE_FORMAT_IDENTITY,
            }),
        ),
        ("release-body-cpu", Value::Null),
    ] {
        let fixture = Fixture::with_provider(label, AuxiliaryRuntime::OnnxRuntimeCpu);
        let server = Server::start_with_identity(identity, Some(fixture.config(true)), None);
        assert_refusal(
            &fixture,
            server,
            1,
            "ml-worker: release identity refused: MalformedReleaseIdentity\n",
            &[IDENTITY_PATH],
        );
    }
}

#[test]
#[ignore = "requires genuine schema-2 identity via SEEON_TEST_ENGINE_IDENTITY with only live_pose and CPU ONNX hashes, real supplied model/engine inputs and isolated ml-api loopback alias"]
fn config_503_without_lkg_exits_2_with_cpu_auxiliaries() {
    let _guard = relay_guard();
    let fixture = Fixture::with_provider("no-config-cpu", AuxiliaryRuntime::OnnxRuntimeCpu);
    let server = Server::start(None, None);
    assert_refusal(
        &fixture,
        server,
        2,
        "ml-worker: worker config pull refused: NoConfig\n",
        &[IDENTITY_PATH, CONFIG_PATH],
    );
}

fn assert_refusal(
    fixture: &Fixture,
    server: Server,
    expected_exit: i32,
    expected_diagnostic: &str,
    expected_paths: &[&str],
) {
    assert!(
        fs::read_dir(&fixture.state)
            .expect("fresh owned state")
            .next()
            .is_none(),
        "each refusal must start with empty state and no LKG"
    );
    let command = fixture.command("run");
    let arguments: Vec<_> = command
        .get_args()
        .skip(1)
        .map(|arg| arg.to_owned())
        .collect();
    let settings = Settings::parse(
        fixture.env.clone(),
        &arguments,
        BootPolicy {
            device_ordinal: 0,
            stored_pose_threshold: 0.25,
            deployed_batch: None,
            readiness_budget: Duration::from_secs(30),
        },
    )
    .expect("real fixture and actual command must pass settings admission");
    assert_eq!(
        settings.flags().auxiliary_runtime,
        AuxiliaryRuntime::OnnxRuntimeCpu
    );
    assert_eq!(settings.state_dir(), fixture.state.as_path());

    let (status, stderr) = run(command);
    let requests = server.finish();
    let diagnostic = std::str::from_utf8(&stderr).expect("UTF-8 worker diagnostic");
    assert_eq!(status.code(), Some(expected_exit), "{diagnostic}");
    assert_eq!(
        diagnostic, expected_diagnostic,
        "only the causal refusal diagnostic is allowed; no policy-loop readiness or later boot output"
    );
    assert_eq!(
        requests
            .iter()
            .map(|request| (request.method.as_str(), request.path.as_str()))
            .collect::<Vec<_>>(),
        expected_paths
            .iter()
            .map(|path| ("GET", *path))
            .collect::<Vec<_>>(),
        "exact startup GET sequence; no POST, status, event, score or readiness traffic"
    );
    fixture.assert_no_source_activation();
    assert!(
        fs::read_dir(&fixture.state)
            .expect("retained owned state")
            .next()
            .is_none(),
        "refusal must not publish LKG, acquire a GPU lease or reach model readiness"
    );
}
