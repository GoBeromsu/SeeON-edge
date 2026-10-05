//! Boot reporting uses admitted configuration, never a fabricated site identity.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::time::{Duration, Instant};

use seeon_ml_worker::config::pull::{ConfigSource, PulledConfig};
use seeon_ml_worker::json::Json;
use seeon_ml_worker::relay::cameras::{
    CameraConfigError, WorkerConfigPayload, policies::resolve_detection_policies,
};
use seeon_ml_worker::run::BootStatusContext;
use seeon_ml_worker::run::status::{BootReason, ReportContextError, ReportOutcome};
use seeon_ml_worker::telemetry::gpu::GpuStatus;

fn config(facilities: &[&str]) -> PulledConfig {
    config_with_enrollment(facilities, None)
}

fn config_with_enrollment(facilities: &[&str], enrolled: Option<&str>) -> PulledConfig {
    let cameras: Vec<_> = facilities
        .iter()
        .enumerate()
        .map(|(index, facility)| {
            serde_json::json!({"camera_id": format!("camera-{index}"), "facility_id": facility,
            "rtsp_url": "rtsp://camera.example/live"})
        })
        .collect();
    let payload = Json::from(&serde_json::json!({"cameras": cameras,
        "clip_export_enabled": true, "clip_export_version": 17, "registry_version": 4,
        "enrolled_facility_id": enrolled}));
    let config = WorkerConfigPayload::parse(&payload).expect("admitted payload");
    let cameras = config.runtime_cameras().expect("admitted roster");
    let ids = cameras
        .iter()
        .map(|camera| camera.camera_id.clone())
        .collect::<Vec<_>>();
    PulledConfig {
        policies: resolve_detection_policies(config.detection_policies(), &ids).expect("policies"),
        directive: config.directive(),
        payload,
        config,
        cameras,
        windows: Default::default(),
        source: ConfigSource::Pulled,
        stale: false,
    }
}

fn gpu() -> GpuStatus {
    GpuStatus {
        nvml_available: false,
        cuda_context_ok: false,
        driver_version: None,
        device_name: None,
        captured_at_sec: 12.0,
        nvml_error: None,
    }
}

#[test]
fn enrollment_identity_is_optional_but_strict_when_supplied() {
    let absent = serde_json::json!({"registry_version": 1, "cameras": []});
    assert_eq!(
        WorkerConfigPayload::parse(&Json::from(&absent))
            .unwrap()
            .enrolled_facility_id(),
        None
    );
    for value in [serde_json::Value::Null, serde_json::json!(" genuine-site ")] {
        let mut payload = absent.clone();
        payload["enrolled_facility_id"] = value.clone();
        let parsed = WorkerConfigPayload::parse(&Json::from(&payload)).unwrap();
        assert_eq!(parsed.enrolled_facility_id(), value.as_str());
        assert!(parsed.runtime_cameras().unwrap().is_empty());
    }
    for value in [
        serde_json::json!(""),
        serde_json::json!(" \t\n"),
        serde_json::json!(false),
        serde_json::json!(17),
        serde_json::json!([]),
        serde_json::json!({}),
    ] {
        let mut payload = absent.clone();
        payload["enrolled_facility_id"] = value;
        assert_eq!(
            WorkerConfigPayload::parse(&Json::from(&payload)).unwrap_err(),
            CameraConfigError::Field("enrolled_facility_id")
        );
    }
}

#[test]
fn nonempty_invalid_facility_is_not_treated_as_an_unassigned_roster() {
    let mut admitted = config(&["site-a"]);
    admitted.cameras[0].facility_id.clear();
    assert!(matches!(
        BootStatusContext::for_config("http://127.0.0.1:1", "test-token", &admitted, 10.0),
        Err(ReportContextError::Facility)
    ));
}
#[test]
fn empty_roster_has_a_typed_unconstructible_report_without_contacting_relay() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    listener.set_nonblocking(true).expect("nonblocking");
    let context = BootStatusContext::for_config(
        &format!("http://{}", listener.local_addr().expect("address")),
        "test-token",
        &config(&[]),
        10.0,
    )
    .expect("empty roster is not a configuration error");
    assert_eq!(
        context.send(BootReason::CudaUnavailable, &gpu()),
        ReportOutcome::NoFacility
    );
    assert_eq!(
        listener.accept().expect_err("no invented report").kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn admitted_facility_and_export_policy_reach_exactly_one_failure_post() {
    assert_failure_post(config(&["site-z", "site-a"]), "site-a");
}

#[test]
fn enrolled_empty_roster_reports_the_genuine_facility_once() {
    assert_failure_post(
        config_with_enrollment(&[], Some("enrolled-site")),
        "enrolled-site",
    );
}

#[test]
fn camera_facility_selection_is_not_replaced_by_enrollment() {
    assert_failure_post(
        config_with_enrollment(&["site-z", "site-a"], Some("enrolled-site")),
        "site-a",
    );
}

fn assert_failure_post(admitted: PulledConfig, expected_facility: &str) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    listener.set_nonblocking(true).expect("nonblocking");
    let address = listener.local_addr().expect("address");
    let server_listener = listener.try_clone().expect("listener clone");
    let server = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(4);
        let stream = loop {
            match server_listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("missing report: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read deadline");
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .expect("write deadline");
        let mut reader = BufReader::new(stream);
        let mut first = String::new();
        reader.read_line(&mut first).expect("request line");
        assert_eq!(first, "POST /api/v1/relay/runtime-status HTTP/1.1\r\n");
        let mut length = None;
        loop {
            let mut line = String::new();
            assert!(reader.read_line(&mut line).expect("header") > 0);
            if line == "\r\n" {
                break;
            }
            if let Some((key, value)) = line.split_once(':')
                && key.eq_ignore_ascii_case("content-length")
            {
                length = Some(value.trim().parse::<usize>().expect("length"));
            }
        }
        let length = length.expect("body length");
        assert!(length <= 65_536);
        let mut body = vec![0; length];
        reader.read_exact(&mut body).expect("body");
        reader.get_mut().write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").expect("response");
        serde_json::from_slice::<serde_json::Value>(&body).expect("JSON")
    });
    let context =
        BootStatusContext::for_config(&format!("http://{address}"), "test-token", &admitted, 10.0)
            .expect("report context");
    assert_eq!(
        context.send(BootReason::EngineIdentity, &gpu()),
        ReportOutcome::Status(503)
    );
    let body = server.join().expect("server");
    assert_eq!(body["facility_id"], expected_facility);
    assert_eq!(
        body["clip_export"],
        serde_json::json!({"enabled": true, "version": 17})
    );
    assert_eq!(body["cameras"], serde_json::json!([]));
    assert_eq!(body["worker"]["profile_boot_error"], "engine_identity");
    assert_eq!(body["worker"]["alive"], false);
    assert_eq!(body["gpu"]["cuda_context_ok"], false);
    assert_eq!(body["seq"], 1);
    assert!(body["generation"].is_null());
    assert_eq!(
        listener.accept().expect_err("no retry").kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn empty_roster_still_reaches_the_lease_gate_and_preserves_its_exit() {
    use seeon_ml_worker::cli::Flags;
    use seeon_ml_worker::config::env::Env;
    use seeon_ml_worker::exit::Exit;
    use seeon_ml_worker::gpu::lease;
    use seeon_ml_worker::run::boot::{BootError, boot};
    use seeon_ml_worker::run::{BootPolicy, Settings};
    use seeon_ml_worker::seam::SystemClock;
    use seeon_ml_worker::shutdown::ShutdownDeadline;
    use std::sync::Arc;

    let parent = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&parent).expect("test parent");
    let state = parent.join(format!("empty-roster-lease-{}", std::process::id()));
    std::fs::create_dir(&state).expect("owned state");
    let held = lease::acquire(&state).expect("contended lease");
    let mut env = Env::new();
    env.insert("RELAY_TOKEN".into(), "test-token".into());
    for key in [
        "ML_WORKER_FALL_ENGINE_PATH",
        "ML_WORKER_BED_ENGINE_PATH",
        "ML_WORKER_STORED_POSE_ENGINE_PATH",
    ] {
        env.insert(
            key.into(),
            state.join("unopened-engine").display().to_string(),
        );
    }
    let settings = Settings::from_flags(
        env,
        Flags {
            heartbeat_on_start: false,
            state_dir: Some(state.clone()),
            auxiliary_runtime:
                seeon_ml_worker::config::model_bundle::identity::AuxiliaryRuntime::TensorRt,
        },
        BootPolicy {
            device_ordinal: 0,
            stored_pose_threshold: 0.5,
            deployed_batch: None,
            readiness_budget: Duration::from_secs(1),
        },
    )
    .expect("settings");
    let report =
        BootStatusContext::for_config("http://127.0.0.1:1", "test-token", &config(&[]), 10.0)
            .expect("empty context");
    let clock = SystemClock::default();
    let error = boot(
        settings,
        report,
        &clock,
        Arc::new(ShutdownDeadline::new(Duration::from_secs(25)).expect("deadline")),
    )
    .err()
    .expect("real lease gate refused");
    assert_eq!(error.exit(&clock), Exit::RefuseToStart);
    let BootError::Failed(failure) = error else {
        panic!("not a cancellation")
    };
    assert_eq!(failure.reason, BootReason::GpuLease);
    assert_eq!(failure.report, ReportOutcome::NoFacility);
    drop(failure);
    drop(held);
    std::fs::remove_dir_all(&state).expect("owned cleanup");
}

#[test]
fn explicit_cpu_settings_do_not_require_unused_engines_or_relax_other_policy() {
    use seeon_ml_worker::config::env::Env;
    use seeon_ml_worker::run::{BootPolicy, Settings, SettingsError};
    use std::ffi::OsString;

    let mut env = Env::new();
    env.insert("RELAY_TOKEN".into(), "test-token".into());
    let policy = BootPolicy {
        device_ordinal: 0,
        stored_pose_threshold: 0.5,
        deployed_batch: None,
        readiness_budget: Duration::from_secs(1),
    };
    let cpu = [
        OsString::from("--auxiliary-runtime=onnxruntime-cpu"),
        OsString::from("--state-dir=/unopened-provider-test-state"),
    ];
    assert!(Settings::parse(env.clone(), &cpu, policy).is_ok());
    assert_eq!(
        Settings::parse(env.clone(), &cpu[1..], policy).err(),
        Some(SettingsError::Required("ML_WORKER_FALL_ENGINE_PATH"))
    );
    for key in [
        "ML_WORKER_FALL_ENGINE_PATH",
        "ML_WORKER_BED_ENGINE_PATH",
        "ML_WORKER_STORED_POSE_ENGINE_PATH",
    ] {
        env.insert(key.into(), " ".into());
    }
    assert!(Settings::parse(env.clone(), &cpu, policy).is_ok());
    let invalid = BootPolicy {
        readiness_budget: Duration::ZERO,
        ..policy
    };
    assert_eq!(
        Settings::parse(env.clone(), &cpu, invalid).err(),
        Some(SettingsError::Policy("readiness_budget"))
    );
    env.insert("ML_WORKER_PROFILE".into(), "retired".into());
    assert_eq!(
        Settings::parse(env, &cpu, policy).err(),
        Some(SettingsError::Policy("ML_WORKER_PROFILE"))
    );
}
