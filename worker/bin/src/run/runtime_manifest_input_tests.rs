//! Schema-2 content from typed admitted facts. Values are unit-test identities,
//! not a GPU execution and not a live inference claim.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use seeon_deepstream_native::RuntimeVersions;
use seeon_worker_runtime::evidence::EngineDigest;

use super::{Manifest, ManifestError, SCHEMA_VERSION};
use crate::cli::Flags;
use crate::config::Checked;
use crate::config::env::Env;
use crate::config::model_bundle::bundle::BundleProof;
use crate::config::pull::{ConfigSource, PulledConfig};
use crate::config::selection::{ModelSelection, Publication};
use crate::json::Json;
use crate::relay::cameras::WorkerConfigPayload;
use crate::relay::cameras::policies::{
    PolicySource, PolicyValues, make_effective_policy, resolve_detection_policies,
};
use crate::run::cameras::{PolicyNumberSource, resolve_fall_policy};
use crate::run::config_digest::config_digest;
use crate::run::media_config::{self, MediaAssembly};
use crate::run::settings::BootPolicy;
use crate::run::{
    Admitted, EngineArtifact, EngineFiles, FlowSettings, ModelEngines, Settings, calibration,
    fall_evidence::FallEvidence,
};
use crate::telemetry::gpu::GpuStatus;

const REVISION: &str = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678";
const IMAGE: &str = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const ONNX: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const WEIGHTS: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const CALIBRATION_SHA: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const BUNDLE_SHA: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const BOOT: &str = "018f6c6a-7b2e-7c3d-8e4f-1a2b3c4d5e6f";

struct Facts {
    settings: Settings,
    admitted: Admitted,
    gpu: GpuStatus,
    config: PulledConfig,
    media: MediaAssembly,
    versions: RuntimeVersions,
}

fn text(value: &str) -> Json {
    Json::Str(value.to_owned())
}

fn object(members: Vec<(&str, Json)>) -> Json {
    Json::Object(
        members
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn field<'a>(value: &'a Json, key: &str) -> &'a Json {
    let Json::Object(members) = value else {
        panic!("{key} parent is not an object");
    };
    members
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, item)| item)
        .unwrap_or_else(|| panic!("missing {key}"))
}

fn versions(trt: i32, cuda: i32) -> RuntimeVersions {
    RuntimeVersions {
        trt_version: trt,
        cuda_runtime_version: cuda,
    }
}

fn settings(revision: Option<&str>) -> Settings {
    let mut env = Env::new();
    if let Some(revision) = revision {
        env.insert("ML_WORKER_BUILD_REVISION".to_owned(), revision.to_owned());
    }
    Settings {
        build_revision: crate::config::build_revision::resolve(revision, &env, Ok(None)),
        env,
        flags: Flags::default(),
        state_dir: PathBuf::from("/tmp/seeon-unit-manifest"),
        execution_records: None,
        engines: EngineFiles {
            fall: PathBuf::from("/models/unit/fall.engine"),
            bed: PathBuf::from("/models/unit/bed.engine"),
            stored_pose: PathBuf::from("/models/unit/pose.engine"),
        },
        policy: BootPolicy {
            device_ordinal: 0,
            stored_pose_threshold: 0.5,
            deployed_batch: Some(1),
            readiness_budget: Duration::from_secs(1),
        },
    }
}

fn engine(byte: u8) -> EngineArtifact {
    EngineArtifact {
        path: PathBuf::from(format!("/models/unit/engine-{byte:02x}.engine")),
        digest: EngineDigest::new([byte; 32]),
    }
}

fn calibration(temperature: f64, promoted: bool) -> crate::run::calibration::Calibration {
    let mut members = vec![
        (
            "class_order",
            Json::Array(vec![text("non_fall"), text("fall_transition_proxy")]),
        ),
        (
            "preprocessing_identity_digest",
            text("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ),
        ("temperature", Json::Float(temperature)),
        (
            "temporal_rule",
            object(vec![("m", Json::Int(2)), ("n", Json::Int(5))]),
        ),
        ("promotion_eligible", Json::Bool(promoted)),
    ];
    if promoted {
        members.push(("threshold", Json::Float(0.81)));
    }
    calibration::parse(
        &object(members),
        "",
        calibration::CalibrationSource::Packaged,
    )
    .expect("unit calibration")
}

fn digest(fill: char) -> String {
    fill.to_string().repeat(64)
}

fn publication(content: &str) -> Publication {
    Publication {
        source_locator: "unit://admitted-selection".to_owned(),
        revision: "unit-rev".to_owned(),
        content: content.to_owned(),
    }
}

fn selection() -> ModelSelection {
    ModelSelection {
        model_publication: publication(BUNDLE_SHA),
        bundle_members_digest: digest('5'),
        dataset_publication: publication(&digest('6')),
        evaluation_receipt_digest: digest('7'),
        field_evaluation_receipt_digest: digest('8'),
        calibration_digest: CALIBRATION_SHA.to_owned(),
        conformance_digest: digest('9'),
        input_observation_schema: digest('a'),
        output_class_count: 2,
        output_class_semantics_digest: digest('b'),
        policy_digest: digest('c'),
        runtime_format: "tensorrt".to_owned(),
        bundle_format: "seeon-model-bundle-v1".to_owned(),
        preprocessing_identity: "pose-bbox56/unit".to_owned(),
        transition_threshold: 0.5,
        threshold_source: "default".to_owned(),
    }
}

fn proof() -> BundleProof {
    let mut identities = BTreeMap::new();
    identities.insert("calibration".to_owned(), CALIBRATION_SHA.to_owned());
    identities.insert("members".to_owned(), digest('5'));
    BundleProof {
        bundle_sha256: BUNDLE_SHA.to_owned(),
        members: vec!["model.onnx".to_owned(), "model.pt".to_owned()],
        receipts: Vec::new(),
        member_digests: BTreeMap::new(),
        calibration: Vec::new(),
        conformance: ("conformance/case.json".to_owned(), Vec::new()),
        identities,
    }
}

fn admitted(selected: bool, promoted: bool, temperature: f64) -> Admitted {
    let mut identity = BTreeMap::new();
    identity.insert("image_digest".to_owned(), IMAGE.to_owned());
    identity.insert("engine_sha256".to_owned(), digest('e'));
    identity.insert("batch_size".to_owned(), "1".to_owned());
    Admitted {
        checked: Checked {
            state_dir: PathBuf::from("/tmp/seeon-unit-manifest"),
            execution_records: None,
            selection: selected.then(|| (selection(), proof())),
            engine_identity: None,
            last_known_good: Ok(None),
        },
        flow: FlowSettings {
            engine: PathBuf::from("/models/unit/pose.engine"),
            identity_path: PathBuf::from("/models/unit/pose.identity.json"),
            infer_config: PathBuf::from("/models/unit/pose-infer.txt"),
            tracker_config: PathBuf::from("/models/unit/tracker.yml"),
            tracker_library: PathBuf::from(
                "/opt/nvidia/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            ),
            onnx: PathBuf::from("/models/unit/pose.onnx"),
            parser_library: PathBuf::from("/models/unit/libpose_parser.so"),
            record_dir: PathBuf::from("/var/lib/seeon/records"),
            record_cache_seconds: 17,
            frame_width: 1280,
            frame_height: 720,
            batch_size: 1,
            rtsp_reconnect_interval_sec: 9,
            identity,
        },
        engines: ModelEngines {
            fall: engine(0xaa),
            bed: engine(0xbb),
            stored_pose: engine(0xcc),
        },
        fall: FallEvidence {
            calibration: calibration(temperature, promoted),
            model_version: ONNX.to_owned(),
            published_weights_digest: WEIGHTS.to_owned(),
            calibration_digest: CALIBRATION_SHA.to_owned(),
            preprocessing_identity: "pose-bbox56/unit".to_owned(),
        },
    }
}

fn policy(threshold: f64, source: PolicySource) -> Json {
    let (facility, camera) = match source {
        PolicySource::ImageDefault => (None, None),
        PolicySource::FacilityDefault => (Some(4), None),
        PolicySource::CameraOverride => (Some(4), Some(9)),
    };
    make_effective_policy(
        "fall",
        2,
        &PolicyValues::FallV2 {
            transition_threshold: threshold,
        },
        source,
        facility,
        camera,
    )
    .expect("effective policy")
    .as_json()
}

fn modules(fall: Json) -> Json {
    let bed = make_effective_policy(
        "bed_exit",
        1,
        &PolicyValues::BED_EXIT_DEFAULT,
        PolicySource::ImageDefault,
        None,
        None,
    )
    .expect("bed policy")
    .as_json();
    object(vec![("bed_exit", bed), ("fall", fall)])
}

fn camera_entry(camera_id: &str, facility_id: &str) -> Json {
    object(vec![
        ("camera_id", text(camera_id)),
        ("facility_id", text(facility_id)),
        ("rtsp_url", text("rtsp://relay.example/live")),
    ])
}
fn camera_domains(camera_id: &str, facility_id: &str, domains: Vec<&str>) -> Json {
    object(vec![
        ("camera_id", text(camera_id)),
        ("facility_id", text(facility_id)),
        ("rtsp_url", text("rtsp://relay.example/live")),
        (
            "domains",
            Json::Array(domains.into_iter().map(text).collect()),
        ),
    ])
}

fn with_domains(base: PulledConfig, domains: Json) -> PulledConfig {
    let Json::Object(mut members) = base.payload else {
        panic!("payload object");
    };
    if let Some(entry) = members.iter_mut().find(|(key, _)| key == "domains") {
        entry.1 = domains;
    } else {
        members.push(("domains".to_owned(), domains));
    }
    let payload = Json::Object(members);
    let config = WorkerConfigPayload::parse(&payload).expect("payload");
    PulledConfig {
        directive: config.directive(),
        payload,
        cameras: base.cameras,
        policies: base.policies,
        windows: base.windows,
        config,
        source: base.source,
        stale: base.stale,
    }
}

fn roster(config: PulledConfig, admitted: Admitted) -> Facts {
    let media = media_config::assemble(&admitted.flow, &config.cameras, BOOT).expect("media");
    Facts {
        settings: settings(Some(REVISION)),
        admitted,
        gpu: gpu(Some("580.65.06"), Some("unit-test-device"), true),
        config,
        media,
        versions: versions(10_03_02, 12_04_01),
    }
}

fn pulled(
    cameras: Vec<Json>,
    policies: Json,
    version: i128,
    generation: i128,
    registry: i128,
) -> PulledConfig {
    let payload = object(vec![
        ("config_version", Json::Int(version)),
        ("restart_epoch", Json::Int(generation)),
        ("registry_version", Json::Int(registry)),
        ("cameras", Json::Array(cameras)),
        ("detection_policies", policies),
    ]);
    let config = WorkerConfigPayload::parse(&payload).expect("payload");
    let ids: Vec<String> = config
        .cameras()
        .into_iter()
        .map(|camera| camera.camera_id)
        .collect();
    PulledConfig {
        directive: config.directive(),
        payload,
        cameras: config.runtime_cameras().expect("runtime cameras"),
        policies: resolve_detection_policies(config.detection_policies(), &ids).expect("policies"),
        config,
        windows: Default::default(),
        source: ConfigSource::Pulled,
        stale: false,
    }
}

fn one_camera(threshold: f64, source: PolicySource) -> PulledConfig {
    pulled(
        vec![camera_entry("cam-a", "facility-7")],
        object(vec![
            ("schema_version", Json::Int(1)),
            ("defaults", modules(policy(0.5, PolicySource::ImageDefault))),
            (
                "cameras",
                object(vec![("cam-a", modules(policy(threshold, source)))]),
            ),
        ]),
        7,
        3,
        11,
    )
}

fn empty_roster(version: i128) -> PulledConfig {
    pulled(Vec::new(), Json::Null, version, 0, 0)
}

fn gpu(driver: Option<&str>, device: Option<&str>, cuda_ok: bool) -> GpuStatus {
    GpuStatus {
        nvml_available: driver.is_some(),
        cuda_context_ok: cuda_ok,
        driver_version: driver.map(str::to_owned),
        device_name: device.map(str::to_owned),
        captured_at_sec: 1_700_000_000.0,
        nvml_error: None,
    }
}

fn facts() -> Facts {
    let config = one_camera(0.62, PolicySource::CameraOverride);
    let admitted = admitted(false, false, 1.75);
    let media = media_config::assemble(&admitted.flow, &config.cameras, BOOT).expect("media");
    Facts {
        settings: settings(Some(REVISION)),
        admitted,
        gpu: gpu(Some("580.65.06"), Some("unit-test-device"), true),
        config,
        media,
        versions: versions(10_03_02, 12_04_01),
    }
}

fn content(facts: &Facts) -> Result<Json, ManifestError> {
    super::content::build(
        &facts.settings,
        &facts.admitted,
        &facts.gpu,
        &facts.config,
        &facts.media,
        facts.versions,
    )
}

fn frozen(facts: &Facts) -> Manifest {
    Manifest::freeze(&content(facts).expect("manifest content")).expect("canonical")
}

fn source_name(source: PolicyNumberSource) -> &'static str {
    match source {
        PolicyNumberSource::Receipt => "receipt",
        PolicyNumberSource::Default => "default",
    }
}

#[test]
fn packaged_inputs_populate_supplied_identities_without_omission() {
    let facts = facts();
    let value = content(&facts).expect("packaged content");
    let manifest = Manifest::freeze(&value).expect("freeze");
    assert!(
        manifest
            .canonical()
            .contains("\"manifest_schema_version\":2")
    );
    assert_eq!(manifest.sha256().len(), 64);
    assert!(!manifest.canonical().contains("rtsp://"));
    assert!(!manifest.canonical().contains("/models/"));

    assert_eq!(
        field(&value, "manifest_schema_version"),
        &Json::Int(SCHEMA_VERSION)
    );
    assert_eq!(field(&value, "profile"), &text("flow"));
    let build = field(&value, "build");
    assert_eq!(field(build, "worker_build_revision"), &text(REVISION));
    assert_eq!(field(build, "worker_image_digest"), &text(IMAGE));
    assert_eq!(field(build, "implementation_language"), &text("rust"));
    assert_eq!(
        field(build, "package_version"),
        &text(env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(
        field(build, "inference_runtime_version_encoded"),
        &Json::Int(i128::from(facts.versions.trt_version))
    );
    assert_eq!(
        field(build, "cuda_runtime_version_encoded"),
        &Json::Int(i128::from(facts.versions.cuda_runtime_version))
    );
    assert_eq!(field(build, "driver_version"), &text("580.65.06"));
    assert_eq!(field(build, "device_name"), &text("unit-test-device"));

    let configuration = field(&value, "configuration");
    assert_eq!(field(configuration, "config_version"), &Json::Int(7));
    assert_eq!(field(configuration, "restart_generation"), &Json::Int(3));
    assert_eq!(field(configuration, "registry_version"), &Json::Int(11));
    assert_eq!(
        field(configuration, "pulled_config_sha256"),
        &text(&config_digest(&facts.config.payload).expect("digest"))
    );

    let Json::Array(components) = field(&value, "components") else {
        panic!("components");
    };
    let fall_engine = engine(0xaa).digest.to_string();
    assert_ne!(fall_engine, ONNX);
    assert_ne!(fall_engine, WEIGHTS);
    assert_ne!(fall_engine, CALIBRATION_SHA);
    assert_eq!(
        field(&components[0], "admitted_engine_sha256"),
        &text(&fall_engine)
    );
    assert_eq!(field(&components[0], "role"), &text("fall"));
    assert_eq!(field(&components[0], "use"), &text("live_score"));
    assert_eq!(field(&components[1], "role"), &text("bed"));
    assert_eq!(field(&components[1], "use"), &text("warm_owner"));
    assert_eq!(
        field(&components[2], "admitted_engine_sha256"),
        &text(&engine(0xcc).digest.to_string())
    );
    assert_eq!(field(&components[3], "role"), &text("pose"));
    assert_eq!(
        field(&components[3], "admitted_engine_sha256"),
        &text(&digest('e'))
    );
    assert_eq!(field(&components[3], "use"), &text("media_plane_infer"));

    let evidence = field(&value, "fall_evidence");
    assert_eq!(field(evidence, "published_onnx_sha256"), &text(ONNX));
    assert_eq!(field(evidence, "published_weights_sha256"), &text(WEIGHTS));
    assert_eq!(
        field(evidence, "calibration_sha256"),
        &text(CALIBRATION_SHA)
    );
    assert_eq!(
        field(evidence, "preprocessing_identity"),
        &text("pose-bbox56/unit")
    );
    assert_eq!(
        field(&value, "bundle"),
        &object(vec![("authority", text("packaged"))])
    );

    let MediaAssembly::Configured(media) = &facts.media else {
        panic!("one camera is configured");
    };
    let plan = field(&value, "media_plan");
    assert_eq!(field(plan, "state"), &text("configured"));
    assert_eq!(
        field(plan, "mux_width"),
        &Json::Int(i128::from(media.mux_width))
    );
    assert_eq!(
        field(plan, "mux_height"),
        &Json::Int(i128::from(media.mux_height))
    );
    assert_eq!(
        field(plan, "record_cache_seconds"),
        &Json::Int(i128::from(media.record_cache_seconds))
    );
    assert_eq!(
        field(plan, "rtsp_reconnect_interval_sec"),
        &Json::Int(i128::from(media.rtsp_reconnect_interval_sec))
    );

    let resolved = resolve_fall_policy(
        &facts.config.policies,
        "cam-a",
        &facts.admitted.fall.calibration,
    )
    .expect("resolved policy");
    let Json::Array(cameras) = field(&value, "cameras") else {
        panic!("cameras");
    };
    assert_eq!(cameras.len(), 1);
    assert_eq!(field(&cameras[0], "camera_id"), &text("cam-a"));
    assert_eq!(field(&cameras[0], "facility_id"), &text("facility-7"));
    assert_eq!(field(&cameras[0], "enabled"), &Json::Bool(true));
    assert_eq!(field(&cameras[0], "module_qualified_id"), &text("fall.v2"));
    assert_eq!(field(&cameras[0], "source_id"), &Json::Int(0));
    let applied = field(&cameras[0], "applied_policy");
    assert_eq!(
        field(applied, "temperature"),
        &Json::Float(resolved.temperature)
    );
    assert_eq!(resolved.temperature, 1.75);
    assert_eq!(
        field(applied, "transition_threshold"),
        &Json::Float(resolved.transition_threshold)
    );
    assert_eq!(
        field(applied, "threshold_source"),
        &text(source_name(resolved.threshold_source))
    );
    assert_eq!(
        field(applied, "confirmation_rule_source"),
        &text(source_name(resolved.confirmation_rule_source))
    );
    assert_eq!(
        field(applied, "transition_votes"),
        &Json::Int(i128::try_from(resolved.transition_votes).unwrap())
    );
    assert_ne!(resolved.threshold_source, PolicyNumberSource::Receipt);
}
#[test]
fn disabled_fall_keeps_roster_identity_and_names_no_applied_policy() {
    let enabled = facts();
    let disabled = roster(
        with_domains(
            one_camera(0.62, PolicySource::CameraOverride),
            object(vec![("fall", object(vec![("enabled", Json::Bool(false))]))]),
        ),
        admitted(false, false, 1.75),
    );
    assert!(
        enabled
            .config
            .config
            .domain_selection()
            .resolve()
            .expect("default fall")
            .fall
    );
    assert!(
        !disabled
            .config
            .config
            .domain_selection()
            .resolve()
            .expect("disabled fall")
            .fall
    );
    let enabled_value = content(&enabled).expect("enabled content");
    let value = content(&disabled).expect("disabled content");
    assert_ne!(
        Manifest::freeze(&enabled_value).expect("enabled").sha256(),
        Manifest::freeze(&value).expect("disabled").sha256()
    );
    let Json::Array(cameras) = field(&value, "cameras") else {
        panic!("cameras");
    };
    assert_eq!(cameras.len(), 1);
    assert_eq!(field(&cameras[0], "camera_id"), &text("cam-a"));
    assert_eq!(field(&cameras[0], "facility_id"), &text("facility-7"));
    assert_eq!(field(&cameras[0], "source_id"), &Json::Int(0));
    assert_eq!(field(&cameras[0], "module_qualified_id"), &text("fall.v2"));
    assert_eq!(field(&cameras[0], "enabled"), &Json::Bool(false));
    assert_eq!(field(&cameras[0], "declared_policy"), &Json::Null);
    assert_eq!(field(&cameras[0], "applied_policy"), &Json::Null);
    let canonical = Manifest::freeze(&value)
        .expect("disabled canonical")
        .canonical()
        .to_owned();
    assert!(!canonical.contains("0.62"));
    assert!(!canonical.contains("1.75"));
    assert!(!canonical.contains("transition_threshold"));
    let Json::Array(components) = field(&value, "components") else {
        panic!("components");
    };
    assert_eq!(field(&components[0], "role"), &text("fall"));
    assert_eq!(field(&components[0], "use"), &text("warm_owner"));
    assert_eq!(field(&components[1], "role"), &text("bed"));
    assert_eq!(field(&components[1], "use"), &text("warm_owner"));
    assert_eq!(field(&components[2], "role"), &text("stored_pose"));
    assert_eq!(field(&components[2], "use"), &text("warm_owner"));
    assert_eq!(
        field(field(&value, "fall_evidence"), "published_onnx_sha256"),
        &text(ONNX)
    );
    assert_eq!(
        field(field(&value, "fall_evidence"), "calibration_sha256"),
        &text(CALIBRATION_SHA)
    );
}

#[test]
fn empty_camera_domain_list_claims_no_live_fall() {
    let admitted = admitted(false, false, 1.75);
    let config = pulled(
        vec![camera_domains("cam-b", "facility-9", Vec::new())],
        object(vec![
            ("schema_version", Json::Int(1)),
            ("defaults", modules(policy(0.5, PolicySource::ImageDefault))),
            (
                "cameras",
                object(vec![(
                    "cam-b",
                    modules(policy(0.44, PolicySource::CameraOverride)),
                )]),
            ),
        ]),
        7,
        3,
        11,
    );
    assert!(
        !config
            .config
            .domain_selection()
            .resolve()
            .expect("empty list")
            .fall
    );
    let facts = roster(config, admitted);
    let value = content(&facts).expect("empty replacement");
    let Json::Array(cameras) = field(&value, "cameras") else {
        panic!("cameras");
    };
    assert_eq!(cameras.len(), 1);
    assert_eq!(field(&cameras[0], "camera_id"), &text("cam-b"));
    assert_eq!(field(&cameras[0], "facility_id"), &text("facility-9"));
    assert_eq!(field(&cameras[0], "enabled"), &Json::Bool(false));
    assert_eq!(field(&cameras[0], "declared_policy"), &Json::Null);
    assert_eq!(field(&cameras[0], "applied_policy"), &Json::Null);
    let Json::Array(components) = field(&value, "components") else {
        panic!("components");
    };
    assert_eq!(field(&components[0], "use"), &text("warm_owner"));
    assert!(
        !Manifest::freeze(&value)
            .expect("canonical")
            .canonical()
            .contains("0.44")
    );
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical provenance implementation"]
fn python_reads_the_actual_disabled_domain_manifest_without_rewriting() {
    let disabled = roster(
        with_domains(
            one_camera(0.62, PolicySource::CameraOverride),
            object(vec![("fall", object(vec![("enabled", Json::Bool(false))]))]),
        ),
        admitted(false, false, 1.75),
    );
    let manifest = Manifest::freeze(&content(&disabled).expect("actual disabled projection"))
        .expect("canonical disabled manifest");
    super::tests::assert_python_accepts(&manifest);
}

#[test]
fn unknown_camera_domain_list_refuses_without_domain_text() {
    let mut facts = facts();
    // Corrupt an admitted fixture to reach the producer's defensive check;
    // normal pull admission refuses this before building runtime cameras.
    let Json::Object(members) = &mut facts.config.payload else {
        panic!("payload object");
    };
    members
        .iter_mut()
        .find(|(key, _)| key == "cameras")
        .expect("cameras")
        .1 = Json::Array(vec![camera_domains(
        "cam-a",
        "facility-7",
        vec!["not-a-domain"],
    )]);
    facts.config.config = WorkerConfigPayload::parse(&facts.config.payload).expect("wire shape");
    let error = content(&facts).expect_err("unknown domain");
    assert_eq!(error, ManifestError::Invalid("domain_selection"));
    let rendered = error.to_string();
    assert!(!rendered.contains("not-a-domain"));
    assert!(!rendered.contains("UnknownDomain"));
}

#[test]
fn missing_and_malformed_fall_policies_remain_distinct_refusals() {
    let mut missing = facts();
    missing.config.policies.cameras.clear();
    missing.config.policies.defaults.clear();
    assert_eq!(
        content(&missing),
        Err(ManifestError::Missing("fall_policy"))
    );

    let mut malformed = facts();
    malformed
        .config
        .policies
        .cameras
        .get_mut("cam-a")
        .expect("camera policy")
        .get_mut("fall")
        .expect("selected fall policy")
        .module_version = 99;
    assert_eq!(
        content(&malformed),
        Err(ManifestError::Invalid("fall_policy"))
    );
}

#[test]
fn selected_bundle_names_proof_without_equating_engine_and_published_bytes() {
    let mut facts = facts();
    facts.admitted = admitted(true, false, 1.75);
    facts.media =
        media_config::assemble(&facts.admitted.flow, &facts.config.cameras, BOOT).expect("media");
    let value = content(&facts).expect("selected content");
    let bundle = field(&value, "bundle");
    assert_eq!(field(bundle, "authority"), &text("selected"));
    assert_eq!(field(bundle, "bundle_sha256"), &text(BUNDLE_SHA));
    let identities = field(bundle, "verified_identities");
    assert_eq!(field(identities, "calibration"), &text(CALIBRATION_SHA));
    let Json::Array(components) = field(&value, "components") else {
        panic!("components");
    };
    assert_ne!(
        field(&components[0], "admitted_engine_sha256"),
        field(field(&value, "fall_evidence"), "published_onnx_sha256")
    );
}

#[test]
fn applied_temperature_and_authoritative_threshold_change_the_digest() {
    let baseline = frozen(&facts());
    let mut warmer = facts();
    warmer.admitted.fall.calibration = calibration(2.5, false);
    let warmer = frozen(&warmer);
    assert_ne!(baseline.sha256(), warmer.sha256());
    assert!(warmer.canonical().contains("2.5"));

    let mut promoted = facts();
    promoted.admitted.fall.calibration = calibration(1.75, true);
    let promoted_value = content(&promoted).expect("promoted");
    let Json::Array(cameras) = field(&promoted_value, "cameras") else {
        panic!("cameras");
    };
    let applied = field(&cameras[0], "applied_policy");
    let resolved = resolve_fall_policy(
        &promoted.config.policies,
        "cam-a",
        &promoted.admitted.fall.calibration,
    )
    .expect("promoted resolution");
    assert_eq!(resolved.threshold_source, PolicyNumberSource::Receipt);
    assert_eq!(field(applied, "threshold_source"), &text("receipt"));
    assert_eq!(field(applied, "transition_threshold"), &Json::Float(0.81));
    assert_ne!(frozen(&promoted).sha256(), baseline.sha256());

    let mut revised = facts();
    revised.config = pulled(
        vec![camera_entry("cam-a", "facility-7")],
        object(vec![
            ("schema_version", Json::Int(1)),
            ("defaults", modules(policy(0.5, PolicySource::ImageDefault))),
            (
                "cameras",
                object(vec![(
                    "cam-a",
                    modules(policy(0.62, PolicySource::CameraOverride)),
                )]),
            ),
        ]),
        8,
        3,
        11,
    );
    let revised_manifest = frozen(&revised);
    assert_ne!(revised_manifest.sha256(), baseline.sha256());
    let revised_content = content(&revised).expect("revised");
    let configuration = field(&revised_content, "configuration");
    assert_eq!(field(configuration, "config_version"), &Json::Int(8));
    assert_eq!(field(configuration, "restart_generation"), &Json::Int(3));
    assert_eq!(field(configuration, "registry_version"), &Json::Int(11));
}

#[test]
fn empty_idle_roster_invents_no_camera_and_keeps_zero_revisions() {
    let mut facts = facts();
    facts.config = empty_roster(0);
    facts.media =
        media_config::assemble(&facts.admitted.flow, &facts.config.cameras, BOOT).expect("idle");
    assert!(matches!(facts.media, MediaAssembly::Idle));
    assert!(facts.config.cameras.is_empty());
    let value = content(&facts).expect("idle content");
    assert_eq!(field(&value, "cameras"), &Json::Array(Vec::new()));
    assert_eq!(field(field(&value, "media_plan"), "state"), &text("idle"));
    assert!(json_strings(&value).all(|item| !item.contains("facility") && !item.contains("cam-")));
    let configuration = field(&value, "configuration");
    assert_eq!(field(configuration, "config_version"), &Json::Int(0));
    assert_eq!(field(configuration, "restart_generation"), &Json::Int(0));
    assert_eq!(field(configuration, "registry_version"), &Json::Int(0));

    facts.config = empty_roster(19);
    let nonzero = content(&facts).expect("nonzero idle");
    assert_eq!(
        field(field(&nonzero, "configuration"), "config_version"),
        &Json::Int(19)
    );
    assert_ne!(
        Manifest::freeze(&nonzero).expect("nonzero").sha256(),
        Manifest::freeze(&value).expect("zero").sha256()
    );
}

#[test]
fn tags_are_not_image_digests_and_media_engine_identity_is_required() {
    for reference in ["ghcr.io/example/worker:latest", "sha256:not-a-digest"] {
        let mut input = facts();
        input
            .admitted
            .flow
            .identity
            .insert("image_digest".into(), reference.into());
        assert_eq!(
            content(&input),
            Err(ManifestError::Invalid("worker_image_digest"))
        );
    }
    let mut pinned = facts();
    pinned.admitted.flow.identity.insert(
        "image_digest".into(),
        format!("ghcr.io/example/worker@{IMAGE}"),
    );
    let value = content(&pinned).unwrap();
    assert_eq!(
        field(field(&value, "build"), "worker_image_digest"),
        &text(IMAGE)
    );
    assert_eq!(
        field(field(&value, "build"), "edge_database_schema_version"),
        &Json::Int(19)
    );
    let mut missing = facts();
    missing.admitted.flow.identity.remove("engine_sha256");
    assert_eq!(
        content(&missing),
        Err(ManifestError::Missing("pose_engine_sha256"))
    );
    missing
        .admitted
        .flow
        .identity
        .insert("engine_sha256".into(), "not-a-hash".into());
    assert_eq!(
        content(&missing),
        Err(ManifestError::Invalid("pose_engine_sha256"))
    );
}

#[test]
fn manifest_uses_the_resolved_revision_not_a_later_environment_value() {
    let mut input = facts();
    let original = content(&input).expect("admitted unit manifest");
    input.settings.env.insert(
        "ML_WORKER_BUILD_REVISION".to_owned(),
        "not-the-compiled-revision".to_owned(),
    );
    assert_eq!(content(&input), Ok(original));
    input.settings.build_revision =
        Err(crate::config::build_revision::BuildRevisionError::Mismatch);
    assert_eq!(
        content(&input),
        Err(ManifestError::Contradictory("worker_build_revision"))
    );
}

#[test]
fn missing_and_invalid_identities_are_precise_refusals() {
    let mut missing_revision = facts();
    missing_revision.settings = settings(None);
    assert_eq!(
        content(&missing_revision),
        Err(ManifestError::Missing("worker_build_revision"))
    );

    for revision in ["", "   "] {
        let mut absent = facts();
        absent.settings = settings(Some(revision));
        assert_eq!(
            content(&absent),
            Err(ManifestError::Missing("worker_build_revision"))
        );
    }
    for revision in ["abc", &"0".repeat(40), &"A".repeat(40), &"g".repeat(40)] {
        let mut invalid = facts();
        invalid.settings = settings(Some(revision));
        assert_eq!(
            content(&invalid),
            Err(ManifestError::Invalid("worker_build_revision")),
            "{revision:?}"
        );
    }

    let mut missing_driver = facts();
    missing_driver.gpu = gpu(Some("  "), Some("unit-test-device"), true);
    assert_eq!(
        content(&missing_driver),
        Err(ManifestError::Missing("driver_version"))
    );
    let mut missing_device = facts();
    missing_device.gpu = gpu(Some("580.65.06"), None, true);
    assert_eq!(
        content(&missing_device),
        Err(ManifestError::Missing("device_name"))
    );

    let mut bad_cuda = facts();
    bad_cuda.gpu.cuda_context_ok = false;
    assert_eq!(
        content(&bad_cuda),
        Err(ManifestError::Invalid("accelerator_runtime"))
    );
    for versions in [versions(0, 12), versions(-1, 12), versions(10, 0)] {
        let mut bad_runtime = facts();
        bad_runtime.versions = versions;
        assert_eq!(
            content(&bad_runtime),
            Err(ManifestError::Invalid("accelerator_runtime"))
        );
    }
    let mut negative = facts();
    negative.config.directive.version = -1;
    assert_eq!(
        content(&negative),
        Err(ManifestError::Invalid("configuration_revision"))
    );
}

#[test]
fn roster_and_source_mismatch_are_contradictions_not_generic_failure() {
    let mut idle_with_camera = facts();
    idle_with_camera.media = MediaAssembly::Idle;
    assert_eq!(
        content(&idle_with_camera),
        Err(ManifestError::Contradictory("media_roster"))
    );

    let mut configured_without_camera = facts();
    configured_without_camera.config = empty_roster(7);
    assert_eq!(
        content(&configured_without_camera),
        Err(ManifestError::Contradictory("media_roster"))
    );

    let mut shifted = facts();
    let MediaAssembly::Configured(media) = &mut shifted.media else {
        panic!("configured");
    };
    media.sources[0].source_id = 1;
    assert_eq!(
        content(&shifted),
        Err(ManifestError::Contradictory("media_source"))
    );
}

fn json_strings(value: &Json) -> impl Iterator<Item = &str> {
    let mut pending = vec![value];
    let mut found = Vec::new();
    while let Some(current) = pending.pop() {
        match current {
            Json::Str(item) => found.push(item.as_str()),
            Json::Array(items) => pending.extend(items),
            Json::Object(members) => {
                for (key, item) in members {
                    found.push(key.as_str());
                    pending.push(item);
                }
            }
            Json::Null | Json::Bool(_) | Json::Int(_) | Json::Float(_) => {}
        }
    }
    found.into_iter()
}
