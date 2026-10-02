//! Applied fall numbers follow registry.py `_effective_transition_threshold`.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::sync_channel;

use seeon_deepstream_native::{
    FrameIdentity, GpuMetrics, MEDIA_MAX_SOURCES, MEDIA_POSE_COLUMNS, TrackedObject,
};
use seeon_ml_worker::config::pull::{ConfigSource, PulledConfig};
use seeon_ml_worker::json::Json;
use seeon_ml_worker::msg::{FallResponse, PosePacket};
use seeon_ml_worker::policy::fall::FallStage;
use seeon_ml_worker::relay::cameras::CameraConfigError;
use seeon_ml_worker::relay::cameras::RuntimeCamera;
use seeon_ml_worker::relay::cameras::WorkerConfigPayload;
use seeon_ml_worker::relay::cameras::policies::{
    PolicySource, PolicyValues, make_effective_policy, resolve_detection_policies,
};
use seeon_ml_worker::run::calibration::{Calibration, CalibrationSource, parse};
use seeon_ml_worker::run::cameras::{
    CameraPolicyError, PolicyNumberSource, ResolvedFallPolicy, camera_policies, resolve_fall_policy,
};
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallCapacities, FallCapacity, FallFailure};
use seeon_worker::trace::{DecisionTraceReason, DecisionTraceValueName, NumericTraceValue};
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, Precision};
use seeon_worker_runtime::fall_gpu::FallScore;

const BOOT: &str = "boot-카메라";
const EPOCH: &str = "epoch-1";
const GENERATION: u64 = 7;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with the canonical Python fall policy resolver"]
fn precedence_and_audit_match_the_python_registry() {
    let mut rows = Vec::new();
    let source_name = |source| match source {
        PolicyNumberSource::Receipt => "receipt",
        PolicyNumberSource::Default => "default",
    };
    for promoted in [false, true] {
        for (source, label, threshold) in [
            (PolicySource::ImageDefault, "image-default", 0.7),
            (PolicySource::FacilityDefault, "facility-default", 0.2),
            (PolicySource::CameraOverride, "camera-override", 0.9),
        ] {
            for receipt in [None, Some(0.8)] {
                let mut config = pulled(vec![camera_entry("camera-a", "facility-a")], Json::Null);
                config
                    .policies
                    .cameras
                    .get_mut("camera-a")
                    .expect("camera policy")
                    .insert(
                        "fall".into(),
                        make_effective_policy(
                            "fall",
                            2,
                            &PolicyValues::FallV2 {
                                transition_threshold: threshold,
                            },
                            source,
                            (source != PolicySource::ImageDefault).then_some(1),
                            (source == PolicySource::CameraOverride).then_some(2),
                        )
                        .expect("effective policy"),
                    );
                let result = resolved(
                    &config,
                    "camera-a",
                    &calibration(1.0, receipt, 1, 8, promoted),
                );
                rows.push(serde_json::json!({
                    "source": label, "threshold": threshold, "receipt": receipt, "promoted": promoted,
                    "actual": {
                        "transition_threshold": result.transition_threshold,
                        "threshold_source": source_name(result.threshold_source),
                        "receipt_threshold": result.receipt_threshold,
                        "unapplied_policy_threshold": result.unapplied_policy_threshold,
                        "transition_votes": result.transition_votes,
                        "transition_window": result.transition_window,
                        "confirmation_rule_source": source_name(result.confirmation_rule_source),
                        "receipt_transition_votes": result.receipt_transition_votes,
                        "receipt_transition_window": result.receipt_transition_window,
                        "unapplied_transition_votes": result.unapplied_transition_votes,
                        "unapplied_transition_window": result.unapplied_transition_window
                    }
                }));
            }
        }
    }
    let script = r"
import json, sys
from types import SimpleNamespace
from shared.detection_policies import FallPolicyV2, make_effective_policy
from worker.domains.registry import _effective_transition_threshold
rows = json.load(sys.stdin)
assert len(rows) == 12
for row in rows:
    source = row['source']
    policy = make_effective_policy(module_id='fall', module_version=2,
        values=FallPolicyV2(transition_threshold=row['threshold']), source=source,
        facility_revision_id=None if source == 'image-default' else 1,
        camera_revision_id=2 if source == 'camera-override' else None)
    model = SimpleNamespace(promotion_eligible=row['promoted'],
        receipt_threshold=row['receipt'], receipt_transition_votes=1, receipt_transition_window=8)
    expected = _effective_transition_threshold(model, policy)._asdict()
    assert row['actual'] == expected, (row, expected)
";
    let mut child =
        Command::new(std::env::var_os("SEEON_TEST_PYTHON").expect("canonical Python interpreter"))
            .arg("-c")
            .arg(script)
            .env(
                "PYTHONPATH",
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
            )
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("Python policy oracle");
    child
        .stdin
        .take()
        .expect("oracle stdin")
        .write_all(&serde_json::to_vec(&rows).expect("cases"))
        .expect("oracle cases");
    assert!(child.wait().expect("oracle exit").success());
}

fn object(members: Vec<(&str, Json)>) -> Json {
    Json::Object(
        members
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn text(value: &str) -> Json {
    Json::Str(value.to_owned())
}

fn policy(
    threshold: f64,
    source: PolicySource,
    facility: Option<i128>,
    camera: Option<i128>,
) -> Json {
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
    .expect("fixture policy")
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
    .expect("fixture bed policy")
    .as_json();
    object(vec![("bed_exit", bed), ("fall", fall)])
}

fn camera_entry(camera_id: &str, facility_id: &str) -> Json {
    camera_with_domains(camera_id, facility_id, None)
}

fn camera_with_domains(camera_id: &str, facility_id: &str, domains: Option<Vec<&str>>) -> Json {
    let mut members = vec![
        ("camera_id", text(camera_id)),
        ("facility_id", text(facility_id)),
        ("rtsp_url", text("rtsp://relay.example/live")),
    ];
    if let Some(domains) = domains {
        members.push((
            "domains",
            Json::Array(domains.into_iter().map(text).collect()),
        ));
    }
    object(members)
}

fn disabled_fall() -> Json {
    object(vec![(
        "domains",
        object(vec![("fall", object(vec![("enabled", Json::Bool(false))]))]),
    )])
}

fn pulled(cameras: Vec<Json>, policies: Json) -> PulledConfig {
    pulled_with(cameras, policies, Json::Null)
}

fn pulled_with(cameras: Vec<Json>, policies: Json, extra: Json) -> PulledConfig {
    let Json::Object(mut members) = object(vec![
        ("config_version", Json::Int(3)),
        ("cameras", Json::Array(cameras)),
        ("detection_policies", policies),
    ]) else {
        unreachable!("object");
    };
    if let Json::Object(extra) = extra {
        members.extend(extra);
    }
    let payload = Json::Object(members);
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

fn document(
    temperature: f64,
    threshold: Option<f64>,
    votes: i128,
    window: i128,
    promoted: bool,
) -> Json {
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
            object(vec![("m", Json::Int(votes)), ("n", Json::Int(window))]),
        ),
        ("promotion_eligible", Json::Bool(promoted)),
    ];
    if let Some(threshold) = threshold {
        members.push(("threshold", Json::Float(threshold)));
    }
    object(members)
}

fn calibration(
    temperature: f64,
    threshold: Option<f64>,
    votes: i128,
    window: i128,
    promoted: bool,
) -> Calibration {
    parse(
        &document(temperature, threshold, votes, window, promoted),
        "",
        CalibrationSource::Packaged,
    )
    .expect("calibration")
}

fn capacities(vote_window: usize) -> FallCapacities {
    FallCapacities {
        retained_tracks: 4,
        generation_identities: 4,
        episodes: 4,
        vote_window,
    }
}

fn two_cameras(first: f64, first_source: PolicySource, second: f64) -> PulledConfig {
    let cameras = vec![
        camera_entry("camera-a", "facility-a"),
        camera_entry("camera-b", "facility-b"),
    ];
    pulled(
        cameras,
        object(vec![
            ("schema_version", Json::Int(1)),
            (
                "defaults",
                modules(policy(0.5, PolicySource::ImageDefault, None, None)),
            ),
            (
                "cameras",
                object(vec![
                    (
                        "camera-b",
                        modules(policy(second, PolicySource::FacilityDefault, Some(2), None)),
                    ),
                    (
                        "camera-a",
                        modules(policy(first, first_source, Some(1), Some(11))),
                    ),
                ]),
            ),
        ]),
    )
}

fn resolved(
    config: &PulledConfig,
    camera_id: &str,
    calibration: &Calibration,
) -> ResolvedFallPolicy {
    resolve_fall_policy(&config.policies, camera_id, calibration).expect("resolved")
}

fn trace_float(stage: &FallStage, name: DecisionTraceValueName) -> f64 {
    match stage.decider().last_trace_snapshots()[0]
        .values()
        .get(&name)
    {
        Some(NumericTraceValue::Float(value)) => value.get(),
        other => panic!("expected float trace value, got {other:?}"),
    }
}

fn trace_count(stage: &FallStage, name: DecisionTraceValueName) -> usize {
    match stage.decider().last_trace_snapshots()[0]
        .values()
        .get(&name)
    {
        Some(NumericTraceValue::Integer(value)) => *value,
        other => panic!("expected integer trace value, got {other:?}"),
    }
}

fn evidence() -> AcceleratorEvidence {
    AcceleratorEvidence::from_delta(
        &GpuMetrics::default(),
        &GpuMetrics {
            attempted: 1,
            succeeded: 1,
            host_to_device_bytes: 1,
            device_to_host_bytes: 1,
            ..GpuMetrics::default()
        },
        0,
        EngineDigest::new([9; 32]),
        Precision::Fp32,
    )
    .expect("evidence")
}

fn pose_row() -> [f32; MEDIA_POSE_COLUMNS] {
    let mut row = [0.0; MEDIA_POSE_COLUMNS];
    row[0] = 8.0;
    row[1] = 8.0;
    row[2] = 40.0;
    row[3] = 40.0;
    row[4] = 0.9;
    for point in 0..17 {
        row[6 + point * 3] = 16.0;
        row[7 + point * 3] = 16.0;
        row[8 + point * 3] = 0.9;
    }
    row
}

fn packet(sequence: u64) -> PosePacket {
    PosePacket {
        frame: FrameIdentity {
            sequence,
            pts_ns: sequence.saturating_mul(200_000_000),
            pts_valid: 1,
            source_width: 640,
            source_height: 640,
            analysis_width: 640,
            analysis_height: 640,
            ..FrameIdentity::default()
        },
        tensor_present: true,
        rows: vec![pose_row()],
        objects: vec![TrackedObject {
            track_id: 4,
            left: 8.0,
            top: 8.0,
            width: 32.0,
            height: 32.0,
            confidence: 0.9,
        }],
    }
}

fn consume(stage: &mut FallStage, logit: f32) -> Vec<BusinessEvent> {
    let (sender, receiver) = sync_channel(1);
    let mut request = None;
    for sequence in 0..40 {
        let frame =
            seeon_ml_worker::policy::ingest::ingest(&packet(sequence), None).expect("frame");
        assert_eq!(frame.rows.len(), 1);
        stage
            .observe(&frame, &sender, &mut |_| {})
            .expect("observe");
        if let Ok(due) = receiver.try_recv() {
            request = Some(due);
            break;
        }
    }
    let request = request.expect("stride opens a window");
    let mut events = Vec::new();
    stage
        .consume(
            FallResponse {
                frame: request.frame,
                track_id: request.track_id,
                score: Ok(FallScore {
                    logit,
                    evidence: evidence(),
                }),
            },
            &mut |update| events.extend(update.events.iter().cloned()),
        )
        .expect("consume");
    events
}

#[test]
fn unpromoted_overrides_stay_canonical_and_promoted_receipt_applies() {
    let config = two_cameras(0.2, PolicySource::CameraOverride, 0.9);
    let idle = calibration(1.0, Some(0.8), 1, 8, false);
    let first = resolved(&config, "camera-a", &idle);
    let second = resolved(&config, "camera-b", &idle);
    assert_eq!(first.transition_threshold, 0.5);
    assert_eq!(second.transition_threshold, 0.5);
    assert_eq!(first.threshold_source, PolicyNumberSource::Default);
    assert_eq!(first.unapplied_policy_threshold, Some(0.2));
    assert_eq!(second.unapplied_policy_threshold, Some(0.9));
    assert_eq!(first.receipt_threshold, Some(0.8));
    assert_eq!((first.transition_votes, first.transition_window), (3, 5));
    assert_eq!(first.confirmation_rule_source, PolicyNumberSource::Default);
    assert_eq!(first.unapplied_transition_votes, Some(1));
    assert_eq!(first.unapplied_transition_window, Some(8));

    let stages =
        camera_policies(&config, &idle, BOOT, EPOCH, GENERATION, capacities(5)).expect("stages");
    assert_eq!(
        stages
            .iter()
            .map(|camera| camera.source_id)
            .collect::<Vec<_>>(),
        vec![0, 1],
    );
    let mut stages: Vec<FallStage> = stages
        .into_iter()
        .map(|camera| camera.stage.expect("fall enabled"))
        .collect();
    assert!(consume(&mut stages[0], 0.0).is_empty());
    assert!(consume(&mut stages[1], 0.0).is_empty());
    assert_eq!(
        trace_float(&stages[0], DecisionTraceValueName::TransitionThreshold),
        0.5
    );
    assert_eq!(
        trace_float(&stages[1], DecisionTraceValueName::TransitionThreshold),
        0.5
    );
    assert_eq!(
        trace_count(&stages[0], DecisionTraceValueName::TransitionVotes),
        3
    );
    assert_eq!(
        trace_float(
            &stages[0],
            DecisionTraceValueName::FallTransitionProbability
        ),
        0.5
    );

    let promoted = calibration(1.0, Some(0.8), 1, 8, true);
    let applied = resolved(&config, "camera-a", &promoted);
    assert_eq!(applied.transition_threshold, 0.8);
    assert_eq!(applied.threshold_source, PolicyNumberSource::Receipt);
    assert_eq!(applied.unapplied_policy_threshold, None);
    assert_eq!(
        (applied.transition_votes, applied.transition_window),
        (1, 8)
    );
    assert_eq!(
        applied.confirmation_rule_source,
        PolicyNumberSource::Receipt
    );
    assert_eq!(applied.unapplied_transition_votes, None);
    let promoted_stages =
        camera_policies(&config, &promoted, BOOT, EPOCH, GENERATION, capacities(8))
            .expect("promoted");
    let mut promoted_stage = promoted_stages
        .into_iter()
        .next()
        .expect("camera-a")
        .stage
        .expect("fall enabled");
    assert!(
        consume(&mut promoted_stage, 0.0).is_empty(),
        "0.5 is below receipt 0.8"
    );
    assert_eq!(
        trace_float(&promoted_stage, DecisionTraceValueName::TransitionThreshold),
        0.8
    );
    assert_eq!(
        promoted_stage.decider().last_trace_snapshots()[0].reason,
        DecisionTraceReason::BelowThreshold
    );
    let mut qualifying =
        camera_policies(&config, &promoted, BOOT, EPOCH, GENERATION, capacities(8))
            .expect("promoted positive case")
            .into_iter()
            .next()
            .expect("camera-a")
            .stage
            .expect("fall enabled");
    let confirmed = consume(&mut qualifying, 20.0);
    assert_eq!(
        confirmed.len(),
        1,
        "one qualifying vote meets the receipt rule"
    );
    assert_eq!(confirmed[0].camera_id, "camera-a");
    assert_eq!(confirmed[0].facility_id, "facility-a");
    assert!(
        confirmed[0]
            .identity
            .starts_with(&format!("{BOOT}:{EPOCH}:"))
    );
    assert!(confirmed[0].identity.contains(&format!(":{GENERATION}:")));
}

#[test]
fn image_default_applies_without_promotion_and_temperature_changes_probability() {
    let config = pulled(vec![camera_entry("camera-a", "facility-a")], Json::Null);
    let idle = calibration(1.0, Some(0.8), 1, 8, false);
    let image = resolved(&config, "camera-a", &idle);
    assert_eq!(image.transition_threshold, 0.5);
    assert_eq!(image.unapplied_policy_threshold, None);
    assert_eq!(image.threshold_source, PolicyNumberSource::Default);

    let custom = pulled(
        vec![camera_entry("camera-a", "facility-a")],
        object(vec![
            ("schema_version", Json::Int(1)),
            (
                "defaults",
                modules(policy(0.25, PolicySource::ImageDefault, None, None)),
            ),
            (
                "cameras",
                object(vec![(
                    "camera-a",
                    modules(policy(0.25, PolicySource::ImageDefault, None, None)),
                )]),
            ),
        ]),
    );
    let accepted = resolved(&custom, "camera-a", &idle);
    assert_eq!(accepted.transition_threshold, 0.25);
    assert_eq!(accepted.unapplied_policy_threshold, None);
    let cool_calibration = calibration(1.0, None, 3, 5, false);
    let mut cool = camera_policies(
        &custom,
        &cool_calibration,
        BOOT,
        EPOCH,
        GENERATION,
        capacities(5),
    )
    .expect("cool")
    .into_iter()
    .next()
    .expect("camera")
    .stage
    .expect("fall enabled");
    let hot_calibration = calibration(100.0, None, 3, 5, false);
    let mut hot = camera_policies(
        &custom,
        &hot_calibration,
        BOOT,
        EPOCH,
        GENERATION,
        capacities(5),
    )
    .expect("hot")
    .into_iter()
    .next()
    .expect("camera")
    .stage
    .expect("fall enabled");
    consume(&mut cool, 20.0);
    consume(&mut hot, 20.0);
    let cool_probability = trace_float(&cool, DecisionTraceValueName::FallTransitionProbability);
    let hot_probability = trace_float(&hot, DecisionTraceValueName::FallTransitionProbability);
    assert!(cool_probability > 0.9);
    assert!(hot_probability > 0.5 && hot_probability < 0.6);
    assert!(cool_probability > hot_probability);
    assert_eq!(
        trace_float(&cool, DecisionTraceValueName::TransitionThreshold),
        0.25
    );
}

#[test]
fn malformed_receipt_and_capacities_refuse_even_when_unpromoted() {
    let config = pulled(vec![camera_entry("camera-a", "facility-a")], Json::Null);
    let valid = calibration(1.0, Some(0.5), 3, 5, false);
    let overflow = Calibration {
        transition_votes: i128::MAX,
        transition_window: i128::MAX,
        promotion_eligible: false,
        ..valid
    };
    assert_eq!(
        camera_policies(&config, &overflow, BOOT, EPOCH, GENERATION, capacities(5))
            .err()
            .expect("bound"),
        CameraPolicyError::CalibrationBound
    );
    assert_eq!(
        camera_policies(&config, &valid, BOOT, EPOCH, GENERATION, capacities(0))
            .err()
            .expect("capacity"),
        CameraPolicyError::Decider(FallFailure::InvalidCapacities)
    );
    assert_eq!(
        camera_policies(&config, &valid, BOOT, EPOCH, GENERATION, capacities(4))
            .err()
            .expect("window"),
        CameraPolicyError::Decider(FallFailure::Capacity(FallCapacity::VoteWindow))
    );
    assert!(matches!(
        camera_policies(
            &config,
            &Calibration {
                temperature: f64::NAN,
                ..valid
            },
            BOOT,
            EPOCH,
            GENERATION,
            capacities(5),
        )
        .err()
        .expect("temperature"),
        CameraPolicyError::Stage(_)
    ));
    let empty = camera_policies(
        &pulled(Vec::new(), Json::Null),
        &valid,
        BOOT,
        EPOCH,
        GENERATION,
        capacities(5),
    )
    .expect("empty");
    assert!(empty.is_empty());
    let cameras = (0..=MEDIA_MAX_SOURCES)
        .map(|index| camera_entry(&format!("camera-{index}"), "facility-a"))
        .collect();
    assert_eq!(
        camera_policies(
            &pulled(cameras, Json::Null),
            &valid,
            BOOT,
            EPOCH,
            GENERATION,
            capacities(5),
        )
        .err()
        .expect("sources"),
        CameraPolicyError::SourceId
    );
}

#[test]
fn disabled_fall_retains_every_source_index_without_a_stage() {
    let cameras = vec![
        camera_with_domains("camera-a", "facility-a", Some(vec!["bed_exit"])),
        camera_with_domains("camera-b", "facility-b", Some(vec!["bed_exit"])),
    ];
    let built = camera_policies(
        &pulled_with(cameras, Json::Null, disabled_fall()),
        &calibration(1.0, Some(0.5), 3, 5, false),
        BOOT,
        EPOCH,
        GENERATION,
        capacities(5),
    )
    .expect("disabled roster");
    assert_eq!(
        built
            .iter()
            .map(|camera| camera.source_id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(built.iter().all(|camera| camera.stage.is_none()));
}

#[test]
fn partial_global_override_keeps_fall_and_camera_replacement_disables_it() {
    let cameras = vec![
        camera_with_domains("camera-a", "facility-a", Some(vec!["bed_exit"])),
        camera_entry("camera-b", "facility-b"),
    ];
    let partial = pulled_with(
        cameras.clone(),
        Json::Null,
        object(vec![(
            "domains",
            object(vec![(
                "bed_exit",
                object(vec![("enabled", Json::Bool(false))]),
            )]),
        )]),
    );
    let retained = camera_policies(
        &partial,
        &calibration(1.0, None, 3, 5, false),
        BOOT,
        EPOCH,
        GENERATION,
        capacities(5),
    )
    .expect("partial override defaults fall on");
    assert_eq!(retained.len(), 2);
    assert!(retained.iter().all(|camera| camera.stage.is_some()));

    let replaced = pulled_with(
        vec![
            camera_with_domains("camera-a", "facility-a", Some(vec!["bed_exit"])),
            camera_with_domains("camera-b", "facility-b", Some(vec!["bed_exit"])),
        ],
        Json::Null,
        Json::Null,
    );
    let disabled = camera_policies(
        &replaced,
        &calibration(1.0, None, 3, 5, false),
        BOOT,
        EPOCH,
        GENERATION,
        capacities(5),
    )
    .expect("camera list replaces defaults");
    assert_eq!(
        disabled
            .iter()
            .map(|camera| (camera.source_id, camera.stage.is_none()))
            .collect::<Vec<_>>(),
        vec![(0, true), (1, true)]
    );

    let payload = object(vec![
        ("config_version", Json::Int(3)),
        (
            "cameras",
            Json::Array(vec![camera_with_domains(
                "camera-a",
                "facility-a",
                Some(vec!["not-a-domain"]),
            )]),
        ),
    ]);
    let config = WorkerConfigPayload::parse(&payload).expect("payload parses");
    assert_eq!(
        config.domain_selection().resolve(),
        Err(CameraConfigError::UnknownDomain)
    );
    let ids = vec!["camera-a".to_owned()];
    let refused = PulledConfig {
        directive: config.directive(),
        policies: resolve_detection_policies(config.detection_policies(), &ids).expect("policies"),
        cameras: vec![RuntimeCamera {
            camera_id: "camera-a".to_owned(),
            facility_id: "facility-a".to_owned(),
            rtsp_url: "rtsp://relay.example/live".to_owned(),
            fps: 30.0,
            frame_stride: 1,
            decode_backend: None,
            label: None,
            bed_zone_regions: Vec::new(),
            bed_zone_image_width: None,
            bed_zone_image_height: None,
        }],
        payload,
        config,
        windows: Default::default(),
        source: ConfigSource::Pulled,
        stale: false,
    };
    assert!(matches!(
        camera_policies(
            &refused,
            &calibration(1.0, None, 3, 5, false),
            BOOT,
            EPOCH,
            GENERATION,
            capacities(5),
        ),
        Err(CameraPolicyError::Domains)
    ));
}
