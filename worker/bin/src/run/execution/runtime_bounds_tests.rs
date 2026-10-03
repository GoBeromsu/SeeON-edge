use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};

use seeon_deepstream_native::{FrameIdentity, GpuMetrics, MediaBinding, TrackedObject};
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider};
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, Precision};

use crate::msg::{FallRequest, FallResponse, FallScore, PosePacket};
use crate::records::lanes::Lanes;
use crate::run::execution::publication;
use crate::run::pump::{CameraPolicy, PolicyPump, PolicySink};
use crate::seam::{IdSource, RandomIds, SystemClock};

#[path = "runtime_receipt_tests.rs"]
mod receipt_tests;
#[path = "runtime_replay_tests.rs"]
mod replay_tests;

fn budget(poses: usize, responses: usize) -> super::TurnBudget {
    super::TurnBudget { poses, responses }
}

use super::{LiveSink, RuntimeError, pump_queue_turn};

fn ungated(cameras: Vec<CameraPolicy>) -> PolicyPump {
    PolicyPump::new(
        cameras,
        None,
        std::sync::Arc::new(crate::seam::SystemClock::new()),
    )
    .unwrap()
}

fn camera(source_id: u32) -> CameraPolicy {
    let decider = FallPolicyDecider::new(
        format!("camera-{source_id}"),
        "facility-1",
        "boot-1",
        "epoch-1",
        0,
        FallPolicy::default(),
        FallCapacities {
            retained_tracks: 8,
            generation_identities: 8,
            episodes: 8,
            vote_window: 5,
        },
    )
    .expect("valid test policy");
    CameraPolicy {
        source_id,
        stage: Some(crate::policy::fall::FallStage::new(decider, 1.0).expect("valid calibration")),
    }
}

fn packet(source_id: u32, sequence: u64, tracks: &[u64]) -> PosePacket {
    let mut rows = Vec::new();
    let mut objects = Vec::new();
    for (slot, &track_id) in tracks.iter().enumerate() {
        let left = 10.0 + slot as f32 * 100.0;
        let mut row = [0.0; 57];
        row[..5].copy_from_slice(&[left, 20.0, left + 80.0, 200.0, 0.9]);
        for point in 0..17 {
            row[6 + point * 3] = left + 40.0;
            row[7 + point * 3] = 30.0 + point as f32 * 8.0;
            row[8 + point * 3] = 0.9;
        }
        rows.push(row);
        objects.push(TrackedObject {
            track_id,
            left,
            top: 20.0,
            width: 80.0,
            height: 180.0,
            confidence: 0.9,
        });
    }
    PosePacket {
        frame: FrameIdentity {
            source_id,
            sequence,
            pts_ns: sequence.saturating_mul(100_000_000),
            pts_valid: 1,
            frame_number: i64::try_from(sequence).unwrap_or(i64::MAX),
            source_width: 640,
            source_height: 360,
            analysis_width: 640,
            analysis_height: 360,
            ..FrameIdentity::default()
        },
        tensor_present: true,
        rows,
        objects,
    }
}

fn score(logit: f32) -> FallScore {
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
    seeon_worker_runtime::fall_gpu::FallScore {
        logit,
        evidence: AcceleratorEvidence::from_delta(
            &before,
            &after,
            2,
            EngineDigest::new(std::array::from_fn(|index| index as u8)),
            Precision::Fp32,
        )
        .expect("public CPU test evidence"),
    }
    .into()
}

fn answered(request: &FallRequest, logit: f32) -> FallResponse {
    FallResponse {
        frame: request.frame,
        track_id: request.track_id,
        score: Ok(score(logit)),
    }
}

struct Ends {
    pose_tx: SyncSender<PosePacket>,
    poses: Receiver<PosePacket>,
    requests: SyncSender<FallRequest>,
    request_rx: Receiver<FallRequest>,
    response_tx: SyncSender<FallResponse>,
    responses: Receiver<FallResponse>,
}

fn ends(pose_capacity: usize, response_capacity: usize) -> Ends {
    let (pose_tx, poses) = mpsc::sync_channel(pose_capacity);
    let (requests, request_rx) = mpsc::sync_channel(64);
    let (response_tx, responses) = mpsc::sync_channel(response_capacity);
    Ends {
        pose_tx,
        poses,
        requests,
        request_rx,
        response_tx,
        responses,
    }
}

fn sink() -> LiveSink {
    LiveSink {
        scores: Vec::new(),
        ready: Vec::new(),
        triggered: Vec::new(),
        failures: Default::default(),
        failure_receipts: Vec::new(),
    }
}

fn due(
    pump: &mut PolicyPump,
    ends: &Ends,
    stop: &AtomicBool,
    sink: &mut LiveSink,
    source_id: u32,
    tracks: &[u64],
) -> Vec<FallRequest> {
    for sequence in 0..120 {
        while ends.responses.try_recv().is_ok() {}
        ends.pose_tx
            .send(packet(source_id, sequence, tracks))
            .expect("pose channel accepts the synthetic frame");
        pump_queue_turn(
            pump,
            &ends.poses,
            &ends.requests,
            &ends.responses,
            stop,
            sink,
            budget(1, 4),
        )
        .expect("warmup pose");
        let requests: Vec<_> = ends.request_rx.try_iter().collect();
        if !requests.is_empty() {
            assert_eq!(requests.len(), tracks.len());
            return requests;
        }
    }
    panic!("fixed synthetic sequence did not produce a due window");
}

#[test]
fn accepted_prefix_survives_capacity_refusal() {
    let mut pump = ungated(vec![camera(19)]);
    let ends = ends(4, 4);
    let stop = AtomicBool::new(false);
    let mut held = sink();
    for track_id in 0..(super::SCORE_RETENTION - 1) as u64 {
        held.score(
            seeon_deepstream_native::FrameIdentity::default(),
            track_id,
            &score(0.0),
        )
        .expect("accepted prefix fits");
    }
    let requests = due(&mut pump, &ends, &stop, &mut held, 19, &[7, 8]);
    let accepted = answered(&requests[0], 1.25);
    let refused = answered(&requests[1], 4.5);
    ends.response_tx.send(accepted).unwrap();
    ends.response_tx.send(refused).unwrap();
    let error = pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut held,
        budget(0, 2),
    )
    .expect_err("second accepted score must refuse");
    assert!(matches!(
        error,
        RuntimeError::Pump(crate::run::pump::PumpError::ScoreRetention)
    ));
    let retained = &held.scores[super::SCORE_RETENTION - 1..];
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].frame, requests[0].frame);
    assert_eq!(retained[0].track_id, requests[0].track_id);
    assert_eq!(retained[0].logit, 1.25);
    assert!(
        held.ready
            .iter()
            .all(|ready| ready.frame != requests[0].frame)
    );
    assert_eq!(pump.pending_scores(), 1);
}

#[test]
fn completing_a_frame_drains_every_accepted_track_with_post_update_generation() {
    let mut pump = ungated(vec![camera(19)]);
    let ends = ends(4, 4);
    let stop = AtomicBool::new(false);
    let mut held = sink();
    let requests = due(&mut pump, &ends, &stop, &mut held, 19, &[7, 8]);
    for request in &requests {
        ends.response_tx.send(answered(request, 0.5)).unwrap();
    }
    pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut held,
        budget(0, requests.len()),
    )
    .expect("both scores fit");
    let accepted = held
        .scores
        .iter()
        .filter(|score| score.frame == requests[0].frame)
        .count();
    assert_eq!(accepted, 0);
    let ready = &held.ready;
    let decided = ready
        .iter()
        .find(|item| item.frame == requests[0].frame)
        .expect("completed frame");
    assert_eq!(decided.scores.len(), requests.len());
    assert!(
        decided
            .scores
            .iter()
            .all(|score| score.generation.is_some())
    );
    assert_eq!(
        decided
            .scores
            .iter()
            .map(|score| score.track_id)
            .collect::<Vec<_>>(),
        requests
            .iter()
            .map(|request| request.track_id)
            .collect::<Vec<_>>()
    );
    assert_eq!(pump.pending_scores(), 0);
}

#[test]
fn response_budget_exhaustion_preserves_a_newer_queued_pose() {
    let mut pump = ungated(vec![camera(19)]);
    let ends = ends(4, 4);
    let stop = AtomicBool::new(false);
    let mut held = sink();
    let requests = due(&mut pump, &ends, &stop, &mut held, 19, &[7]);
    let pending = requests[0].frame;
    ends.response_tx.send(answered(&requests[0], 1.25)).unwrap();
    ends.pose_tx
        .send(packet(19, pending.sequence + 1, &[8]))
        .unwrap();
    pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut held,
        budget(1, 1),
    )
    .expect("queued response then newer pose");
    let ready = &held.ready;
    let decided = ready
        .iter()
        .find(|item| item.frame == pending)
        .expect("queued frame was decided");
    assert_eq!(decided.scores.len(), 1);
    assert_eq!(decided.scores[0].track_id, requests[0].track_id);
    assert!(decided.scores[0].generation.is_some());
    assert!(ends.responses.try_recv().is_err());
    assert_eq!(
        ends.poses.try_recv().unwrap().frame.sequence,
        pending.sequence + 1
    );
}
#[test]
fn response_arriving_between_poses_is_consumed_before_the_next_pose() {
    struct InjectResponse<'a> {
        held: &'a mut LiveSink,
        sender: &'a SyncSender<FallResponse>,
        response: Option<FallResponse>,
    }
    impl PolicySink for InjectResponse<'_> {
        fn frame_failure(
            &mut self,
            frame: FrameIdentity,
            error: &crate::run::pump::PumpError,
        ) -> Result<(), crate::run::pump::PumpError> {
            self.held.frame_failure(frame, error)
        }
        fn decision(&mut self, update: crate::policy::fall::DecisionUpdate<'_>) {
            let inject = update.frame.source_id == 20;
            self.held.decision(update);
            if inject {
                self.sender
                    .send(self.response.take().expect("first pose decision"))
                    .unwrap();
            }
        }
        fn score(
            &mut self,
            frame: FrameIdentity,
            track_id: u64,
            score: &FallScore,
        ) -> Result<(), crate::run::pump::PumpError> {
            self.held.score(frame, track_id, score)
        }
    }
    let mut pump = ungated(vec![camera(19), camera(20)]);
    let ends = ends(4, 4);
    let stop = AtomicBool::new(false);
    let mut held = sink();
    let requests = due(&mut pump, &ends, &stop, &mut held, 19, &[7]);
    let pending = requests[0].frame;
    ends.pose_tx.send(packet(20, 1, &[])).unwrap();
    ends.pose_tx
        .send(packet(19, pending.sequence + 1, &[9]))
        .unwrap();
    let mut injecting = InjectResponse {
        held: &mut held,
        sender: &ends.response_tx,
        response: Some(answered(&requests[0], 1.25)),
    };
    pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut injecting,
        budget(2, 2),
    )
    .expect("response queued between poses");
    let decided = held
        .ready
        .iter()
        .find(|item| item.frame == pending)
        .expect("inter-pose response completed the pending frame");
    assert_eq!(decided.scores.len(), 1);
    assert!(ends.poses.try_recv().is_err());
}

#[test]
fn continuous_pose_refill_cannot_exceed_the_turn_bound() {
    let stop_fill = std::sync::Arc::new(AtomicBool::new(false));
    let filling = std::sync::Arc::clone(&stop_fill);
    let mut pump = ungated(vec![camera(19)]);
    let ends = ends(8, 1);
    let stop = AtomicBool::new(false);
    let mut held = sink();
    let bound = 2;
    let sender = ends.pose_tx.clone();
    let filler = std::thread::spawn(move || {
        let mut offered = 0_u64;
        while !filling.load(Ordering::SeqCst) {
            match sender.try_send(packet(19, offered, &[7])) {
                Ok(()) => offered += 1,
                Err(mpsc::TrySendError::Full(_)) => std::thread::yield_now(),
                Err(mpsc::TrySendError::Disconnected(_)) => break,
            }
        }
    });
    std::thread::sleep(std::time::Duration::from_millis(20));
    pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut held,
        budget(bound, 0),
    )
    .expect("bounded pose turn");
    stop_fill.store(true, Ordering::SeqCst);
    drop(ends.pose_tx);
    let _ = filler.join();
    let left = ends.poses.try_iter().count();
    assert!(left > 0, "refill left work beyond the frozen allowance");
    assert!(left + bound < 40);
}
#[test]
fn zero_pose_budget_ignores_disconnected_pose_channel_and_drains_live_responses() {
    let mut pump = ungated(Vec::new());
    let ends = ends(1, 1);
    drop(ends.pose_tx);
    let stop = AtomicBool::new(false);
    let mut held = sink();
    pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut held,
        budget(0, 1),
    )
    .expect("unused disconnected pose channel is not media fatal");
    assert!(ends.responses.try_recv().is_err());
    assert!(matches!(
        pump_queue_turn(
            &mut pump,
            &ends.poses,
            &ends.requests,
            &ends.responses,
            &stop,
            &mut held,
            budget(1, 1),
        ),
        Err(RuntimeError::MediaFatal)
    ));
}

#[test]
fn frame_local_ingest_and_gap_failures_preserve_both_sources_pending_work() {
    let mut pump = ungated(vec![camera(19), camera(20)]);
    let ends = ends(4, 4);
    let stop = AtomicBool::new(false);
    let mut held = sink();
    let first = due(&mut pump, &ends, &stop, &mut held, 19, &[7]);
    let second = due(&mut pump, &ends, &stop, &mut held, 20, &[7]);
    assert_eq!(pump.pending_scores(), 2);
    held.ready.clear();
    let mut invalid = packet(19, first[0].frame.sequence + 1, &[7]);
    invalid.frame.source_width = 0;
    let mut gap = packet(19, first[0].frame.sequence + 2, &[7]);
    gap.frame.pts_ns = 120_000_000_000;
    let failed_frame = gap.frame;
    ends.pose_tx.send(invalid).unwrap();
    ends.pose_tx.send(gap).unwrap();
    pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut held,
        budget(2, 4),
    )
    .expect("Python frame-local failures continue");
    assert_eq!(
        pump.pending_scores(),
        2,
        "neither source loses its original pending request"
    );
    assert!(!stop.load(Ordering::SeqCst));
    assert!(held.ready.is_empty());
    assert!(held.triggered.is_empty());
    assert!(ends.request_rx.try_recv().is_err());
    assert_eq!(held.failures[&19], 2);
    assert_eq!(
        held.failure_receipts.len(),
        2,
        "every failure keeps its diagnostic"
    );
    assert!(
        held.failure_receipts[0]
            .diagnostic("actual-camera-id")
            .contains("SourceSize")
    );
    assert_eq!(held.failure_receipts[0].count, 1);
    let failure = &held.failure_receipts[1];
    assert_eq!(failure.count, 2);
    assert_eq!(failure.frame, failed_frame);
    assert!(matches!(
        &failure.error,
        crate::run::pump::PumpError::Response(crate::run::policy::FallResponseError::Policy(
            crate::policy::fall::FallStageError::Gap(_)
        ))
    ));
    let message = failure.diagnostic("actual-camera-id");
    assert!(message.contains("camera_id=actual-camera-id"));
    assert!(message.contains("failure_count=2"));
    assert!(message.contains("source_id=19"));
    assert!(message.contains("Gap"));
    for request in first.iter().chain(&second) {
        ends.response_tx.send(answered(request, 0.5)).unwrap();
    }
    pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut held,
        budget(0, 4),
    )
    .unwrap();
    assert_eq!(pump.pending_scores(), 0);
    assert_eq!(held.ready.len(), 2);
    assert_eq!(held.ready[0].frame, first[0].frame);
    assert_eq!(held.ready[1].frame, second[0].frame);
}

#[test]
fn routing_and_accelerator_faults_never_become_frame_local_diagnostics() {
    let mut pump = ungated(vec![camera(19)]);
    let ends = ends(2, 2);
    let stop = AtomicBool::new(false);
    let mut held = sink();
    ends.pose_tx.send(packet(88, 0, &[7])).unwrap();
    assert!(matches!(
        pump_queue_turn(
            &mut pump,
            &ends.poses,
            &ends.requests,
            &ends.responses,
            &stop,
            &mut held,
            budget(1, 4)
        ),
        Err(RuntimeError::Pump(
            crate::run::pump::PumpError::UnknownSource(88)
        ))
    ));
    assert!(held.failures.is_empty());
    ends.pose_tx.send(packet(19, u64::MAX, &[7])).unwrap();
    assert!(matches!(
        pump_queue_turn(
            &mut pump,
            &ends.poses,
            &ends.requests,
            &ends.responses,
            &stop,
            &mut held,
            budget(1, 4),
        ),
        Err(RuntimeError::Pump(crate::run::pump::PumpError::Ingest {
            cause: crate::policy::ingest::IngestRefusal::Sequence,
            ..
        }))
    ));
    assert!(
        held.failures.is_empty(),
        "Rust-only identity overflow is not a Python ValueError fallback"
    );
    ends.response_tx
        .send(FallResponse {
            frame: packet(88, 0, &[7]).frame,
            track_id: 7,
            score: Err(seeon_worker_runtime::fall_gpu::FallGpuError::Poisoned.into()),
        })
        .unwrap();
    let error = pump_queue_turn(
        &mut pump,
        &ends.poses,
        &ends.requests,
        &ends.responses,
        &stop,
        &mut held,
        budget(0, 4),
    )
    .unwrap_err();
    assert_eq!(error.exit(), crate::exit::Exit::FatalAccelerator);
    assert!(stop.load(Ordering::SeqCst));
    assert!(held.failures.is_empty());
}

#[test]
fn frame_failure_counters_are_bounded_without_eviction_or_overflow() {
    use crate::run::pump::PumpError;
    let mut held = sink();
    let error = PumpError::Ingest {
        source_id: 0,
        cause: crate::policy::ingest::IngestRefusal::SourceSize,
    };
    for source_id in 0..seeon_deepstream_native::MEDIA_MAX_SOURCES {
        held.frame_failure(packet(source_id as u32, 0, &[]).frame, &error)
            .unwrap();
    }
    assert_eq!(
        held.frame_failure(packet(99, 0, &[]).frame, &error),
        Err(PumpError::FrameFailureRetention)
    );
    assert_eq!(
        held.failures.len(),
        seeon_deepstream_native::MEDIA_MAX_SOURCES
    );
    held.frame_failure(packet(0, 1, &[]).frame, &error).unwrap();
    assert_eq!(held.failures[&0], 2);
    held.failures.insert(0, u64::MAX);
    let prior = held.failure_receipts.clone();
    assert_eq!(
        held.frame_failure(packet(0, 2, &[]).frame, &error),
        Err(PumpError::FrameFailureRetention)
    );
    assert_eq!(held.failures[&0], u64::MAX);
    assert_eq!(held.failure_receipts, prior);
    held.failures.insert(0, 2);
    while held.failure_receipts.len()
        < seeon_deepstream_native::MEDIA_MAX_SOURCES * crate::msg::POSE_PER_CAMERA
    {
        held.frame_failure(packet(0, 3, &[]).frame, &error).unwrap();
    }
    let count = held.failures[&0];
    let prefix = held.failure_receipts.clone();
    assert_eq!(
        held.frame_failure(packet(0, 4, &[]).frame, &error),
        Err(PumpError::FrameFailureRetention)
    );
    assert_eq!(held.failures[&0], count);
    assert_eq!(held.failure_receipts, prefix);
}

#[test]
fn failed_score_handoff_retains_pending_suffix_and_unrelated_events() {
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let path = std::env::temp_dir().join(RandomIds.uuid4().unwrap());
    std::fs::create_dir(&path).unwrap();
    let root = Scratch(path);
    let clock = SystemClock::new();
    let boot_id = "00000000-0000-4000-8000-000000000146";
    let binding = MediaBinding {
        token: 11,
        generation: 3,
        epoch: 5,
    };
    let cameras = [(
        19_u32,
        binding,
        "camera-19".to_owned(),
        "facility-19".to_owned(),
    )];
    let (commands, _command_rx, _receipts, records) = publication::channels();
    let publications = publication::open(
        publication::PublicationConfig {
            boot_id,
            state_dir: &root.0.join("state"),
            record_dir: &root.0.join("records"),
            store_root: &root.0.join("store"),
            cameras: &cameras,
            config_version: 1,
            manifest_sha: None,
        },
        commands,
        records,
        Arc::new(SystemClock::new()),
    )
    .unwrap();
    let lanes = Arc::new(Lanes::new(8).unwrap());
    let mut session = crate::run::execution::output::tests::session(
        PolicyPump::new(
            vec![CameraPolicy {
                source_id: 19,
                stage: None,
            }],
            None,
            Arc::new(SystemClock::new()),
        )
        .unwrap(),
        publications,
        boot_id.to_owned(),
    );
    session.records = Some(Arc::clone(&lanes));
    let evidence = score(1.0).accelerator().copied();
    let frame = |sequence, source_id| {
        let mut identity = packet(source_id, sequence, &[1]).frame;
        identity.binding = binding;
        identity
    };
    let ready = |frame, track_id, logit| super::FrameScores {
        frame,
        scores: vec![super::ReadyScore {
            track_id,
            logit,
            evidence,
            generation: Some(1),
        }],
    };
    let event = |frame| {
        super::PendingEvent::held(
            frame,
            BusinessEvent {
                domain: "fall".into(),
                event_type: "fall".into(),
                identity: "unrelated-event".into(),
                camera_id: "camera-19".into(),
                facility_id: "facility-19".into(),
                time_sec: 1.0,
                probability: Some(0.5),
                person_id: Some(7),
                bed_id: None,
            },
        )
    };
    let mut held = sink();
    held.ready = vec![
        ready(frame(41, 19), 7, 1.0),
        ready(frame(42, 99), 8, 1.0),
        ready(frame(43, 19), 9, 1.0),
    ];
    held.triggered.push(event(frame(41, 19)));
    let unrelated = held.triggered.clone();
    assert!(matches!(
        super::apply_sink(&mut session, &clock, &mut held),
        Err(RuntimeError::Identity)
    ));
    assert!(held.failure_receipts.is_empty());
    assert_eq!(held.triggered, unrelated);
    assert_eq!(
        held.ready
            .iter()
            .map(|item| item.frame.sequence)
            .collect::<Vec<_>>(),
        vec![42, 43]
    );
    let first = lanes
        .drain_for("camera-19", boot_id, 8)
        .unwrap()
        .expect("emitted prefix");
    let first_id = first.records[0].record_id().to_owned();
    assert_eq!(first.records[0].body().frame_seq, Some(41));
    assert!(matches!(
        super::apply_sink(&mut session, &clock, &mut held),
        Err(RuntimeError::Identity)
    ));
    assert_eq!(held.triggered, unrelated);
    assert!(lanes.drain_for("camera-19", boot_id, 8).unwrap().is_none());
    // Repair only the synthetic attribution fault; no pending score is dropped.
    held.ready[0].frame.source_id = 19;
    // This slice tests score handoff, not the independent event staging path.
    held.triggered.clear();
    super::apply_sink(&mut session, &clock, &mut held).unwrap();
    let resumed = lanes.drain_for("camera-19", boot_id, 8).unwrap().unwrap();
    assert!(
        resumed
            .records
            .iter()
            .all(|record| record.record_id() != first_id)
    );
    assert_eq!(
        resumed
            .records
            .iter()
            .map(|record| record.body().frame_seq)
            .collect::<Vec<_>>(),
        vec![Some(42), Some(43)]
    );
    super::apply_sink(&mut session, &clock, &mut held).unwrap();
    assert!(lanes.drain_for("camera-19", boot_id, 8).unwrap().is_none());
    let mut record_failure = sink();
    record_failure.ready = vec![
        ready(frame(51, 19), 11, 1.0),
        super::FrameScores {
            frame: frame(52, 19),
            scores: vec![
                super::ReadyScore {
                    track_id: 12,
                    logit: 1.0,
                    evidence,
                    generation: Some(1),
                },
                super::ReadyScore {
                    track_id: 120,
                    logit: f32::NAN,
                    evidence,
                    generation: Some(1),
                },
                super::ReadyScore {
                    track_id: 121,
                    logit: 1.0,
                    evidence,
                    generation: Some(1),
                },
            ],
        },
        ready(frame(53, 19), 13, 1.0),
    ];
    record_failure.triggered.push(event(frame(51, 19)));
    let events = record_failure.triggered.clone();
    assert!(matches!(
        super::apply_sink(&mut session, &clock, &mut record_failure),
        Err(RuntimeError::Identity)
    ));
    assert_eq!(record_failure.triggered, events);
    assert_eq!(
        record_failure.ready[0]
            .scores
            .iter()
            .map(|score| score.track_id)
            .collect::<Vec<_>>(),
        vec![120, 121]
    );
    assert_eq!(record_failure.ready[1].frame.sequence, 53);
    let second = lanes.drain_for("camera-19", boot_id, 8).unwrap().unwrap();
    assert!(
        second
            .records
            .iter()
            .all(|record| record.record_id() != first_id)
    );
    assert_eq!(second.records[1].body().frame_seq, Some(52));
    assert!(matches!(
        super::apply_sink(&mut session, &clock, &mut record_failure),
        Err(RuntimeError::Identity)
    ));
    assert!(lanes.drain_for("camera-19", boot_id, 8).unwrap().is_none());
    assert_eq!(record_failure.ready[0].scores[0].track_id, 120);
    assert_eq!(record_failure.triggered, events);
}
#[test]
fn recording_admission_failure_retains_failed_event_and_does_not_restage() {
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let path = std::env::temp_dir().join(RandomIds.uuid4().unwrap());
    std::fs::create_dir(&path).unwrap();
    let root = Scratch(path);
    let clock = SystemClock::new();
    let boot_id = "00000000-0000-4000-8000-000000000147";
    let binding = MediaBinding {
        token: 11,
        generation: 3,
        epoch: 5,
    };
    let cameras = [(
        19_u32,
        binding,
        "camera-19".to_owned(),
        "facility-19".to_owned(),
    )];
    let (commands, command_rx, _receipts, records) = publication::channels();
    let publications = publication::open(
        publication::PublicationConfig {
            boot_id,
            state_dir: &root.0.join("state"),
            record_dir: &root.0.join("records"),
            store_root: &root.0.join("store"),
            cameras: &cameras,
            config_version: 1,
            manifest_sha: None,
        },
        commands,
        records,
        Arc::new(SystemClock::new()),
    )
    .unwrap();
    let lanes = Arc::new(Lanes::new(8).unwrap());
    let mut session = crate::run::execution::output::tests::session(
        PolicyPump::new(
            vec![CameraPolicy {
                source_id: 19,
                stage: None,
            }],
            None,
            Arc::new(SystemClock::new()),
        )
        .unwrap(),
        publications,
        boot_id.to_owned(),
    );
    session.records = Some(Arc::clone(&lanes));
    let frame = |sequence| {
        let mut identity = packet(19, sequence, &[1]).frame;
        identity.binding = binding;
        identity
    };
    let event = |sequence, identity: &str| {
        super::PendingEvent::held(
            frame(sequence),
            BusinessEvent {
                domain: "fall".into(),
                event_type: "fall".into(),
                identity: identity.into(),
                camera_id: "camera-19".into(),
                facility_id: "facility-19".into(),
                time_sec: 1.0,
                probability: Some(0.5),
                person_id: Some(7),
                bed_id: None,
            },
        )
    };
    let mut held = sink();
    held.triggered = vec![
        event(41, "00000000-0000-4000-8000-0000000000a1"),
        event(42, "00000000-0000-4000-8000-0000000000a2"),
    ];
    let suffix = held.triggered[1].clone();
    let capacity = crate::clips::recorder::MAX_PENDING_ALERTS;
    let detected = crate::clips::time::Utc::parse("2026-01-01T00:00:00Z").unwrap();
    session.publications.recorders[0].quiesce();
    for index in 0..capacity {
        session.publications.recorders[0]
            .admit(&format!("prior-{index}"), detected)
            .unwrap();
    }
    assert!(matches!(
        super::apply_sink(&mut session, &clock, &mut held),
        Err(RuntimeError::Publication(
            publication::PublicationError::Recorder(
                crate::clips::recorder::RecorderError::PendingFull
            )
        ))
    ));
    assert_eq!(held.triggered.len(), 2);
    assert_eq!(held.triggered[1], suffix);
    assert_eq!(held.triggered[0].frame.sequence, 41);
    assert_eq!(
        held.triggered[0].event.identity,
        "00000000-0000-4000-8000-0000000000a1"
    );
    assert!(held.triggered[0].prepared.is_some());
    let first_receipt = held.triggered[0]
        .staged
        .clone()
        .expect("durable acceptance precedes recorder admission");
    assert!(first_receipt.admission.accepted);
    assert!(!first_receipt.admission.already_admitted);
    assert_eq!(session.publications.recorders[0].pending(), capacity);
    assert_eq!(session.publications.queue.entries().unwrap().len(), 1);
    let before = lanes.drain_for("camera-19", boot_id, 8).unwrap().unwrap();
    assert_eq!(before.records.len(), 1);
    assert_eq!(before.records[0].body().frame_seq, Some(41));
    assert!(matches!(
        super::apply_sink(&mut session, &clock, &mut held),
        Err(RuntimeError::Publication(
            publication::PublicationError::Recorder(
                crate::clips::recorder::RecorderError::PendingFull
            )
        ))
    ));
    assert_eq!(held.triggered.len(), 2);
    assert_eq!(held.triggered[1], suffix);
    assert_eq!(held.triggered[0].staged.as_ref(), Some(&first_receipt));
    assert_eq!(session.publications.recorders[0].pending(), capacity);
    assert_eq!(session.publications.queue.entries().unwrap().len(), 1);
    assert!(lanes.drain_for("camera-19", boot_id, 8).unwrap().is_none());
    let prior = session.publications.recorders[0].take_unstarted().unwrap();
    assert_eq!(prior.len(), capacity);
    super::apply_sink(&mut session, &clock, &mut held).unwrap();
    assert!(held.triggered.is_empty());
    assert_eq!(session.publications.recorders[0].pending(), 2);
    assert_eq!(session.publications.queue.entries().unwrap().len(), 2);
    let resumed = lanes.drain_for("camera-19", boot_id, 8).unwrap().unwrap();
    assert_eq!(resumed.records.len(), 1);
    assert_eq!(resumed.records[0].body().frame_seq, Some(42));
    assert!(
        matches!(
            command_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ),
        "quiesced admission never starts native media"
    );
}
