//! Shared real fall-stage and publication fixtures.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::mpsc;

use seeon_deepstream_native::{FrameIdentity, MediaBinding};
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider, FallPolicyParameters};
use seeon_worker::trace::DecisionTraceSnapshot;

use crate::msg::{FallRequest, FallResponse, FallScore};
use crate::policy::fall::FallStage;
use crate::policy::ingest::Frame;
use crate::records::Record;
use crate::records::lanes::Lanes;
use crate::relay::cameras::policies::{
    PolicySource, PolicyValues, default_policy_bundle, make_effective_policy,
};
use crate::run::execution::{output, publication};
use crate::run::pump::{CameraPolicy, PolicyPump, PolicySink};
use crate::seam::{Clock, IdSource, RandomIds, SystemClock};

use super::super::apply_sink;
use super::{BOOT, LiveSink, RuntimeError};

struct Scratch(std::path::PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) struct Fixture {
    pub(super) session: output::Session,
    pub(super) lanes: Arc<Lanes>,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) policy_ids: [String; 2],
    _root: Scratch,
}

impl Fixture {
    pub(super) fn new(capacity: u64) -> Self {
        Self::with_clock(capacity, Arc::new(SystemClock::new()))
    }

    pub(super) fn with_clock(capacity: u64, clock: Arc<dyn Clock>) -> Self {
        let root = Scratch(std::env::temp_dir().join(RandomIds.uuid4().unwrap()));
        std::fs::create_dir(&root.0).unwrap();
        let cameras = [19_u32, 20].map(|source_id| {
            (
                source_id,
                input(source_id, 0, &[]).identity.binding,
                format!("camera-{source_id}"),
                "facility-1".to_owned(),
            )
        });
        let (commands, _command_rx, _receipts, records) = publication::channels();
        let publications = publication::open(
            publication::PublicationConfig {
                boot_id: BOOT,
                state_dir: &root.0.join("state"),
                record_dir: &root.0.join("records"),
                store_root: &root.0.join("store"),
                cameras: &cameras,
                config_version: 1,
                manifest_sha: None,
            },
            commands,
            records,
            Arc::clone(&clock),
        )
        .unwrap();
        let mut session = output::tests::session(
            PolicyPump::new(
                cameras
                    .iter()
                    .map(|(source_id, ..)| CameraPolicy {
                        source_id: *source_id,
                        stage: None,
                    })
                    .collect(),
                None,
                Arc::clone(&clock),
            )
            .unwrap(),
            publications,
            BOOT.to_owned(),
        );
        let ids: Vec<_> = cameras
            .iter()
            .map(|(_, _, camera, _)| camera.clone())
            .collect();
        let mut policies = default_policy_bundle(&ids).unwrap();
        let override_policy = make_effective_policy(
            "fall",
            2,
            &PolicyValues::FallV2 {
                transition_threshold: 0.65,
            },
            PolicySource::CameraOverride,
            None,
            Some(7),
        )
        .unwrap();
        let policy_ids = [
            policies.defaults["fall"].effective_policy_id.clone(),
            override_policy.effective_policy_id.clone(),
        ];
        policies
            .cameras
            .get_mut("camera-19")
            .unwrap()
            .remove("fall");
        policies
            .cameras
            .get_mut("camera-20")
            .unwrap()
            .insert("fall".into(), override_policy);
        session.decision_policies =
            output::decision_policies(&policies, ids.iter().map(String::as_str)).unwrap();
        let lanes = Arc::new(Lanes::new(capacity).unwrap());
        session.records = Some(Arc::clone(&lanes));
        // Exercise the same handoff used by shutdown drains without native media.
        session.request_media_stop();
        Self {
            session,
            lanes,
            clock,
            policy_ids,
            _root: root,
        }
    }

    pub(super) fn deliver(&mut self, sink: &mut LiveSink) -> Result<(), RuntimeError> {
        apply_sink(&mut self.session, self.clock.as_ref(), sink)
    }

    pub(super) fn drain(&self, source_id: u32) -> Vec<Record> {
        self.lanes
            .drain_for(&format!("camera-{source_id}"), BOOT, 256)
            .unwrap()
            .map_or_else(Vec::new, |drained| drained.records)
    }
}

pub(super) fn input(source_id: u32, sequence: u64, tracks: &[u64]) -> Frame {
    Frame {
        identity: FrameIdentity {
            binding: MediaBinding {
                token: u64::from(source_id) + 1,
                generation: 3,
                epoch: 5,
            },
            source_id,
            sequence,
            pts_ns: sequence * 100_000_000,
            pts_valid: 1,
            frame_number: sequence as i64,
            source_width: 640,
            source_height: 360,
            analysis_width: 640,
            analysis_height: 360,
            ..FrameIdentity::default()
        },
        width: 640,
        height: 360,
        live_track_ids: tracks.to_vec(),
        rows: tracks
            .iter()
            .map(|&id| (id, [0.5; 56]))
            .collect::<BTreeMap<_, _>>(),
        time_sec: Some(sequence as f64 / 10.0),
        frame_index: sequence as i64,
    }
}

pub(super) fn stage(source_id: u32) -> FallStage {
    FallStage::new(
        FallPolicyDecider::new(
            format!("camera-{source_id}"),
            "facility-1",
            BOOT,
            "5",
            3,
            FallPolicy::new(FallPolicyParameters {
                transition_votes: 1,
                ..Default::default()
            })
            .unwrap(),
            FallCapacities {
                retained_tracks: 8,
                generation_identities: 8,
                episodes: 8,
                vote_window: 5,
            },
        )
        .unwrap(),
        1.0,
    )
    .unwrap()
}

pub(super) fn next_due(
    stage: &mut FallStage,
    source_id: u32,
    sequence: &mut u64,
    tracks: &[u64],
) -> Vec<FallRequest> {
    let (sender, receiver) = mpsc::sync_channel(64);
    for _ in 0..120 {
        stage
            .observe(&input(source_id, *sequence, tracks), &sender, &mut |_| {})
            .unwrap();
        *sequence += 1;
        let requests: Vec<_> = receiver.try_iter().collect();
        if !requests.is_empty() {
            assert_eq!(requests.len(), tracks.len());
            return requests;
        }
    }
    panic!("synthetic pose fixture produced no due fall window");
}

pub(super) fn consume(
    stage: &mut FallStage,
    requests: &[FallRequest],
    logit: f32,
    sink: &mut LiveSink,
) -> Vec<DecisionTraceSnapshot> {
    let mut snapshots = Vec::new();
    for request in requests {
        let score = FallScore::Cpu(logit);
        sink.score(request.frame, request.track_id, &score).unwrap();
        stage
            .consume(
                FallResponse {
                    frame: request.frame,
                    track_id: request.track_id,
                    score: Ok(score),
                },
                &mut |update| {
                    snapshots.extend_from_slice(update.snapshots);
                    sink.decision(update);
                },
            )
            .unwrap();
    }
    snapshots
}
