//! Synchronous CPU policy composition; the root owns channels and drain deadlines.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::SyncSender;

use seeon_deepstream_native::FrameIdentity;
use seeon_worker::detection_window::{AwareDateTime, DetectionWindow, DetectionWindowError};

use super::policy::{FallResponseError, validate_fall_response};
use crate::exit::Exit;
use crate::msg::{FallRequest, FallResponse, FallScore, PosePacket};
use crate::policy::fall::{DecisionUpdate, FallStage, FallStageError};
use crate::policy::ingest::{IngestRefusal, ingest};
use crate::seam::Clock;

/// `stage` is `None` when the effective fall domain is disabled. The source
/// remains admitted; absence is not a placeholder stage.
pub struct CameraPolicy {
    pub source_id: u32,
    pub stage: Option<FallStage>,
}

/// Synchronous, non-reentrant, receipt-only callbacks. The root supplies a real
/// sink. Events are delivered only through `decision`, never again via returns.
pub trait PolicySink {
    fn decision(&mut self, update: DecisionUpdate<'_>);
    fn score(
        &mut self,
        frame: FrameIdentity,
        track_id: u64,
        score: &FallScore,
    ) -> Result<(), PumpError>;
    /// Root queue exception boundary. `PolicyPump::observe` does not call this;
    /// it returns the typed error. The root records a source-size or PTS-gap
    /// failure and continues; identity overflow, routing, GPU, clock and
    /// capacity faults remain fatal.
    fn frame_failure(&mut self, frame: FrameIdentity, error: &PumpError) -> Result<(), PumpError>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum PumpError {
    ScoreRetention,
    /// The root could not retain another frame-local failure receipt.
    FrameFailureRetention,
    DuplicateSource(u32),
    /// A validated response arrived for an admitted source whose fall domain is off.
    InactiveSource(u32),
    UnknownSource(u32),
    Ingest {
        source_id: u32,
        cause: IngestRefusal,
    },
    Response(FallResponseError),
    /// Injected wall time could not be classified against the owned window.
    Window(DetectionWindowError),
}

impl PumpError {
    /// Duplicate roster entries are constructor configuration refusals (2).
    /// Routing/ingest/policy composition and sink capacity failures are runtime errors (1);
    /// fatal accelerator faults retain exit 4.
    pub const fn exit(&self) -> Exit {
        match self {
            Self::DuplicateSource(_) => Exit::Config,
            Self::UnknownSource(_) | Self::Ingest { .. } => Exit::Runtime,
            Self::InactiveSource(_) => Exit::Runtime,
            Self::Window(_) => Exit::Runtime,
            Self::ScoreRetention => Exit::Runtime,
            Self::FrameFailureRetention => Exit::Runtime,
            Self::Response(error) => error.exit(),
        }
    }
}

impl From<FallStageError> for PumpError {
    fn from(error: FallStageError) -> Self {
        Self::Response(FallResponseError::Policy(error))
    }
}

pub struct PolicyPump {
    cameras: Vec<CameraPolicy>,
    window: Option<DetectionWindow>,
    clock: Arc<dyn Clock>,
    replay: Option<super::replay::ReplayCapture>,
}

/// One admitted source, read before lifecycle flush. Counters are the stage's
/// existing totals; `pending_scores` is outstanding work, not a completed decision.
/// `last_decision` is present only when the decider already evaluated an update
/// and retained an explicit snapshot. Absence is not a negative decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PolicyExitObservation {
    pub(crate) source_id: u32,
    pub(crate) fall_stage: bool,
    pub(crate) counters: crate::policy::fall::FallCounters,
    pub(crate) pending_scores: usize,
    pub(crate) last_decision: Option<PolicyExitDecision>,
}

/// Bounded copy of one existing snapshot. Tokens come from `as_str`; no values,
/// subjects, or events are copied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PolicyExitDecision {
    pub(crate) reason: &'static str,
    pub(crate) previous_state: &'static str,
    pub(crate) current_state: &'static str,
    pub(crate) triggered: bool,
    pub(crate) samples: usize,
}

const EXIT_SUMMARY_BOUND: usize = 512;

impl PolicyExitObservation {
    /// Pre-finalization line. `phase` must name that moment; this does not
    /// sample again and does not classify the run.
    pub(crate) fn summary(&self, phase: &str, camera_id: Option<&str>) -> String {
        let camera_id = camera_id.unwrap_or("-");
        let decision = self.last_decision.as_ref().map_or_else(
            || "none".to_owned(),
            |sample| {
                format!(
                    "reason={} previous={} current={} triggered={} samples={}",
                    sample.reason,
                    sample.previous_state,
                    sample.current_state,
                    sample.triggered,
                    sample.samples
                )
            },
        );
        let mut line = format!(
            "ml-worker: policy exit {phase} source_id={} camera_id={camera_id} fall_stage={} pending_scores={} missing_observations={} resample_gap_rows={} stale_responses={} last_decision={decision}",
            self.source_id,
            self.fall_stage,
            self.pending_scores,
            self.counters.missing_observations,
            self.counters.resample_gap_rows,
            self.counters.stale_responses,
        );
        if line.len() > EXIT_SUMMARY_BOUND {
            let mut end = EXIT_SUMMARY_BOUND;
            while !line.is_char_boundary(end) {
                end -= 1;
            }
            line.truncate(end);
            let _ = write!(line, " truncated=1");
        }
        line
    }
}

impl PolicyPump {
    /// Keeps declared roster order for flush, independently of source ID order.
    /// An empty roster is a valid idle pump.
    pub fn new(
        cameras: Vec<CameraPolicy>,
        window: Option<DetectionWindow>,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, PumpError> {
        let mut sources = BTreeSet::new();
        for camera in &cameras {
            if !sources.insert(camera.source_id) {
                return Err(PumpError::DuplicateSource(camera.source_id));
            }
        }
        Ok(Self {
            cameras,
            window,
            clock,
            replay: None,
        })
    }

    pub(crate) fn set_replay(&mut self, replay: super::replay::ReplayCapture) {
        self.replay = Some(replay);
    }

    /// Called only after the runtime actually observed an empty metadata queue.
    /// An exhausted pose/response budget is not evidence of unavailable input.
    pub(crate) fn replay_wait_empty(&mut self) {
        if let Some(replay) = &mut self.replay {
            replay.wait_empty(self.clock.monotonic());
        }
    }

    pub fn observe(
        &mut self,
        packet: &PosePacket,
        requests: &SyncSender<FallRequest>,
        sink: &mut dyn PolicySink,
    ) -> Result<(), PumpError> {
        let result = self.observe_frame(packet, requests, sink);
        if let Some(replay) = &mut self.replay {
            // Metadata arrived even when its processing refused this frame.
            // Processing time itself is not a metadata-wait timeout.
            replay.processed(packet.frame.source_id, self.clock.monotonic());
        }
        result
    }

    fn observe_frame(
        &mut self,
        packet: &PosePacket,
        requests: &SyncSender<FallRequest>,
        sink: &mut dyn PolicySink,
    ) -> Result<(), PumpError> {
        let source_id = packet.frame.source_id;
        let index = self
            .cameras
            .iter()
            .position(|camera| camera.source_id == source_id)
            .ok_or(PumpError::UnknownSource(source_id))?;
        let mut observed = Vec::new();
        let capture = self.replay.is_some().then_some(&mut observed);
        let frame =
            ingest(packet, capture).map_err(|cause| PumpError::Ingest { source_id, cause })?;
        if let Some(replay) = &mut self.replay {
            replay.capture(&frame, &observed, self.clock.as_ref());
        }
        let Some(stage) = self.cameras[index].stage.as_mut() else {
            return Ok(());
        };
        let outside = match &self.window {
            None => false,
            Some(window) => {
                let now = AwareDateTime::from_utc_system_time(self.clock.wall())
                    .map_err(PumpError::Window)?;
                !window.contains(now).map_err(PumpError::Window)?
            }
        };
        if outside {
            stage.skip_outside_window(&frame, &mut |update| sink.decision(update))?;
        } else {
            stage.observe(&frame, requests, &mut |update| sink.decision(update))?;
        }
        Ok(())
    }

    pub fn consume(
        &mut self,
        response: FallResponse,
        stop: &AtomicBool,
        sink: &mut dyn PolicySink,
    ) -> Result<(), PumpError> {
        validate_fall_response(&response, stop).map_err(PumpError::Response)?;
        let source_id = response.frame.source_id;
        let Some(stage) = self.stage(source_id)? else {
            return Err(PumpError::InactiveSource(source_id));
        };
        if let Ok(score) = &response.score
            && stage.expects_response(&response)
        {
            sink.score(response.frame, response.track_id, score)?;
        }
        stage.consume(response, &mut |update| sink.decision(update))?;
        Ok(())
    }

    /// Completes real pending updates in roster order, stopping at the first
    /// failure without retracting any earlier successful receipt.
    pub fn flush(&mut self, sink: &mut dyn PolicySink) -> Result<(), PumpError> {
        for camera in &mut self.cameras {
            if let Some(stage) = &mut camera.stage {
                stage.flush(&mut |update| sink.decision(update))?;
            }
        }
        Ok(())
    }

    pub fn source_count(&self) -> usize {
        self.cameras.len()
    }

    pub fn pending_scores(&self) -> usize {
        self.cameras
            .iter()
            .filter_map(|camera| camera.stage.as_ref())
            .map(FallStage::pending_scores)
            .sum()
    }

    /// Roster-order read of admitted policy state. Does not flush, observe,
    /// consume, or otherwise change a decision.
    pub(crate) fn exit_observations(&self) -> impl Iterator<Item = PolicyExitObservation> + '_ {
        self.cameras.iter().map(|camera| {
            let Some(stage) = camera.stage.as_ref() else {
                return PolicyExitObservation {
                    source_id: camera.source_id,
                    fall_stage: false,
                    counters: crate::policy::fall::FallCounters::default(),
                    pending_scores: 0,
                    last_decision: None,
                };
            };
            let snapshots = stage.decider().last_trace_snapshots();
            let last_decision = stage.decider().last_update_evaluated().then(|| {
                let sample = snapshots.first();
                PolicyExitDecision {
                    reason: sample.map_or("absent", |item| item.reason.as_str()),
                    previous_state: sample.map_or("absent", |item| item.previous_state.as_str()),
                    current_state: sample.map_or("absent", |item| item.current_state.as_str()),
                    triggered: sample.is_some_and(|item| item.triggered),
                    samples: snapshots.len(),
                }
            });
            PolicyExitObservation {
                source_id: camera.source_id,
                fall_stage: true,
                counters: stage.counters(),
                pending_scores: stage.pending_scores(),
                last_decision,
            }
        })
    }

    fn stage(&mut self, source_id: u32) -> Result<Option<&mut FallStage>, PumpError> {
        self.cameras
            .iter_mut()
            .find(|camera| camera.source_id == source_id)
            .map(|camera| camera.stage.as_mut())
            .ok_or(PumpError::UnknownSource(source_id))
    }
}

#[cfg(test)]
mod exit_observation_tests {
    use super::PolicyPump;
    use crate::policy::fall::{FallCounters, FallStage};
    use crate::seam::SystemClock;
    use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider};
    use std::sync::Arc;

    fn stage(camera_id: &str) -> FallStage {
        let decider = FallPolicyDecider::new(
            camera_id,
            "facility-1",
            "boot-1",
            "epoch-1",
            0,
            FallPolicy::default(),
            FallCapacities {
                retained_tracks: 1,
                generation_identities: 1,
                episodes: 1,
                vote_window: 5,
            },
        )
        .expect("valid test policy");
        FallStage::new(decider, 1.0).expect("valid calibration")
    }

    #[test]
    fn exit_observations_copy_real_counters_without_inventing_a_decision() {
        let mut active = stage("camera-7");
        active
            .consume(
                crate::msg::FallResponse {
                    frame: seeon_deepstream_native::FrameIdentity::default(),
                    track_id: 1,
                    score: Err(seeon_worker_runtime::fall_gpu::FallGpuError::Window.into()),
                },
                &mut |_| panic!("a stale response must not decide"),
            )
            .expect("stale response counted");
        let pump = PolicyPump::new(
            vec![
                super::CameraPolicy {
                    source_id: 7,
                    stage: Some(active),
                },
                super::CameraPolicy {
                    source_id: 9,
                    stage: None,
                },
            ],
            None,
            Arc::new(SystemClock::new()),
        )
        .expect("roster");
        let observed: Vec<_> = pump.exit_observations().collect();
        assert_eq!(observed.len(), 2);
        assert!(observed[0].fall_stage);
        assert_eq!(observed[0].source_id, 7);
        assert_eq!(observed[0].pending_scores, 0);
        assert_eq!(
            observed[0].counters,
            FallCounters {
                stale_responses: 1,
                ..FallCounters::default()
            }
        );
        assert_eq!(observed[0].last_decision, None);
        assert!(!observed[1].fall_stage);
        assert_eq!(observed[1].source_id, 9);
        assert_eq!(observed[1].counters, FallCounters::default());
        assert_eq!(observed[1].last_decision, None);
        let line = observed[0].summary("pre-finalization", Some("camera-7"));
        assert!(line.starts_with("ml-worker: policy exit pre-finalization "));
        assert!(line.contains("source_id=7"));
        assert!(line.contains("camera_id=camera-7"));
        assert!(line.contains("fall_stage=true"));
        assert!(line.contains("pending_scores=0"));
        assert!(line.contains("missing_observations=0"));
        assert!(line.contains("stale_responses=1"));
        assert!(line.contains("last_decision=none"));
        assert!(line.len() <= super::EXIT_SUMMARY_BOUND);
        let absent = observed[1].summary("pre-finalization", None);
        assert!(absent.contains("camera_id=-"));
        assert!(absent.contains("fall_stage=false"));
        let unicode = observed[0].summary("pre-finalization", Some(&"카메라".repeat(100)));
        assert!(unicode.ends_with(" truncated=1"));
        assert!(unicode.len() <= super::EXIT_SUMMARY_BOUND + " truncated=1".len());
        let again: Vec<_> = pump.exit_observations().collect();
        assert_eq!(again, observed);
    }
}
