//! Production loop. Fall responses are classified by the existing pump before
//! consume. Restart requests process shutdown and never reopens media.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};

use seeon_deepstream_native::{MEDIA_MAX_OBJECTS, MEDIA_MAX_SOURCES, PosePacket};

use crate::config::pull;
use crate::config::restart::{RESTART_POLL_INTERVAL, RestartCheck, RestartDirective};
use crate::exit::Exit;
use crate::msg::{FALL_RESPONSE_CAPACITY, FallRequest, FallResponse, POSE_PER_CAMERA};
use crate::poll::POLL_INTERVAL;
use crate::run::Booted;
use crate::run::pump::{PolicyPump, PolicySink};
use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;

use super::RELAY_URL;
use crate::run::events::{PreparedEvent, StagedEvent};

use super::output::Session;
use super::publication::PublicationError;

#[derive(Debug)]
pub enum RuntimeError {
    Pump(crate::run::pump::PumpError),
    Publication(PublicationError),
    Delivery,
    MediaFatal,
    MediaOwner(Exit),
    ModelOwner(Exit),
    Identity,
    Configuration,
    WindowAsset(crate::config::windows::WindowError),
}

impl RuntimeError {
    pub const fn exit(&self) -> Exit {
        match self {
            Self::Pump(error) => error.exit(),
            Self::MediaFatal => Exit::FatalAccelerator,
            Self::MediaOwner(exit) | Self::ModelOwner(exit) => *exit,
            Self::Identity | Self::Configuration | Self::WindowAsset(_) => Exit::Runtime,
            Self::Publication(_) | Self::Delivery => Exit::Runtime,
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pump(_) => formatter.write_str("policy pump refused a live frame or response"),
            Self::Publication(_) => formatter.write_str("publication drain refused"),
            Self::Delivery => formatter.write_str("delivery sender failed"),
            Self::MediaFatal => formatter.write_str("media plane reported fatal"),
            Self::MediaOwner(exit) => {
                write!(formatter, "media owner failed with exit {}", exit.code())
            }
            Self::ModelOwner(exit) => {
                write!(formatter, "model owner failed with exit {}", exit.code())
            }
            Self::Identity => {
                formatter.write_str("event identity does not match the trigger frame")
            }
            Self::Configuration => formatter.write_str("runtime configuration is invalid"),
            Self::WindowAsset(error) => {
                write!(formatter, "configuration poll zone asset refused: {error}")
            }
        }
    }
}

pub enum RunOutcome {
    Restart(RestartDirective),
    Stopped,
    Failed(RuntimeError),
}

pub(super) struct TurnBudget {
    poses: usize,
    responses: usize,
}

struct ReadyScore {
    track_id: u64,
    logit: f32,
    evidence: Option<seeon_worker_runtime::evidence::AcceleratorEvidence>,
    generation: Option<u64>,
}

struct FrameScores {
    frame: seeon_deepstream_native::FrameIdentity,
    scores: Vec<ReadyScore>,
}

const SCORE_RETENTION: usize = MEDIA_MAX_SOURCES * MEDIA_MAX_OBJECTS;

#[derive(Clone, Debug, PartialEq)]
struct FrameFailure {
    count: u64,
    frame: seeon_deepstream_native::FrameIdentity,
    error: crate::run::pump::PumpError,
}

impl FrameFailure {
    fn diagnostic(&self, camera_id: &str) -> String {
        let Self {
            frame,
            error,
            count,
        } = self;
        format!(
            "native policy frame failed: camera_id={camera_id} source_id={} frame_seq={} failure_count={} reason={error:?}",
            frame.source_id, frame.sequence, count,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PendingEvent {
    frame: seeon_deepstream_native::FrameIdentity,
    event: seeon_worker::episode::BusinessEvent,
    prepared: Option<PreparedEvent>,
    staged: Option<StagedEvent>,
}
#[cfg(test)]
impl PendingEvent {
    fn held(
        frame: seeon_deepstream_native::FrameIdentity,
        event: seeon_worker::episode::BusinessEvent,
    ) -> Self {
        Self {
            frame,
            event,
            prepared: None,
            staged: None,
        }
    }
}

pub(super) struct LiveSink {
    ready: Vec<FrameScores>,
    triggered: Vec<PendingEvent>,

    scores: Vec<super::output::HeldScore>,
    /// Cumulative per-source counts plus every failure from the bounded turn.
    failures: BTreeMap<u32, u64>,
    failure_receipts: Vec<FrameFailure>,
}

impl LiveSink {
    /// Before policy execution there are no accepted responses or events.
    pub(super) fn new() -> Self {
        Self {
            ready: Vec::new(),
            triggered: Vec::new(),
            scores: Vec::new(),
            failures: BTreeMap::new(),
            failure_receipts: Vec::new(),
        }
    }
}

impl PolicySink for LiveSink {
    fn frame_failure(
        &mut self,
        frame: seeon_deepstream_native::FrameIdentity,
        error: &crate::run::pump::PumpError,
    ) -> Result<(), crate::run::pump::PumpError> {
        use crate::run::pump::PumpError;
        if (!self.failures.contains_key(&frame.source_id)
            && self.failures.len() >= MEDIA_MAX_SOURCES)
            || self.failure_receipts.len() >= MEDIA_MAX_SOURCES * POSE_PER_CAMERA
        {
            return Err(PumpError::FrameFailureRetention);
        }
        let count = self
            .failures
            .get(&frame.source_id)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(PumpError::FrameFailureRetention)?;
        self.failures.insert(frame.source_id, count);
        self.failure_receipts.push(FrameFailure {
            count,
            frame,
            error: error.clone(),
        });
        Ok(())
    }

    fn decision(&mut self, update: crate::policy::fall::DecisionUpdate<'_>) {
        let frame = update.frame;
        let ready = self
            .scores
            .iter()
            .filter(|score| score.frame == frame)
            .map(|score| ReadyScore {
                track_id: score.track_id,
                logit: score.logit,
                evidence: score.evidence,
                generation: update.generation_for(score.track_id),
            })
            .collect();
        self.scores.retain(|score| score.frame != frame);
        self.ready.push(FrameScores {
            frame,
            scores: ready,
        });
        self.triggered
            .extend(update.events.iter().cloned().map(|event| PendingEvent {
                frame,
                event,
                prepared: None,
                staged: None,
            }));
    }

    fn score(
        &mut self,
        frame: seeon_deepstream_native::FrameIdentity,
        track_id: u64,
        score: &crate::msg::FallScore,
    ) -> Result<(), crate::run::pump::PumpError> {
        if self.scores.len() == SCORE_RETENTION {
            return Err(crate::run::pump::PumpError::ScoreRetention);
        }
        self.scores.push(super::output::HeldScore {
            frame,
            track_id,
            evidence: score.accelerator().copied(),
            logit: score.logit(),
        });
        Ok(())
    }
}

pub(super) fn apply_sink(
    session: &mut Session,
    clock: &dyn Clock,
    sink: &mut LiveSink,
) -> Result<(), RuntimeError> {
    deliver(session, clock, sink)
}

pub fn run(
    session: &mut Session,
    booted: &Booted,
    clock: &dyn Clock,
    shutdown: &Arc<ShutdownDeadline>,
    boot_directive: RestartDirective,
) -> (RunOutcome, LiveSink) {
    let mut restart = RestartCheck::new(boot_directive, RESTART_POLL_INTERVAL);
    let mut sink = LiveSink::new();
    let mut announced_ready = false;
    loop {
        if let Some(exit) = booted.models.failure() {
            session.request_media_stop();
            emit_pre_finalization_exit(session);
            return (RunOutcome::Failed(RuntimeError::ModelOwner(exit)), sink);
        }
        if shutdown.deadline().is_some() || session.stop.load(Ordering::SeqCst) {
            let failure = session
                .media
                .as_ref()
                .and_then(|media| media.diagnostics.snapshot().failure);
            session.request_media_stop();
            let outcome = failure.map_or(RunOutcome::Stopped, |exit| {
                RunOutcome::Failed(RuntimeError::MediaOwner(exit))
            });
            // Sample before lifecycle flush. Flush turns unanswered scores into
            // missing observations; these lines are the earlier counters.
            emit_pre_finalization_exit(session);
            return (outcome, sink);
        }
        if let Err(error) = check_sender(session) {
            session.request_media_stop();
            return (RunOutcome::Failed(error), sink);
        }
        match restart.check(clock, || poll_restart(booted)) {
            Ok(Some(directive)) => {
                session.request_media_stop();
                return (RunOutcome::Restart(directive), sink);
            }
            Ok(None) => {}
            Err(error) => {
                session.request_media_stop();
                return (RunOutcome::Failed(error), sink);
            }
        }
        if let Err(error) = turn(session, booted, clock, &mut sink) {
            session.request_media_stop();
            return (RunOutcome::Failed(error), sink);
        }
        if let Err(error) = deliver(session, clock, &mut sink) {
            session.request_media_stop();
            return (RunOutcome::Failed(error), sink);
        }
        if !announced_ready {
            eprintln!(
                "ml-worker: policy loop ready cameras={}",
                session.pump.source_count()
            );
            announced_ready = true;
        }
        clock.pause(POLL_INTERVAL);
    }
}

fn emit_pre_finalization_exit(session: &Session) {
    let diagnostics = session
        .media
        .as_ref()
        .map(|media| media.diagnostics.snapshot());
    for observation in session.pump.exit_observations() {
        let camera_id = session
            .publications
            .camera_for_diagnostic(observation.source_id);
        eprintln!("{}", observation.summary("pre-finalization", camera_id));
        let Some(diagnostics) = diagnostics.as_ref() else {
            eprintln!(
                "ml-worker: media exit pre-finalization source_id={} camera_id={} absent=1",
                observation.source_id,
                camera_id.unwrap_or("-")
            );
            continue;
        };
        let camera = usize::try_from(observation.source_id).ok();
        let counters = camera.and_then(|index| diagnostics.cameras.get(index));
        match counters {
            Some(counters) => eprintln!(
                "ml-worker: media exit pre-finalization source_id={} camera_id={} published_frames={} objects={} tensor_absent={} handoff_dropped_frames={}",
                observation.source_id,
                camera_id.unwrap_or("-"),
                counters.published_frames,
                counters.objects_observed,
                counters.frames_without_pose_tensor,
                counters.handoff_dropped_frames,
            ),
            None => eprintln!(
                "ml-worker: media exit pre-finalization source_id={} camera_id={} roster_absent=1",
                observation.source_id,
                camera_id.unwrap_or("-")
            ),
        }
    }
}

pub(super) fn pump_queue_turn(
    pump: &mut PolicyPump,
    poses: &Receiver<PosePacket>,
    requests: &SyncSender<FallRequest>,
    responses: &Receiver<FallResponse>,
    stop: &std::sync::atomic::AtomicBool,
    sink: &mut dyn PolicySink,
    budget: TurnBudget,
) -> Result<(), RuntimeError> {
    let TurnBudget {
        poses: pose_capacity,
        responses: response_capacity,
    } = budget;
    let mut poses_left = pose_capacity;
    let mut responses_left = response_capacity;
    if pose_capacity == 0 {
        return drain_responses(pump, responses, stop, sink, &mut responses_left);
    }
    while poses_left > 0 {
        drain_responses(pump, responses, stop, sink, &mut responses_left)?;
        if responses_left == 0 {
            return Ok(());
        }
        match poses.try_recv() {
            Ok(packet) => {
                poses_left -= 1;
                match pump.observe(&packet, requests, sink) {
                    Ok(()) => {}
                    Err(error) if frame_local(&error) => {
                        sink.frame_failure(packet.frame, &error)
                            .map_err(RuntimeError::Pump)?;
                    }
                    Err(error) => return Err(RuntimeError::Pump(error)),
                }
            }
            Err(TryRecvError::Empty) => {
                if !stop.load(Ordering::SeqCst) {
                    pump.replay_wait_empty();
                }
                break;
            }
            Err(TryRecvError::Disconnected) => return Err(RuntimeError::MediaFatal),
        }
    }
    drain_responses(pump, responses, stop, sink, &mut responses_left)?;
    Ok(())
}

/// Python NativePolicyPump catches these observe-path ValueError analogues
/// per frame. Routing, clocks, capacity and accelerator faults do not share
/// that boundary. In particular, this never resets a resampler on reentry.
fn frame_local(error: &crate::run::pump::PumpError) -> bool {
    use crate::policy::fall::FallStageError;
    use crate::run::policy::FallResponseError;
    use crate::run::pump::PumpError;
    matches!(
        error,
        PumpError::Ingest {
            cause: crate::policy::ingest::IngestRefusal::SourceSize,
            ..
        } | PumpError::Response(FallResponseError::Policy(FallStageError::Gap(_)))
    )
}

fn drain_responses(
    pump: &mut PolicyPump,
    responses: &Receiver<FallResponse>,
    stop: &std::sync::atomic::AtomicBool,
    sink: &mut dyn PolicySink,
    responses_left: &mut usize,
) -> Result<(), RuntimeError> {
    while *responses_left > 0 {
        match responses.try_recv() {
            Ok(response) => {
                *responses_left -= 1;
                pump.consume(response, stop, sink)
                    .map_err(RuntimeError::Pump)?;
            }
            Err(TryRecvError::Empty) => return Ok(()),
            Err(TryRecvError::Disconnected) => return Err(RuntimeError::MediaFatal),
        }
    }
    Ok(())
}

fn turn(
    session: &mut Session,
    booted: &Booted,
    clock: &dyn Clock,
    sink: &mut LiveSink,
) -> Result<(), RuntimeError> {
    let budget = TurnBudget {
        poses: POSE_PER_CAMERA.saturating_mul(session.pump.source_count()),
        responses: FALL_RESPONSE_CAPACITY,
    };
    let owner = booted.models.fall().ok_or(RuntimeError::MediaFatal)?;
    let responses = booted
        .models
        .fall_responses()
        .ok_or(RuntimeError::MediaFatal)?;
    let (pump, poses, stop) = session.queue_ends();
    pump_queue_turn(pump, poses, &owner.requests, responses, stop, sink, budget)?;
    advance_recordings(session, clock)?;
    if let Some(media) = &session.media {
        let snapshot = media.diagnostics.snapshot();
        if let Some(exit) = snapshot.failure {
            return Err(RuntimeError::MediaOwner(exit));
        }
        if snapshot.fatal {
            return Err(RuntimeError::MediaFatal);
        }
    }
    Ok(())
}

fn drain_recordings(session: &mut Session, clock: &dyn Clock) -> Result<(), RuntimeError> {
    let published = session
        .publications
        .drain_records(clock)
        .map_err(RuntimeError::Publication)?;
    if published && let Some(sender) = &session.sender {
        sender.wake();
    }
    Ok(())
}

fn advance_recordings(session: &mut Session, clock: &dyn Clock) -> Result<(), RuntimeError> {
    let receipt_observation_cutoff = clock.monotonic();
    drain_recordings(session, clock)?;
    session
        .publications
        .tick_recorders(receipt_observation_cutoff)
        .map_err(RuntimeError::Publication)
}

fn deliver(
    session: &mut Session,
    clock: &dyn Clock,
    sink: &mut LiveSink,
) -> Result<(), RuntimeError> {
    for failure in &sink.failure_receipts {
        let camera_id = session
            .publications
            .camera_for_diagnostic(failure.frame.source_id)
            .ok_or(RuntimeError::Identity)?;
        eprintln!("ml-worker: {}", failure.diagnostic(camera_id));
    }
    sink.failure_receipts.clear();
    let lanes = session.records.clone();
    let mut ready_index = 0usize;
    let mut ready_error = None;
    while ready_index < sink.ready.len() {
        let stream = match attributed(session, &sink.ready[ready_index].frame) {
            Ok(stream) => stream,
            Err(error) => {
                ready_error = Some(error);
                break;
            }
        };
        if let Some(lanes) = &lanes {
            let ready = &mut sink.ready[ready_index];
            let mut emitted = 0;
            for score in &ready.scores {
                let record =
                    match score_record(&stream, &ready.frame, score, clock, session.temperature) {
                        Some(record) => record,
                        None => {
                            ready_error = Some(RuntimeError::Identity);
                            break;
                        }
                    };
                let _ = lanes.try_emit(record);
                emitted += 1;
            }
            ready.scores.drain(..emitted);
        }
        if ready_error.is_some() {
            break;
        }
        ready_index += 1;
    }
    sink.ready.drain(..ready_index);
    if let Some(error) = ready_error {
        return Err(error);
    }
    let mut retired = 0;
    let delivered = (|| {
        for pending in &mut sink.triggered {
            let stream = attributed(session, &pending.frame)?;
            if pending.event.camera_id != stream.camera_id {
                return Err(RuntimeError::Identity);
            }
            let stager = session
                .publications
                .stager_index(&pending.event.camera_id)
                .ok_or(RuntimeError::Identity)?;
            if pending.prepared.is_none() {
                let record_frame = observed_frame(&pending.frame)?;
                pending.prepared = Some(
                    session
                        .publications
                        .prepare(stager, &pending.event, &stream, record_frame, None)
                        .map_err(RuntimeError::Publication)?,
                );
            }
            if pending.staged.is_none() {
                pending.staged = Some(
                    session
                        .publications
                        .stage(
                            stager,
                            pending.prepared.as_mut().ok_or(RuntimeError::Identity)?,
                            &mut |record| {
                                if let Some(lanes) = &lanes {
                                    let _ = lanes.try_emit(record);
                                }
                            },
                        )
                        .map_err(RuntimeError::Publication)?,
                );
                if let Some(sender) = &session.sender {
                    sender.wake();
                }
            }
            // Staging can span the receipt deadline. Observe queued evidence
            // before admission checks the immutable delivery budget, using a
            // cutoff sampled before the drain rather than after its work.
            let receipt_observation_cutoff = clock.monotonic();
            drain_recordings(session, clock)?;
            let recording = session
                .publications
                .admit_recording(
                    stager,
                    &pending.event,
                    pending.staged.as_ref().ok_or(RuntimeError::Identity)?,
                    receipt_observation_cutoff,
                )
                .map_err(RuntimeError::Publication)?;
            if let crate::clips::recorder::Admit::Refused(refusal) = &recording {
                eprintln!(
                    "ml-worker: recording start refused camera_id={} reason={refusal:?}",
                    pending.event.camera_id
                );
            }
            retired += 1;
        }
        Ok(())
    })();
    sink.triggered.drain(..retired);
    delivered
}

fn attributed(
    session: &Session,
    frame: &seeon_deepstream_native::FrameIdentity,
) -> Result<crate::records::builder::Stream, RuntimeError> {
    let camera = session
        .publications
        .camera_for(frame.binding, frame.source_id)
        .ok_or(RuntimeError::Identity)?;
    Ok(crate::records::builder::Stream {
        camera_id: camera.to_owned(),
        worker_boot_id: session.boot_id.clone(),
        source_generation: frame.binding.generation,
        stream_epoch: frame.binding.epoch,
    })
}
fn observed_frame(
    frame: &seeon_deepstream_native::FrameIdentity,
) -> Result<crate::records::builder::Frame, RuntimeError> {
    let source_pts_ns = match frame.pts_valid {
        0 => None,
        1 => Some(i64::try_from(frame.pts_ns).map_err(|_| RuntimeError::Identity)?),
        _ => return Err(RuntimeError::Identity),
    };
    Ok(crate::records::builder::Frame {
        frame_seq: frame.sequence,
        source_pts_ns,
    })
}

fn score_record(
    stream: &crate::records::builder::Stream,
    frame: &seeon_deepstream_native::FrameIdentity,
    score: &ReadyScore,
    clock: &dyn Clock,
    temperature: f64,
) -> Option<crate::records::Record> {
    let ReadyScore {
        track_id,
        logit,
        evidence,
        generation,
    } = score;
    let transition =
        crate::policy::fall::score::transition_probability(*logit, temperature as f32)?;
    let emitted = crate::policy::emit::ModelScore {
        track_id: i64::try_from(*track_id).ok()?,
        generation: generation.map(i64::try_from).transpose().ok()?,
        fall_transition: Some(transition),
        background: Some(1.0 - transition),
        fallen: Some(0.0),
        evidence: Some(crate::policy::emit::ModelEvidence {
            raw_logit: f64::from(*logit),
            applied_temperature: temperature,
        }),
    };
    let observed = clock
        .wall()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let observed_at_ns = u64::try_from(observed).ok()?;
    let record_frame = observed_frame(frame).ok()?;
    crate::run::records::model_score_record(
        stream,
        record_frame,
        observed_at_ns,
        &emitted,
        evidence.as_ref(),
    )
    .ok()
}

fn check_sender(session: &mut Session) -> Result<(), RuntimeError> {
    let sender = session.sender.as_mut().ok_or(RuntimeError::Delivery)?;
    if !sender.is_finished() {
        return Ok(());
    }
    match sender.join() {
        Ok(report) => eprintln!("ml-worker: delivery sender stopped unexpectedly: {report:?}"),
        Err(error) => eprintln!("ml-worker: delivery sender failed: {error:?}"),
    }
    Err(RuntimeError::Delivery)
}

fn poll_restart(booted: &Booted) -> Result<Option<RestartDirective>, RuntimeError> {
    let token = booted
        .settings
        .relay_token()
        .map_err(|_| RuntimeError::Configuration)?;
    match pull::poll_worker_config(
        RELAY_URL,
        token,
        std::path::Path::new(crate::config::windows::ZONEINFO_DIR),
    ) {
        Ok(polled) => Ok(polled.map(|polled| polled.restart_candidate())),
        // Python make_restart_check catches OSError/TimeoutError, preserving
        // the running snapshot; malformed TZif body exceptions are not caught.
        Err(crate::config::windows::WindowError::Io(_)) => Ok(None),
        Err(error) => Err(RuntimeError::WindowAsset(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::{LiveSink, ReadyScore, observed_frame, score_record};
    use crate::records::builder::Stream;
    use crate::records::id::canonical;
    use crate::run::pump::PolicySink;
    use crate::seam::SystemClock;
    use seeon_deepstream_native::{FrameIdentity, GpuMetrics};
    use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, Precision};

    #[test]
    fn invalid_pts_never_becomes_an_absent_timestamp() {
        let mut frame = FrameIdentity {
            sequence: 31,
            pts_ns: u64::MAX,
            pts_valid: 1,
            ..FrameIdentity::default()
        };
        assert!(observed_frame(&frame).is_err());
        frame.pts_valid = 0;
        assert_eq!(observed_frame(&frame).unwrap().source_pts_ns, None);
        frame.pts_valid = 2;
        assert!(observed_frame(&frame).is_err());
        frame.pts_valid = 1;
        frame.pts_ns = 0;
        assert_eq!(observed_frame(&frame).unwrap().source_pts_ns, Some(0));
    }

    #[test]
    fn cpu_channel_score_retains_no_accelerator_through_record_conversion() {
        let stream = Stream {
            camera_id: "cpu-camera".into(),
            worker_boot_id: "cpu-boot".into(),
            source_generation: 7,
            stream_epoch: 9,
        };
        let frame = FrameIdentity {
            sequence: 143,
            pts_ns: 123_456,
            pts_valid: 1,
            ..FrameIdentity::default()
        };
        let mut sink = LiveSink::new();
        sink.score(frame, 42, &crate::msg::FallScore::Cpu(2.0))
            .expect("CPU score retained");
        let held = sink.scores.pop().expect("one retained score");
        assert_eq!(held.frame, frame);
        assert_eq!(held.track_id, 42);
        assert!(held.evidence.is_none());
        let record = score_record(
            &stream,
            &held.frame,
            &ReadyScore {
                track_id: held.track_id,
                logit: held.logit,
                evidence: held.evidence,
                generation: Some(13),
            },
            &SystemClock::new(),
            2.0,
        )
        .expect("CPU score record");
        let wire: serde_json::Value =
            serde_json::from_str(&canonical(&record.to_json()).unwrap()).unwrap();
        let payload = &wire["payload"];
        assert!(payload.get("accelerator").is_none());
        assert_eq!(wire["frame_seq"], 143);
        assert_eq!(payload["track_id"], 42);
        assert_eq!(payload["generation"], 13);
        assert_eq!(payload["raw_logit"], 2.0);
        assert_eq!(payload["applied_temperature"], 2.0);
        assert!((payload["fall_transition"].as_f64().unwrap() - 0.731_058_6).abs() < 1e-7);
    }

    #[test]
    fn score_record_uses_admitted_identity_temperature_and_receipt() {
        let stream = Stream {
            camera_id: "camera-real".into(),
            worker_boot_id: "boot-real".into(),
            source_generation: 7,
            stream_epoch: 9,
        };
        let frame = FrameIdentity {
            sequence: 143,
            pts_ns: 123_456,
            pts_valid: 1,
            ..FrameIdentity::default()
        };
        let before = GpuMetrics {
            attempted: 4,
            succeeded: 4,
            failed: 0,
            host_to_device_bytes: 100,
            device_to_host_bytes: 20,
            elapsed_ns: 1000,
            device: 0,
        };
        let after = GpuMetrics {
            attempted: 5,
            succeeded: 5,
            failed: 0,
            host_to_device_bytes: 700,
            device_to_host_bytes: 80,
            elapsed_ns: 1789,
            device: 0,
        };
        let evidence = AcceleratorEvidence::from_delta(
            &before,
            &after,
            0,
            EngineDigest::new([0xab; 32]),
            Precision::Fp32,
        )
        .unwrap();
        let record = score_record(
            &stream,
            &frame,
            &ReadyScore {
                track_id: 42,
                logit: 2.0,
                evidence: Some(evidence),
                generation: Some(13),
            },
            &SystemClock::new(),
            2.0,
        )
        .unwrap();
        let wire: serde_json::Value =
            serde_json::from_str(&canonical(&record.to_json()).unwrap()).unwrap();
        assert_eq!(wire["camera_id"], "camera-real");
        assert_eq!(wire["worker_boot_id"], "boot-real");
        assert_eq!(wire["source_generation"], 7);
        assert_eq!(wire["stream_epoch"], 9);
        assert_eq!(wire["frame_seq"], 143);
        assert_eq!(wire["source_pts_ns"], 123_456);
        assert_eq!(wire["payload"]["track_id"], 42);
        assert_eq!(wire["payload"]["generation"], 13);
        assert_eq!(wire["payload"]["applied_temperature"], 2.0);
        assert_eq!(wire["payload"]["raw_logit"], 2.0);
        let probability = wire["payload"]["fall_transition"].as_f64().unwrap();
        assert!((probability - 0.731_058_6).abs() < 1e-7);
        assert_eq!(wire["payload"]["accelerator"]["call_seq"], 5);
        assert_eq!(wire["payload"]["accelerator"]["elapsed_ns"], 789);
    }
}
#[cfg(test)]
#[path = "runtime_bounds_tests.rs"]
mod bounds_tests;
