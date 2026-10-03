//! Owners after model boot and before source activation. Empty roster omits
//! media without a dummy runtime. Records require attribution; optional replay
//! captures actual accepted metadata. Status/heartbeat use existing publishers.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::thread::JoinHandle;
use std::time::Duration;

use seeon_deepstream_native::MediaBinding;

use crate::config::pull::PulledConfig;
use crate::exit::Exit;
use crate::media::Command;
use crate::media::diagnostics::Diagnostics;
use crate::media::owner::{self, MediaParams};
use crate::media::shutdown::ShutdownControl;
use crate::msg::{POSE_PER_CAMERA, PREVIEW_CAPACITY, PosePacket, PreviewPacket};
use crate::poll::poll_until;
use crate::records::compose::{self, Composed};
use crate::records::provenance::Identities;
use crate::relay::RelayClient;
use crate::run::Booted;
use crate::run::cameras::{self, CameraPolicyError};
use crate::run::delivery;
use crate::run::exporter::{self, Handle as ExporterHandle};
use crate::run::media_config::{self, MediaAssembly, MediaConfigError};
use crate::run::pump::{PolicyPump, PumpError};
use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;
use crate::telemetry::heartbeat::{Heartbeat, HeartbeatSender};
use crate::telemetry::status::{
    CameraStatus, ClipExportStatus, ClipRecorderStatus, DecodeStatus, FacilityStatus, StatusSender,
    WorkerStatus,
};
use crate::telemetry::{self, LoopHandle, Schedule};

use super::RELAY_URL;
use super::publication::{self, PublicationError, Publications};

const OPEN_BUDGET_MS: u32 = 10_000;
pub struct HeldScore {
    pub frame: seeon_deepstream_native::FrameIdentity,
    pub track_id: u64,
    pub evidence: Option<seeon_worker_runtime::evidence::AcceleratorEvidence>,
    pub logit: f32,
}
const MEDIA_READY_WAIT: Duration = Duration::from_secs(30);

#[derive(Debug)]
pub enum OutputError {
    Policy(CameraPolicyError),
    Pump(PumpError),
    Media(MediaConfigError),
    Publication(PublicationError),
    Records,
    RecordsComposition(compose::ComposeError),
    Manifest(crate::run::runtime_manifest::ManifestError),
    Relay,
    Spawn,
    Join,
    Delivery,
    Clock,
    Stopped,
    Readiness(Exit),
    Trace,
}

impl OutputError {
    pub const fn exit(&self) -> Exit {
        match self {
            Self::Policy(_) | Self::Media(_) => Exit::Config,
            Self::Pump(error) => error.exit(),
            Self::Readiness(exit) => *exit,
            Self::Stopped => Exit::CleanShutdown,
            _ => Exit::Runtime,
        }
    }
}

impl fmt::Display for OutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Policy(_) => "admitted camera policy refused",
            Self::Pump(_) => "policy pump refused",
            Self::Media(_) => "media configuration refused",
            Self::Publication(_) => "publication owner refused",
            Self::Records => "execution records refused",
            Self::RecordsComposition(compose::ComposeError::Provenance(
                crate::records::provenance::ProvenanceError::Missing(fields),
            )) => {
                return write!(
                    formatter,
                    "execution records refused: missing {}",
                    fields.join(", ")
                );
            }
            Self::RecordsComposition(_) => "execution-record composition refused",
            Self::Manifest(error) => return write!(formatter, "{error}"),
            Self::Relay => "relay client refused",
            Self::Spawn => "owner thread did not start",
            Self::Join => "owner thread did not stop cleanly",
            Self::Delivery => "delivery sender refused",
            Self::Clock => "output clock is not representable",
            Self::Stopped => "output startup stopped",
            Self::Readiness(_) => "media readiness refused",
            Self::Trace => "replay trace writer refused",
        })
    }
}

pub struct MediaSession {
    pub thread: Option<JoinHandle<()>>,
    pub diagnostics: Arc<Diagnostics>,
    pub shutdown: Arc<ShutdownControl>,
    pub commands: SyncSender<Command>,
}

pub struct Session {
    pub pump: PolicyPump,
    pub publications: Publications,
    pub media: Option<MediaSession>,
    pub records: Option<Arc<crate::records::lanes::Lanes>>,
    pub exporter: Option<ExporterHandle>,
    pub heartbeat: Option<LoopHandle>,
    pub status: Option<LoopHandle>,
    pub stop: Arc<AtomicBool>,
    /// Accelerator classification cannot stop media ahead of recorder quiescence.
    policy_stop: AtomicBool,
    pub boot_id: String,
    pub temperature: f64,
    pose_rx: Receiver<PosePacket>,
    preview_rx: Receiver<PreviewPacket>,
    pub sender: Option<delivery::Handle>,
}

/// A failed start still transfers every owner already created to root cleanup.
pub(super) struct PrepareFailure {
    pub error: OutputError,
    pub session: Option<Box<Session>>,
}

impl From<OutputError> for PrepareFailure {
    fn from(error: OutputError) -> Self {
        Self {
            error,
            session: None,
        }
    }
}

pub(super) fn prepare(
    booted: &Booted,
    config: &PulledConfig,
    boot_id: &str,
    clock: Arc<dyn Clock>,
    process: Arc<ShutdownDeadline>,
) -> Result<Session, PrepareFailure> {
    let selected = booted.admitted.checked.selection.is_some();
    let policies = cameras::camera_policies(
        config,
        &booted.admitted.fall.calibration,
        boot_id,
        boot_id,
        1,
        default_capacities(),
    )
    .map_err(OutputError::Policy)?;
    let mut pump = PolicyPump::new(
        policies,
        config
            .windows
            .get("fall")
            .map(|admitted| admitted.window.clone()),
        Arc::clone(&clock),
    )
    .map_err(OutputError::Pump)?;
    let assembly = media_config::assemble(&booted.admitted.flow, &config.cameras, boot_id)
        .map_err(OutputError::Media)?;
    let manifest =
        match crate::run::runtime_manifest::build(booted, config, &assembly).and_then(|manifest| {
            manifest.persist(booted.settings.state_dir())?;
            Ok(manifest)
        }) {
            Ok(manifest) => Some(manifest),
            Err(error) if selected => return Err(OutputError::Manifest(error).into()),
            Err(error) => {
                eprintln!("ml-worker: packaged runtime manifest unavailable: {error}");
                None
            }
        };
    let token = booted
        .settings
        .relay_token()
        .map_err(|_| OutputError::Relay)?;
    let (command_tx, command_rx, record_tx, record_rx) = publication::channels();
    let commands = command_tx.clone();
    let roster = roster(config);
    let publications = publication::open(
        publication::PublicationConfig {
            boot_id,
            state_dir: booted.settings.state_dir(),
            record_dir: &booted.admitted.flow.record_dir,
            store_root: std::path::Path::new(publication::CLIP_STORE_ROOT),
            cameras: &roster,
            config_version: config_version(config),
            manifest_sha: manifest
                .as_ref()
                .map(crate::run::runtime_manifest::Manifest::sha256),
        },
        command_tx,
        record_rx,
        Arc::clone(&clock),
    )
    .map_err(OutputError::Publication)?;
    let camera_ids: Vec<_> = config
        .cameras
        .iter()
        .map(|camera| camera.camera_id.clone())
        .collect();
    let replay = publication::replay_before_media(
        booted.settings.state_dir(),
        &camera_ids,
        &publications.store,
        &publications.queue,
        clock.as_ref(),
    )
    .map_err(OutputError::Publication)?;
    if replay.malformed > 0 {
        eprintln!(
            "ml-worker: retained malformed sealed sidecars count={}",
            replay.malformed
        );
    }
    for (camera_id, report) in camera_ids.iter().zip(&replay.reports) {
        if report.failed > 0 || report.missing_media > 0 {
            eprintln!(
                "ml-worker: sealed replay camera_id={camera_id} failed={} missing_media={}",
                report.failed, report.missing_media
            );
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    let (pose_tx, pose_rx) = std::sync::mpsc::sync_channel(pose_capacity(config.cameras.len()));
    let (preview_tx, preview_rx) = std::sync::mpsc::sync_channel(PREVIEW_CAPACITY);
    let composed = compose_records(booted, config, token)?;
    let records = composed.as_ref().map(|records| Arc::clone(&records.lanes));
    let delivery_client = RelayClient::new(
        RELAY_URL,
        token,
        crate::relay::client::ALERT_DELIVERY_TIMEOUT,
    )
    .map_err(|_| OutputError::Relay)?;
    let started_at = super::unix_seconds(clock.wall()).ok_or(OutputError::Clock)?;
    if let Some(directory) = booted.settings.replay_trace_dir() {
        pump.set_replay(
            crate::run::replay::ReplayCapture::new(&directory, config)
                .map_err(|_| OutputError::Trace)?,
        );
    }
    let mut session = Session {
        pump,
        publications,
        media: None,
        records,
        exporter: None,
        heartbeat: None,
        status: None,
        stop,
        policy_stop: AtomicBool::new(false),
        boot_id: boot_id.to_owned(),
        temperature: booted.admitted.fall.calibration.temperature,
        pose_rx,
        preview_rx,
        sender: None,
    };
    let started = (|| -> Result<(), OutputError> {
        if process.deadline().is_some() {
            return Err(OutputError::Stopped);
        }
        if let Some(composed) = composed {
            session.exporter = Some(
                exporter::spawn_composed(composed, Arc::clone(&clock))
                    .map_err(|_| OutputError::Records)?,
            );
        }
        session.status = spawn_status(booted, config, token, started_at, Arc::clone(&clock))?;
        if booted.settings.flags().heartbeat_on_start {
            session.heartbeat = Some(spawn_heartbeat(config, token)?);
        }
        session.sender = Some(
            delivery::spawn(
                Arc::clone(&session.publications.queue),
                delivery_client,
                config.config.clip_export_enabled(),
                Arc::clone(&clock),
                Arc::clone(&process),
            )
            .map_err(|error| {
                eprintln!("ml-worker: delivery sender could not start: {error}");
                OutputError::Spawn
            })?,
        );
        if process.deadline().is_some() {
            return Err(OutputError::Stopped);
        }
        if let MediaAssembly::Configured(media_config) = assembly {
            let diagnostics = Arc::new(Diagnostics::new(media_config.sources.len()));
            let shutdown = Arc::new(ShutdownControl::new(Arc::clone(&process)));
            let (media, readiness) = spawn_media(
                MediaParams {
                    config: media_config,
                    open_budget_ms: OPEN_BUDGET_MS,
                    shutdown,
                    stop: Arc::clone(&session.stop),
                    clock: Arc::clone(&clock),
                    pose_tx,
                    preview_tx,
                    record_tx,
                    commands: command_rx,
                    diagnostics,
                },
                commands,
            )?;
            session.media = Some(media);
            wait_media_ready(&readiness, clock.as_ref(), &process)?;
        }
        Ok(())
    })();
    match started {
        Ok(()) => Ok(session),
        Err(error) => Err(PrepareFailure {
            error,
            session: Some(Box::new(session)),
        }),
    }
}

fn default_capacities() -> seeon_worker::fall::FallCapacities {
    seeon_worker::fall::FallCapacities {
        retained_tracks: 64,
        generation_identities: 64,
        episodes: 64,
        vote_window: 5,
    }
}

fn roster(config: &PulledConfig) -> Vec<(u32, MediaBinding, String, String)> {
    config
        .cameras
        .iter()
        .enumerate()
        .filter_map(|(index, camera)| {
            Some((
                u32::try_from(index).ok()?,
                MediaBinding {
                    token: u64::try_from(index).ok()? + 1,
                    generation: 1,
                    epoch: 1,
                },
                camera.camera_id.clone(),
                camera.facility_id.clone(),
            ))
        })
        .collect()
}

fn config_version(config: &PulledConfig) -> i64 {
    i64::try_from(config.directive.version).unwrap_or(i64::MAX)
}

fn pose_capacity(cameras: usize) -> usize {
    POSE_PER_CAMERA.saturating_mul(cameras.max(1))
}

fn spawn_media(
    params: MediaParams,
    command_tx: SyncSender<Command>,
) -> Result<(MediaSession, Receiver<crate::msg::Readiness>), OutputError> {
    let diagnostics = Arc::clone(&params.diagnostics);
    let shutdown = Arc::clone(&params.shutdown);
    let (thread, readiness) = owner::spawn(params).map_err(|_| OutputError::Spawn)?;
    Ok((
        MediaSession {
            thread: Some(thread),
            diagnostics,
            shutdown,
            commands: command_tx,
        },
        readiness,
    ))
}

fn wait_media_ready(
    readiness: &Receiver<crate::msg::Readiness>,
    clock: &dyn Clock,
    process: &ShutdownDeadline,
) -> Result<(), OutputError> {
    let limit = clock
        .monotonic()
        .checked_add(MEDIA_READY_WAIT)
        .ok_or(OutputError::Clock)?;
    let mut result = None;
    let waited = poll_until(clock, limit, "media readiness", || {
        let now = clock.monotonic();
        result = match readiness.try_recv() {
            Ok(ready) => Some(ready.map_err(OutputError::Readiness)),
            Err(TryRecvError::Disconnected) => Some(Err(OutputError::Spawn)),
            Err(TryRecvError::Empty) => None,
        };
        // Release can publish shutdown immediately after sending a refusal.
        // Observe the actual refusal before treating that publication as stop.
        if matches!(result, Some(Err(_))) {
            return true;
        }
        if process.deadline().is_some() {
            result = Some(Err(OutputError::Stopped));
        } else if now >= limit {
            result = Some(Err(OutputError::Readiness(Exit::Runtime)));
        }
        result.is_some()
    });
    // As with model readiness, an observed refusal outranks a later signal.
    if matches!(
        result,
        Some(Err(OutputError::Readiness(_) | OutputError::Spawn))
    ) {
        return result.unwrap_or(Err(OutputError::Spawn));
    }
    if process.deadline().is_some() {
        return Err(OutputError::Stopped);
    }
    waited.map_err(|_| OutputError::Readiness(Exit::Runtime))?;
    result.unwrap_or(Err(OutputError::Spawn))
}

fn compose_records(
    booted: &Booted,
    config: &PulledConfig,
    token: &str,
) -> Result<Option<Composed>, OutputError> {
    if booted.settings.execution_records().is_none() {
        return Ok(None);
    }
    let digest = crate::run::config_digest::config_digest(&config.payload)
        .map_err(|_| OutputError::Records)?;
    let identities = record_identities(booted, config);
    compose::compose(&booted.settings.env, RELAY_URL, token, &identities, &digest)
        .map_err(OutputError::RecordsComposition)
}

fn record_identities(booted: &Booted, config: &PulledConfig) -> Identities {
    Identities {
        build_revision: booted
            .settings
            .build_revision()
            .ok()
            .flatten()
            .map(str::to_owned),
        image_digest: booted.admitted.flow.identity.get("image_digest").cloned(),
        model_digest: Some(booted.admitted.fall.published_weights_digest.clone()),
        calibration_digest: Some(booted.admitted.fall.calibration_digest.clone()),
        preprocessing_identity: Some(booted.admitted.fall.preprocessing_identity.clone()),
        policy_identity: config
            .policies
            .defaults
            .get("fall")
            .map(|policy| policy.effective_policy_id.clone()),
    }
}

fn spawn_status(
    booted: &Booted,
    config: &PulledConfig,
    token: &str,
    started_at: f64,
    clock: Arc<dyn Clock>,
) -> Result<Option<LoopHandle>, OutputError> {
    if config.cameras.is_empty() {
        return Ok(None);
    }
    let client = RelayClient::new(RELAY_URL, token, crate::records::RELAY_TIMEOUT)
        .map_err(|_| OutputError::Relay)?;
    let facility = config.cameras[0].facility_id.clone();
    let cameras = config.cameras.clone();
    let gpu = booted.gpu.clone();
    let export = ClipExportStatus {
        enabled: config.config.clip_export_enabled(),
        version: i64::try_from(config.config.clip_export_version()).unwrap_or(0),
    };
    let sender = StatusSender::new(client, move || {
        vec![FacilityStatus {
            facility_id: facility.clone(),
            cameras: cameras
                .iter()
                .map(|camera| CameraStatus {
                    camera_id: camera.camera_id.clone(),
                    decode: DecodeStatus {
                        requested: camera
                            .decode_backend
                            .clone()
                            .unwrap_or_else(|| "auto".to_owned()),
                        selected: None,
                        fallback_count: 0,
                        last_reason: None,
                        updated_at_sec: started_at,
                    },
                    measured_fps: None,
                    detection: None,
                })
                .collect(),
            clip_recorder: ClipRecorderStatus::unavailable(),
            clip_export: export,
            gpu: Some(gpu.clone()),
            worker: Some(WorkerStatus::current(true, started_at, None)),
            delivery_queue: None,
        }]
    });
    let _ = clock;
    telemetry::spawn("runtime-status", sender, Schedule::STATUS)
        .map(Some)
        .map_err(|_| OutputError::Spawn)
}

fn spawn_heartbeat(config: &PulledConfig, token: &str) -> Result<LoopHandle, OutputError> {
    let client = RelayClient::new(RELAY_URL, token, crate::records::RELAY_TIMEOUT)
        .map_err(|_| OutputError::Relay)?;
    let version = config_version(config);
    let ready: Vec<_> = config
        .cameras
        .iter()
        .map(|camera| Heartbeat {
            camera_id: camera.camera_id.clone(),
            facility_id: camera.facility_id.clone(),
            config_version: version,
        })
        .collect();
    telemetry::spawn(
        "heartbeat",
        HeartbeatSender::new(client, move || ready.clone()),
        Schedule::HEARTBEAT,
    )
    .map_err(|_| OutputError::Spawn)
}

impl Session {
    pub fn previews(&self) -> &Receiver<PreviewPacket> {
        &self.preview_rx
    }
    pub(super) fn queue_ends(&mut self) -> (&mut PolicyPump, &Receiver<PosePacket>, &AtomicBool) {
        (&mut self.pump, &self.pose_rx, &self.policy_stop)
    }
    pub fn request_media_stop(&mut self) {
        self.publications.quiesce();
        self.stop.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;
    use std::thread::ThreadId;
    use std::time::Instant;

    use seeon_deepstream_native::{MediaConfig, SourceConfig};

    use super::*;
    use crate::media::diagnostics::{CameraCounters, Snapshot};
    use crate::msg::{ONESHOT_CAPACITY, Readiness, RecordReceipt};
    use crate::poll::POLL_INTERVAL;
    use crate::seam::{IdSource, RandomIds, SystemClock};

    #[test]
    fn record_composition_reports_all_missing_names_without_changing_exit() {
        let error = OutputError::RecordsComposition(compose::ComposeError::Provenance(
            crate::records::provenance::ProvenanceError::Missing(vec![
                "worker_build_revision",
                "policy_identity",
            ]),
        ));
        assert_eq!(
            error.to_string(),
            "execution records refused: missing worker_build_revision, policy_identity"
        );
        assert_eq!(error.exit(), Exit::Runtime);
        let other = OutputError::RecordsComposition(compose::ComposeError::RelayMissing);
        assert_eq!(other.to_string(), "execution-record composition refused");
        assert_eq!(other.exit(), Exit::Runtime);
    }

    pub(in crate::run::execution) fn session(
        pump: PolicyPump,
        publications: Publications,
        boot_id: String,
    ) -> Session {
        let (_pose_tx, pose_rx) = std::sync::mpsc::sync_channel(1);
        let (_preview_tx, preview_rx) = std::sync::mpsc::sync_channel(1);
        Session {
            pump,
            publications,
            media: None,
            records: None,
            exporter: None,
            heartbeat: None,
            status: None,
            stop: Arc::new(AtomicBool::new(false)),
            policy_stop: AtomicBool::new(false),
            boot_id,
            temperature: 1.0,
            pose_rx,
            preview_rx,
            sender: None,
        }
    }

    #[test]
    fn empty_pose_capacity_never_uses_a_zero_channel() {
        assert!(pose_capacity(0) >= 1);
    }

    #[test]
    fn policy_fault_cannot_stop_media_before_all_recorders_quiesce() {
        use crate::run::pump::CameraPolicy;
        use crate::seam::{IdSource, RandomIds, SystemClock};
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                std::fs::remove_dir_all(&self.0).unwrap();
            }
        }
        let root = Scratch(std::env::temp_dir().join(RandomIds.uuid4().unwrap()));
        std::fs::create_dir(&root.0).unwrap();
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let boot_id = RandomIds.uuid4().unwrap();
        let cameras = [3_u32, 7].map(|source_id| {
            (
                source_id,
                MediaBinding {
                    token: u64::from(source_id),
                    generation: 1,
                    epoch: 1,
                },
                format!("camera-{source_id}"),
                "facility-real".to_owned(),
            )
        });
        let (commands, command_rx, _receipts, records) = publication::channels();
        let publications = publication::open(
            publication::PublicationConfig {
                boot_id: &boot_id,
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
        let mut session = session(
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
            boot_id,
        );
        let fault = crate::msg::FallResponse {
            frame: seeon_deepstream_native::FrameIdentity::default(),
            track_id: 7,
            score: Err(seeon_worker_runtime::fall_gpu::FallGpuError::Poisoned.into()),
        };
        let (_, _, classifier_stop) = session.queue_ends();
        let error =
            crate::run::policy::validate_fall_response(&fault, classifier_stop).unwrap_err();
        assert_eq!(error.exit(), Exit::FatalAccelerator);
        assert!(classifier_stop.load(Ordering::SeqCst));
        assert!(
            !session.stop.load(Ordering::SeqCst),
            "classification cannot bypass quiescence"
        );
        assert!(
            session
                .publications
                .recorders
                .iter()
                .all(|recorder| !recorder.is_quiesced())
        );
        session.request_media_stop();
        assert!(session.stop.load(Ordering::SeqCst));
        assert!(
            session
                .publications
                .recorders
                .iter()
                .all(|recorder| recorder.is_quiesced())
        );
        let detected = crate::clips::time::Utc::parse("2026-01-01T00:00:00Z").unwrap();
        for recorder in &mut session.publications.recorders {
            assert_eq!(
                recorder.admit("late-event", detected).unwrap(),
                crate::clips::recorder::Admit::Queued
            );
            assert_eq!(recorder.pending(), 1);
            recorder.tick().expect("quiesced unstarted recorder");
        }
        session.request_media_stop();
        assert!(
            command_rx.try_recv().is_err(),
            "quiesced late events never start native recording"
        );
        assert!(
            session
                .publications
                .recorders
                .iter()
                .all(|recorder| recorder.pending() == 1)
        );
    }

    /// `wait_media_ready` only. These tests do not call `prepare` or
    /// `failed_preparation`, so they do not cover root integration: a refused
    /// open must still come back as `PrepareFailure { session: Some(...) }`
    /// and be joined by lifecycle shutdown with publications and telemetry.
    fn zero_budget_config() -> MediaConfig {
        MediaConfig {
            sources: vec![SourceConfig {
                source_id: 0,
                binding: MediaBinding {
                    token: 1,
                    generation: 1,
                    epoch: 1,
                },
                uri: "file:///not-a-native-open.mp4".into(),
                record_prefix: "camera".into(),
            }],
            infer_config_path: "infer.txt".into(),
            tracker_config_path: "tracker.yml".into(),
            tracker_library_path: "tracker.so".into(),
            record_directory: "records".into(),
            record_cache_seconds: 30,
            record_capacity: 4,
            mux_width: 640,
            mux_height: 360,
            mux_batch_timeout_us: 40000,
            mux_live_source: false,
            tracker_width: 960,
            tracker_height: 544,
            queue_max_buffers: 4,
            preview_enabled: false,
            max_preview_bytes: 0,
            allow_file_uris: true,
            rtsp_reconnect_interval_sec: 0,
        }
    }

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "seeon-output-ready-{}-{}",
                std::process::id(),
                RandomIds.uuid4().expect("scratch id")
            ));
            std::fs::create_dir(&path).expect("scratch directory");
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct Fixture {
        session: Session,
        process: Arc<ShutdownDeadline>,
        commands: SyncSender<Command>,
        command_rx: Option<Receiver<Command>>,
        record_tx: Option<SyncSender<RecordReceipt>>,
        release: Option<Arc<AtomicBool>>,
        clock: SystemClock,
        scratch: Scratch,
    }

    impl Fixture {
        fn idle() -> Self {
            let scratch = Scratch::new();
            let boot_id = RandomIds.uuid4().expect("boot id");
            let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
            let (commands, command_rx, record_tx, records) = publication::channels();
            let cameras: [(u32, MediaBinding, String, String); 0] = [];
            let publications = publication::open(
                publication::PublicationConfig {
                    boot_id: &boot_id,
                    state_dir: &scratch.0.join("state"),
                    record_dir: &scratch.0.join("records"),
                    store_root: &scratch.0.join("store"),
                    cameras: &cameras,
                    config_version: 1,
                    manifest_sha: None,
                },
                commands.clone(),
                records,
                Arc::clone(&clock),
            )
            .expect("empty publication fixture");
            let pump = PolicyPump::new(Vec::new(), None, clock).expect("empty pump");
            let process =
                Arc::new(ShutdownDeadline::new(Duration::from_secs(2)).expect("shutdown budget"));
            Self {
                session: session(pump, publications, boot_id),
                process,
                commands,
                command_rx: Some(command_rx),
                record_tx: Some(record_tx),
                release: None,
                clock: SystemClock::new(),
                scratch,
            }
        }

        fn install(
            &mut self,
            thread: std::thread::JoinHandle<()>,
            diagnostics: Arc<Diagnostics>,
            shutdown: Arc<ShutdownControl>,
        ) {
            self.session.media = Some(MediaSession {
                thread: Some(thread),
                diagnostics,
                shutdown,
                commands: self.commands.clone(),
            });
        }

        fn hold_thread(&mut self) -> ThreadId {
            let release = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&release);
            let thread = std::thread::Builder::new()
                .name("output-ready-hold".to_owned())
                .spawn(move || {
                    let started = Instant::now();
                    while !flag.load(Ordering::SeqCst) {
                        if started.elapsed() >= Duration::from_secs(30) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                })
                .expect("hold thread spawns");
            let id = thread.thread().id();
            self.release = Some(release);
            let shutdown = Arc::new(ShutdownControl::new(Arc::clone(&self.process)));
            self.install(thread, Arc::new(Diagnostics::new(0)), shutdown);
            id
        }

        fn spawn_refused(&mut self) -> Receiver<Readiness> {
            let config = zero_budget_config();
            let diagnostics = Arc::new(Diagnostics::new(config.sources.len()));
            let (pose_tx, pose_rx) = std::sync::mpsc::sync_channel(POSE_PER_CAMERA);
            let (preview_tx, preview_rx) = std::sync::mpsc::sync_channel(PREVIEW_CAPACITY);
            let record_tx = self.record_tx.take().expect("record sender");
            let command_rx = self.command_rx.take().expect("command receiver");
            let shutdown = Arc::new(ShutdownControl::new(Arc::clone(&self.process)));
            let media_clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
            let (thread, readiness) = owner::spawn(MediaParams {
                config,
                open_budget_ms: 0,
                shutdown: Arc::clone(&shutdown),
                stop: Arc::clone(&self.session.stop),
                clock: media_clock,
                pose_tx,
                preview_tx,
                record_tx,
                commands: command_rx,
                diagnostics: Arc::clone(&diagnostics),
            })
            .expect("media thread spawns");
            self.install(thread, diagnostics, shutdown);
            self.session.pose_rx = pose_rx;
            self.session.preview_rx = preview_rx;
            readiness
        }

        fn join_media(&mut self) -> Result<(), crate::gpu::owners::JoinError> {
            if let Some(flag) = &self.release {
                flag.store(true, Ordering::SeqCst);
            }
            let Some(thread) = self
                .session
                .media
                .as_mut()
                .and_then(|media| media.thread.take())
            else {
                return Ok(());
            };
            let deadline = self
                .clock
                .monotonic()
                .saturating_add(Duration::from_secs(2));
            owner::join(thread, &self.clock, deadline)
        }

        fn finish(&mut self) {
            assert!(
                self.session
                    .media
                    .as_ref()
                    .and_then(|media| media.thread.as_ref())
                    .is_some(),
                "readiness wait must leave the media JoinHandle owned"
            );
            self.join_media().expect("media thread joins");
            std::fs::remove_dir_all(&self.scratch.0).expect("remove owned readiness fixture");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = self.join_media();
        }
    }

    fn retained(session: &Session, id: ThreadId, live: bool) {
        let handle = session
            .media
            .as_ref()
            .and_then(|media| media.thread.as_ref())
            .expect("readiness wait must keep the media JoinHandle");
        assert_eq!(handle.thread().id(), id);
        if live {
            assert!(
                !handle.is_finished(),
                "readiness wait must not join or detach the media thread"
            );
        }
    }

    struct FrozenClock;

    impl Clock for FrozenClock {
        fn monotonic(&self) -> Duration {
            Duration::ZERO
        }
        fn wall(&self) -> std::time::SystemTime {
            std::time::SystemTime::UNIX_EPOCH
        }
        fn pause(&self, _: Duration) {
            panic!("this readiness result is available without waiting");
        }
    }

    #[derive(Default)]
    struct ExpiringClock {
        reads: AtomicUsize,
    }

    impl Clock for ExpiringClock {
        fn monotonic(&self) -> Duration {
            let read = self.reads.fetch_add(1, Ordering::SeqCst);
            if read == 0 {
                Duration::ZERO
            } else {
                MEDIA_READY_WAIT
            }
        }
        fn wall(&self) -> std::time::SystemTime {
            std::time::SystemTime::UNIX_EPOCH
        }
        fn pause(&self, _: Duration) {
            panic!("timeout is observed from the clock, not a pause");
        }
    }

    struct BoundClock {
        start: Instant,
        pauses: AtomicUsize,
    }

    impl BoundClock {
        fn new() -> Self {
            Self {
                start: Instant::now(),
                pauses: AtomicUsize::new(0),
            }
        }
    }

    impl Clock for BoundClock {
        fn monotonic(&self) -> Duration {
            if self.pauses.load(Ordering::SeqCst) > 40 {
                return MEDIA_READY_WAIT.saturating_add(Duration::from_secs(1));
            }
            self.start.elapsed()
        }
        fn wall(&self) -> std::time::SystemTime {
            std::time::SystemTime::UNIX_EPOCH
        }
        fn pause(&self, limit: Duration) {
            self.pauses.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(limit.min(POLL_INTERVAL));
        }
    }

    #[test]
    fn refused_readiness_keeps_the_media_thread_and_open_marker() {
        let mut fixture = Fixture::idle();
        let readiness = fixture.spawn_refused();
        let id = fixture
            .session
            .media
            .as_ref()
            .and_then(|media| media.thread.as_ref())
            .expect("spawned media thread")
            .thread()
            .id();
        let error = wait_media_ready(&readiness, &BoundClock::new(), fixture.process.as_ref());
        assert!(matches!(error, Err(OutputError::Readiness(Exit::Config))));
        retained(&fixture.session, id, false);
        assert_eq!(
            fixture
                .session
                .media
                .as_ref()
                .expect("media session")
                .diagnostics
                .snapshot(),
            Snapshot {
                open_refused: true,
                failure: Some(Exit::Config),
                cameras: vec![CameraCounters::default()],
                ..Snapshot::default()
            }
        );
        fixture.finish();
    }

    #[test]
    fn disconnected_readiness_keeps_the_media_thread() {
        let mut fixture = Fixture::idle();
        let id = fixture.hold_thread();
        let (sender, readiness) = std::sync::mpsc::sync_channel(ONESHOT_CAPACITY);
        drop(sender);
        let error = wait_media_ready(&readiness, &FrozenClock, fixture.process.as_ref());
        assert!(matches!(error, Err(OutputError::Spawn)));
        retained(&fixture.session, id, true);
        assert_eq!(
            fixture
                .session
                .media
                .as_ref()
                .expect("media session")
                .diagnostics
                .snapshot(),
            Snapshot::default()
        );
        fixture.finish();
    }

    #[test]
    fn queued_refusal_outranks_requested_stop_and_keeps_the_media_thread() {
        let mut fixture = Fixture::idle();
        let id = fixture.hold_thread();
        fixture
            .process
            .request_at(Duration::ZERO)
            .expect("shutdown already requested");
        let (sender, readiness) = std::sync::mpsc::sync_channel(ONESHOT_CAPACITY);
        assert!(sender.try_send(Err(Exit::Config)).is_ok());
        let error = wait_media_ready(&readiness, &FrozenClock, fixture.process.as_ref());
        assert!(matches!(error, Err(OutputError::Readiness(Exit::Config))));
        assert_eq!(readiness.try_recv(), Err(TryRecvError::Empty));
        retained(&fixture.session, id, true);
        drop(sender);
        fixture.finish();
    }

    #[test]
    fn requested_stop_without_refusal_keeps_the_media_thread() {
        let mut fixture = Fixture::idle();
        let id = fixture.hold_thread();
        fixture.process.request_at(Duration::ZERO).expect("stop");
        let (sender, readiness) = std::sync::mpsc::sync_channel(ONESHOT_CAPACITY);
        let error = wait_media_ready(&readiness, &FrozenClock, fixture.process.as_ref());
        assert!(matches!(error, Err(OutputError::Stopped)));
        retained(&fixture.session, id, true);
        drop(sender);
        fixture.finish();
    }

    #[test]
    fn readiness_timeout_retains_the_original_media_thread() {
        let mut fixture = Fixture::idle();
        let id = fixture.hold_thread();
        let (sender, readiness) = std::sync::mpsc::sync_channel(ONESHOT_CAPACITY);
        let error = wait_media_ready(
            &readiness,
            &ExpiringClock::default(),
            fixture.process.as_ref(),
        );
        assert!(matches!(error, Err(OutputError::Readiness(Exit::Runtime))));
        retained(&fixture.session, id, true);
        drop(sender);
        fixture.finish();
    }
}
