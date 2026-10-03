//! Sealed replay before source activation, then live queue and recorder owners.
//! Completed ENOSPC fallback stays inside `ReservePool`; unresolved saves retain ownership.
//! A missing runtime-manifest SHA is not invented here.

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender};
use std::time::Duration;

use seeon_deepstream_native::MediaBinding;

use crate::clips::recorder::plane::CommandPlane;
use crate::clips::recorder::{Admit, Recorder, RecorderError};
use crate::clips::reserve::ReservePool;
use crate::clips::sealed::{ReplayOutcome, ReplayReport, SIDECAR_DIR, SealedSidecars};
use crate::clips::store::ClipStore;
use crate::clips::time::Utc;
use crate::delivery::DeliveryQueue;
use crate::media::COMMAND_CAPACITY;
use crate::media::Command;
use crate::msg::{RECORD_CAPACITY, RecordReceipt};
use crate::policy::emit::{EmitError, Stager};
use crate::run::clip_output::{self, ClipOutputError};
use crate::run::events::{EventDelivery, EventDeliveryError, PreparedEvent, StagedEvent};
use crate::seam::Clock;

pub const CLIP_STORE_ROOT: &str = "/var/lib/clip-store";
const COMMAND_WAIT: Duration = Duration::from_secs(2);

#[derive(Debug)]
pub enum PublicationError {
    Sidecar,
    Queue,
    Reserve,
    Stager(EmitError),
    Event(EventDeliveryError),
    Clip(ClipOutputError),
    Recorder(RecorderError),
    Identity,
    Timestamp,
    MediaPath,
    RecordDirectory(std::io::ErrorKind),
}

impl std::fmt::Display for PublicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Sidecar => "sealed sidecar replay failed",
            Self::Queue => "delivery queue could not be opened",
            Self::Reserve => "clip reserve could not be armed",
            Self::Stager(_) => "event stager refused admitted identity",
            Self::Event(_) => "event staging retained the original identity",
            Self::Clip(_) => "sealed clip publication retained its sidecar",
            Self::Recorder(_) => "recording admission failed",
            Self::Identity => "recording identity does not match admitted camera",
            Self::Timestamp => "recording timestamp is not the staged event timestamp",
            Self::MediaPath => "recording path is not an admitted regular media file",
            Self::RecordDirectory(kind) => {
                return write!(
                    formatter,
                    "recording directory could not be prepared: {kind}"
                );
            }
        })
    }
}

pub struct Replay {
    pub reports: Vec<ReplayReport>,
    pub malformed: usize,
}

pub struct Publications {
    boot_id: String,
    cameras: Vec<(u32, seeon_deepstream_native::MediaBinding, String, String)>,
    events: BTreeMap<String, crate::clips::sealed::SealedEvent>,
    pub store: ClipStore,
    pub reserve: ReservePool,
    pub queue: Arc<DeliveryQueue>,
    pub sidecars: SealedSidecars,
    pub stagers: Vec<Stager>,
    pub recorders: Vec<Recorder<CommandPlane>>,
    records: Receiver<RecordReceipt>,
    pub record_dir: PathBuf,
    clock: Arc<dyn Clock>,
}

pub fn replay_before_media(
    state_dir: &Path,
    camera_ids: &[String],
    store: &ClipStore,
    queue: &DeliveryQueue,
    clock: &dyn Clock,
) -> Result<Replay, PublicationError> {
    let sidecars = SealedSidecars::new(state_dir.join(SIDECAR_DIR));
    let mut reports = Vec::with_capacity(camera_ids.len());
    let mut malformed = BTreeSet::new();
    for camera_id in camera_ids {
        let pending = sidecars
            .pending(camera_id)
            .map_err(|_| PublicationError::Sidecar)?;
        malformed.extend(pending.malformed.iter().cloned());
        let mut report = ReplayReport {
            malformed: pending.malformed.len(),
            ..ReplayReport::default()
        };
        for recovery in pending.recoveries {
            let outcome = sidecars
                .replay_one(&recovery, |recovery| {
                    match clip_output::resume_terminal(recovery, store, queue) {
                        Ok(Some(published)) => return ReplayOutcome::Published(published),
                        Err(error) => return ReplayOutcome::Failed(error),
                        Ok(None) => {}
                    }
                    let media = match recovery_media(recovery, store) {
                        Ok(Some(media)) => media,
                        Ok(None) => return ReplayOutcome::MissingMedia,
                        Err(error) => return ReplayOutcome::Failed(error),
                    };
                    match super::recording::measured_codec(&media)
                        .map_err(ClipOutputError::from)
                        .and_then(|codec| {
                            clip_output::publish_recovery(
                                recovery,
                                store,
                                queue,
                                &codec,
                                Utc::from_system(clock.wall()),
                            )
                        }) {
                        Ok(published) => ReplayOutcome::Published(published),
                        Err(error) => ReplayOutcome::Failed(error),
                    }
                })
                .map_err(|_| PublicationError::Sidecar)?;
            match outcome {
                ReplayOutcome::Published(_) => report.published += 1,
                ReplayOutcome::MissingMedia => report.missing_media += 1,
                ReplayOutcome::Failed(_) => report.failed += 1,
            }
        }
        reports.push(report);
    }
    Ok(Replay {
        reports,
        malformed: malformed.len(),
    })
}

fn recovery_media(
    recovery: &crate::clips::sealed::Recovery,
    store: &ClipStore,
) -> Result<Option<PathBuf>, ClipOutputError> {
    for path in [
        store
            .staging_dir(&recovery.sealed.clip_id)
            .join(crate::clips::store::ARTIFACT_FILE),
        store
            .clip_dir(&recovery.sealed.clip_id)
            .join(crate::clips::publish::MEDIA_FILE),
        PathBuf::from(&recovery.sealed.path),
    ] {
        match path.symlink_metadata() {
            Ok(metadata) if metadata.is_file() => return Ok(Some(path)),
            Ok(_) => return Err(crate::clips::publish::PublishError::Conflict.into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(None)
}

pub struct PublicationConfig<'a> {
    pub boot_id: &'a str,
    pub state_dir: &'a Path,
    pub record_dir: &'a Path,
    pub store_root: &'a Path,
    pub cameras: &'a [(u32, MediaBinding, String, String)],
    pub config_version: i64,
    pub manifest_sha: Option<&'a str>,
}

pub fn open(
    config: PublicationConfig<'_>,
    commands: SyncSender<Command>,
    records: Receiver<RecordReceipt>,
    clock: Arc<dyn Clock>,
) -> Result<Publications, PublicationError> {
    let PublicationConfig {
        boot_id,
        state_dir,
        record_dir,
        store_root,
        cameras,
        config_version,
        manifest_sha,
    } = config;
    if !crate::run::event_payload::is_uuid(boot_id) {
        return Err(PublicationError::Identity);
    }
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(record_dir)
        .map_err(|error| PublicationError::RecordDirectory(error.kind()))?;
    let queue = DeliveryQueue::open(&state_dir.join("delivery-queue"), true)
        .map_err(|_| PublicationError::Queue)?;
    let store = ClipStore::new(store_root);
    let reserve = ReservePool::arm(&store, cameras.len()).map_err(|_| PublicationError::Reserve)?;
    let mut stagers = Vec::with_capacity(cameras.len());
    let mut recorders = Vec::with_capacity(cameras.len());
    for (source_id, binding, camera_id, facility_id) in cameras {
        stagers.push(
            Stager::new(camera_id, facility_id, config_version, manifest_sha)
                .map_err(PublicationError::Stager)?,
        );
        recorders.push(Recorder::new(
            *source_id,
            CommandPlane::new(commands.clone(), *source_id, *binding, COMMAND_WAIT),
            Arc::clone(&clock),
        ));
    }
    Ok(Publications {
        boot_id: boot_id.to_owned(),
        cameras: cameras.to_vec(),
        events: BTreeMap::new(),
        store,
        reserve,
        queue: Arc::new(queue),
        sidecars: SealedSidecars::new(state_dir.join(SIDECAR_DIR)),
        stagers,
        recorders,
        records,
        record_dir: record_dir.to_path_buf(),
        clock,
    })
}

impl Publications {
    /// Stop new native start/extend work without consuming active receipts or
    /// pending contributors. Shutdown publication owns their eventual handoff.
    pub fn quiesce(&mut self) {
        for recorder in &mut self.recorders {
            recorder.quiesce();
        }
    }

    /// Whether a completed save published work that can wake the sender.
    pub fn drain_records(&mut self, clock: &dyn Clock) -> Result<bool, PublicationError> {
        let mut published = false;
        let mut failure = None;
        while let Ok(receipt) = self.records.try_recv() {
            let Some(index) = self.cameras.iter().position(|(source_id, binding, _, _)| {
                *source_id == receipt.ticket.source_id && *binding == receipt.ticket.binding
            }) else {
                eprintln!(
                    "ml-worker: recording receipt refused source_id={} request={} binding={:?}: no admitted camera",
                    receipt.ticket.source_id, receipt.ticket.request_id, receipt.ticket.binding
                );
                continue;
            };
            if receipt.result == seeon_deepstream_native::MediaResult::Ok
                && receipt.contains_video
                && !super::recording::admits_path(&self.record_dir, &receipt)
            {
                failure.get_or_insert(PublicationError::MediaPath);
                self.quiesce();
                continue;
            }
            let now = Utc::from_system(clock.wall());
            let clip_id = format!(
                "{}-{}-{}",
                self.boot_id, receipt.ticket.source_id, receipt.ticket.request_id
            );
            let events = std::mem::take(&mut self.events);
            let mut retired = Vec::new();
            let saved = {
                let recorder = &mut self.recorders[index];
                let mut publisher = super::recording::RecordingPublisher {
                    store: &self.store,
                    queue: &self.queue,
                    reserve: &mut self.reserve,
                    sidecars: &self.sidecars,
                    events: &events,
                };
                recorder.on_receipt(&receipt, |sealed| {
                    let saved = publisher.save(&clip_id, sealed, now)?;
                    if !matches!(
                        saved,
                        crate::clips::reserve::SaveOutcome::FinalizeFailed(None)
                    ) {
                        published = true;
                        retired.extend(
                            sealed
                                .contributors
                                .iter()
                                .map(|item| item.event_ref.clone()),
                        );
                    }
                    Ok(saved)
                })
            };
            self.events = events;
            for event_ref in retired {
                self.events.remove(&event_ref);
            }
            match saved {
                Ok(_) => {}
                Err(
                    error @ (RecorderError::WrongSource { .. }
                    | RecorderError::WrongBinding { .. }
                    | RecorderError::WrongGeneration { .. }
                    | RecorderError::WrongEpoch { .. }
                    | RecorderError::DuplicateRequest(_)
                    | RecorderError::UnexpectedRequest(_)
                    | RecorderError::SessionContradiction { .. }
                    | RecorderError::InvalidReceiptTicket),
                ) => {
                    eprintln!(
                        "ml-worker: recording receipt refused camera_id={} request={} result={:?} reason={error}",
                        self.cameras[index].2, receipt.ticket.request_id, receipt.result
                    );
                }
                Err(crate::clips::recorder::RecorderError::Save(error)) => {
                    failure.get_or_insert(PublicationError::Clip(
                        crate::run::clip_output::ClipOutputError::from(error),
                    ));
                    self.quiesce();
                }
                Err(error) => {
                    failure.get_or_insert(PublicationError::Recorder(error));
                    self.quiesce();
                }
            }
        }
        // A failed camera must not strand other genuine queued receipts during
        // shutdown. Keep the first failure, and prevent any subsequent save
        // from starting another recording while handing off the remaining seals.
        failure.map_or(Ok(published), Err)
    }

    pub fn camera_for(
        &self,
        binding: seeon_deepstream_native::MediaBinding,
        source_id: u32,
    ) -> Option<&str> {
        self.cameras
            .iter()
            .find_map(|(id, admitted, camera_id, _)| {
                (*id == source_id && *admitted == binding).then_some(camera_id.as_str())
            })
    }

    /// Diagnostic attribution only: even malformed frame bindings need a
    /// camera-labelled failure. Event publication must use `camera_for`.
    pub fn camera_for_diagnostic(&self, source_id: u32) -> Option<&str> {
        self.cameras
            .iter()
            .find_map(|(id, _, camera_id, _)| (*id == source_id).then_some(camera_id.as_str()))
    }

    pub fn stager_index(&self, camera_id: &str) -> Option<usize> {
        self.cameras
            .iter()
            .position(|(_, _, id, _)| id == camera_id)
    }

    /// The receipt cutoff must be sampled before the preceding receipt drain.
    pub fn admit_recording(
        &mut self,
        index: usize,
        event: &seeon_worker::episode::BusinessEvent,
        staged: &StagedEvent,
        receipt_observation_cutoff: Duration,
    ) -> Result<Admit, PublicationError> {
        let (_, _, camera_id, facility_id) =
            self.cameras.get(index).ok_or(PublicationError::Identity)?;
        if event.camera_id != *camera_id
            || event.facility_id != *facility_id
            || staged.event_ref != event.identity
        {
            return Err(PublicationError::Identity);
        }
        let detected_at =
            Utc::parse(&staged.detected_at).map_err(|_| PublicationError::Timestamp)?;
        let outcome = self.recorders[index]
            .admit_with_receipt_observation_cutoff(
                &staged.event_ref,
                detected_at,
                receipt_observation_cutoff,
            )
            .map_err(PublicationError::Recorder)?;
        self.events.insert(
            event.identity.clone(),
            crate::clips::sealed::SealedEvent {
                identity: event.identity.clone(),
                time_sec: event.time_sec,
                probability: event.probability,
                camera_id: event.camera_id.clone(),
                facility_id: event.facility_id.clone(),
                domain: event.domain.clone(),
                event_type: event.event_type.clone(),
            },
        );
        Ok(outcome)
    }

    /// One pre-drain cutoff covers all recorders, not their content scheduling.
    pub fn tick_recorders(
        &mut self,
        receipt_observation_cutoff: Duration,
    ) -> Result<(), PublicationError> {
        for recorder in &mut self.recorders {
            recorder
                .tick_with_receipt_observation_cutoff(receipt_observation_cutoff)
                .map_err(PublicationError::Recorder)?;
        }
        Ok(())
    }

    pub fn prepare(
        &self,
        index: usize,
        event: &seeon_worker::episode::BusinessEvent,
        stream: &crate::records::builder::Stream,
        frame: crate::records::builder::Frame,
        audit: Option<&crate::policy::emit::Payload>,
    ) -> Result<PreparedEvent, PublicationError> {
        EventDelivery::new(self.clock.as_ref(), &self.stagers[index], &self.queue)
            .prepare(event, stream, frame, audit)
            .map_err(PublicationError::Event)
    }

    pub fn stage(
        &mut self,
        index: usize,
        prepared: &mut PreparedEvent,
        observer: &mut dyn FnMut(crate::records::Record),
    ) -> Result<StagedEvent, PublicationError> {
        EventDelivery::new(self.clock.as_ref(), &self.stagers[index], &self.queue)
            .stage(prepared, observer)
            .map_err(PublicationError::Event)
    }
}

pub fn channels() -> (
    SyncSender<Command>,
    Receiver<Command>,
    SyncSender<RecordReceipt>,
    Receiver<RecordReceipt>,
) {
    let commands = std::sync::mpsc::sync_channel(COMMAND_CAPACITY);
    let records = std::sync::mpsc::sync_channel(RECORD_CAPACITY);
    (commands.0, commands.1, records.0, records.1)
}
#[cfg(test)]
#[path = "publication/native_tests.rs"]
mod native_tests;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
    use crate::clips::manifest::{ClipMetadata, Contributor, Extension, MediaFacts};
    use crate::clips::publish::{MANIFEST_FILE, MEDIA_FILE, Published, Publisher, TERMINAL_MARKER};
    use crate::clips::reserve::FINALIZE_FAILED;
    use crate::clips::sealed::{Recovery, SealedClip, SealedContributor, SealedEvent};
    use crate::records::builder::{Frame, Stream};
    use crate::seam::{IdSource, RandomIds, SystemClock};
    use seeon_deepstream_native::{MediaPoll, MediaResult, RecordTicket};

    struct NoReplayClock;

    impl Clock for NoReplayClock {
        fn monotonic(&self) -> Duration {
            panic!("terminal recovery must not consult the clock")
        }
        fn wall(&self) -> std::time::SystemTime {
            panic!("only fresh READY publication needs a clock")
        }
        fn pause(&self, _: Duration) {
            panic!("startup recovery must not pause")
        }
    }

    struct ReplayFixture {
        root: PathBuf,
        state: PathBuf,
        store: ClipStore,
        queue: DeliveryQueue,
        recovery: Recovery,
        meta: ClipMetadata,
    }

    impl Drop for ReplayFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).expect("owned replay fixture cleanup");
        }
    }

    impl ReplayFixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(RandomIds.uuid4().unwrap());
            std::fs::create_dir(&root).unwrap();
            let state = root.join("state");
            std::fs::create_dir(&state).unwrap();
            let queue = DeliveryQueue::open(&state.join("delivery-queue"), true).unwrap();
            let store = ClipStore::new(root.join("store"));
            let detected = Utc::parse("2026-01-02T03:04:05.123456Z").unwrap();
            let event = SealedEvent {
                domain: "fall".into(),
                event_type: "FALL_DETECTED".into(),
                identity: "event-replay".into(),
                camera_id: "camera-replay".into(),
                facility_id: "facility-replay".into(),
                time_sec: 12.0,
                probability: Some(0.9),
            };
            let attributed = ContributorEvent {
                camera_id: event.camera_id.clone(),
                facility_id: event.facility_id.clone(),
                domain: event.domain.clone(),
                event_type: event.event_type.clone(),
            };
            let sealed = SealedClip {
                clip_id: "clip-replay".into(),
                path: root.join("original.mp4").to_str().unwrap().into(),
                duration_ms: 4_000,
                boundary: "end".into(),
                contributors: vec![SealedContributor {
                    event_ref: event.identity.clone(),
                    detected_at: detected.iso_micros(),
                }],
            };
            let events = BTreeMap::from([(event.identity.clone(), event)]);
            let sidecars = SealedSidecars::new(state.join(SIDECAR_DIR));
            sidecars.persist(&sealed, &events).unwrap();
            let recovery = sidecars
                .pending("camera-replay")
                .unwrap()
                .recoveries
                .remove(0);
            let meta = flow_metadata(
                &sealed.clip_id,
                &BTreeMap::from([("event-replay".into(), attributed)]),
                Extension {
                    boundary: sealed.boundary,
                    duration_ms: sealed.duration_ms,
                    contributors: vec![Contributor {
                        event_ref: "event-replay".into(),
                        detected_at: detected,
                    }],
                },
                FLOW_ENCODER,
                Utc::parse("2026-01-02T03:05:00.000Z").unwrap(),
            )
            .unwrap();
            Self {
                root,
                state,
                store,
                queue,
                recovery,
                meta,
            }
        }

        fn unavailable(&self) -> Published {
            let reservation = self
                .store
                .reserve(&self.meta.camera_id, &self.meta.clip_id)
                .unwrap();
            Publisher::new(&self.queue)
                .publish_unavailable(&reservation, &self.meta, FINALIZE_FAILED, None)
                .unwrap()
        }

        fn replay(&self) -> Replay {
            replay_before_media(
                &self.state,
                std::slice::from_ref(&self.recovery.camera_id),
                &self.store,
                &self.queue,
                &NoReplayClock,
            )
            .unwrap()
        }
    }

    #[test]
    fn startup_orphan_terminal_marker_retains_attribution_without_readmission() {
        let fixture = ReplayFixture::new();
        let published = fixture.unavailable();
        let marker = fixture
            .store
            .clip_dir(&published.clip_id)
            .join(TERMINAL_MARKER);
        let marker_bytes = std::fs::read(&marker).unwrap();
        let sidecar_bytes = std::fs::read(&fixture.recovery.sidecar_path).unwrap();
        assert!(
            fixture
                .queue
                .acknowledge_backend(published.entry.entry_id(), 204)
                .unwrap()
        );
        std::fs::remove_file(published.manifest_path).unwrap();
        assert_eq!(
            fixture.replay().reports,
            vec![ReplayReport {
                failed: 1,
                ..ReplayReport::default()
            }]
        );
        assert_eq!(std::fs::read(marker).unwrap(), marker_bytes);
        assert_eq!(
            std::fs::read(&fixture.recovery.sidecar_path).unwrap(),
            sidecar_bytes
        );
        assert!(fixture.queue.entries().unwrap().is_empty());
    }

    #[test]
    fn startup_probe_selects_the_same_owned_media_as_publication() {
        for staged in [true, false] {
            let fixture = ReplayFixture::new();
            let original = Path::new(&fixture.recovery.sealed.path);
            std::fs::write(original, b"different original bytes").unwrap();
            let reservation = fixture
                .store
                .reserve(&fixture.meta.camera_id, &fixture.meta.clip_id)
                .unwrap();
            let selected = if staged {
                reservation.artifact_path()
            } else {
                reservation.final_dir.join(MEDIA_FILE)
            };
            std::fs::create_dir_all(selected.parent().unwrap()).unwrap();
            std::fs::write(&selected, b"owned publication bytes").unwrap();
            assert_eq!(
                recovery_media(&fixture.recovery, &fixture.store).unwrap(),
                Some(selected)
            );
            // Synthetic bytes establish owner selection, not measured codec.
            let published = clip_output::publish_recovery(
                &fixture.recovery,
                &fixture.store,
                &fixture.queue,
                "h264",
                Utc::parse("2026-01-02T03:05:00.000Z").unwrap(),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(published.video_path.unwrap()).unwrap(),
                b"owned publication bytes"
            );
            assert_eq!(
                std::fs::read(original).unwrap(),
                b"different original bytes"
            );
        }
    }

    #[test]
    fn startup_nonregular_media_is_not_missing_attribution() {
        for symlink in [false, true] {
            let fixture = ReplayFixture::new();
            let path = Path::new(&fixture.recovery.sealed.path);
            if symlink {
                std::os::unix::fs::symlink(fixture.root.join("absent"), path).unwrap();
            } else {
                std::fs::create_dir(path).unwrap();
            }
            let bytes = std::fs::read(&fixture.recovery.sidecar_path).unwrap();
            assert_eq!(
                fixture.replay().reports,
                vec![ReplayReport {
                    failed: 1,
                    ..ReplayReport::default()
                }]
            );
            assert_eq!(
                std::fs::read(&fixture.recovery.sidecar_path).unwrap(),
                bytes
            );
            assert!(fixture.queue.entries().unwrap().is_empty());
        }
    }

    #[test]
    fn startup_reconciles_finalize_failed_without_media_clock_or_codec_probe() {
        let fixture = ReplayFixture::new();
        let first = fixture.unavailable();
        let marker = fixture.store.clip_dir(&first.clip_id).join(TERMINAL_MARKER);
        let marker_bytes = std::fs::read(&marker).unwrap();
        std::fs::remove_file(&marker).unwrap();
        assert!(
            fixture
                .queue
                .acknowledge_backend(first.entry.entry_id(), 204)
                .unwrap()
        );
        assert!(!Path::new(&fixture.recovery.sealed.path).exists());
        assert!(
            !fixture
                .store
                .staging_dir(&first.clip_id)
                .join(crate::clips::store::ARTIFACT_FILE)
                .exists()
        );
        assert!(
            !fixture
                .store
                .clip_dir(&first.clip_id)
                .join(MEDIA_FILE)
                .exists()
        );

        let replay = fixture.replay();

        assert_eq!(
            replay.reports,
            vec![ReplayReport {
                published: 1,
                ..ReplayReport::default()
            }]
        );
        assert!(
            !fixture.recovery.sidecar_path.exists(),
            "terminal handoff retires attribution"
        );
        assert_eq!(
            std::fs::read(first.manifest_path).unwrap(),
            first.manifest_bytes
        );
        assert_eq!(std::fs::read(marker).unwrap(), marker_bytes);
        let queued = fixture.queue.entries().unwrap();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0]["entry_id"], first.entry.entry_id());
        assert_eq!(queued[0]["finalized_at"], "2026-01-02T03:05:00.000Z");
        assert_eq!(queued[0]["unavailable_reason"], FINALIZE_FAILED);
    }

    #[test]
    fn startup_terminal_conflicts_keep_sidecar_manifest_marker_and_entry() {
        for (field, replacement) in [("camera_id", "other-camera"), ("state", "UNKNOWN")] {
            let fixture = ReplayFixture::new();
            let first = fixture.unavailable();
            let sidecar_bytes = std::fs::read(&fixture.recovery.sidecar_path).unwrap();
            let marker = fixture.store.clip_dir(&first.clip_id).join(TERMINAL_MARKER);
            let marker_bytes = std::fs::read(&marker).unwrap();
            let queued = fixture.queue.entries().unwrap();
            let mut value: serde_json::Value =
                serde_json::from_slice(&first.manifest_bytes).unwrap();
            value[field] = serde_json::json!(replacement);
            let body = crate::json::Serialiser::ModelSelection
                .canonical(&crate::json::Json::from(&value))
                .unwrap();
            let conflict = format!("{body}\n").into_bytes();
            std::fs::write(&first.manifest_path, &conflict).unwrap();

            let replay = fixture.replay();

            assert_eq!(
                replay.reports,
                vec![ReplayReport {
                    failed: 1,
                    ..ReplayReport::default()
                }]
            );
            assert_eq!(
                std::fs::read(&fixture.recovery.sidecar_path).unwrap(),
                sidecar_bytes
            );
            assert_eq!(std::fs::read(first.manifest_path).unwrap(), conflict);
            assert_eq!(std::fs::read(marker).unwrap(), marker_bytes);
            assert_eq!(fixture.queue.entries().unwrap(), queued);
        }
    }

    #[test]
    fn startup_ready_checks_actual_final_bytes_without_codec_probe() {
        for change in ["unchanged", "changed", "missing"] {
            let fixture = ReplayFixture::new();
            let reservation = fixture
                .store
                .reserve(&fixture.meta.camera_id, &fixture.meta.clip_id)
                .unwrap();
            // Opaque bytes cannot pass a codec probe; an identical terminal must not try one.
            let media = b"opaque sealed recording bytes";
            std::fs::write(reservation.artifact_path(), media).unwrap();
            let facts = MediaFacts {
                sha256: crate::clips::durable::sha256_hex(media),
                size_bytes: i64::try_from(media.len()).unwrap(),
                codec: "h264".into(),
                duration_ms: fixture.recovery.sealed.duration_ms,
            };
            let first = Publisher::new(&fixture.queue)
                .publish_ready(&reservation, &fixture.meta, &facts)
                .unwrap();
            let video = first.video_path.as_ref().unwrap();
            let sidecar_bytes = std::fs::read(&fixture.recovery.sidecar_path).unwrap();
            match change {
                "changed" => {
                    let mut altered = media.to_vec();
                    altered[0] ^= 1;
                    std::fs::write(video, altered).unwrap();
                }
                "missing" => std::fs::remove_file(video).unwrap(),
                _ => {}
            }

            let replay = fixture.replay();

            let expected = if change == "unchanged" {
                assert!(!fixture.recovery.sidecar_path.exists());
                ReplayReport {
                    published: 1,
                    ..ReplayReport::default()
                }
            } else {
                assert_eq!(
                    std::fs::read(&fixture.recovery.sidecar_path).unwrap(),
                    sidecar_bytes
                );
                ReplayReport {
                    failed: 1,
                    ..ReplayReport::default()
                }
            };
            assert_eq!(replay.reports, vec![expected]);
            assert_eq!(
                std::fs::read(first.manifest_path).unwrap(),
                first.manifest_bytes
            );
            assert_eq!(fixture.queue.entries().unwrap().len(), 1);
        }
    }

    #[test]
    fn startup_retires_truly_missing_ready_only_without_a_terminal() {
        let fixture = ReplayFixture::new();
        let replay = fixture.replay();
        assert_eq!(
            replay.reports,
            vec![ReplayReport {
                missing_media: 1,
                ..ReplayReport::default()
            }]
        );
        assert!(!fixture.recovery.sidecar_path.exists());
        assert!(
            !fixture
                .store
                .clip_dir(&fixture.recovery.sealed.clip_id)
                .join(MANIFEST_FILE)
                .exists()
        );
        assert!(fixture.queue.entries().unwrap().is_empty());
    }

    #[test]
    fn recording_directory_is_prepared_without_replacing_existing_contents() {
        let root = std::env::temp_dir().join(RandomIds.uuid4().unwrap());
        std::fs::create_dir(&root).unwrap();
        let state = root.join("state");
        let store = root.join("store");
        let record_dir = root.join("nested/records");
        let open_at = |record_dir: &Path, state_dir: &Path| {
            let (commands, _, _, records) = channels();
            open(
                PublicationConfig {
                    boot_id: "00000000-0000-4000-8000-000000000099",
                    state_dir,
                    record_dir,
                    store_root: &store,
                    cameras: &[],
                    config_version: 1,
                    manifest_sha: None,
                },
                commands,
                records,
                Arc::new(SystemClock::new()),
            )
        };
        drop(open_at(&record_dir, &state).unwrap());
        let retained = record_dir.join("retained.mp4");
        std::fs::write(&retained, b"existing recording bytes").unwrap();
        drop(open_at(&record_dir, &state).unwrap());
        assert_eq!(
            std::fs::read(&retained).unwrap(),
            b"existing recording bytes"
        );
        let refused_state = root.join("refused-state");
        assert!(matches!(
            open_at(&retained, &refused_state),
            Err(PublicationError::RecordDirectory(_))
        ));
        assert!(!refused_state.exists());
        assert_eq!(
            std::fs::read(&retained).unwrap(),
            b"existing recording bytes"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn durable_events_start_their_camera_and_receipts_finish_only_that_recorder() {
        struct Scratch(PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                std::fs::remove_dir_all(&self.0).expect("owned recording fixture cleanup");
            }
        }
        let root = Scratch(std::env::temp_dir().join(RandomIds.uuid4().unwrap()));
        std::fs::create_dir(&root.0).unwrap();
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let binding = |token| MediaBinding {
            token,
            generation: 3,
            epoch: 5,
        };
        let cameras = [
            (
                3,
                binding(10),
                "camera-a".to_owned(),
                "facility-a".to_owned(),
            ),
            (
                7,
                binding(20),
                "camera-b".to_owned(),
                "facility-b".to_owned(),
            ),
        ];
        let (commands, inbox, receipts, records) = channels();
        let worker = std::thread::spawn(move || {
            let mut tickets = Vec::new();
            for (offset, expected) in [3, 7].into_iter().enumerate() {
                let command = inbox.recv_timeout(Duration::from_secs(5)).unwrap();
                let Command::RecordStart {
                    source_id,
                    binding,
                    reply,
                    ..
                } = command
                else {
                    panic!("expected actual record start");
                };
                assert_eq!(source_id, expected);
                let ticket = RecordTicket {
                    binding,
                    request_id: u64::try_from(offset).unwrap() + 1,
                    source_id,
                    session_id: source_id + 40,
                    session_valid: 1,
                    coalesced: 0,
                };
                reply.send(Ok(MediaPoll::Ready(ticket))).unwrap();
                tickets.push(ticket);
            }
            tickets
        });
        let state = root.0.join("state");
        let record_dir = root.0.join("records");
        let store = root.0.join("store");
        let mut output = open(
            PublicationConfig {
                boot_id: "00000000-0000-4000-8000-000000000099",
                state_dir: &state,
                record_dir: &record_dir,
                store_root: &store,
                cameras: &cameras,
                config_version: 8,
                manifest_sha: None,
            },
            commands,
            records,
            Arc::clone(&clock),
        )
        .unwrap();
        for (index, (_, _, camera, facility)) in cameras.iter().enumerate() {
            let event = seeon_worker::episode::BusinessEvent {
                domain: "fall".into(),
                event_type: "FALL_DETECTED".into(),
                identity: format!("00000000-0000-4000-8000-00000000000{index}"),
                camera_id: camera.clone(),
                facility_id: facility.clone(),
                time_sec: 12.0,
                probability: Some(0.9),
                person_id: Some(7),
                bed_id: None,
            };
            let stream = Stream {
                camera_id: camera.clone(),
                worker_boot_id: "boot-recording-test".into(),
                source_generation: 3,
                stream_epoch: 5,
            };
            let prepared = output
                .prepare(
                    index,
                    &event,
                    &stream,
                    Frame {
                        frame_seq: 90,
                        source_pts_ns: Some(12_000_000_000),
                    },
                    None,
                )
                .unwrap();
            let mut prepared = prepared;
            let staged = output.stage(index, &mut prepared, &mut |_| {}).unwrap();
            assert!(matches!(
                output.admit_recording(index, &event, &staged, clock.monotonic()),
                Ok(Admit::Started(_))
            ));
        }
        let tickets = worker.join().unwrap();
        assert!(
            record_dir.is_dir(),
            "output preparation owns recording directory creation"
        );
        let outside = root.0.join("outside.mp4");
        std::fs::write(&outside, b"not owned by recorder").unwrap();
        receipts
            .send(RecordReceipt {
                ticket: tickets[1],
                result: MediaResult::Ok,
                error: 0,
                duration_ms: 30_000,
                width: 1280,
                height: 720,
                contains_video: true,
                contains_audio: false,
                directory: root.0.clone(),
                filename: "outside.mp4".into(),
            })
            .unwrap();
        assert!(matches!(
            output.drain_records(clock.as_ref()),
            Err(PublicationError::MediaPath)
        ));
        assert_eq!(std::fs::read(outside).unwrap(), b"not owned by recorder");
        assert_eq!(
            output.recorders[1].state(),
            crate::clips::recorder::State::Recording
        );
        let retained_events = output.events.len();
        receipts
            .send(RecordReceipt {
                ticket: RecordTicket {
                    request_id: 999,
                    ..tickets[1]
                },
                result: MediaResult::Ok,
                error: 0,
                duration_ms: 0,
                width: 0,
                height: 0,
                contains_video: false,
                contains_audio: false,
                directory: record_dir.clone(),
                filename: "unmatched.mp4".into(),
            })
            .unwrap();
        assert!(!output.drain_records(clock.as_ref()).unwrap());
        assert_eq!(output.events.len(), retained_events);
        assert_eq!(
            output.recorders[1].state(),
            crate::clips::recorder::State::Recording
        );
        receipts
            .send(RecordReceipt {
                ticket: tickets[1],
                result: MediaResult::Ok,
                error: 0,
                duration_ms: 30_000,
                width: 1280,
                height: 720,
                contains_video: false,
                contains_audio: false,
                directory: record_dir,
                filename: "no-frames.mp4".into(),
            })
            .unwrap();
        output.drain_records(clock.as_ref()).unwrap();
        assert_eq!(
            output.recorders[0].state(),
            crate::clips::recorder::State::Recording
        );
        assert_eq!(
            output.recorders[1].state(),
            crate::clips::recorder::State::Idle
        );
        assert_eq!(output.recorders[0].pending(), 0);
        assert_eq!(output.recorders[1].pending(), 0);
        let clip_id = "00000000-0000-4000-8000-000000000099-7-2";
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(output.store.clip_dir(clip_id).join("manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["clip_id"], clip_id);
        assert_eq!(manifest["camera_id"], "camera-b");
        assert!(!output.store.clip_dir("47").exists());
        assert!(
            !output
                .store
                .clip_dir("00000000-0000-4000-8000-000000000099-7-47")
                .exists()
        );
    }
}
