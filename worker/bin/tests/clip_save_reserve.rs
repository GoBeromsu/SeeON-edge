//! T37: saving a sealed recording when the clip volume is full (G5, X18).
//!
//! The ENOSPC cases run only in the lane that mounts a real 8 MiB tmpfs
//! volume at `$SEEON_TEST_ENOSPC_DIR`; the clip store lives there while the
//! delivery queue and the sealed recording stay on the state volume, so the
//! save copies the recording across filesystems as it does in production.
//! The recorder talks to a channel-level stand-in for the media thread. The
//! backend oracle `parse_manifest_bytes` judges the FINALIZE_FAILED manifest
//! through `$SEEON_TEST_PYTHON`. The remaining cases check the binding error
//! mapping without a full volume.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command as Process;
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::Value;

use seeon_deepstream_native::{MediaBinding, MediaPoll, MediaResult, RecordTicket};
use seeon_ml_worker::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use seeon_ml_worker::clips::publish::{
    MANIFEST_FILE, MEDIA_FILE, PublishError, Publisher, TERMINAL_MARKER,
};
use seeon_ml_worker::clips::recorder::{
    Admit, ClipSealed, CommandPlane, PlaneRefusal, Recorder, RecorderError, State,
};
use seeon_ml_worker::clips::reserve::{FINALIZE_FAILED, ReservePool, SaveOutcome};
use seeon_ml_worker::clips::store::ClipStore;
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::delivery::DeliveryQueue;
use seeon_ml_worker::media::{COMMAND_CAPACITY, Command};
use seeon_ml_worker::msg::RecordReceipt;
use seeon_ml_worker::seam::SystemClock;

const SOURCE_ID: u32 = 4;
const BINDING: MediaBinding = MediaBinding {
    token: 73,
    generation: 7,
    epoch: 11,
};
const CAMERA: &str = "cmsnw6rjc01vhlh01oswn99yq";
const CLIP: &str = "cmsnw6rjc01vhlh01oswn99yq-20260820T172057197192Z-00b200000001";
const EVENTS: [&str; 2] = [
    "00000000-0000-4000-8000-00000000e0a1",
    "00000000-0000-4000-8000-00000000e0b2",
];
/// Larger than anything a full 8 MiB tmpfs has left, smaller than one slot.
const MEDIA_BYTES: usize = 64 * 1024;

static ENOSPC_VOLUME: Mutex<()> = Mutex::new(());

fn at(text: &str) -> Utc {
    Utc::parse(text).expect("RFC 3339 timestamp")
}

/// Answers every record start with a fresh session until the recorder goes.
fn media_stand_in() -> (SyncSender<Command>, JoinHandle<()>) {
    let (commands, inbox) = mpsc::sync_channel(COMMAND_CAPACITY);
    let thread = thread::spawn(move || {
        let mut session = 1_u32;
        let mut request_id = 0_u64;
        while let Ok(command) = inbox.recv() {
            if let Command::RecordStart {
                source_id,
                binding,
                reply,
                ..
            } = command
            {
                request_id = request_id
                    .checked_add(1)
                    .expect("test media request sequence does not exhaust");
                let ticket = RecordTicket {
                    binding,
                    request_id,
                    source_id,
                    session_id: session,
                    session_valid: 1,
                    coalesced: 0,
                };
                session += 1;
                let _ = reply.send(Ok(MediaPoll::Ready(ticket)));
            }
        }
    });
    (commands, thread)
}

fn recorder(commands: SyncSender<Command>) -> Recorder<CommandPlane> {
    let plane = CommandPlane::new(commands, SOURCE_ID, BINDING, Duration::from_secs(5));
    Recorder::new(SOURCE_ID, plane, Arc::new(SystemClock::new()))
}

fn started(recorder: &mut Recorder<CommandPlane>, event_ref: &str) -> RecordTicket {
    match recorder.admit(event_ref, at("2026-08-20T17:20:58.197192Z")) {
        Ok(Admit::Started(ticket)) => ticket,
        other => panic!("expected a started recording, got {other:?}"),
    }
}

fn receipt(ticket: RecordTicket, sealed: &Path) -> RecordReceipt {
    RecordReceipt {
        ticket,
        result: MediaResult::Ok,
        error: 0,
        duration_ms: 30_000,
        width: 1280,
        height: 720,
        contains_video: true,
        contains_audio: false,
        directory: sealed.parent().expect("sealed directory").to_path_buf(),
        filename: sealed.file_name().expect("sealed name").to_os_string(),
    }
}

/// A test's directories: the clip store under `store_base`, and the queue
/// and sealed recording on the state volume. Removed again on drop.
struct Bench {
    store_dir: PathBuf,
    state_dir: PathBuf,
    store: ClipStore,
    queue: DeliveryQueue,
    sealed: PathBuf,
}

impl Bench {
    fn new(store_base: &Path, test: &str) -> Self {
        let store_dir = store_base.join(test);
        let state_dir =
            PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("clip_save_reserve-{test}"));
        for dir in [&store_dir, &state_dir] {
            if dir.exists() {
                fs::remove_dir_all(dir).expect("clear previous run");
            }
            fs::create_dir_all(dir).expect("test directory");
        }
        let queue = DeliveryQueue::open(&state_dir.join("delivery-queue"), true).expect("queue");
        let sealed = state_dir.join("sealed.mp4");
        fs::write(&sealed, vec![0x5a_u8; MEDIA_BYTES]).expect("sealed recording");
        Self {
            store: ClipStore::new(store_dir.join("store")),
            store_dir,
            state_dir,
            queue,
            sealed,
        }
    }

    /// Fills the clip volume until the kernel reports it full.
    fn fill(&self) {
        let mut filler = File::create(self.store_dir.join("filler")).expect("filler");
        for block in [vec![0_u8; 64 * 1024], vec![0_u8; 4096]] {
            loop {
                match filler.write_all(&block) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::StorageFull => break,
                    Err(error) => panic!("filling the volume failed: {error}"),
                }
            }
        }
    }

    fn save(
        &self,
        pool: &mut ReservePool,
        sealed: &ClipSealed,
    ) -> Result<SaveOutcome, PublishError> {
        let event = ContributorEvent {
            camera_id: CAMERA.to_owned(),
            facility_id: "facility-1".to_owned(),
            domain: "fall".to_owned(),
            event_type: "fall".to_owned(),
        };
        let events: BTreeMap<String, ContributorEvent> = sealed
            .contributors
            .iter()
            .map(|c| (c.event_ref.clone(), event.clone()))
            .collect();
        let meta = flow_metadata(
            CLIP,
            &events,
            sealed.extension(),
            FLOW_ENCODER,
            at("2026-08-20T17:21:30Z"),
        )
        .map_err(PublishError::Manifest)?;
        pool.save(
            &self.store,
            &Publisher::new(&self.queue),
            &meta,
            sealed,
            "h264",
        )
    }

    fn final_dir(&self) -> PathBuf {
        self.store.root().join("clips").join(CLIP)
    }
}

impl Drop for Bench {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.store_dir);
        let _ = fs::remove_dir_all(&self.state_dir);
    }
}

fn enospc_volume() -> (PathBuf, MutexGuard<'static, ()>) {
    let dir = std::env::var_os("SEEON_TEST_ENOSPC_DIR").expect("SEEON_TEST_ENOSPC_DIR is required");
    let guard = ENOSPC_VOLUME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    (PathBuf::from(dir), guard)
}

const BACKEND_PARSE: &str = "
import json, sys
from backend.app.features.clips.manifest import parse_manifest_bytes
out = []
for path in sys.argv[1:]:
    with open(path, 'rb') as handle:
        manifest = parse_manifest_bytes(handle.read())
    if manifest is None:
        out.append({'verdict': 'rejected'})
    else:
        out.append({'as_response': manifest.as_response(),
                    'event_refs': list(manifest.event_refs), 'verdict': 'parsed'})
print(json.dumps(out))
";

fn backend_view(manifest: &Path) -> Value {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON is required");
    let output = Process::new(python)
        .arg("-c")
        .arg(BACKEND_PARSE)
        .arg(manifest)
        .output()
        .expect("python runs");
    assert!(
        output.status.success(),
        "backend parser exited with {:?}",
        output.status
    );
    let parsed: Value = serde_json::from_slice(&output.stdout).expect("parser output is JSON");
    parsed[0].clone()
}

#[test]
#[ignore = "requires SEEON_TEST_ENOSPC_DIR"]
fn full_volume_publishes_finalize_failed_and_the_recorder_admits_again() {
    let (volume, _serial) = enospc_volume();
    let bench = Bench::new(&volume, "full");
    let mut pool = ReservePool::arm(&bench.store, 1).expect("reserve armed");
    assert_eq!(pool.available(), ReservePool::slots_for(1));
    bench.fill();
    let (commands, media) = media_stand_in();
    let mut recorder = recorder(commands);
    let ticket = started(&mut recorder, EVENTS[0]);
    assert_eq!(
        recorder
            .admit(EVENTS[1], at("2026-08-20T17:21:01Z"))
            .expect("extend"),
        Admit::Extended
    );

    let outcome = recorder
        .on_receipt(&receipt(ticket, &bench.sealed), |sealed| {
            bench.save(&mut pool, sealed)
        })
        .expect("a full volume is a save outcome, not an error");

    let SaveOutcome::FinalizeFailed(Some(published)) = outcome else {
        panic!("expected FINALIZE_FAILED with a reserve-slot manifest, got {outcome:?}");
    };
    assert_eq!(recorder.state(), State::Idle);
    assert!(
        matches!(recorder.admit("00000000-0000-4000-8000-00000000e0c3", at("2026-08-20T17:22:00Z")), Ok(Admit::Started(next)) if next.session_id != ticket.session_id)
    );
    assert_eq!(pool.available(), ReservePool::slots_for(1) - 1);

    let view = backend_view(&published.manifest_path);
    assert_eq!(view["verdict"], "parsed");
    assert_eq!(view["as_response"]["video_available"], false);
    assert_eq!(view["as_response"]["video_error"], FINALIZE_FAILED);
    assert_eq!(view["as_response"]["path"], Value::Null);
    assert_eq!(view["as_response"]["finalized"], true);
    assert_eq!(view["event_refs"], Value::from(EVENTS.to_vec()));

    let fields = published.entry.fields();
    assert_eq!(fields.local_state, "UNAVAILABLE");
    assert_eq!(fields.unavailable_reason.as_deref(), Some(FINALIZE_FAILED));
    let queued = bench.queue.entries().expect("queue entries");
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0]["entry_id"], published.entry.entry_id());
    assert_eq!(queued[0]["local_state"], "UNAVAILABLE");
    assert_eq!(queued[0]["unavailable_reason"], FINALIZE_FAILED);

    assert!(!bench.final_dir().join(MEDIA_FILE).exists());
    assert!(
        !bench
            .store
            .root()
            .join("clips/.staging")
            .join(CLIP)
            .exists()
    );
    assert_eq!(
        fs::read(&bench.sealed)
            .expect("sealed recording kept")
            .len(),
        MEDIA_BYTES
    );
    drop(recorder);
    media.join().expect("media stand-in");
}

#[test]
#[ignore = "requires SEEON_TEST_ENOSPC_DIR"]
fn same_volume_with_room_publishes_the_clip() {
    let (volume, _serial) = enospc_volume();
    let bench = Bench::new(&volume, "room");
    let mut pool = ReservePool::arm(&bench.store, 1).expect("reserve armed");
    let (commands, media) = media_stand_in();
    let mut recorder = recorder(commands);
    let ticket = started(&mut recorder, EVENTS[0]);

    let outcome = recorder
        .on_receipt(&receipt(ticket, &bench.sealed), |sealed| {
            bench.save(&mut pool, sealed)
        })
        .expect("save");

    let SaveOutcome::Saved(published) = outcome else {
        panic!("expected a published clip, got {outcome:?}");
    };
    assert_eq!(published.entry.fields().local_state, "VERIFIED");
    let media_bytes = fs::read(bench.final_dir().join(MEDIA_FILE)).expect("published media");
    assert_eq!(media_bytes, vec![0x5a_u8; MEDIA_BYTES]);
    assert_eq!(pool.available(), ReservePool::slots_for(1));
    assert_eq!(recorder.state(), State::Idle);
    drop(recorder);
    media.join().expect("media stand-in");
}

#[test]
#[ignore = "requires SEEON_TEST_ENOSPC_DIR"]
fn volume_full_before_arming_retains_unpublished_and_refuses_new_admission() {
    let (volume, _serial) = enospc_volume();
    let bench = Bench::new(&volume, "unarmed");
    fs::create_dir_all(bench.store.root()).expect("store root");
    bench.fill();
    let mut pool = ReservePool::arm(&bench.store, 1).expect("arming a full store is not an error");
    assert_eq!(pool.available(), 0);
    let (commands, media) = media_stand_in();
    let mut recorder = recorder(commands);
    let ticket = started(&mut recorder, EVENTS[0]);

    let result = recorder.on_receipt(&receipt(ticket, &bench.sealed), |sealed| {
        bench.save(&mut pool, sealed)
    });

    assert!(
        matches!(result, Err(RecorderError::Unpublished { ticket: retained }) if retained == ticket),
        "{result:?}"
    );
    assert!(!bench.final_dir().join(MANIFEST_FILE).exists());
    assert!(!bench.final_dir().join(TERMINAL_MARKER).exists());
    assert!(bench.queue.entries().expect("queue entries").is_empty());
    assert_eq!(
        fs::read(&bench.sealed).expect("unpublished media"),
        vec![0x5a_u8; MEDIA_BYTES]
    );
    assert_eq!(recorder.state(), State::Finalizing);
    assert!(matches!(
        recorder.admit(EVENTS[1], at("2026-08-20T17:22:00Z")),
        Err(RecorderError::Unpublished { ticket: retained }) if retained == ticket
    ));
    drop(recorder);
    media.join().expect("media stand-in");
}

fn cpu_bench(test: &str) -> Bench {
    Bench::new(
        &PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("clip_save_reserve-store"),
        test,
    )
}

#[test]
fn other_io_failure_is_returned_and_the_recorder_admits_again() {
    let bench = cpu_bench("missing");
    let mut pool = ReservePool::arm(&bench.store, 1).expect("reserve armed");
    let (commands, media) = media_stand_in();
    let mut recorder = recorder(commands);
    let ticket = started(&mut recorder, EVENTS[0]);
    fs::remove_file(&bench.sealed).expect("lose the sealed recording");

    let result = recorder.on_receipt(&receipt(ticket, &bench.sealed), |sealed| {
        bench.save(&mut pool, sealed)
    });

    match result {
        Err(RecorderError::Save(PublishError::Io(error))) => {
            assert_eq!(error.kind(), io::ErrorKind::NotFound)
        }
        other => panic!("expected the I/O error, got {other:?}"),
    }
    assert!(!bench.final_dir().join(MANIFEST_FILE).exists());
    assert_eq!(pool.available(), ReservePool::slots_for(1));
    assert_eq!(recorder.state(), State::Idle);
    assert!(matches!(
        recorder.admit(EVENTS[1], at("2026-08-20T17:22:00Z")),
        Ok(Admit::Started(_))
    ));
    drop(recorder);
    media.join().expect("media stand-in");
}

#[test]
fn a_second_receipt_for_a_sealed_session_is_refused_without_saving() {
    let bench = cpu_bench("duplicate");
    let mut pool = ReservePool::arm(&bench.store, 1).expect("reserve armed");
    let (commands, media) = media_stand_in();
    let mut recorder = recorder(commands);
    let ticket = started(&mut recorder, EVENTS[0]);
    let sealed = receipt(ticket, &bench.sealed);
    let first = recorder
        .on_receipt(&sealed, |clip| bench.save(&mut pool, clip))
        .expect("first save");
    assert!(matches!(first, SaveOutcome::Saved(_)), "{first:?}");

    let saved_again = Cell::new(false);
    let second = recorder.on_receipt(&sealed, |_| {
        saved_again.set(true);
        Err(PublishError::MissingMedia)
    });

    assert!(
        matches!(second, Err(RecorderError::DuplicateRequest(request)) if request == ticket.request_id)
    );
    assert!(!saved_again.get());
    assert_eq!(recorder.state(), State::Idle);
    drop(recorder);
    media.join().expect("media stand-in");
}

#[test]
fn a_gone_media_thread_refuses_starts_and_bounds_waiting_alerts() {
    let (commands, inbox) = mpsc::sync_channel(COMMAND_CAPACITY);
    drop(inbox);
    let mut recorder = recorder(commands);
    for index in 0..128 {
        let alert = format!("00000000-0000-4000-8000-{index:012x}");
        assert!(matches!(
            recorder.admit(&alert, at("2026-08-20T17:20:58Z")),
            Ok(Admit::Refused(PlaneRefusal::Closed))
        ));
    }

    let overflow = recorder.admit(
        "00000000-0000-4000-8000-0000000000ff",
        at("2026-08-20T17:20:59Z"),
    );

    assert!(
        matches!(overflow, Err(RecorderError::PendingFull)),
        "{overflow:?}"
    );
    assert_eq!(recorder.pending(), 128);
    assert_eq!(recorder.counters().refused, 128);
    assert_eq!(recorder.state(), State::Idle);
}
