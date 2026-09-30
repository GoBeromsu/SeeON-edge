//! T10-T15: the Rust delivery queue and the Python `shared/events/delivery_queue.py`
//! share one on-disk format and one lock. The oracles are the reviewed Python
//! goldens under `tests/fixtures/worker-wire/d/` and, in the xlang lane, the
//! Python queue itself run as a child through `$SEEON_TEST_PYTHON`.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::Duration;

use seeon_ml_worker::delivery::{
    AdmissionFault, AdmissionResult, ClipEntry, ClipFields, DeliveryEntry, DeliveryQueue,
    EntryKind, EventEntry, EventFields, LOCK_FILE_NAME, MAX_ACCEPTED_BYTES, MAX_ACCEPTED_ENTRIES,
    SnapshotAttachmentEntry, SnapshotAttachmentFields, SnapshotDispositionEntry,
    SnapshotDispositionFields,
};
use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::seam::{Clock, SystemClock};
use serde_json::{Value, json};

const EVENT_ENTRY_ID: &str = "event-a5e15ff2-90fd-4764-be74-a7da4f573cc9";
const EVENT_FILE: &str = "event-a5e15ff2-90fd-4764-be74-a7da4f573cc9.json";
const DEAD_LETTER_FILE: &str = "422.0.event-a5e15ff2-90fd-4764-be74-a7da4f573cc9.json";
const ORPHAN_TEMP: &str = ".0123abcd.tmp";

const ADMITTED: AdmissionResult = AdmissionResult {
    accepted: true,
    fault: None,
    already_admitted: false,
};
const ALREADY_ADMITTED: AdmissionResult = AdmissionResult {
    accepted: true,
    fault: None,
    already_admitted: true,
};

/// Holds the Python queue lock, reports `held` on stdout, releases on stdin EOF.
const HOLD: &str = "\
import sys
from pathlib import Path
from shared.events.delivery_queue import DeliveryQueue
queue = DeliveryQueue(Path(sys.argv[1]), recover=False)
with queue._locked():
    print('held', flush=True)
    sys.stdin.read()
";

/// Reads a Rust-written queue through the Python queue and prints one JSON report.
const READ: &str = "\
import base64, json, sys
from pathlib import Path
from shared.events.delivery_queue import DeliveryQueue, EventEntry
live, retained, golden, full = (Path(arg) for arg in sys.argv[1:5])
def admission(result):
    fault = None if result.fault is None else str(result.fault)
    return {'accepted': result.accepted, 'fault': fault, 'already_admitted': result.already_admitted}
queue = DeliveryQueue(live)
snapshot = queue.capacity_snapshot
report = {
    'entries': [[entry['kind'], entry['entry_id']] for entry in queue.entries()],
    'accepted_count': snapshot.accepted_count,
    'accepted_bytes': snapshot.accepted_bytes,
    'by_kind': {str(kind): count for kind, count in snapshot.by_kind.items()},
    'dead_lettered_count': snapshot.dead_lettered_count,
    'dead_lettered_bytes': snapshot.dead_lettered_bytes,
}
report['requeued'] = queue.requeue_dead_lettered(retained)
report['count_after_requeue'] = queue.accepted_count
fields = json.loads(golden.read_bytes())
del fields['kind'], fields['entry_id']
fields['decision_trace'] = base64.b64decode(fields.pop('decision_trace_b64'))
fields['values'] = base64.b64decode(fields.pop('values_b64'))
fields['shed_detail_keys'] = tuple(fields['shed_detail_keys'])
report['readmit'] = admission(queue.try_admit(EventEntry(**fields)))
fields['edge_event_id'] = 'b2c0dec0-0000-4000-8000-0000000000ff'
report['full'] = admission(DeliveryQueue(full).try_admit(EventEntry(**fields)))
print(json.dumps(report, sort_keys=True))
";

/// Interpreter start plus one import; an upper bound, not a pause.
const HOLDER_WAIT: Duration = Duration::from_secs(30);

fn refused(fault: AdmissionFault) -> AdmissionResult {
    AdmissionResult {
        accepted: false,
        fault: Some(fault),
        already_admitted: false,
    }
}

fn worker_wire() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/worker-wire")
}

/// A new empty directory owned by one test.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("queue_python_interop")
        .join(name);
    match fs::remove_dir_all(&dir) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => panic!("clear {}: {error}", dir.display()),
    }
    fs::create_dir_all(&dir).expect("create the test directory");
    dir
}

/// Every regular file directly under `dir` whose name passes `keep`, by name.
fn files_where(dir: &Path, keep: impl Fn(&str) -> bool) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(dir).expect("list the directory") {
        let entry = entry.expect("directory entry");
        let name = entry.file_name().into_string().expect("UTF-8 file name");
        if keep(&name) {
            files.insert(name, fs::read(entry.path()).expect("read the file"));
        }
    }
    files
}

fn published(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    files_where(dir, |name| name.ends_with(".json"))
}

/// Every name under `dir` with its size, for no-change comparisons.
fn listing(dir: &Path) -> BTreeMap<OsString, u64> {
    fs::read_dir(dir)
        .expect("list the directory")
        .map(|entry| {
            let entry = entry.expect("directory entry");
            let size = entry.metadata().expect("stat the entry").len();
            (entry.file_name(), size)
        })
        .collect()
}

fn golden_queue_files() -> BTreeMap<String, Vec<u8>> {
    let files = published(&worker_wire().join("d/delivery-queue"));
    assert_eq!(files.len(), 6, "the six reviewed queue goldens");
    files
}

fn is_missing(path: &Path) -> bool {
    matches!(fs::symlink_metadata(path), Err(error) if error.kind() == io::ErrorKind::NotFound)
}

fn b64_decode(text: &str) -> Vec<u8> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let (mut bits, mut width, mut bytes) = (0_u32, 0_u32, Vec::new());
    for byte in text.bytes().filter(|byte| *byte != b'=') {
        let sextet = ALPHABET
            .iter()
            .position(|symbol| *symbol == byte)
            .unwrap_or_else(|| panic!("base64 symbol {byte:#04x}"));
        bits = (bits << 6) | u32::try_from(sextet).expect("sextet");
        width += 6;
        if width >= 8 {
            width -= 8;
            bytes.push(u8::try_from(bits >> width).expect("one byte"));
            bits &= (1 << width) - 1;
        }
    }
    bytes
}

fn hex_decode(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("hex pair"))
        .collect()
}

fn text(value: &Value, key: &str) -> String {
    match &value[key] {
        Value::String(text) => text.clone(),
        other => panic!("{key} is a string, not {other}"),
    }
}

fn optional_text(value: &Value, key: &str) -> Option<String> {
    match &value[key] {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        other => panic!("{key} is a string or null, not {other}"),
    }
}

fn optional_integer(value: &Value, key: &str) -> Option<i64> {
    match &value[key] {
        Value::Null => None,
        other => Some(
            other
                .as_i64()
                .unwrap_or_else(|| panic!("{key} is an integer")),
        ),
    }
}

fn integer(value: &Value, key: &str) -> i64 {
    optional_integer(value, key).unwrap_or_else(|| panic!("{key} is present"))
}

fn strings(value: &Value, key: &str) -> Vec<String> {
    let items = value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} is a list"));
    let item = |item: &Value| {
        item.as_str()
            .unwrap_or_else(|| panic!("{key} holds strings"))
            .to_owned()
    };
    items.iter().map(item).collect()
}

/// Rebuilds one golden through the Rust constructors with the default id.
fn entry_from_golden(value: &Value) -> DeliveryEntry {
    let id = String::new();
    match text(value, "kind").as_str() {
        "EVENT" => EventEntry::new(EventFields {
            edge_event_id: text(value, "edge_event_id"),
            event_type: text(value, "event_type"),
            detected_at: text(value, "detected_at"),
            camera_id: text(value, "camera_id"),
            facility_id: text(value, "facility_id"),
            decision_trace: b64_decode(&text(value, "decision_trace_b64")),
            values: b64_decode(&text(value, "values_b64")),
            shed_detail_keys: strings(value, "shed_detail_keys"),
            entry_id: id,
        })
        .expect("the golden event constructs")
        .into(),
        "SNAPSHOT_ATTACHMENT" => SnapshotAttachmentEntry::new(SnapshotAttachmentFields {
            edge_event_id: text(value, "edge_event_id"),
            snapshot_id: text(value, "snapshot_id"),
            sha256: text(value, "sha256"),
            media_reference: text(value, "media_reference"),
            size_bytes: integer(value, "size_bytes"),
            mime_type: text(value, "mime_type"),
            entry_id: id,
        })
        .expect("the golden attachment constructs")
        .into(),
        "CLIP" => ClipEntry::new(ClipFields {
            clip_id: text(value, "clip_id"),
            event_ids: strings(value, "event_ids"),
            camera_id: text(value, "camera_id"),
            facility_id: text(value, "facility_id"),
            local_state: text(value, "local_state"),
            state_version: integer(value, "state_version"),
            media_reference: optional_text(value, "media_reference"),
            sha256: optional_text(value, "sha256"),
            size_bytes: optional_integer(value, "size_bytes"),
            mime_type: optional_text(value, "mime_type"),
            codec: optional_text(value, "codec"),
            duration_ms: optional_integer(value, "duration_ms"),
            clip_start_at: optional_text(value, "clip_start_at"),
            clip_end_at: optional_text(value, "clip_end_at"),
            finalized_at: optional_text(value, "finalized_at"),
            unavailable_reason: optional_text(value, "unavailable_reason"),
            entry_id: id,
        })
        .expect("the golden clip constructs")
        .into(),
        "SNAPSHOT_DISPOSITION" => SnapshotDispositionEntry::new(SnapshotDispositionFields {
            edge_event_id: text(value, "edge_event_id"),
            snapshot_id: text(value, "snapshot_id"),
            disposition: text(value, "disposition"),
            reason: text(value, "reason"),
            entry_id: id,
        })
        .expect("the golden disposition constructs")
        .into(),
        other => panic!("unknown golden kind {other}"),
    }
}

fn golden_entry(bytes: &[u8]) -> DeliveryEntry {
    entry_from_golden(&serde_json::from_slice(bytes).expect("golden JSON"))
}

/// Codec-edge `fields` (hex byte strings) as Rust event fields.
fn codec_edge_fields(fields: &Value) -> EventFields {
    EventFields {
        edge_event_id: text(fields, "edge_event_id"),
        event_type: text(fields, "event_type"),
        detected_at: text(fields, "detected_at"),
        camera_id: text(fields, "camera_id"),
        facility_id: text(fields, "facility_id"),
        decision_trace: hex_decode(&text(fields, "decision_trace_hex")),
        values: hex_decode(&text(fields, "values_hex")),
        shed_detail_keys: strings(fields, "shed_detail_keys"),
        entry_id: String::new(),
    }
}

/// The test's own mapping of a Rust admission onto the Python golden shape.
fn admission_json(result: AdmissionResult) -> Value {
    let fault = match result.fault {
        None => Value::Null,
        Some(AdmissionFault::EntryCapacity) => json!("entry_capacity"),
        Some(AdmissionFault::ByteCapacity) => json!("byte_capacity"),
        Some(AdmissionFault::Conflict) => json!("conflict"),
        Some(AdmissionFault::LockUnavailable) => json!("lock_unavailable"),
    };
    json!({"accepted": result.accepted, "already_admitted": result.already_admitted, "fault": fault})
}

/// Fills `dir` with `count` distinct published files holding the event golden.
fn prefill(dir: &Path, count: usize) {
    let bytes = &golden_queue_files()[EVENT_FILE];
    for ordinal in 0..count {
        fs::write(dir.join(format!("event-fill-{ordinal:04}.json")), bytes).expect("prefill");
    }
}

#[test]
fn rust_serialises_python_bytes() {
    let base = fresh_dir("t10");
    let dir = base.join("queue");
    let queue = DeliveryQueue::open(&dir, true).expect("open a new queue");
    let goldens = golden_queue_files();
    for (name, bytes) in &goldens {
        assert_eq!(
            queue.try_admit(&golden_entry(bytes)).unwrap(),
            ADMITTED,
            "{name}"
        );
    }
    assert_eq!(
        published(&dir),
        goldens,
        "file names and bytes equal the goldens"
    );
    let others = files_where(&dir, |name| !name.ends_with(".json"));
    assert_eq!(
        others.keys().collect::<Vec<_>>(),
        [LOCK_FILE_NAME],
        "only the lock beside them"
    );

    assert!(queue.dead_letter(EVENT_ENTRY_ID, 422).unwrap());
    let golden_dead = worker_wire().join("d/delivery-queue-dead-letter/queue-dead-letter");
    let rust_dead = base.join("queue-dead-letter");
    assert_eq!(queue.dead_letter_directory(), rust_dead);
    assert_eq!(
        files_where(&rust_dead, |_| true),
        files_where(&golden_dead, |_| true)
    );
    assert!(files_where(&golden_dead, |_| true).contains_key(DEAD_LETTER_FILE));
    assert!(
        is_missing(&dir.join(EVENT_FILE)),
        "dead-lettering moves the entry"
    );

    let codec: Value = serde_json::from_slice(
        &fs::read(worker_wire().join("d/codec-edges.json")).expect("read codec-edges"),
    )
    .expect("codec-edges JSON");
    let codec_dir = base.join("codec");
    let codec_queue = DeliveryQueue::open(&codec_dir, true).expect("open the codec queue");
    let cases = codec["queue"]["entries"].as_array().expect("queue entries");
    assert_eq!(cases.len(), 6, "the six codec-edge queue cases");
    for case in cases {
        let label = text(case, "case");
        let entry: DeliveryEntry = EventEntry::new(codec_edge_fields(&case["fields"]))
            .unwrap_or_else(|error| panic!("{label}: {error:?}"))
            .into();
        let result = codec_queue.try_admit(&entry).unwrap();
        assert_eq!(admission_json(result), case["admission"], "{label}");
        let golden_path = text(&case["file"], "path");
        let name = Path::new(&golden_path)
            .file_name()
            .expect("golden file name");
        assert_eq!(
            name,
            format!("{}.json", text(case, "entry_id")).as_str(),
            "{label}"
        );
        let written = fs::read(codec_dir.join(name)).expect("the Rust-written file");
        assert_eq!(
            written,
            fs::read(worker_wire().join(&golden_path)).unwrap(),
            "{label}"
        );
    }
    let leftovers = files_where(&codec_dir, |name| !name.ends_with(".json"));
    let expected = &codec["queue"]["non_entry_files_left_in_queue_dir"];
    assert_eq!(json!(leftovers.keys().collect::<Vec<_>>()), *expected);

    let base_fields = &cases[0]["fields"];
    for refusal in codec["queue"]["constructor_refusals"]
        .as_array()
        .expect("refusals")
    {
        let label = text(refusal, "case");
        let mut fields = codec_edge_fields(base_fields);
        match text(refusal, "field").as_str() {
            "event_type" => fields.event_type = text(refusal, "value"),
            "shed_detail_keys" => fields.shed_detail_keys = strings(refusal, "value"),
            other => panic!("{label}: unexpected field {other}"),
        }
        match (text(refusal, "verdict").as_str(), EventEntry::new(fields)) {
            ("constructed", Ok(_)) => {}
            ("refused", Err(error)) => assert_eq!(error.field, text(refusal, "field"), "{label}"),
            (verdict, outcome) => panic!("{label}: expected {verdict}, got {outcome:?}"),
        }
    }
}

#[test]
fn rust_opens_python_queue() {
    let base = fresh_dir("t11");
    let dir = base.join("queue");
    fs::create_dir(&dir).expect("create the Python queue directory");
    let goldens = golden_queue_files();
    for (name, bytes) in &goldens {
        fs::write(dir.join(name), bytes).expect("copy a golden");
    }
    fs::write(dir.join(ORPHAN_TEMP), b"{\"partial").expect("write an orphan temp");

    DeliveryQueue::open(&dir, false).expect("open without recovery");
    assert!(
        !is_missing(&dir.join(ORPHAN_TEMP)),
        "recover=false leaves temps"
    );

    let queue = DeliveryQueue::open(&dir, true).expect("open with recovery");
    assert!(
        is_missing(&dir.join(ORPHAN_TEMP)),
        "recovery removes the orphan temp"
    );
    assert_eq!(
        published(&dir),
        goldens,
        "recovery keeps every published entry"
    );
    let snapshot = queue.capacity_snapshot().expect("snapshot");
    let total: usize = goldens.values().map(Vec::len).sum();
    assert_eq!(snapshot.accepted_count, 6);
    assert_eq!(snapshot.accepted_bytes, u64::try_from(total).unwrap());
    let by_kind = BTreeMap::from([
        (EntryKind::Event, 1),
        (EntryKind::Clip, 3),
        (EntryKind::SnapshotAttachment, 1),
        (EntryKind::SnapshotDisposition, 1),
    ]);
    assert_eq!(snapshot.by_kind, by_kind);
    assert_eq!(
        (snapshot.dead_lettered_count, snapshot.dead_lettered_bytes),
        (0, 0)
    );
    let ids: Vec<Value> = queue
        .entries()
        .unwrap()
        .iter()
        .map(|e| e["entry_id"].clone())
        .collect();
    let golden_ids: Vec<Value> = goldens
        .keys()
        .map(|name| json!(name.trim_end_matches(".json")))
        .collect();
    assert_eq!(ids, golden_ids);
}

#[test]
fn readmit_same_bytes_is_already_admitted() {
    let base = fresh_dir("t12");
    let dir = base.join("queue");
    let queue = DeliveryQueue::open(&dir, true).expect("open a new queue");
    let golden = golden_queue_files()[EVENT_FILE].clone();
    let entry = golden_entry(&golden);
    assert_eq!(queue.try_admit(&entry).unwrap(), ADMITTED);
    assert_eq!(queue.try_admit(&entry).unwrap(), ALREADY_ADMITTED);

    let mut value: Value = serde_json::from_slice(&golden).unwrap();
    value["event_type"] = json!("fall");
    let changed = entry_from_golden(&value);
    assert_eq!(
        changed.entry_id(),
        EVENT_ENTRY_ID,
        "same id, different bytes"
    );
    assert_eq!(
        queue.try_admit(&changed).unwrap(),
        refused(AdmissionFault::Conflict)
    );
    assert_eq!(
        fs::read(dir.join(EVENT_FILE)).unwrap(),
        golden,
        "conflict leaves the file"
    );

    assert!(!queue.acknowledge_backend(EVENT_ENTRY_ID, 422).unwrap());
    assert_eq!(
        fs::read(dir.join(EVENT_FILE)).unwrap(),
        golden,
        "422 is not an acknowledgement"
    );
    assert!(queue.acknowledge_backend(EVENT_ENTRY_ID, 409).unwrap());
    assert!(
        is_missing(&dir.join(EVENT_FILE)),
        "409 acknowledges and deletes"
    );
}

#[test]
fn full_queue_refuses_without_eviction() {
    let goldens = golden_queue_files();
    let event = golden_entry(&goldens[EVENT_FILE]);
    let other = golden_entry(goldens.values().next().expect("a non-event golden"));
    assert_ne!(other.entry_id(), event.entry_id());

    let entries_dir = fresh_dir("t13").join("entries");
    let queue = DeliveryQueue::open(&entries_dir, true).expect("open the entry-cap queue");
    prefill(&entries_dir, MAX_ACCEPTED_ENTRIES - 1);
    assert_eq!(
        queue.try_admit(&event).unwrap(),
        ADMITTED,
        "the last free slot"
    );
    let before = listing(&entries_dir);
    assert_eq!(
        queue.try_admit(&other).unwrap(),
        refused(AdmissionFault::EntryCapacity)
    );
    assert_eq!(
        listing(&entries_dir),
        before,
        "nothing written, nothing evicted"
    );

    let bytes_dir = fresh_dir("t13").join("bytes");
    let queue = DeliveryQueue::open(&bytes_dir, true).expect("open the byte-cap queue");
    let event_len = u64::try_from(goldens[EVENT_FILE].len()).unwrap();
    File::create(bytes_dir.join("filler.json"))
        .and_then(|filler| filler.set_len(MAX_ACCEPTED_BYTES - event_len))
        .expect("create the sparse filler");
    assert_eq!(
        queue.try_admit(&event).unwrap(),
        ADMITTED,
        "exactly at the byte cap"
    );
    let before = listing(&bytes_dir);
    assert_eq!(
        queue.try_admit(&other).unwrap(),
        refused(AdmissionFault::ByteCapacity)
    );
    assert_eq!(
        listing(&bytes_dir),
        before,
        "nothing written, nothing evicted"
    );
}

fn python() -> PathBuf {
    let value = std::env::var_os("SEEON_TEST_PYTHON")
        .unwrap_or_else(|| panic!("SEEON_TEST_PYTHON is required"));
    assert!(
        !value.is_empty(),
        "SEEON_TEST_PYTHON must be a nonblank path"
    );
    PathBuf::from(value)
}

/// Starts the Python lock holder and waits until it reports that it holds.
fn python_holds(python: &Path, dir: &Path) -> Child {
    let mut holder = Command::new(python)
        .arg("-c")
        .arg(HOLD)
        .arg(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the Python holder");
    let stdout = holder.stdout.take().expect("holder stdout");
    let (line_tx, line_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut line = String::new();
        let _ = line_tx.send(BufReader::new(stdout).read_line(&mut line).map(|_| line));
    });
    let mut first_line = None;
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + HOLDER_WAIT;
    poll_until(&clock, deadline, "python queue lock holder", || {
        match line_rx.try_recv() {
            Ok(line) => first_line = Some(line.ok()),
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => first_line = Some(None),
        }
        true
    })
    .expect("the Python holder answers before its deadline");
    assert_eq!(
        first_line.flatten().as_deref(),
        Some("held\n"),
        "the Python holder locked"
    );
    holder
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn python_lock_blocks_rust_nonblocking() {
    let python = python();
    let dir = fresh_dir("t14").join("queue");
    let queue = DeliveryQueue::open(&dir, true).expect("open a new queue");
    let entry = golden_entry(&golden_queue_files()[EVENT_FILE]);

    let mut holder = python_holds(&python, &dir);
    let result = queue.try_admit_nonblocking(&entry).unwrap();
    assert_eq!(result, refused(AdmissionFault::LockUnavailable));
    assert!(
        is_missing(&dir.join(EVENT_FILE)),
        "nothing written while Python holds"
    );
    drop(holder.stdin.take());
    assert!(holder.wait().expect("wait for the holder").success());

    assert_eq!(queue.try_admit_nonblocking(&entry).unwrap(), ADMITTED);
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn python_reads_rust_written_queue() {
    let python = python();
    let base = fresh_dir("t15");
    let dir = base.join("queue");
    let queue = DeliveryQueue::open(&dir, true).expect("open a new queue");
    let goldens = golden_queue_files();
    for bytes in goldens.values() {
        assert_eq!(queue.try_admit(&golden_entry(bytes)).unwrap(), ADMITTED);
    }
    assert!(queue.dead_letter(EVENT_ENTRY_ID, 422).unwrap());
    let full = base.join("full");
    let full_queue = DeliveryQueue::open(&full, true).expect("open the full queue");
    prefill(&full, MAX_ACCEPTED_ENTRIES - 1);
    assert_eq!(
        full_queue
            .try_admit(&golden_entry(&goldens[EVENT_FILE]))
            .unwrap(),
        ADMITTED
    );

    let output = Command::new(&python)
        .arg("-c")
        .arg(READ)
        .arg(&dir)
        .arg(base.join("queue-dead-letter").join(DEAD_LETTER_FILE))
        .arg(worker_wire().join("d/delivery-queue").join(EVENT_FILE))
        .arg(&full)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .expect("run the Python reader");
    assert!(output.status.success(), "the Python reader exits cleanly");
    let report: Value = serde_json::from_slice(&output.stdout).expect("one JSON report");

    let live: BTreeMap<&String, &Vec<u8>> = goldens
        .iter()
        .filter(|(name, _)| name.as_str() != EVENT_FILE)
        .collect();
    let entries: Vec<Value> = live
        .iter()
        .map(|(name, bytes)| {
            let value: Value = serde_json::from_slice(bytes).unwrap();
            json!([value["kind"], name.trim_end_matches(".json")])
        })
        .collect();
    let live_bytes: usize = live.values().map(|bytes| bytes.len()).sum();
    let dead_bytes = goldens[EVENT_FILE].len();
    let expected = json!({
        "entries": entries,
        "accepted_count": 5,
        "accepted_bytes": live_bytes,
        "by_kind": {"EVENT": 0, "CLIP": 3, "SNAPSHOT_ATTACHMENT": 1, "SNAPSHOT_DISPOSITION": 1},
        "dead_lettered_count": 1,
        "dead_lettered_bytes": dead_bytes,
        "requeued": true,
        "count_after_requeue": 6,
        "readmit": {"accepted": true, "already_admitted": true, "fault": null},
        "full": {"accepted": false, "already_admitted": false, "fault": "entry_capacity"},
    });
    assert_eq!(report, expected);
}
