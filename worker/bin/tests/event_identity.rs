//! T36: `policy/identity.rs` (G20) against `d/event-identities/cases.json`,
//! recorded from the Python `EventIdentityStore` at 030aaf1 with a fixed wall
//! clock and a patched `uuid4`; `FakeClock` and `FakeIds` are that recipe.
//! Every journal is compared byte for byte with its `.lines.json` rebuild,
//! and a refusal by its typed variant, mapped from the golden's detail.

use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use seeon_ml_worker::policy::identity::{
    EventIdentityStore, IdentityError, Limits, MAX_JOURNAL_BYTES, RETENTION_SEC,
    event_identity_path,
};
use seeon_ml_worker::seam::{Clock, IdSource};
use serde_json::Value;
use sha2::{Digest, Sha256};

const CASES_SHA256_PREFIX: &str = "25325a46d02a";
/// Reviewed sha256 prefixes of every `.lines.json` golden the cases name.
const LINES_SHA256_PREFIXES: [(&str, &str); 9] = [
    ("fresh/0-after-resolve-new", "69a0944f7ceb"),
    ("refusal-conflicting-source-key/0-input", "6b755a7fc482"),
    ("refusal-future-dated/0-input", "4a9cd2459fe9"),
    ("refusal-malformed-extra-field/0-input", "dba175eae346"),
    ("refusal-malformed-syntax/0-input", "9f1cb04213eb"),
    ("retention/0-input", "98ab083a4328"),
    ("retention/1-after-open", "e0662606a41e"),
    ("retention/2-after-resolve-older", "985f928473e3"),
    ("retention/3-after-resolve-new", "f6951db27886"),
];
const CASE_ORDER: [&str; 6] = [
    "retention",
    "fresh",
    "refusal-malformed-syntax",
    "refusal-malformed-extra-field",
    "refusal-future-dated",
    "refusal-conflicting-source-key",
];
/// The recipe's NOW: 1787000000.125 s after the epoch, for every call.
const NOW: Duration = Duration::new(1_787_000_000, 125_000_000);
const CAMERA_ID: &str = "cmsnw6rjc01vhlh01oswn99yq";
/// The recipe mints `UUID(int=0xB0 + n, version=4)` for n = 1, 2, ...
const MINTED_BASE: u64 = 0xB0;
/// Another id source, so a store that forgot a key would mint a new id.
const OTHER_BASE: u64 = 0xC0;
const OPEN: &str = "EventIdentityStore(path, clock=NOW)";

struct FakeClock;

impl Clock for FakeClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }

    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + NOW
    }

    fn pause(&self, _limit: Duration) {}
}

struct FakeIds {
    base: u64,
    minted: Mutex<u64>,
}

impl FakeIds {
    fn new(base: u64) -> Arc<Self> {
        Arc::new(Self {
            base,
            minted: Mutex::new(0),
        })
    }
}

impl IdSource for FakeIds {
    fn uuid4(&self) -> io::Result<String> {
        let mut minted = self
            .minted
            .lock()
            .map_err(|_| io::Error::other("id counter poisoned"))?;
        *minted += 1;
        Ok(uuid(self.base + *minted))
    }
}

/// `str(UUID(int=value, version=4))` for a value below 2^48.
fn uuid(value: u64) -> String {
    format!("00000000-0000-4000-8000-{value:012x}")
}

fn key(track: u32) -> String {
    format!("boot-a:1:fall:none:track-{track}:1:1:1")
}

/// One journal line in the goldens' shape, newline included.
fn line(source_key: &str, id: u64, recorded_at: &str) -> String {
    let edge_event_id = uuid(id);
    format!(
        "{{\"source_key\":\"{source_key}\",\"edge_event_id\":\"{edge_event_id}\",\"recorded_at\":{recorded_at}}}\n"
    )
}

fn open(path: Option<PathBuf>, limits: Limits, base: u64) -> EventIdentityStore {
    EventIdentityStore::open(path, limits, Arc::new(FakeClock), FakeIds::new(base))
        .unwrap_or_else(|error| panic!("open refused: {error:?}"))
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/worker-wire")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn text(value: &Value, key: &str) -> String {
    value[key].as_str().expect(key).to_owned()
}

fn cases() -> Value {
    let bytes = fs::read(fixtures().join("d/event-identities/cases.json"))
        .expect("event-identities cases are readable");
    let digest = sha256_hex(&bytes);
    assert!(
        digest.starts_with(CASES_SHA256_PREFIX),
        "cases sha256 {digest}"
    );
    serde_json::from_slice(&bytes).expect("cases are JSON")
}

/// The journal a `.lines.json` golden describes, after checking the golden
/// file and the recorded sha256 and size of the rebuilt bytes.
fn golden_journal(journal: &Value) -> String {
    let golden = text(journal, "golden");
    let name = golden
        .strip_prefix("d/event-identities/")
        .and_then(|name| name.strip_suffix(".lines.json"))
        .unwrap_or_else(|| panic!("golden path {golden}"));
    let (_, prefix) = LINES_SHA256_PREFIXES
        .iter()
        .find(|(reviewed, _)| *reviewed == name)
        .unwrap_or_else(|| panic!("unreviewed golden {golden}"));
    let bytes = fs::read(fixtures().join(&golden)).expect("lines golden is readable");
    let digest = sha256_hex(&bytes);
    assert!(digest.starts_with(prefix), "{golden} sha256 {digest}");
    let lines: Value = serde_json::from_slice(&bytes).expect("lines golden is JSON");
    assert_eq!(lines["file_name"], journal["file_name"], "{golden}");
    let mut rebuilt = lines["lines"]
        .as_array()
        .expect("lines")
        .iter()
        .map(|line| line.as_str().expect("line"))
        .collect::<Vec<_>>()
        .join("\n");
    if lines["trailing_newline"]
        .as_bool()
        .expect("trailing_newline")
    {
        rebuilt.push('\n');
    }
    assert_eq!(
        sha256_hex(rebuilt.as_bytes()),
        text(journal, "sha256"),
        "{golden}"
    );
    assert_eq!(
        Some(rebuilt.len() as u64),
        journal["size_bytes"].as_u64(),
        "{golden}"
    );
    rebuilt
}

/// A cleared `<work>/<case>` directory; the store state dir under it.
fn fresh_state(test: &str, case: &str) -> PathBuf {
    let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("event_identity")
        .join(test)
        .join(case);
    match fs::remove_dir_all(&work) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => panic!("clear {}: {error}", work.display()),
    }
    work.join("state")
}

fn write_journal(state: &Path, lines: &[&str]) -> PathBuf {
    let path = event_identity_path(CAMERA_ID, state);
    fs::create_dir_all(path.parent().expect("journal dir")).expect("create journal dir");
    fs::write(&path, lines.concat()).expect("write journal");
    path
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).expect("journal is readable UTF-8")
}

/// The journal's inode and contents, `None` when it does not exist.
fn snapshot(path: &Path) -> Option<(u64, String)> {
    match fs::metadata(path) {
        Ok(metadata) => Some((metadata.ino(), read(path))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => panic!("stat {}: {error}", path.display()),
    }
}

/// `(name, is_file, size)` per entry, `None` when the directory is absent.
fn listing(dir: &Path) -> Option<Vec<(String, bool, u64)>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return None,
        Err(error) => panic!("list {}: {error}", dir.display()),
    };
    let mut listed: Vec<(String, bool, u64)> = entries
        .map(|entry| {
            let entry = entry.expect("directory entry");
            let metadata = entry.metadata().expect("entry metadata");
            let name = entry.file_name().into_string().expect("UTF-8 name");
            (name, metadata.is_file(), metadata.len())
        })
        .collect();
    listed.sort();
    Some(listed)
}

fn golden_listing(value: &Value) -> Option<Vec<(String, bool, u64)>> {
    value.as_array().map(|entries| {
        let mut listed: Vec<(String, bool, u64)> = entries
            .iter()
            .map(|entry| {
                let is_file = entry["is_file"].as_bool().expect("is_file");
                let size = entry["size_bytes"].as_u64().expect("size_bytes");
                (text(entry, "name"), is_file, size)
            })
            .collect();
        listed.sort();
        listed
    })
}

/// The typed refusal a golden step records. Python's message ends with the
/// detail after the journal path; only that recorded suffix is mapped.
fn refusal(step: &Value) -> IdentityError {
    assert_eq!(text(step, "class"), "EventIdentityStoreError");
    let message = text(step, "message");
    let (_, detail) = message.rsplit_once(": ").expect("golden detail");
    let number = |line: &str| line.parse().expect("golden line number");
    if let Some(line) = detail.strip_prefix("malformed line ") {
        return IdentityError::Malformed { line: number(line) };
    }
    match detail.strip_prefix("conflicting source key at line ") {
        Some(line) => IdentityError::ConflictingSourceKey { line: number(line) },
        None => panic!("unmapped golden detail {detail}"),
    }
}

fn replay(cases: &Value, case: &Value) {
    let name = text(case, "case");
    let state = fresh_state("replay", &name);
    let path = event_identity_path(CAMERA_ID, &state);
    let dir = path.parent().expect("journal dir").to_path_buf();
    let journal_path = &cases["journal_path"];
    assert_eq!(
        path.strip_prefix(&state).ok(),
        Some(Path::new(&text(journal_path, "relative_to_state_dir")))
    );
    if !case["input_journal"].is_null() {
        fs::create_dir_all(&dir).expect("create journal dir");
        fs::write(&path, golden_journal(&case["input_journal"])).expect("write input");
    }
    let clock: Arc<dyn Clock> = Arc::new(FakeClock);
    let ids: Arc<dyn IdSource> = FakeIds::new(MINTED_BASE);
    let mut store: Option<EventIdentityStore> = None;
    for (index, step) in case["steps"].as_array().expect("steps").iter().enumerate() {
        let call = text(step, "call");
        let label = format!("{name} step {index}: {call}");
        let before = snapshot(&path);
        let call_name = call.strip_suffix(" [restart]").unwrap_or(&call);
        if call_name == OPEN {
            drop(store.take());
            let opened = EventIdentityStore::open(
                Some(path.clone()),
                Limits::default(),
                clock.clone(),
                ids.clone(),
            );
            match text(step, "verdict").as_str() {
                "constructed" => {
                    store = Some(opened.unwrap_or_else(|error| panic!("{label}: {error:?}")));
                }
                "refused" => assert_eq!(opened.err(), Some(refusal(step)), "{label}"),
                verdict => panic!("{label}: unknown verdict {verdict}"),
            }
        } else {
            let role = call_name
                .strip_prefix("resolve(")
                .and_then(|role| role.strip_suffix(')'))
                .unwrap_or_else(|| panic!("unknown golden call {call}"));
            let source_key = text(&cases["fixed_inputs"]["source_keys"], role);
            let resolved = store.as_mut().expect("an open store").resolve(&source_key);
            let expected = match step.get("class") {
                Some(_) => Err(refusal(step)),
                None => Ok(text(step, "edge_event_id")),
            };
            assert_eq!(resolved, expected, "{label}");
        }
        let after = snapshot(&path);
        let expected = match &step["journal_after"] {
            Value::Null => None,
            journal => {
                assert_eq!(journal["file_name"], journal_path["file_name"], "{label}");
                Some(golden_journal(journal))
            }
        };
        assert_eq!(
            after.as_ref().map(|(_, bytes)| bytes),
            expected.as_ref(),
            "{label}"
        );
        let inodes = (before.as_ref().map(|b| b.0), after.as_ref().map(|a| a.0));
        match text(step, "journal_file").as_str() {
            "absent" => assert_eq!(inodes, (None, None), "{label}"),
            "created" => assert!(matches!(inodes, (None, Some(_))), "{label}"),
            "replaced" => assert!(
                matches!(inodes, (Some(old), Some(new)) if old != new),
                "{label}: a new inode"
            ),
            "unchanged" => {
                assert!(after.is_some(), "{label}");
                assert_eq!(before, after, "{label}");
            }
            outcome => panic!("{label}: unknown journal_file {outcome}"),
        }
        assert_eq!(listing(&dir), golden_listing(&step["listing"]), "{label}");
    }
}

#[test]
fn every_golden_case_replays_to_the_recorded_ids_refusals_and_journal_bytes() {
    let cases = cases();
    let fixed = &cases["fixed_inputs"];
    assert_eq!(fixed["clock_now"].as_f64(), Some(NOW.as_secs_f64()));
    assert_eq!(text(fixed, "camera_id"), CAMERA_ID);
    assert_eq!(
        fixed["retention_sec_observed"].as_f64(),
        Some(RETENTION_SEC)
    );
    assert_eq!(
        fixed["max_journal_bytes_observed"].as_u64(),
        Some(MAX_JOURNAL_BYTES as u64)
    );
    let all = cases["cases"].as_array().expect("cases");
    let names: Vec<String> = all.iter().map(|case| text(case, "case")).collect();
    assert_eq!(names, CASE_ORDER);
    for case in all {
        replay(&cases, case);
    }
}

#[test]
fn the_same_source_key_resolves_to_the_same_id_across_calls_stores_and_restarts() {
    let path = event_identity_path(CAMERA_ID, &fresh_state("same_key", "journal"));
    let mut first = open(Some(path.clone()), Limits::default(), MINTED_BASE);
    let mut second = open(Some(path.clone()), Limits::default(), OTHER_BASE);
    let minted = first.resolve(&key(2)).expect("first store mints");
    assert_eq!(minted, uuid(0xB1));
    assert_eq!(first.resolve(&key(2)), Ok(minted.clone()));
    assert_eq!(second.resolve(&key(2)), Ok(minted.clone()));
    drop((first, second));
    let mut restarted = open(Some(path), Limits::default(), OTHER_BASE);
    assert_eq!(restarted.resolve(&key(2)), Ok(minted));

    let mut memory = open(None, Limits::default(), MINTED_BASE);
    assert_eq!(memory.resolve(&key(2)), Ok(uuid(0xB1)));
    assert_eq!(memory.resolve(&key(0)), Ok(uuid(0xB2)));
    assert_eq!(memory.resolve(&key(2)), Ok(uuid(0xB1)));
}

#[test]
fn a_record_exactly_at_the_cutoff_is_kept_and_one_a_second_older_is_dropped() {
    assert_eq!(NOW.as_secs_f64() - RETENTION_SEC, 1_779_224_000.125);
    let at_cutoff = line(&key(4), 0xA4, "1779224000.125");
    let older = line(&key(1), 0xA3, "1779223999.125");
    let state = fresh_state("retention_boundary", "journal");
    let path = write_journal(&state, &[&older, &at_cutoff]);
    let mut store = open(Some(path.clone()), Limits::default(), MINTED_BASE);
    assert_eq!(read(&path), at_cutoff);
    assert_eq!(store.resolve(&key(4)), Ok(uuid(0xA4)));
    assert_eq!(store.resolve(&key(1)), Ok(uuid(0xB1)));
}

#[test]
fn the_byte_budget_keeps_the_newest_records_and_evicts_the_oldest_first() {
    let oldest = line(&key(1), 0xA1, "1786999970.125");
    let middle = line(&key(2), 0xA2, "1786999980.125");
    let newest = line(&key(3), 0xA3, "1786999990.125");
    let limits = Limits {
        retention_sec: RETENTION_SEC,
        max_bytes: middle.len() + newest.len(),
    };
    let state = fresh_state("eviction_order", "journal");
    let path = write_journal(&state, &[&newest, &oldest, &middle]);
    let mut store = open(Some(path.clone()), limits, MINTED_BASE);
    assert_eq!(read(&path), format!("{middle}{newest}"));
    assert_eq!(store.resolve(&key(3)), Ok(uuid(0xA3)));
    assert_eq!(store.resolve(&key(2)), Ok(uuid(0xA2)));
    assert_eq!(store.resolve(&key(1)), Ok(uuid(0xB1)));
    let reminted = line(&key(1), 0xB1, "1787000000.125");
    assert_eq!(read(&path), format!("{newest}{reminted}"));
}

#[test]
fn eviction_stops_at_the_first_record_that_does_not_fit() {
    let oldest = line(&key(1), 0xA1, "1786999970.125");
    let long_key = format!("{}:{}", key(2), "x".repeat(32));
    let middle = line(&long_key, 0xA2, "1786999980.125");
    let newest = line(&key(3), 0xA3, "1786999990.125");
    let limits = Limits {
        retention_sec: RETENTION_SEC,
        max_bytes: newest.len() + oldest.len(),
    };
    let state = fresh_state("eviction_stop", "journal");
    let path = write_journal(&state, &[&oldest, &middle, &newest]);
    let mut store = open(Some(path.clone()), limits, MINTED_BASE);
    assert_eq!(read(&path), newest);
    assert_eq!(store.resolve(&key(1)), Ok(uuid(0xB1)));
}

/// Resolves one key through the Python store with the recipe's clock and
/// `uuid4` patched to one fixed id, and prints the id it returns.
const RESOLVE: &str = "\
import sys
from pathlib import Path
from uuid import UUID
import worker.pipeline.decision.event_identity as event_identity
path, now, key, minted = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4], 16)
event_identity.uuid4 = lambda: UUID(int=minted, version=4)
store = event_identity.EventIdentityStore(path, clock=lambda: now)
print(store.resolve(key))
";

/// The id Python mints when the journal does not know the key.
const PYTHON_MINTED: u64 = 0xD1;

fn python() -> PathBuf {
    let value = std::env::var_os("SEEON_TEST_PYTHON")
        .unwrap_or_else(|| panic!("SEEON_TEST_PYTHON is required"));
    assert!(
        !value.is_empty(),
        "SEEON_TEST_PYTHON must be a nonblank path"
    );
    PathBuf::from(value)
}

fn python_resolve(python: &Path, journal: &Path, source_key: &str) -> String {
    let output = Command::new(python)
        .arg("-c")
        .arg(RESOLVE)
        .arg(journal)
        .arg(NOW.as_secs_f64().to_string())
        .arg(source_key)
        .arg(format!("{PYTHON_MINTED:x}"))
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .expect("run the Python store");
    assert!(output.status.success(), "the Python store exits cleanly");
    String::from_utf8(output.stdout)
        .expect("UTF-8 Python output")
        .trim()
        .to_owned()
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn rust_and_python_resolve_the_same_id_from_each_others_journal() {
    let python = python();

    let rust_first = event_identity_path(CAMERA_ID, &fresh_state("xlang", "rust-first"));
    let mut rust = open(Some(rust_first.clone()), Limits::default(), MINTED_BASE);
    let minted = rust.resolve(&key(2)).expect("Rust mints");
    drop(rust);
    let written = read(&rust_first);
    assert_eq!(python_resolve(&python, &rust_first, &key(2)), minted);
    assert_eq!(read(&rust_first), written, "Python rewrites the same bytes");

    let python_first = event_identity_path(CAMERA_ID, &fresh_state("xlang", "python-first"));
    let minted = python_resolve(&python, &python_first, &key(2));
    assert_eq!(minted, uuid(PYTHON_MINTED));
    let written = read(&python_first);
    let mut rust = open(Some(python_first.clone()), Limits::default(), MINTED_BASE);
    assert_eq!(rust.resolve(&key(2)), Ok(minted));
    assert_eq!(read(&python_first), written, "Rust rewrites the same bytes");
}
