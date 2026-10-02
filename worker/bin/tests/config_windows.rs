//! Canonical window admission, strict startup, and real timezone IO boundaries.
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use seeon_ml_worker::config::windows::{
    AdmissionMode, AdmittedWindow, DropReason, WindowDrop, WindowError, admit_at,
};
use seeon_ml_worker::json::Json;
use seeon_ml_worker::relay::cameras::WorkerConfigPayload;
use seeon_worker::detection_window::{AwareDateTime, DetectionWindowError, MAX_TZIF_BYTES};
use serde_json::{Value, json};

static SERIAL: AtomicUsize = AtomicUsize::new(0);
struct ZoneDir(PathBuf);
impl std::ops::Deref for ZoneDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}
impl Drop for ZoneDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove owned zone fixture");
    }
}
fn scratch(label: &str) -> ZoneDir {
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&parent).expect("test scratch parent");
    let directory = parent.join(format!(
        "config-windows-{}-{stamp}-{label}-{serial}",
        std::process::id()
    ));
    fs::create_dir(&directory).expect("create new owned directory, never remove a collision");
    ZoneDir(directory)
}
fn parse(document: Value) -> WorkerConfigPayload {
    WorkerConfigPayload::parse(&Json::from(&document)).expect("legal window payload")
}
fn payload(windows: Value) -> WorkerConfigPayload {
    parse(json!({"config_version":0,"cameras":[],"detection_windows":windows}))
}
fn definition(start: &str, end: &str, tz: &str) -> Value {
    json!({"start":start,"end":end,"tz":tz})
}
fn slots(entries: Vec<(&str, Value)>) -> WorkerConfigPayload {
    // Json::Object preserves this explicit order; serde_json::Map does not.
    let windows = Json::Object(
        entries
            .into_iter()
            .map(|(name, value)| (name.to_owned(), Json::from(&value)))
            .collect(),
    );
    WorkerConfigPayload::parse(&Json::Object(vec![
        ("config_version".into(), Json::Int(0)),
        ("cameras".into(), Json::Array(vec![])),
        ("detection_windows".into(), windows),
    ]))
    .expect("ordered candidates")
}
fn copy_zone(root: &Path, key: &str) {
    let destination = root.join(key);
    fs::create_dir_all(destination.parent().unwrap()).unwrap();
    fs::copy(Path::new("/usr/share/zoneinfo").join(key), destination).expect("real TZif copy");
}
struct Admission {
    windows: Result<BTreeMap<String, AdmittedWindow>, WindowError>,
    drops: Vec<(String, DropReason)>,
}
fn admit(config: &WorkerConfigPayload, root: &Path, mode: AdmissionMode) -> Admission {
    let mut drops = Vec::new();
    let windows = admit_at(config, root, mode, &mut |drop| {
        drops.push((drop.domain.to_owned(), drop.reason))
    });
    Admission { windows, drops }
}
fn utc(seconds: u64) -> AwareDateTime {
    AwareDateTime::from_utc_system_time(UNIX_EPOCH + Duration::from_secs(seconds)).unwrap()
}

#[test]
fn explicit_candidates_report_in_input_order_and_override_legacy() {
    let root = scratch("precedence");
    copy_zone(&root, "UTC");
    let config = slots(vec![
        ("zeta", definition("01:00", "02:00", "UTC")),
        ("silent", Value::Null),
        ("shape", json!({"start":"01:00","end":"02:00"})),
        ("equal", definition("01:00", "01:00", "UTC")),
        ("unknown", definition("01:00", "02:00", "No/Such")),
    ]);
    let outcome = admit(&config, &root, AdmissionMode::Poll);
    assert_eq!(
        outcome
            .windows
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["zeta"]
    );
    assert_eq!(
        outcome.drops,
        [
            ("shape".into(), DropReason::Shape),
            ("equal".into(), DropReason::Time),
            ("unknown".into(), DropReason::Missing),
        ]
    );
    let old = definition("21:00", "05:00", "UTC");
    for explicit in [json!({}), json!({"fall":null,"bed_exit":null})] {
        let config = parse(
            json!({"config_version":0,"cameras":[],"night_window":old,"detection_windows":explicit}),
        );
        let outcome = admit(&config, &root, AdmissionMode::Startup);
        assert!(
            outcome.windows.unwrap().is_empty(),
            "explicit map supersedes a real legacy window"
        );
        assert!(outcome.drops.is_empty(), "null is silent");
    }
    let config =
        parse(json!({"config_version":0,"cameras":[],"night_window":old,"detection_windows":null}));
    let outcome = admit(&config, &root, AdmissionMode::Startup);
    assert_eq!(
        outcome
            .windows
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["bed_exit"]
    );
    assert!(outcome.drops.is_empty());
}

#[test]
fn mixed_unicode_normalizes_rules_but_strictness_counts_original_characters() {
    let root = scratch("unicode");
    copy_zone(&root, "Asia/Seoul");
    copy_zone(&root, "UTC");
    let config = payload(json!({"fall":definition("0１:00", "02:00", "Asia/Seoul")}));
    for mode in [AdmissionMode::Startup, AdmissionMode::Poll] {
        let outcome = admit(&config, &root, mode);
        assert!(outcome.drops.is_empty());
        let windows = outcome.windows.unwrap();
        assert_eq!(windows["fall"].definition.start, "0１:00");
        // 16:30 UTC is 01:30 next day in Seoul, not in the UTC 01:00..02:00 window.
        assert!(
            windows["fall"]
                .window
                .contains(utc(16 * 3600 + 1800))
                .unwrap()
        );
    }
    for start in ["1:00", "01:0", "١:٠"] {
        let config = payload(json!({"fall":definition(start, "02:00", "UTC")}));
        let poll = admit(&config, &root, AdmissionMode::Poll);
        assert_eq!(poll.windows.unwrap()["fall"].definition.start, start);
        assert!(poll.drops.is_empty());
        let startup = admit(&config, &root, AdmissionMode::Startup);
        assert_eq!(startup.windows, Err(WindowError::StrictClock), "{start:?}");
        assert!(
            startup.drops.is_empty(),
            "strict refusal is not a canonical drop"
        );
    }
}

#[test]
fn complete_canonical_walk_precedes_strict_startup_refusal() {
    let root = scratch("order");
    copy_zone(&root, "UTC");
    let config = slots(vec![
        ("loose", definition("1:00", "02:00", "UTC")),
        ("later", definition("03:00", "04:00", "No/Such")),
        ("tail", json!("not-an-object")),
    ]);
    let outcome = admit(&config, &root, AdmissionMode::Startup);
    assert_eq!(outcome.windows, Err(WindowError::StrictClock));
    assert_eq!(
        outcome.drops,
        [
            ("later".into(), DropReason::Missing),
            ("tail".into(), DropReason::Shape)
        ]
    );
    let unknown = payload(json!({"fall":definition("1:00", "02:00", "No/Such")}));
    let outcome = admit(&unknown, &root, AdmissionMode::Startup);
    assert!(
        outcome.windows.unwrap().is_empty(),
        "dropped loose windows never reach strict validation"
    );
    assert_eq!(outcome.drops, [("fall".into(), DropReason::Missing)]);
}

#[test]
fn captured_rules_survive_source_mutation_and_removal() {
    let root = scratch("capture");
    copy_zone(&root, "UTC");
    let config = payload(json!({"fall":definition("00:00", "01:00", "UTC")}));
    let outcome = admit(&config, &root, AdmissionMode::Poll);
    assert!(outcome.drops.is_empty());
    let windows = outcome.windows.unwrap();
    let captured = &windows["fall"].window;
    assert!(captured.contains(utc(1800)).unwrap());
    assert!(!captured.contains(utc(3600)).unwrap());
    fs::write(root.join("UTC"), b"broken").unwrap();
    assert!(captured.contains(utc(1800)).unwrap());
    let fresh = admit(&config, &root, AdmissionMode::Poll);
    assert!(fresh.windows.unwrap().is_empty());
    assert_eq!(fresh.drops, [("fall".into(), DropReason::Header)]);
    fs::remove_file(root.join("UTC")).unwrap();
    assert!(captured.contains(utc(1800)).unwrap());
    let fresh = admit(&config, &root, AdmissionMode::Poll);
    assert!(fresh.windows.unwrap().is_empty());
    assert_eq!(fresh.drops, [("fall".into(), DropReason::Missing)]);
    assert_eq!(captured.source_path(), root.join("UTC").as_path());
}

#[test]
fn header_value_errors_drop_but_truncated_bodies_and_size_remain_faults() {
    let root = scratch("header");
    for bytes in [&b"XXXX"[..], &b"TZif"[..], &b"TZifa"[..]] {
        fs::write(root.join("Header"), bytes).unwrap();
        let config = payload(json!({"fall":definition("00:00", "01:00", "Header")}));
        let outcome = admit(&config, &root, AdmissionMode::Poll);
        assert!(outcome.windows.unwrap().is_empty());
        assert_eq!(outcome.drops, [("fall".into(), DropReason::Header)]);
    }
    let real = fs::read("/usr/share/zoneinfo/UTC").unwrap();
    fs::write(root.join("Truncated"), &real[..20]).unwrap();
    let config = payload(json!({"fall":definition("00:00", "01:00", "Truncated")}));
    let outcome = admit(&config, &root, AdmissionMode::Poll);
    assert_eq!(outcome.windows, Err(WindowError::CorruptTzif));
    assert!(outcome.drops.is_empty());
    let second = real
        .windows(4)
        .enumerate()
        .skip(44)
        .find(|(_, bytes)| *bytes == b"TZif")
        .map(|(index, _)| index)
        .expect("UTC 64-bit header");
    for length in [44, second + 4, second + 5] {
        fs::write(root.join("Second"), &real[..length]).unwrap();
        let config = payload(json!({"fall":definition("00:00", "01:00", "Second")}));
        let outcome = admit(&config, &root, AdmissionMode::Poll);
        if length <= second + 4 {
            assert!(outcome.windows.unwrap().is_empty());
            assert_eq!(outcome.drops, [("fall".into(), DropReason::Header)]);
        } else {
            assert_eq!(outcome.windows, Err(WindowError::CorruptTzif));
            assert!(outcome.drops.is_empty());
        }
    }
    let mut huge = real;
    huge.resize(MAX_TZIF_BYTES + 1, 0);
    fs::write(root.join("Huge"), &huge).unwrap();
    let config = payload(json!({"fall":definition("00:00", "01:00", "Huge")}));
    let outcome = admit(&config, &root, AdmissionMode::Poll);
    assert_eq!(outcome.windows, Err(WindowError::TooLarge));
    assert!(outcome.drops.is_empty());
}

#[test]
fn unreadable_regular_file_is_io_and_fifo_is_a_bounded_missing_drop() {
    assert_ne!(
        fs::metadata("/proc/self").unwrap().uid(),
        0,
        "permission proof requires an unprivileged process"
    );
    let root = scratch("io");
    copy_zone(&root, "UTC");
    fs::set_permissions(root.join("UTC"), fs::Permissions::from_mode(0o000)).unwrap();
    let config = payload(json!({"fall":definition("00:00", "01:00", "UTC")}));
    let outcome = admit(&config, &root, AdmissionMode::Poll);
    assert_eq!(
        outcome.windows,
        Err(WindowError::Io(std::io::ErrorKind::PermissionDenied))
    );
    assert!(
        outcome.drops.is_empty(),
        "known regular-file IO must not become ALWAYS"
    );
    let fifo_root = scratch("fifo");
    rustix::fs::mknodat(
        rustix::fs::CWD,
        fifo_root.join("FIFO"),
        rustix::fs::FileType::Fifo,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        0,
    )
    .unwrap();
    let config = payload(json!({"fall":definition("00:00", "01:00", "FIFO")}));
    let (sender, receiver) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        sender
            .send(admit(&config, &fifo_root, AdmissionMode::Poll))
            .unwrap()
    });
    let outcome = receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("nonregular zone must not block");
    reader.join().unwrap();
    assert!(outcome.windows.unwrap().is_empty());
    assert_eq!(outcome.drops, [("fall".into(), DropReason::Missing)]);
}

#[test]
fn public_errors_do_not_retain_zone_paths_or_configuration_text() {
    let root = scratch("errors");
    let config = payload(json!({"fall":definition("00:00", "01:00", "UTC")}));
    let outcome = admit(
        &config,
        Path::new("private-relative-root"),
        AdmissionMode::Poll,
    );
    let error = outcome.windows.unwrap_err();
    assert_eq!(
        error,
        WindowError::Invariant(DetectionWindowError::InvalidZoneDirectory)
    );
    assert!(!error.to_string().contains("private-relative-root"));
    assert!(!format!("{error:?}").contains("UTC"));
    let secret = "private-zone-spelling";
    let config = payload(json!({"fall":definition("00:00", "01:00", secret)}));
    let mut captured = Vec::new();
    let windows = admit_at(
        &config,
        &root,
        AdmissionMode::Poll,
        &mut |drop: WindowDrop<'_>| {
            assert!(!drop.reason.to_string().contains(secret));
            captured.push(drop.definition.unwrap().tz.clone());
        },
    )
    .unwrap();
    assert!(windows.is_empty());
    assert_eq!(captured, [secret]);
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical worker config models"]
fn actual_python_startup_and_poll_match_window_admission() {
    assert_ne!(
        fs::metadata("/proc/self").unwrap().uid(),
        0,
        "unprivileged oracle required"
    );
    let root = scratch("oracle");
    copy_zone(&root, "UTC");
    fs::write(root.join("BadMagic"), b"XXXX").unwrap();
    fs::write(root.join("MagicOnly"), b"TZif").unwrap();
    let real = fs::read(root.join("UTC")).unwrap();
    fs::write(root.join("Truncated"), &real[..20]).unwrap();
    fs::write(root.join("Unreadable"), &real).unwrap();
    fs::set_permissions(root.join("Unreadable"), fs::Permissions::from_mode(0o000)).unwrap();
    let cases = vec![
        definition("01:00", "02:00", "UTC"),
        definition("0１:00", "02:00", "UTC"),
        definition("1:00", "02:00", "UTC"),
        definition("01:0", "02:00", "UTC"),
        definition("١:٠", "02:00", "UTC"),
        definition("01:00", "01:00", "UTC"),
        definition("1:00", "02:00", "No/Such"),
        definition("０１:００", "02:00", "UTC"),
        json!({"start":"01:00","end":"02:00"}),
        Value::Null,
        definition("01:00", "02:00", "BadMagic"),
        definition("01:00", "02:00", "MagicOnly"),
        definition("01:00", "02:00", "Truncated"),
        definition("01:00", "02:00", "Unreadable"),
    ];
    let expected = python_outcomes(&root, &cases);
    for (index, window) in cases.into_iter().enumerate() {
        let config = payload(json!({"fall":window}));
        for (mode, name) in [
            (AdmissionMode::Startup, "startup"),
            (AdmissionMode::Poll, "poll"),
        ] {
            let outcome = admit(&config, &root, mode);
            let class = match outcome.windows {
                Ok(map) if map.is_empty() => "always",
                Ok(_) => "window",
                Err(WindowError::StrictClock) => "refuse",
                Err(WindowError::Io(_)) => "io",
                Err(WindowError::CorruptTzif) => "corrupt",
                Err(error) => panic!("unexpected admission fault: {error}"),
            };
            assert_eq!(
                expected[index][name]["class"], class,
                "case {index}, {name}"
            );
            assert_eq!(
                expected[index][name]["drop"],
                !outcome.drops.is_empty(),
                "diagnostic case {index}, {name}"
            );
        }
    }
}
struct OracleChild(Option<Child>);

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical worker config models"]
fn second_header_truncations_match_python_for_three_real_zones() {
    let root = scratch("second-header");
    let mut cases = Vec::new();
    for (zone_index, zone) in ["UTC", "Asia/Seoul", "America/New_York"]
        .into_iter()
        .enumerate()
    {
        let bytes = fs::read(Path::new("/usr/share/zoneinfo").join(zone)).unwrap();
        let second = bytes
            .windows(4)
            .enumerate()
            .skip(44)
            .find(|(_, magic)| *magic == b"TZif")
            .map(|(offset, _)| offset)
            .expect("real 64-bit second header");
        for length in [
            44,
            second,
            second + 3,
            second + 4,
            second + 5,
            second + 6,
            second + 43,
        ] {
            let name = format!("Probe-{zone_index}-{length}");
            fs::write(root.join(&name), &bytes[..length]).unwrap();
            cases.push(definition("01:00", "02:00", &name));
        }
    }
    let expected = python_outcomes(&root, &cases);
    for (index, case) in cases.into_iter().enumerate() {
        let config = payload(json!({"fall":case}));
        for (mode, name) in [
            (AdmissionMode::Startup, "startup"),
            (AdmissionMode::Poll, "poll"),
        ] {
            let outcome = admit(&config, &root, mode);
            let actual = match outcome.windows {
                Ok(map) if map.is_empty() => "always",
                Err(WindowError::CorruptTzif) => "corrupt",
                other => panic!("unexpected truncated-zone result: {other:?}"),
            };
            assert_eq!(
                expected[index][name]["class"], actual,
                "case {index}/{name}"
            );
            assert_eq!(expected[index][name]["drop"], !outcome.drops.is_empty());
        }
    }
}

impl Drop for OracleChild {
    fn drop(&mut self) {
        if let Some(child) = &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
fn python_outcomes(root: &Path, cases: &[Value]) -> Value {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON");
    let mut child = OracleChild(Some(
        Command::new(python)
            .arg("-c")
            .arg(ORACLE)
            .arg(root)
            .arg(serde_json::to_string(cases).unwrap())
            .env(
                "PYTHONPATH",
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
            )
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ));
    let process = child.0.as_mut().unwrap();
    let clock = seeon_ml_worker::seam::SystemClock::new();
    seeon_ml_worker::poll::poll_until(&clock, Duration::from_secs(20), "window oracle", || {
        process.try_wait().unwrap().is_some()
    })
    .expect("bounded Python oracle");
    let output = child.0.take().unwrap().wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.len() < 4096, "compact oracle receipt");
    serde_json::from_slice(&output.stdout).unwrap()
}
const ORACLE: &str = r#"
import contextlib, io, json, struct, sys, zoneinfo
from pydantic import ValidationError
from worker.runtime.config.pull_models import BackendWorkerConfigPayload
zoneinfo.reset_tzpath([sys.argv[1]])
zoneinfo.ZoneInfo.clear_cache()
results = []
for window in json.loads(sys.argv[2]):
    row = {}
    for mode in ('startup', 'poll'):
        diagnostic = io.StringIO()
        try:
            with contextlib.redirect_stderr(diagnostic):
                payload = BackendWorkerConfigPayload.model_validate({'config_version':0, 'cameras':[], 'detection_windows':{'fall':window}})
                if mode == 'startup':
                    config = payload.to_worker_config('http://relay.invalid', 'token')
                    windows = config.domains.detection_windows or {}
                else:
                    windows = payload.to_pulled_config().detection_windows or {}
            result = 'window' if 'fall' in windows else 'always'
        except ValidationError:
            result = 'refuse'
        except OSError:
            result = 'io'
        except struct.error:
            result = 'corrupt'
        row[mode] = {'class':result, 'drop':'ALWAYS/24-7' in diagnostic.getvalue()}
    results.append(row)
print(json.dumps(results, separators=(',', ':')))
"#;
