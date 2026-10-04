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
use seeon_worker::detection_window::{
    AwareDateTime, ClockRelation, DetectionWindow, DetectionWindowError, MAX_TZIF_BYTES,
};
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
fn oversized_zone_refusal_precedes_malformed_header_classification() {
    let root = scratch("oversized-headers");
    let mut base = fs::read("/usr/share/zoneinfo/UTC").unwrap();
    base.resize(MAX_TZIF_BYTES + 1, 0);
    let mut bad_magic = base.clone();
    bad_magic[..4].copy_from_slice(b"XXXX");
    let mut corrupt_counts = base.clone();
    corrupt_counts[20..24].copy_from_slice(&(-1_i32).to_be_bytes());
    let mut absent_second = base;
    absent_second[40..44].copy_from_slice(&i32::try_from(MAX_TZIF_BYTES).unwrap().to_be_bytes());
    let mut observed = Vec::new();
    for (name, bytes) in [
        ("BadMagic", bad_magic),
        ("CorruptCounts", corrupt_counts),
        ("AbsentSecond", absent_second),
    ] {
        fs::write(root.join(name), bytes).unwrap();
        let config = payload(json!({"fall":definition("00:00", "01:00", name)}));
        for mode in [AdmissionMode::Startup, AdmissionMode::Poll] {
            let outcome = admit(&config, &root, mode);
            observed.push((name, mode, outcome.windows, outcome.drops));
        }
    }
    assert!(
        observed.iter().all(|(_, _, result, drops)| {
            *result == Err(WindowError::TooLarge) && drops.is_empty()
        }),
        "oversized assets must never fall open or enter header parsing: {observed:#?}"
    );
}

#[test]
fn malformed_zone_at_exact_size_limit_retains_header_drop() {
    let root = scratch("exact-size-header");
    fs::write(root.join("BadMagic"), vec![0; MAX_TZIF_BYTES]).unwrap();
    let config = payload(json!({"fall":definition("00:00", "01:00", "BadMagic")}));
    for mode in [AdmissionMode::Startup, AdmissionMode::Poll] {
        let outcome = admit(&config, &root, mode);
        assert!(outcome.windows.unwrap().is_empty());
        assert_eq!(outcome.drops, [("fall".into(), DropReason::Header)]);
    }
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

fn utc_with_footer(footer: &str) -> Vec<u8> {
    let original = fs::read("/usr/share/zoneinfo/UTC").unwrap();
    let mut bytes = original
        .strip_suffix(b"\nUTC0\n")
        .expect("system UTC footer")
        .to_vec();
    bytes.extend_from_slice(format!("\n{footer}\n").as_bytes());
    bytes
}

#[test]
fn accepted_footer_preserves_original_offset_and_future_dst_rules() {
    let root = scratch("footer-rules");
    for (name, footer, inside, outside) in [
        ("Offset", "EST5", 2_209_012_200, 2_208_994_200),
        (
            "Dst",
            "AAA0BBB,M3.2.0/2,M11.1.0/2",
            2_224_715_400,
            2_224_719_000,
        ),
    ] {
        fs::write(root.join(name), utc_with_footer(footer)).unwrap();
        let config = payload(json!({"fall":definition("01:00", "02:00", name)}));
        for mode in [AdmissionMode::Startup, AdmissionMode::Poll] {
            let outcome = admit(&config, &root, mode);
            assert!(outcome.drops.is_empty());
            let windows = outcome.windows.unwrap();
            assert!(windows["fall"].window.contains(utc(inside)).unwrap());
            assert!(!windows["fall"].window.contains(utc(outside)).unwrap());
        }
    }
}

#[test]
fn footer_drop_is_reported_before_surviving_startup_clock_refusal() {
    let root = scratch("footer-order");
    copy_zone(&root, "UTC");
    fs::write(root.join("InvalidFooter"), utc_with_footer("INVALID!!!")).unwrap();
    let config = slots(vec![
        ("bed_exit", definition("1:00", "02:00", "UTC")),
        ("fall", definition("1:00", "02:00", "InvalidFooter")),
    ]);
    for mode in [AdmissionMode::Startup, AdmissionMode::Poll] {
        let outcome = admit(&config, &root, mode);
        assert_eq!(outcome.drops, [("fall".into(), DropReason::Footer)]);
        match mode {
            AdmissionMode::Startup => {
                assert_eq!(outcome.windows, Err(WindowError::StrictClock));
            }
            AdmissionMode::Poll => {
                let windows = outcome.windows.unwrap();
                assert_eq!(windows.len(), 1);
                assert!(windows.contains_key("bed_exit"));
            }
        }
    }
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
    let mut cases = vec![
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
    for (name, footer) in [
        ("FooterShortName", "A0"),
        ("FooterTwoLetter", "AB0"),
        ("FooterQuotedShort", "<A>0"),
        ("FooterCarryMinute", "ABC0:60"),
        ("FooterCarrySecond", "ABC0:00:60"),
        ("FooterNulValid", "EST5\0garbage"),
        ("FooterNulInvalid", "INVALID!!!\0UTC0"),
        ("FooterNulLeading", "\0UTC0"),
        ("FooterNulShort", "A0\0garbage"),
        ("FooterNulDst", "AAA0BBB,M3.2.0/2,M11.1.0/2\0garbage"),
        ("FooterEmptyName", "<>0"),
        ("FooterLargeCarry", "ABC24:99:99"),
        ("FooterNegativeCarry", "ABC-24:99:99"),
        (
            "FooterTransitionCarry",
            "AAA0BBB,M3.2.0/-167:99:99,M11.1.0/167",
        ),
        ("FooterInvalid", "INVALID!!!"),
        ("FooterLargeOffset", "UTC25"),
        ("FooterPunctuation", "UT!0"),
        ("FooterEmpty", ""),
        ("FooterUtc", "UTC0"),
        ("FooterName", "ABC0"),
        ("FooterOffset", "EST5"),
        ("FooterDst", "AAA0BBB,M3.2.0/2,M11.1.0/2"),
    ] {
        fs::write(root.join(name), utc_with_footer(footer)).unwrap();
        cases.push(definition("01:00", "02:00", name));
    }
    cases.push(definition("1:00", "02:00", "FooterInvalid"));
    cases.push(definition("1:00", "02:00", "FooterUtc"));
    let mut version_one = real;
    version_one[4] = b'1';
    fs::write(root.join("VersionOne"), version_one).unwrap();
    cases.push(definition("01:00", "02:00", "VersionOne"));
    let expected = python_outcomes(&root, &cases, ORACLE);
    let mut mismatches = Vec::new();
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
            let actual = json!({"class":class, "drop":!outcome.drops.is_empty()});
            if expected[index][name] != actual {
                mismatches.push(json!({
                    "case":index, "mode":name,
                    "python":expected[index][name], "rust":actual,
                }));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "window admission mismatches: {mismatches:#?}"
    );
}

const FOOTER_ORACLE: &str =
    include_str!("../../../tests_support/detection_window_footer_oracle.py");

fn compare_temporal_probes(
    window: &DetectionWindow,
    probes: &[Value],
    case: &Value,
    surface: &str,
    mismatches: &mut Vec<Value>,
) {
    for probe in probes {
        let relation = match probe[2].as_str().unwrap() {
            "S" => ClockRelation::SameTargetTzinfo,
            "E" => ClockRelation::DifferentTzinfo,
            other => panic!("unexpected oracle relation: {other}"),
        };
        let clock = AwareDateTime::new(
            probe[0].as_str().unwrap().parse().unwrap(),
            Some(i32::try_from(probe[1].as_i64().unwrap()).unwrap()),
            relation,
        )
        .unwrap_or_else(|error| {
            panic!("oracle source clock outside the existing domain: {probe:?}: {error:?}")
        });
        let actual = window.contains(clock);
        let expected = &probe[3];
        let matches = if let Some(value) = expected.as_bool() {
            actual == Ok(value)
        } else {
            assert!(case["edge"].as_bool().unwrap_or(false));
            assert_eq!(expected["exception"], "OverflowError", "{probe:?}");
            assert!(
                expected["message"]
                    .as_str()
                    .is_some_and(|text| !text.is_empty())
            );
            actual == Err(DetectionWindowError::ClockOutOfRange)
        };
        if !matches {
            mismatches.push(json!({
                "window":case["window"], "surface":surface,
                "clock":probe, "rust":format!("{actual:?}"),
            }));
        }
    }
}

fn compare_python_temporal_cases(root: &Path, cases: &[Value]) {
    let mut captured = BTreeMap::new();
    for case in cases {
        let zone = case["window"]["tz"].as_str().unwrap();
        captured
            .entry(zone.to_owned())
            .or_insert_with(|| fs::read(root.join(zone)).unwrap());
    }
    let mut mismatches = Vec::new();
    // Generated clocks include their civil fields and actual source offsets.
    // One definition per call keeps those receipts below the existing 4096 B.
    for chunk in cases.chunks(1) {
        let expected = python_outcomes(root, chunk, FOOTER_ORACLE);
        assert_eq!(expected.as_array().unwrap().len(), chunk.len());
        for (index, case) in chunk.iter().enumerate() {
            let definition = &case["window"];
            let start = definition["start"].as_str().unwrap();
            let end = definition["end"].as_str().unwrap();
            let zone = definition["tz"].as_str().unwrap();
            let probes = expected[index].as_array().unwrap();
            assert!(!probes.is_empty());
            match DetectionWindow::from_tzif_bytes(start, end, zone, root, &captured[zone]) {
                Ok(window) => {
                    assert_eq!(window.source_path(), root.join(zone).as_path());
                    compare_temporal_probes(
                        &window,
                        probes,
                        case,
                        "from_tzif_bytes",
                        &mut mismatches,
                    );
                }
                Err(error) => mismatches.push(json!({
                    "window":definition, "surface":"from_tzif_bytes",
                    "rust":format!("{error:?}"),
                })),
            }
            let config = payload(json!({"fall":definition}));
            for (mode, name) in [
                (AdmissionMode::Startup, "startup"),
                (AdmissionMode::Poll, "poll"),
            ] {
                let outcome = admit(&config, root, mode);
                match outcome.windows {
                    Ok(windows) if start == end => {
                        assert!(windows.is_empty());
                        assert_eq!(outcome.drops, [("fall".into(), DropReason::Time)]);
                    }
                    Ok(windows) if outcome.drops.is_empty() && windows.contains_key("fall") => {
                        let admitted = &windows["fall"];
                        assert_eq!(admitted.definition.start, start);
                        assert_eq!(admitted.definition.end, end);
                        assert_eq!(admitted.definition.tz, zone);
                        compare_temporal_probes(
                            &admitted.window,
                            probes,
                            case,
                            name,
                            &mut mismatches,
                        );
                    }
                    other => mismatches.push(json!({
                        "window":definition, "surface":name,
                        "rust":format!("{other:?}"), "drops":format!("{:?}", outcome.drops),
                    })),
                }
            }
        }
    }
    for (zone, bytes) in captured {
        assert_eq!(
            fs::read(root.join(zone)).unwrap(),
            bytes,
            "oracle changed TZif bytes"
        );
    }
    assert!(
        mismatches.is_empty(),
        "temporal TZif mismatches: {mismatches:#?}"
    );
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical worker config models"]
fn actual_python_footer_contains_matches_python_valid_assets() {
    let root = scratch("footer-temporal");
    let mut cases = Vec::new();
    for (name, footer) in [
        ("ShortName", "A0"),
        ("TwoLetter", "AB0"),
        ("QuotedShort", "<A>0"),
        ("EmptyName", "<>0"),
        ("Offset", "EST5"),
        ("NulValid", "EST5\0ignored"),
        ("NulShort", "A0\0ignored"),
        ("MinuteCarry", "ABC0:60"),
        ("SecondCarry", "ABC0:00:60"),
        ("LargeNegative", "ABC24:99:99"),
        ("LargePositive", "ABC-24:99:99"),
        ("ExactDay", "ABC24"),
        ("DefaultDstLarge", "AAA-24:99:99BBB,M3.2.0,M11.1.0"),
        ("EqualRules", "AAA0BBB,0/0,0/0"),
    ] {
        fs::write(root.join(name), utc_with_footer(footer)).unwrap();
        cases.push(json!({
            "window":definition("08:00", "09:00", name),
            "clocks":[
                ["2050-01-15T08:30:00",0,"E",0],
                ["2050-07-15T08:30:00",0,"E",0],
                ["2050-07-15T08:29:59.999999",0,"E",0],
            ],
            "sampling":{"kind":"window","anchor":"2050-07-15T08:30:00"},
        }));
    }
    // ABC24 is a valid contains result, not a target-utcoffset range refusal.
    // Equal endpoints remain empty in the primitive and drop in configuration.
    cases.push(json!({
        "window":definition("08:00", "08:00", "EmptyName"),
        "sampling":{"kind":"window","anchor":"2050-07-15T08:30:00"},
    }));
    cases.push(json!({
        "window":definition("23:30", "00:30", "Offset"),
        "sampling":{"kind":"window","anchor":"2050-07-15T08:30:00"},
    }));
    cases.push(json!({
        "window":definition("00:00", "01:00", "EqualRules"),
        "clocks":[
            ["2049-12-31T22:59:59.999999",0,"E",0],
            ["2049-12-31T23:00:00",0,"E",0],
            ["2049-12-31T23:59:59.999999",0,"E",0],
            ["2050-01-01T00:00:00",0,"E",0],
            ["2050-01-01T00:00:00.000001",0,"E",0],
        ],
    }));
    compare_python_temporal_cases(&root, &cases);
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical worker config models"]
fn actual_python_footer_transition_boundaries_match_contains() {
    let root = scratch("footer-transition");
    let mut cases = Vec::new();
    for (name, footer) in [
        ("Northern", "AAA0BBB,M3.2.0/2,M11.1.0/2"),
        ("Southern", "AAA-10BBB-11,M10.1.0/2,M4.1.0/3"),
        ("NegativeDst", "AAA-1BBB0,M3.2.0/2,M11.1.0/2"),
        ("Extended", "AAA0BBB,M3.2.0/-167:99:99,M11.1.0/167:99:99"),
        ("Julian", "AAA0BBB,J60/2,J300/2"),
        ("NumericJulian", "AAA0BBB,59/2,299/2"),
        ("Jan1Start", "AAA0BBB,0/-2,200/26"),
        ("Jan1End", "AAA0BBB,200/2,0/-2"),
        ("DefaultDstLarge", "AAA-24:99:99BBB,M3.2.0,M11.1.0"),
        ("NulDst", "AAA0BBB,M3.2.0/2,M11.1.0/2\0ignored"),
    ] {
        fs::write(root.join(name), utc_with_footer(footer)).unwrap();
        let years: &[i32] = if matches!(name, "Julian" | "NumericJulian") {
            &[2050, 2052]
        } else {
            &[2050]
        };
        let windows: &[(&str, &str)] = match name {
            "Extended" => &[("23:30", "00:30")],
            "Jan1Start" | "Jan1End" => &[("21:30", "22:30"), ("00:30", "01:30")],
            _ => &[("01:30", "02:30")],
        };
        for year in years {
            for (start, end) in windows {
                cases.push(json!({
                    "window":definition(start, end, name),
                    "sampling":{
                        "kind":"transitions",
                        "anchor":format!("{year}-07-15T08:30:00"),
                        "start":format!("{}-12-20T00:00:00", year - 1),
                        "end":format!("{}-01-10T00:00:00", year + 1),
                    },
                }));
            }
        }
    }
    compare_python_temporal_cases(&root, &cases);
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical worker config models"]
fn actual_python_historical_body_and_whole_second_footer_cutoff_match_contains() {
    let root = scratch("footer-history");
    copy_zone(&root, "America/New_York");
    let original = fs::read(root.join("America/New_York")).unwrap();
    assert!(original.ends_with(b"\n"));
    let footer_start = original[..original.len() - 1]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .unwrap();
    let mut changed = original[..footer_start].to_vec();
    changed.extend_from_slice(b"\nA0\n");
    fs::write(root.join("History"), changed).unwrap();
    let mut cases = Vec::new();
    for zone in ["History", "America/New_York"] {
        cases.push(json!({
            "window":definition("01:00", "02:00", zone),
            "clocks":[
                ["1950-01-15T06:30:00",0,"E",0],
                ["1950-07-15T05:30:00",0,"E",0],
                ["2050-01-15T06:30:00",0,"E",0],
                ["2050-07-15T05:30:00",0,"E",0],
            ],
            "sampling":{"kind":"history","anchor":"1950-07-15T05:30:00"},
        }));
    }
    let mut day_rules = original[..footer_start].to_vec();
    day_rules.extend_from_slice(b"\nEST5EDT,60/2,300/2\n");
    fs::write(root.join("NativeDayRules"), day_rules).unwrap();
    cases.push(json!({
        "window":definition("03:00", "04:00", "NativeDayRules"),
        "clocks":[
            ["2050-03-01T06:59:59.999999",0,"E",0],
            ["2050-03-01T07:00:00",0,"E",0],
            ["2050-03-01T07:30:00",0,"E",0],
            ["2050-03-02T07:30:00",0,"E",0],
        ],
        "sampling":{"kind":"history","anchor":"2050-03-01T07:30:00"},
    }));
    compare_python_temporal_cases(&root, &cases);
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical worker config models"]
fn actual_python_same_target_gap_fold_and_civil_edges_match_contains() {
    let root = scratch("footer-clock-domain");
    copy_zone(&root, "UTC");
    copy_zone(&root, "America/New_York");
    fs::write(root.join("Lower"), utc_with_footer("ABC24:99:99")).unwrap();
    fs::write(
        root.join("Upper"),
        utc_with_footer("AAA-24:99:99BBB,M10.1.0,M4.1.0"),
    )
    .unwrap();
    let cases = [
        json!({
            "window":definition("02:00", "03:00", "America/New_York"),
            "clocks":[
                ["2024-03-10T02:30:00",null,"S",0],
                ["2024-03-10T02:30:00",null,"S",1],
                ["2024-03-10T02:30:00",-18000,"E",0],
                ["2024-03-10T02:30:00",-14400,"E",0],
            ],
        }),
        json!({
            "window":definition("01:30", "02:00", "America/New_York"),
            "clocks":[
                ["2024-11-03T01:29:59.999999",null,"S",0],
                ["2024-11-03T01:30:00",null,"S",0],
                ["2024-11-03T01:30:00",null,"S",1],
                ["2024-11-03T01:30:00",-14400,"E",0],
                ["2024-11-03T01:30:00",-18000,"E",0],
                ["2024-11-03T02:00:00",null,"S",0],
            ],
        }),
        json!({
            "window":definition("23:59", "00:01", "America/New_York"),
            "clocks":[
                ["0001-01-01T00:00:00",null,"S",0],
                ["9999-12-31T23:59:59.999999",null,"S",0],
                ["0001-01-01T00:00:00",0,"E",0],
            ],
            "edge":true,
        }),
        json!({
            "window":definition("23:59", "00:01", "UTC"),
            "clocks":[
                ["0001-01-01T00:00:00",0,"E",0],
                ["0001-01-01T00:00:00.000001",0,"E",0],
                ["9999-12-30T00:00:00",0,"E",0],
            ],
        }),
        json!({
            "window":definition("00:00", "01:00", "Lower"),
            "sampling":{"kind":"edge","side":"lower","anchor":"0001-01-03T00:00:00"},
            "edge":true,
        }),
        json!({
            "window":definition("23:59", "00:01", "Upper"),
            // Southern default DST is +96039 s, so the upper civil cutoff's
            // UTC samples remain inside the existing external-clock domain.
            "sampling":{"kind":"edge","side":"upper","anchor":"9999-12-29T12:00:00"},
            "edge":true,
        }),
    ];
    compare_python_temporal_cases(&root, &cases);
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
            second + 44,
        ] {
            let name = format!("Probe-{zone_index}-{length}");
            fs::write(root.join(&name), &bytes[..length]).unwrap();
            cases.push(definition("01:00", "02:00", &name));
        }
    }
    let expected = python_outcomes(&root, &cases, ORACLE);
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
fn python_outcomes(root: &Path, cases: &[Value], program: &str) -> Value {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON");
    let mut child = OracleChild(Some(
        Command::new(python)
            .arg("-c")
            .arg(program)
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
