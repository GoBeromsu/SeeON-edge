//! Port of `tests/test_rust_detection_window_parity.py`. The oracle is the
//! Python `worker.domains.detection_window.DetectionWindow` on real
//! `ZoneInfo.from_file` zones, and `BedExitMonitor` driven by the bed-policy
//! test's `_exercise`: every WINDOWPROBE/BEDPROBE v1 row they rendered is
//! replayed through the public `DetectionWindow` and `BedExitMonitor` and
//! compared row for row. Zones come only from the recorded TZif bytes, written
//! to a private directory after their size and sha256 checks; no machine zone
//! data or ambient environment is read.

#[path = "fixtures/support.rs"]
mod support;

#[path = "fixtures/policy.rs"]
mod policy;

#[path = "fixtures/bed.rs"]
mod bed;

use policy::{Fields, Transcript};
use seeon_worker::detection_window::{self, DetectionWindow, DetectionWindowError};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

const FIXTURE: &str = "detection_window/detection_window.json";
const ORACLE_SOURCES: [(&str, &str); 15] = [
    (
        "tests/test_rust_detection_window_parity.py",
        "3d13e2595e785011844fbda6e13a0e483343554c3bece3e8b3a5b529512cf3ef",
    ),
    (
        "tests/test_rust_bed_policy_parity.py",
        "ffa548537788de88baa0540966b8d57c5806d7349377a2de013401a95f04b041",
    ),
    (
        "worker/domains/bed_exit/detector.py",
        "8cda28e29330e7061b4562840a836dd91b4c4c5772275e189bfb26fff1303cf1",
    ),
    (
        "worker/domains/bed_exit/geometry.py",
        "b6f160a131d95476e48b5b3288b9ac50d7c266d060d0f447933971c9b075a46d",
    ),
    (
        "worker/domains/bed_exit/latch.py",
        "fd3d63afe85941cd77238b336814cb17f463a5cc2eb4bc47c6d9b4754ea9479a",
    ),
    (
        "worker/domains/bed_exit/night_window.py",
        "8b48f6d3d1dc99863ce02a64522f9dea485f32cb9d22ccd17bc5d3a094ca3b56",
    ),
    (
        "worker/domains/bed_exit/schema.py",
        "3802cfaa6209f787524fe2dd4ee8037fad11b43c4f1ab3668a073d4a276e52a3",
    ),
    (
        "worker/domains/detection_window.py",
        "0618af9c8f7df60b502131d214305f98be1adc47b977b2ab2db75d6b903d5b44",
    ),
    (
        "worker/domains/episode/authority.py",
        "e29a9fcbb1a5823d0630378e0dd0a84477f42fa60b33528a75e603df0acd065d",
    ),
    (
        "worker/domains/staleness.py",
        "a204973339fbcb573a7afd023f36d20e6b5ae98e4392d3d7f0491430d5f8019f",
    ),
    (
        "shared/detection_policies.py",
        "6fa6a49832e0aa63da7905bd6d4ebe512ff152999335af32433403649470389c",
    ),
    (
        "contracts/observation.py",
        "75c06ff9ddef6cc93b23d5c3ff515312e1c00e91ca725284fbca40efb53b6823",
    ),
    (
        "worker/types/bed_pose_features.py",
        "9254e6bd498077ecbdce2e9be339d8b874da47dbbee1345bc10fd876df70d2bf",
    ),
    (
        "worker/types/business_event.py",
        "c59eda20a1dbb4fd9aa35c1f41afd9b1d6ec05be6d1905c5a18fb74f3f778b3d",
    ),
    (
        "worker/types/trace.py",
        "ac569c2dfb1a8fa1e057f9e5555b971fce76f606e225f61f1e282601cb2a7b46",
    ),
];

fn fixture() -> &'static Value {
    static FIXTURE_VALUE: OnceLock<Value> = OnceLock::new();
    FIXTURE_VALUE.get_or_init(|| support::load_sources(FIXTURE, &ORACLE_SOURCES))
}

/// A private zoneinfo directory, removed when the test that made it ends.
struct ZoneDir(PathBuf);

impl ZoneDir {
    fn new(label: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "seeon-dw-fixture-{}-{}-{label}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path)
            .unwrap_or_else(|error| panic!("create {}: {error}", path.display()));
        Self(path)
    }

    /// Writes `bytes` at the relative IANA `key` after the recorded size and sha256.
    fn write(&self, key: &str, bytes: &[u8], recorded: &Value) {
        assert_eq!(
            bytes.len(),
            support::usize_of(&recorded["size"]),
            "{key}: size"
        );
        assert_eq!(
            support::sha256(bytes),
            support::text(&recorded["sha256"]),
            "{key}: sha256"
        );
        let path = self.0.join(key);
        let parent = path.parent().expect("zone file has a parent");
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("create {}: {error}", parent.display()));
        std::fs::write(&path, bytes)
            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ZoneDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The frozen UTC, Asia/Seoul and America/New_York TZif files the oracle read.
fn frozen_zones(label: &str) -> ZoneDir {
    let zones = ZoneDir::new(label);
    let files = support::array(&fixture()["zoneinfo"]["files"]);
    let keys: Vec<&str> = files
        .iter()
        .map(|file| support::text(&file["key"]))
        .collect();
    assert_eq!(
        keys,
        ["UTC", "Asia/Seoul", "America/New_York"],
        "frozen zones"
    );
    for file in files {
        let bytes: Vec<u8> = support::array(&file["bytes"])
            .iter()
            .map(|byte| u8::try_from(support::uint(byte)).expect("TZif byte"))
            .collect();
        zones.write(support::text(&file["key"]), &bytes, file);
    }
    zones
}

fn replay(transcript: &Transcript, zoneinfo: &Path) -> Result<String, String> {
    let request = transcript.request();
    match request.first().copied() {
        Some(bed::WINDOW_HEADER) => bed::replay_windows(&request, zoneinfo),
        Some(bed::BED_HEADER) => bed::replay(&request, Some(zoneinfo)),
        header => panic!("{}: request header {header:?}", transcript.context),
    }
}

fn assert_parity(test: &str, cases: usize, exchanges: usize) {
    let zones = frozen_zones(test);
    for transcript in policy::transcripts(fixture(), test, cases, exchanges) {
        let replayed = replay(&transcript, zones.path());
        policy::assert_transcript(&transcript, replayed);
    }
}

fn python_wider(key: &str) {
    assert_eq!(
        fixture()["python_wider"][key].as_bool(),
        Some(true),
        "Python contains() accepted {key}"
    );
}

#[test]
fn half_open_equal_overnight_microsecond_edges() {
    assert_parity("test_half_open_equal_overnight_microsecond_edges", 9, 9);
}

#[test]
fn cross_offset_conversion_uses_injected_source_offset() {
    assert_parity(
        "test_cross_offset_conversion_uses_injected_source_offset",
        3,
        3,
    );
}

#[test]
fn seoul_historical_half_hour_and_future_posix_rules() {
    assert_parity(
        "test_seoul_historical_half_hour_and_future_posix_rules",
        1,
        1,
    );
}

#[test]
fn new_york_gap_from_real_instants_and_fold_both_occurrences() {
    assert_parity(
        "test_new_york_gap_from_real_instants_and_fold_both_occurrences",
        1,
        1,
    );
}

#[test]
fn source_local_gap_preserves_canonical_same_zone_semantics() {
    assert_parity(
        "test_source_local_gap_preserves_canonical_same_zone_semantics",
        2,
        2,
    );
}

#[test]
fn equal_key_distinct_gap_clocks_differ_only_by_identity_tag() {
    assert_parity(
        "test_equal_key_distinct_gap_clocks_differ_only_by_identity_tag",
        4,
        4,
    );
}

#[test]
fn fall_fold_identity_and_actual_offset_when_target_changes_to_utc() {
    assert_parity(
        "test_fall_fold_identity_and_actual_offset_when_target_changes_to_utc",
        4,
        4,
    );
}

#[test]
fn supported_python_calendar_edges() {
    assert_parity("test_supported_python_calendar_edges", 3, 3);
}

#[test]
fn same_target_python_calendar_extremes_are_civil() {
    assert_parity("test_same_target_python_calendar_extremes_are_civil", 6, 6);
}

#[test]
fn external_conversion_outside_python_calendar_is_rejected() {
    assert_parity(
        "test_external_conversion_outside_python_calendar_is_rejected",
        2,
        2,
    );
}

#[test]
fn external_jiff_extreme_timestamp_rejection_is_not_python_equivalence() {
    // Python's contains() answers both cases; the refusal is Rust range safety.
    python_wider("jiff_extreme_contains_from_file");
    python_wider("jiff_extreme_contains_no_cache");
    assert_parity(
        "test_external_jiff_extreme_timestamp_rejection_is_not_python_equivalence",
        2,
        2,
    );
}

#[test]
fn distinct_equal_zone_minimum_with_in_range_intermediate_utc() {
    assert_parity(
        "test_distinct_equal_zone_minimum_with_in_range_intermediate_utc",
        4,
        4,
    );
}

#[test]
fn distinct_equal_zone_can_overflow_before_returning_to_in_range_civil_time() {
    assert_parity(
        "test_distinct_equal_zone_can_overflow_before_returning_to_in_range_civil_time",
        4,
        4,
    );
}

#[test]
fn invalid_window_times_fail_real_reference_and_probe() {
    assert_parity(
        "test_invalid_window_times_fail_real_reference_and_probe",
        9,
        9,
    );
}

#[test]
fn naive_clock_rejected_without_machine_local_fallback() {
    assert_parity(
        "test_naive_clock_rejected_without_machine_local_fallback",
        2,
        2,
    );
}

#[test]
fn clock_transport_admission_not_wider_python_parity() {
    assert_parity(
        "test_clock_transport_admission_not_wider_python_parity",
        16,
        16,
    );
}

#[test]
fn fractional_offset_python_domain_is_explicitly_wider() {
    // Python's contains() accepts a 0.5 s offset; the whole-second transport refuses it.
    python_wider("fractional_offset_contains");
    assert_parity(
        "test_fractional_offset_python_domain_is_explicitly_wider",
        1,
        1,
    );
}

/// The defect a recorded `{'defect': ..., 'relation': ...}` case names.
fn defect(params: &str) -> &str {
    let rest = params
        .split_once("'defect': '")
        .unwrap_or_else(|| panic!("params {params:?} name no defect"))
        .1;
    rest.split_once('\'').expect("quoted defect").0
}

#[test]
fn zone_data_failure_never_falls_back() {
    let test = "test_zone_data_failure_never_falls_back";
    let transcripts = policy::transcripts(fixture(), test, 14, 14);
    let recorded: Vec<(&str, &Value)> = support::array(&fixture()["tests"][test])
        .iter()
        .flat_map(|case| {
            let params = support::text(&case["params"]);
            support::array(&case["transcripts"])
                .iter()
                .map(move |transcript| (defect(params), &transcript["zoneinfo"]))
        })
        .collect();
    let mut seen = Vec::new();
    for (transcript, (defect, zoneinfo)) in transcripts.iter().zip(recorded) {
        let context = &transcript.context;
        let kind = support::text(&zoneinfo["kind"]);
        // The library takes an explicit directory and reads no environment, so
        // the probe's unset variable is the same refusal as an empty one.
        let (dir, expected) = match (defect, kind) {
            ("unset", "unset") | ("empty", "empty") => {
                (None, DetectionWindowError::InvalidZoneDirectory)
            }
            ("relative", "relative") => (None, DetectionWindowError::InvalidZoneDirectory),
            ("unknown-zone", "frozen") => (
                Some(frozen_zones("unknown-zone")),
                DetectionWindowError::ZoneDataUnavailable,
            ),
            (defect, "tmp") => {
                let dir = ZoneDir::new(defect);
                for file in support::array(&zoneinfo["files"]) {
                    let content = &file["content"];
                    let bytes = match (content["ascii"].as_str(), content["repeat"].as_str()) {
                        (Some(ascii), None) => ascii.as_bytes().to_vec(),
                        (None, Some(repeat)) => repeat
                            .repeat(support::usize_of(&content["count"]))
                            .into_bytes(),
                        _ => panic!("{context}: zone file content {content}"),
                    };
                    dir.write(support::text(&file["name"]), &bytes, file);
                }
                let expected = match defect {
                    "missing" => DetectionWindowError::ZoneDataUnavailable,
                    "invalid" => DetectionWindowError::InvalidZoneData,
                    "large" => DetectionWindowError::ZoneDataTooLarge,
                    defect => panic!("{context}: temporary-directory defect {defect:?}"),
                };
                (Some(dir), expected)
            }
            pair => panic!("{context}: zoneinfo {pair:?}"),
        };
        let root = match (&dir, kind) {
            (Some(dir), _) => dir.path().to_path_buf(),
            (None, "relative") => PathBuf::from(support::text(&zoneinfo["path"])),
            (None, _) => PathBuf::new(),
        };
        let mut row = Fields::new(&transcript.request[1]);
        assert_eq!(row.next(), "W", "{context}: window row");
        let (_start, _end, zone) = (row.text(), row.text(), row.text());
        let named = if defect == "unknown-zone" {
            "Unknown/NoSuchZone"
        } else {
            "UTC"
        };
        assert_eq!(zone, named, "{context}: zone");
        let replayed = replay(transcript, &root);
        assert_eq!(
            replayed,
            Err(format!("window: {expected:?}")),
            "{context}: zone {zone:?} under {root:?}"
        );
        policy::assert_transcript(transcript, replayed);
        seen.push(defect.to_owned());
    }
    seen.dedup();
    assert_eq!(
        seen,
        [
            "unset",
            "empty",
            "relative",
            "missing",
            "invalid",
            "large",
            "unknown-zone"
        ],
        "every recorded defect"
    );
}

#[test]
fn bed_window_suppression_consumes_raw_onset_and_uses_wall_clock() {
    assert_parity(
        "test_bed_window_suppression_consumes_raw_onset_and_uses_wall_clock",
        1,
        1,
    );
}

#[test]
fn window_suppressed_recovery_preserves_open_episode() {
    assert_parity(
        "test_window_suppressed_recovery_preserves_open_episode",
        1,
        1,
    );
}

#[test]
fn bed_reused_gap_clock_recomputes_identity_after_target_changes() {
    assert_parity(
        "test_bed_reused_gap_clock_recomputes_identity_after_target_changes",
        2,
        2,
    );
}

#[test]
fn fatal_onset_conserves_prefix_and_exact_release_after_poison() {
    // Rust capacity safety, not Python-domain parity: the Python exercise
    // renders both onsets; capacity 1 admits one, then fails with it.
    let fatal = &fixture()["fatal_onset"];
    let zones = frozen_zones("fatal-onset");
    let exercise = &fatal["exercise"];
    let context = "fatal_onset exercise";
    let request = policy::request(exercise, context);
    let request: Vec<&str> = request.iter().map(String::as_str).collect();
    let replayed = bed::replay(&request, Some(zones.path()))
        .unwrap_or_else(|cause| panic!("{context}: Rust refused: {cause}"));
    policy::assert_lines(context, &replayed, &policy::rows(&exercise["expected"]));

    let context = "fatal_onset capacity 1";
    let request = policy::request(fatal, context);
    let request: Vec<&str> = request.iter().map(String::as_str).collect();
    let (out, success) = bed::replay_monitor(&request, Some(zones.path()))
        .unwrap_or_else(|cause| panic!("{context}: construction refused: {cause}"));
    assert!(!success, "{context}: every call succeeded");
    assert_eq!(
        support::text(&fatal["stderr"]),
        "bed-policy-probe: rejected",
        "{context}: recorded probe verdict"
    );
    let lines: Vec<&str> = out.lines().collect();
    let events: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| line.starts_with("E\t"))
        .collect();
    assert_eq!(events, policy::rows(&fatal["events"]), "{context}: events");
    assert!(
        !out.contains(support::text(&fatal["absent_identity_hex"])),
        "{context}: second onset identity rendered"
    );
    let errors: Vec<[String; 3]> = lines
        .iter()
        .filter(|line| line.starts_with("X\t"))
        .map(|line| {
            let fields: Vec<&str> = line.split('\t').collect();
            [fields[1], fields[2], fields[fields.len() - 1]].map(str::to_owned)
        })
        .collect();
    let expected: Vec<[String; 3]> = support::array(&fatal["errors"])
        .iter()
        .map(|error| {
            let error: Vec<String> = support::array(error)
                .iter()
                .map(|field| support::text(field).to_owned())
                .collect();
            error.try_into().expect("error triple")
        })
        .collect();
    assert_eq!(errors, expected, "{context}: failure rows");
    let prefix = support::text(&fatal["release_after_prefix"]);
    let after = lines
        .iter()
        .position(|line| line.starts_with(prefix))
        .unwrap_or_else(|| panic!("{context}: no {prefix:?} row"));
    let release = support::text(&fatal["release_row"]);
    assert!(
        lines[after..].contains(&release),
        "{context}: {release:?} after {prefix:?}"
    );
    assert!(
        out.ends_with(&format!("{}\n", support::text(&fatal["end"]))),
        "{context}: call count"
    );
}

/// Captured bytes are the rules. `source_path` is the checked caller-supplied
/// name, not evidence that construction read a file.
#[test]
fn captured_tzif_bytes_match_filesystem_and_survive_source_mutation() {
    let zones = frozen_zones("captured-bytes");
    let root = zones.path();
    let clocks = [
        ("UTC", 0_i32, "1970-01-01T00:00:00", true),
        ("UTC", 0, "1970-01-01T00:01:00", false),
        ("Asia/Seoul", 0, "1969-12-31T23:59:59", true),
        ("Asia/Seoul", 0, "1960-01-01T00:00:00", true),
        ("America/New_York", 0, "1969-12-31T23:59:59", true),
        ("America/New_York", 0, "1970-01-01T00:01:00", false),
        ("America/New_York", -4 * 3600, "2024-11-03T01:30:00", true),
        ("America/New_York", -5 * 3600, "2024-11-03T01:30:00", true),
        ("America/New_York", 0, "2024-03-10T06:29:59", false),
        ("America/New_York", 0, "2024-03-10T06:30:00", true),
        ("America/New_York", 0, "2024-03-10T07:00:00", true),
    ];
    for file in support::array(&fixture()["zoneinfo"]["files"]) {
        let key = support::text(&file["key"]);
        let captured = tzif_bytes(file);
        let expected_path = detection_window::zoneinfo_path(key, root).expect("checked path");
        assert_eq!(expected_path, root.join(key));
        let (start, end, membership) = match key {
            "UTC" => ("00:00", "00:01", clocks_for(key, &clocks)),
            "Asia/Seoul" => ("08:30", "09:00", clocks_for(key, &clocks)),
            "America/New_York" => ("01:30", "19:01", clocks_for(key, &clocks)),
            other => panic!("unexpected frozen zone {other}"),
        };
        let from_file =
            DetectionWindow::from_zoneinfo_dir(start, end, key, root).expect("filesystem window");
        let from_bytes =
            DetectionWindow::from_tzif_bytes(start, end, key, root, &captured).expect("captured");
        assert_eq!(from_bytes, from_file, "{key}: captured rules");
        assert_eq!(from_bytes.source_path(), expected_path.as_path());
        for (offset, text, expected) in membership {
            let now = external(text, offset);
            assert_eq!(from_bytes.contains(now).unwrap(), expected, "{key} {text}");
            assert_eq!(from_file.contains(now).unwrap(), expected, "{key} {text}");
        }
    }

    let seoul = tzif_bytes(
        support::array(&fixture()["zoneinfo"]["files"])
            .iter()
            .find(|file| support::text(&file["key"]) == "Asia/Seoul")
            .expect("Seoul record"),
    );
    let seoul_path = root.join("Asia/Seoul");
    std::fs::write(&seoul_path, b"not tzif").expect("mutate captured source");
    let still = DetectionWindow::from_tzif_bytes("21:00", "05:00", "Asia/Seoul", root, &seoul)
        .expect("captured bytes ignore the mutated file");
    assert_eq!(still.source_path(), seoul_path.as_path());
    let in_seoul_but_not_utc = external("1970-01-01T16:00:00", 0);
    assert!(still.contains(in_seoul_but_not_utc).unwrap());
    assert_eq!(
        DetectionWindow::from_zoneinfo_dir("21:00", "05:00", "Asia/Seoul", root),
        Err(DetectionWindowError::InvalidZoneData)
    );
    std::fs::remove_file(&seoul_path).expect("remove source after capture");
    let after_removal =
        DetectionWindow::from_tzif_bytes("21:00", "05:00", "Asia/Seoul", root, &seoul)
            .expect("captured bytes ignore removal");
    assert_eq!(after_removal, still);
    assert!(after_removal.contains(in_seoul_but_not_utc).unwrap());
    assert_eq!(
        DetectionWindow::from_zoneinfo_dir("21:00", "05:00", "Asia/Seoul", root),
        Err(DetectionWindowError::ZoneDataUnavailable)
    );
}

fn tzif_bytes(file: &Value) -> Vec<u8> {
    support::array(&file["bytes"])
        .iter()
        .map(|byte| u8::try_from(support::uint(byte)).expect("TZif byte"))
        .collect()
}

fn clocks_for<'a>(key: &str, clocks: &'a [(&str, i32, &str, bool)]) -> Vec<(i32, &'a str, bool)> {
    clocks
        .iter()
        .filter(|row| row.0 == key)
        .map(|row| (row.1, row.2, row.3))
        .collect()
}

fn external(text: &str, offset: i32) -> seeon_worker::detection_window::AwareDateTime {
    seeon_worker::detection_window::AwareDateTime::new(
        text.parse().unwrap(),
        Some(offset),
        seeon_worker::detection_window::ClockRelation::DifferentTzinfo,
    )
    .unwrap()
}

#[test]
fn zoneinfo_path_refuses_before_io_and_bytes_need_no_file() {
    let directory = ZoneDir::new("byte-nonexistent-root");
    let missing = directory.path().join("not-created");
    assert!(!missing.exists(), "fixture root must not exist");
    for (tz, expected) in [
        ("", DetectionWindowError::InvalidZoneName),
        ("../UTC", DetectionWindowError::InvalidZoneName),
        ("Asia/../Seoul", DetectionWindowError::InvalidZoneName),
        ("UTC/.", DetectionWindowError::InvalidZoneName),
        ("Asia//Seoul", DetectionWindowError::InvalidZoneName),
        ("UTC\0", DetectionWindowError::InvalidZoneName),
        ("not a zone", DetectionWindowError::InvalidZoneName),
    ] {
        assert_eq!(
            detection_window::zoneinfo_path(tz, &missing),
            Err(expected),
            "{tz:?}"
        );
        assert_eq!(
            DetectionWindow::from_tzif_bytes("00:00", "01:00", tz, &missing, b"TZif"),
            Err(expected),
            "{tz:?} must fail before any byte or filesystem use"
        );
    }
    assert_eq!(
        detection_window::zoneinfo_path("UTC", Path::new("relative")),
        Err(DetectionWindowError::InvalidZoneDirectory)
    );
    assert_eq!(
        DetectionWindow::from_zoneinfo_dir("00:00", "01:00", "UTC", Path::new("relative")),
        Err(DetectionWindowError::InvalidZoneDirectory)
    );
    let overlong = PathBuf::from(format!(
        "/{}",
        "x".repeat(detection_window::MAX_ZONE_PATH_BYTES)
    ));
    for root in [Path::new("relative"), overlong.as_path()] {
        assert_eq!(
            detection_window::zoneinfo_path("../UTC", root),
            Err(DetectionWindowError::InvalidZoneName)
        );
    }
    assert_eq!(
        detection_window::zoneinfo_path("UTC", &overlong),
        Err(DetectionWindowError::InvalidZoneDirectory)
    );
    for root in [&missing, &overlong] {
        assert_eq!(
            DetectionWindow::from_zoneinfo_dir("bad", "01:00", "../UTC", root),
            Err(DetectionWindowError::InvalidTime)
        );
        assert_eq!(
            DetectionWindow::from_tzif_bytes("bad", "01:00", "../UTC", root, b"bad"),
            Err(DetectionWindowError::InvalidTime)
        );
    }

    let zones = frozen_zones("byte-admission");
    let utc = tzif_bytes(&fixture()["zoneinfo"]["files"][0]);
    assert_eq!(
        support::text(&fixture()["zoneinfo"]["files"][0]["key"]),
        "UTC"
    );
    let named_only =
        DetectionWindow::from_tzif_bytes("0:0", "1:00", "UTC", &missing, &utc).expect("no file");
    assert_eq!(named_only.source_path(), missing.join("UTC").as_path());
    assert!(
        named_only
            .contains(external("1970-01-01T00:00:00", 0))
            .unwrap()
    );
    assert_eq!(
        DetectionWindow::from_zoneinfo_dir("00:00", "01:00", "UTC", &missing),
        Err(DetectionWindowError::ZoneDataUnavailable)
    );
    let oversized = vec![0_u8; seeon_worker::detection_window::MAX_TZIF_BYTES + 1];
    assert_eq!(
        DetectionWindow::from_tzif_bytes("00:00", "01:00", "UTC", &missing, &oversized),
        Err(DetectionWindowError::ZoneDataTooLarge)
    );
    assert_eq!(
        DetectionWindow::from_tzif_bytes("00:00", "01:00", "UTC", &missing, b"not tzif"),
        Err(DetectionWindowError::InvalidZoneData)
    );
    let real = DetectionWindow::from_zoneinfo_dir("00:00", "01:00", "UTC", zones.path()).unwrap();
    let captured =
        DetectionWindow::from_tzif_bytes("00:00", "01:00", "UTC", zones.path(), &utc).unwrap();
    assert_eq!(captured, real);
}
