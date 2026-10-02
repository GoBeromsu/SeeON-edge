//! T26: replay-trace parity with Python. The writer file is checked against
//! the reviewed golden `d/replay-trace/7ec7c11be0c6ec04.lines.json` and its
//! manifest `writer_file_sha256`; the wire body against
//! `d/replay-trace/replay-wire-trace.canonical.json`. The xlang case feeds
//! the Rust outputs to Python `decode_jsonl` and `decode_replay_trace`, and
//! runs Python's own `ReplayTraceWriter` as the rotation oracle.

use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use seeon_ml_worker::config::env::Env;
use seeon_ml_worker::json::{Json, JsonError};
use seeon_ml_worker::trace_out::{
    BedPolygon, DEFAULT_MAX_BYTES, DEFAULT_ROTATION_COUNT, FrameKey, HEADER_LINE, Lifecycle,
    ReplayRow, ReplayTraceWriter, ReplayTrack, ReplayWire, RowError, Source, SourceEvent,
    TraceError, Truncation, WireError,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CAMERA: &str = "camera-replay";
const TRACE_FILE: &str = "7ec7c11be0c6ec04.jsonl";
const LINES_GOLDEN: &str = "d/replay-trace/7ec7c11be0c6ec04.lines.json";
const WIRE_GOLDEN: &str = "d/replay-trace/replay-wire-trace.canonical.json";

fn worker_wire(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire")
        .join(relative)
}

fn read_json(relative: &str) -> Value {
    let bytes = fs::read(worker_wire(relative)).expect("golden file is readable");
    serde_json::from_slice(&bytes).expect("golden file is JSON")
}

fn golden_entry(path: &str) -> Value {
    read_json("manifest.json")["goldens"]
        .as_array()
        .expect("manifest goldens is an array")
        .iter()
        .find(|entry| entry["path"] == path)
        .cloned()
        .expect("manifest lists the golden")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// A fresh, empty per-test directory; the test removes it when done.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("seeon-t26-{}-{name}", std::process::id()));
    match fs::remove_dir_all(&dir) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
            panic!("stale test directory is removable: {error:?}")
        }
        _ => {}
    }
    fs::create_dir_all(&dir).expect("test directory is creatable");
    dir
}

fn track(track_id: i128) -> ReplayTrack {
    let mut keypoints = [[0.0; 3]; 17];
    for (index, point) in (0u8..).zip(keypoints.iter_mut()) {
        *point = [
            f64::from(30 + index) / 100.0,
            f64::from(20 + 2 * index) / 100.0,
            0.9,
        ];
    }
    ReplayTrack {
        track_id,
        lifecycle: Lifecycle::Tracked,
        bbox: [0.25, 0.2, 0.55, 0.8, 0.91],
        keypoints,
    }
}

/// The golden input: one nvdcf frame row, one tracked COCO-17 track, one bed
/// polygon on 640x480; `seq` seconds after the epoch start.
fn golden_row(seq: u64) -> ReplayRow {
    ReplayRow {
        camera_id: CAMERA.to_owned(),
        seq,
        pts_ns: seq * 1_000_000_000,
        epoch: 0,
        source_event: SourceEvent::Frame,
        source: Source::Nvdcf,
        tracks: vec![track(1)],
        bed: Some(BedPolygon {
            id: "bed-1".to_owned(),
            polygon: vec![[0.1, 0.5], [0.9, 0.5], [0.9, 0.95], [0.1, 0.95]],
            image_size: [640, 480],
        }),
        night_window_active: false,
        frame_width: 640,
        frame_height: 480,
    }
}

/// `{"file_name", "lines", "trailing_newline"}` as the golden rebuilds a file.
fn rebuilt(path: &Path) -> Value {
    let text = fs::read_to_string(path).expect("trace file is UTF-8");
    let (body, trailing) = match text.strip_suffix('\n') {
        Some(body) => (body, true),
        None => (text.as_str(), false),
    };
    json!({
        "file_name": path.file_name().and_then(|name| name.to_str()),
        "lines": body.split('\n').collect::<Vec<_>>(),
        "trailing_newline": trailing,
    })
}

fn to_json(value: &Value) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(flag) => Json::Bool(*flag),
        Value::Number(number) => match (number.as_i64(), number.as_u64()) {
            (Some(int), _) => Json::Int(i128::from(int)),
            (None, Some(int)) => Json::Int(i128::from(int)),
            (None, None) => Json::Float(number.as_f64().expect("finite golden number")),
        },
        Value::String(text) => Json::Str(text.clone()),
        Value::Array(items) => Json::Array(items.iter().map(to_json).collect()),
        Value::Object(members) => Json::Object(
            members
                .iter()
                .map(|(key, item)| (key.clone(), to_json(item)))
                .collect(),
        ),
    }
}

fn frame_key(value: &Value) -> Option<FrameKey> {
    let parts = value.as_array()?;
    Some(FrameKey {
        worker_boot_id: parts[0].as_str().expect("boot id").to_owned(),
        camera_id: parts[1].as_str().expect("camera id").to_owned(),
        stream_epoch: parts[2].as_u64().expect("epoch"),
        seq: parts[3].as_u64().expect("seq"),
    })
}

/// The one-frame trace POSTed to `/replay` in `h/replay.json`.
fn golden_wire() -> ReplayWire {
    let cases = read_json("h/replay.json");
    let trace = &cases["cases"]["bad-body-missing-policy"]["request"]["body"]["trace"];
    let cut = &trace["truncation"];
    let count = |key: &str| cut[key].as_u64().expect("truncation counter");
    ReplayWire {
        camera_id: trace["camera_id"].as_str().expect("camera").to_owned(),
        frames: trace["frames"]
            .as_array()
            .expect("frames")
            .iter()
            .map(to_json)
            .collect(),
        truncation: Truncation {
            handoff_dropped_frames: count("handoff_dropped_frames"),
            pruned_frames: count("pruned_frames"),
            oldest_retained_seq: cut["oldest_retained_seq"].as_u64(),
            newest_retained_seq: cut["newest_retained_seq"].as_u64(),
            persistence_failed_frames: count("persistence_failed_frames"),
            retention_blocked_frames: count("retention_blocked_frames"),
            oldest_retained_key: frame_key(&cut["oldest_retained_key"]),
            newest_retained_key: frame_key(&cut["newest_retained_key"]),
            detail_unavailable_reason: cut["detail_unavailable_reason"].as_str().map(str::to_owned),
        },
    }
}

#[test]
fn writer_file_matches_python_golden() {
    let dir = fresh_dir("golden");
    let mut writer =
        ReplayTraceWriter::new(&dir, CAMERA, DEFAULT_MAX_BYTES, DEFAULT_ROTATION_COUNT)
            .expect("writer opens");
    assert_eq!(writer.append(&golden_row(1)), Ok(true));
    assert_eq!(
        writer.path().file_name().and_then(|name| name.to_str()),
        Some(TRACE_FILE)
    );
    assert_eq!(rebuilt(writer.path()), read_json(LINES_GOLDEN));
    let bytes = fs::read(writer.path()).expect("trace file is readable");
    assert_eq!(
        sha256_hex(&bytes),
        golden_entry(LINES_GOLDEN)["writer_file_sha256"]
    );
    assert_eq!(
        (writer.written_rows_total(), writer.dropped_rows_total()),
        (1, 0)
    );
    fs::remove_dir_all(&dir).expect("test directory is removable");
}

#[test]
fn wire_canonical_matches_python_golden() {
    let canonical = golden_wire()
        .canonical_json()
        .expect("golden trace encodes");
    let golden = fs::read(worker_wire(WIRE_GOLDEN)).expect("wire golden is readable");
    assert_eq!(canonical.as_bytes(), golden.as_slice());
    assert_eq!(
        sha256_hex(canonical.as_bytes()),
        golden_entry(WIRE_GOLDEN)["sha256"]
    );
}
#[test]
fn u64_max_track_id_serialises_as_an_exact_integer() {
    let mut row = golden_row(1);
    row.tracks[0].track_id = i128::from(u64::MAX);
    let line = row
        .encode_line()
        .expect("max native id is a valid track id");
    assert!(line.contains("\"track_id\":18446744073709551615,"));
    assert!(!line.contains("18446744073709552000"));
}

#[test]
fn non_finite_or_incomplete_wire_is_refused() {
    for number in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut wire = golden_wire();
        let Json::Object(members) = &mut wire.frames[0] else {
            panic!("golden frame is an object")
        };
        members.push(("score".to_owned(), Json::Float(number)));
        assert_eq!(
            wire.canonical_json(),
            Err(WireError::Json(JsonError::NonFinite))
        );
    }
    let mut empty_camera = golden_wire();
    empty_camera.camera_id.clear();
    assert_eq!(empty_camera.canonical_json(), Err(WireError::EmptyCamera));
    let mut no_frames = golden_wire();
    no_frames.frames.clear();
    assert_eq!(no_frames.canonical_json(), Err(WireError::NoFrames));
    let mut scalar_frame = golden_wire();
    scalar_frame.frames.push(Json::Int(1));
    assert_eq!(
        scalar_frame.canonical_json(),
        Err(WireError::FrameNotObject)
    );
}

/// One contract violation applied to an otherwise valid golden row.
type Mutation = fn(&mut ReplayRow);

#[test]
fn row_contract_refusals_are_typed() {
    let cases: [(Mutation, RowError); 13] = [
        (|row| row.tracks[0].bbox[4] = 1.5, RowError::UnitValue),
        (|row| row.tracks[0].bbox[0] = f64::NAN, RowError::UnitValue),
        (
            |row| row.tracks[0].keypoints[3][1] = -0.1,
            RowError::UnitValue,
        ),
        (|row| row.tracks[0].bbox[0] = 0.6, RowError::BboxOrder),
        (|row| row.tracks[0].bbox[3] = 0.1, RowError::BboxOrder),
        (|row| row.frame_width = 0, RowError::FrameSize),
        (|row| row.camera_id.clear(), RowError::CameraId),
        (
            |row| row.bed.as_mut().expect("bed").id.clear(),
            RowError::BedPolygonId,
        ),
        (
            |row| row.bed.as_mut().expect("bed").polygon.truncate(2),
            RowError::BedPolygonPoints,
        ),
        (
            |row| row.bed.as_mut().expect("bed").polygon[1][0] = 1.2,
            RowError::UnitValue,
        ),
        (
            |row| row.bed.as_mut().expect("bed").image_size[1] = 0,
            RowError::BedImageSize,
        ),
        (
            |row| row.source_event = SourceEvent::Open,
            RowError::ControlRowTracks,
        ),
        (|row| row.tracks.push(track(1)), RowError::DuplicateTrackId),
    ];
    let dir = fresh_dir("refusals");
    let mut writer =
        ReplayTraceWriter::new(&dir, CAMERA, DEFAULT_MAX_BYTES, DEFAULT_ROTATION_COUNT)
            .expect("writer opens");
    for (mutate, expected) in cases {
        let mut row = golden_row(1);
        mutate(&mut row);
        assert_eq!(row.encode_line(), Err(expected));
        assert_eq!(writer.append(&row), Err(TraceError::Row(expected)));
    }
    assert!(!writer.path().exists(), "a refused row writes nothing");
    assert_eq!(
        (writer.written_rows_total(), writer.dropped_rows_total()),
        (0, 0)
    );
    let mut control = golden_row(2);
    control.source_event = SourceEvent::Reconnect;
    control.tracks.clear();
    assert!(
        control.encode_line().is_ok(),
        "a trackless control row is valid"
    );
    fs::remove_dir_all(&dir).expect("test directory is removable");
}

fn line_len(row: &ReplayRow) -> u64 {
    row.encode_line().expect("valid row").len() as u64
}

/// Runs one bounded writer over `rows`; returns what each append answered,
/// every file under the root by name, and the two counters.
fn run_scenario(root: &Path, max_bytes: u64, rotation_count: u32, rows: &[ReplayRow]) -> Value {
    let mut writer =
        ReplayTraceWriter::new(root, CAMERA, max_bytes, rotation_count).expect("writer opens");
    let appends: Vec<bool> = rows
        .iter()
        .map(|row| writer.append(row).expect("append does not fail"))
        .collect();
    let mut files = serde_json::Map::new();
    let mut entries: Vec<_> = fs::read_dir(root)
        .expect("root is listable")
        .map(|entry| entry.expect("root entry").path())
        .collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("name");
        let text = fs::read_to_string(&path).expect("trace file is UTF-8");
        files.insert(name.to_owned(), Value::String(text));
    }
    json!({
        "appends": appends,
        "files": files,
        "written": writer.written_rows_total(),
        "dropped": writer.dropped_rows_total(),
    })
}

fn file_of(rows: &[&ReplayRow]) -> String {
    let mut text = HEADER_LINE.to_owned();
    for row in rows {
        text.push_str(&row.encode_line().expect("valid row"));
    }
    text
}

#[test]
fn writer_bounds_and_rotation() {
    let dir = fresh_dir("bounds");
    assert_eq!(
        ReplayTraceWriter::new(&dir, CAMERA, 0, DEFAULT_ROTATION_COUNT).map(|_| ()),
        Err(TraceError::Bounds)
    );
    let rows: Vec<ReplayRow> = (1..=7).map(golden_row).collect();
    let header = HEADER_LINE.len() as u64;
    let line = line_len(&rows[0]);
    assert!(
        rows.iter().all(|row| line_len(row) == line),
        "equal-length rows"
    );

    let two_per_file = run_scenario(&dir.join("a"), header + 2 * line, 2, &rows);
    let expected = json!({
        "appends": vec![true; 7],
        "files": {
            TRACE_FILE: file_of(&[&rows[6]]),
            format!("{TRACE_FILE}.1"): file_of(&[&rows[4], &rows[5]]),
            format!("{TRACE_FILE}.2"): file_of(&[&rows[2], &rows[3]]),
        },
        "written": 7,
        "dropped": 0,
    });
    assert_eq!(two_per_file, expected, "oldest rotated file is dropped");

    let no_history = run_scenario(&dir.join("b"), header + line, 0, &rows[..3]);
    let expected = json!({
        "appends": vec![true; 3],
        "files": {TRACE_FILE: file_of(&[&rows[2]])},
        "written": 3,
        "dropped": 0,
    });
    assert_eq!(no_history, expected, "rotation_count 0 truncates in place");

    let oversize = run_scenario(&dir.join("c"), header + line - 1, 1, &rows[..2]);
    let expected = json!({"appends": [false, false], "files": {}, "written": 0, "dropped": 2});
    assert_eq!(
        oversize, expected,
        "a row that cannot fit a fresh file is refused"
    );

    let mut wide = golden_row(8);
    wide.tracks.push(track(2));
    let rotated_then_refused =
        run_scenario(&dir.join("d"), header + line, 1, &[rows[0].clone(), wide]);
    let expected = json!({
        "appends": [true, false],
        "files": {format!("{TRACE_FILE}.1"): file_of(&[&rows[0]])},
        "written": 1,
        "dropped": 1,
    });
    assert_eq!(
        rotated_then_refused, expected,
        "rotation happens before the refusal"
    );
    fs::remove_dir_all(&dir).expect("test directory is removable");
}

#[test]
fn trace_path_escaping_root_is_refused() {
    let dir = fresh_dir("escape");
    let outside = dir.join("outside");
    fs::create_dir_all(&outside).expect("outside directory");
    let root = dir.join("root");
    fs::create_dir_all(&root).expect("root directory");
    symlink(outside.join("missing.jsonl"), root.join(TRACE_FILE)).expect("dangling link");
    assert_eq!(
        ReplayTraceWriter::new(&root, CAMERA, DEFAULT_MAX_BYTES, 1).map(|_| ()),
        Err(TraceError::Escape)
    );
    fs::remove_file(root.join(TRACE_FILE)).expect("link is removable");
    fs::write(outside.join("existing.jsonl"), b"").expect("outside file");
    symlink(outside.join("existing.jsonl"), root.join(TRACE_FILE)).expect("resolving link");
    assert_eq!(
        ReplayTraceWriter::new(&root, CAMERA, DEFAULT_MAX_BYTES, 1).map(|_| ()),
        Err(TraceError::Escape)
    );
    fs::remove_file(root.join(TRACE_FILE)).expect("link is removable");

    let mut writer = ReplayTraceWriter::new(&root, CAMERA, DEFAULT_MAX_BYTES, 1).expect("opens");
    symlink(outside.join("existing.jsonl"), root.join(TRACE_FILE)).expect("late link");
    assert_eq!(writer.append(&golden_row(1)), Err(TraceError::Escape));
    let untouched = fs::read(outside.join("existing.jsonl")).expect("outside file");
    assert!(untouched.is_empty(), "nothing is written through the link");
    assert_eq!(writer.written_rows_total(), 0);
    fs::remove_dir_all(&dir).expect("test directory is removable");
}

#[test]
fn from_env_is_opt_in() {
    let key = "WORKER_REPLAY_TRACE_DIR";
    assert!(matches!(
        ReplayTraceWriter::from_env(&Env::new(), CAMERA),
        Ok(None)
    ));
    let blank = Env::from([(key.to_owned(), " \t".to_owned())]);
    assert!(matches!(
        ReplayTraceWriter::from_env(&blank, CAMERA),
        Ok(None)
    ));
    let dir = fresh_dir("env");
    let set = Env::from([(key.to_owned(), dir.join("trace").display().to_string())]);
    let writer = ReplayTraceWriter::from_env(&set, CAMERA)
        .expect("writer opens")
        .expect("a set directory enables the trace");
    let root = fs::canonicalize(dir.join("trace")).expect("root exists");
    assert_eq!(writer.path(), root.join(TRACE_FILE));
    fs::remove_dir_all(&dir).expect("test directory is removable");
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

/// argv: rows file, wire file, scenarios. Decodes both with the Python
/// contracts, re-encodes, and replays each scenario through Python's writer.
const PYTHON_ORACLE: &str = r#"
import json, sys, tempfile
from pathlib import Path
from contracts.replay_trace import decode_jsonl, encode_jsonl
from shared.events.replay_wire import decode_replay_trace
from worker.pipeline.trace.replay_trace_writer import ReplayTraceWriter

header, rows = decode_jsonl(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert rows[-1].tracks[0].track_id == 18446744073709551615
wire = decode_replay_trace(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
results = []
for scenario in json.loads(sys.argv[3]):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "trace"
        writer = ReplayTraceWriter(
            root,
            scenario["camera_id"],
            max_bytes=scenario["max_bytes"],
            rotation_count=scenario["rotation_count"],
        )
        appends = [writer.append(rows[index]) for index in scenario["rows"]]
        files = {path.name: path.read_text(encoding="utf-8") for path in sorted(root.iterdir())}
        results.append({
            "appends": appends,
            "files": files,
            "written": writer.written_rows_total,
            "dropped": writer.dropped_rows_total,
        })
print(json.dumps({
    "reencoded": encode_jsonl(header, list(rows)),
    "wire": wire.canonical_json(),
    "scenarios": results,
}))
"#;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn python_decodes_rust_outputs_and_rotates_alike() {
    let python = python();
    let dir = fresh_dir("xlang");
    let mut rows: Vec<ReplayRow> = (1..=7).map(golden_row).collect();
    let mut control = golden_row(8);
    control.source_event = SourceEvent::Reconnect;
    control.source = Source::LegacyAssociation;
    control.tracks.clear();
    control.bed = None;
    control.night_window_active = true;
    rows.push(control);
    let mut wide = golden_row(9);
    let mut lost = track(2);
    lost.lifecycle = Lifecycle::Lost;
    wide.tracks.push(lost);
    rows.push(wide);
    let mut native_max = golden_row(10);
    native_max.tracks[0].track_id = i128::from(u64::MAX);
    rows.push(native_max);

    let mut all = ReplayTraceWriter::new(&dir.join("all"), CAMERA, DEFAULT_MAX_BYTES, 1)
        .expect("writer opens");
    for row in &rows {
        assert_eq!(all.append(row), Ok(true));
    }
    let wire_path = dir.join("wire.json");
    let wire = golden_wire()
        .canonical_json()
        .expect("golden trace encodes");
    fs::write(&wire_path, &wire).expect("wire file");

    let header = HEADER_LINE.len() as u64;
    let line = line_len(&rows[0]);
    let scenarios = [
        (header + 2 * line, 2, (0..7).collect::<Vec<usize>>()),
        (header + line, 0, vec![0, 1, 2]),
        (header + line - 1, 1, vec![0, 1]),
        (header + line, 1, vec![0, 8]),
        (header + line, 2, vec![0, 7, 7, 1, 8, 2]),
    ];
    let spec: Vec<Value> = scenarios
        .iter()
        .map(|(max_bytes, count, picks)| {
            json!({"camera_id": CAMERA, "max_bytes": max_bytes, "rotation_count": count, "rows": picks})
        })
        .collect();
    let output = Command::new(&python)
        .arg("-c")
        .arg(PYTHON_ORACLE)
        .arg(all.path())
        .arg(&wire_path)
        .arg(Value::Array(spec).to_string())
        .env(
            "PYTHONPATH",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        )
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .expect("run the Python oracle");
    assert!(output.status.success(), "the Python oracle exits cleanly");
    let report: Value = serde_json::from_slice(&output.stdout).expect("one JSON report");

    let written = fs::read_to_string(all.path()).expect("trace file is UTF-8");
    assert_eq!(
        report["reencoded"],
        Value::String(written),
        "decode_jsonl round-trips"
    );
    assert_eq!(
        report["wire"],
        Value::String(wire.clone()),
        "decode_replay_trace round-trips"
    );
    let golden = fs::read_to_string(worker_wire(WIRE_GOLDEN)).expect("wire golden");
    assert_eq!(wire, golden);
    for (index, (max_bytes, count, picks)) in scenarios.iter().enumerate() {
        let picked: Vec<ReplayRow> = picks.iter().map(|pick| rows[*pick].clone()).collect();
        let rust = run_scenario(&dir.join(format!("s{index}")), *max_bytes, *count, &picked);
        assert_eq!(
            rust, report["scenarios"][index],
            "scenario {index} rotates alike"
        );
    }
    fs::remove_dir_all(&dir).expect("test directory is removable");
}
