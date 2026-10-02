//! T25 (playback rendition and thumbnail) and T30 (durable write order)
//! against the real host tools.
//!
//! The binary is built in the CPU lane and run on the host lane with
//! `SEEON_TEST_FFMPEG`, `SEEON_TEST_FFPROBE`, `SEEON_TEST_PYTHON` (backend
//! oracle, `PYTHONPATH` at the repository), `SEEON_TEST_STRACE` (T30),
//! `SEEON_TEST_REPO` (the repository, for the reviewed goldens) and
//! `SEEON_TEST_WORK` (work directories). Sources are synthetic `lavfi` clips
//! made by the test's own ffmpeg call; rendition facts are read back with the
//! test's own ffprobe call and MP4 atom walk.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};

use seeon_ml_worker::clips::manifest::{ClipMetadata, MediaFacts};
use seeon_ml_worker::clips::publish::{MANIFEST_FILE, MEDIA_FILE, Publisher};
use seeon_ml_worker::clips::rendition::thumbnail::seek_offset;
use seeon_ml_worker::clips::rendition::{
    ATTESTATION_FILE, RENDITION_PREFIX, RenditionError, TEMP_FILE, THUMBNAIL_FILE, Tools,
    write_playback_rendition, write_thumbnail,
};
use seeon_ml_worker::clips::sealed::{
    SIDECAR_DIR, SealedClip, SealedContributor, SealedEvent, SealedSidecars,
};
use seeon_ml_worker::clips::store::ClipStore;
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::delivery::DeliveryQueue;

const CAMERA: &str = "cmsnw6rjc01vhlh01oswn99yq";
const EVENT_REF: &str = "00000000-0000-4000-8000-00000000e0a1";
const T30_CLIP: &str = "cmsnw6rjc01vhlh01oswn99yq-20260820T172057197192Z-00a100000005";

/// Backend oracle: the playback identity the backend would serve, or the
/// backend's view of the clip thumbnail.
const ORACLE: &str = r#"
import json, sys
from pathlib import Path
from backend.app.features.clips.store import ClipStore
mode, root, clip_id = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
store = ClipStore(root)
located = store.locate_manifest(clip_id)
if mode == "playback":
    identity = store.open_located_playback_identity(located)
    out = {
        key: getattr(identity, key)
        for key in ("served_kind", "served_media_sha256", "original_sha256", "served_pts_identical")
    }
    out["playback_codec"] = store.playback_codec(located)
else:
    from backend.app.features.clips import thumbnail_files
    manifest = Path(sys.argv[4])
    out = {
        "available": thumbnail_files.thumbnail_file_state(root, manifest).available,
        "path": str(thumbnail_files.contained_thumbnail_path(root, manifest)),
    }
print(json.dumps(out, sort_keys=True))
"#;

fn required(name: &str) -> PathBuf {
    PathBuf::from(env::var_os(name).unwrap_or_else(|| panic!("{name} is required")))
}

fn tools() -> Tools {
    Tools {
        ffmpeg: required("SEEON_TEST_FFMPEG"),
        ffprobe: required("SEEON_TEST_FFPROBE"),
        ..Tools::default()
    }
}

fn work(test: &str) -> PathBuf {
    let base = env::var_os("SEEON_TEST_WORK")
        .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from);
    let work = base.join(format!("clips_host-{test}"));
    if work.exists() {
        fs::remove_dir_all(&work).expect("clear previous run");
    }
    fs::create_dir_all(&work).expect("work dir");
    work
}

fn wire() -> PathBuf {
    env::var_os("SEEON_TEST_REPO")
        .map_or_else(
            || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
            PathBuf::from,
        )
        .join("tests/fixtures/worker-wire")
}

fn read_json(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_slice(&bytes).expect("JSON document")
}

fn sha256_hex(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn clip_id(n: u8) -> String {
    format!("{CAMERA}-20260820T172057197192Z-00a10000000{n}")
}

#[derive(Clone, Copy)]
enum Source {
    /// Two MPEG-4 Part 2 frames on a 1/90000 track: needs a rendition.
    Mpeg4,
    /// The same two frames already in H.264: needs none.
    H264,
    /// Thirty seconds of 720p MPEG-4: outlives a short transcode deadline.
    Long,
    /// The `d/clip-thumbnail.json` source command.
    Thumbnail,
}

fn make_source(tools: &Tools, out: &Path, source: Source) {
    let spec: &[&str] = match source {
        Source::Mpeg4 => &[
            "testsrc=size=320x240:rate=10",
            "-frames:v",
            "2",
            "-c:v",
            "mpeg4",
            "-video_track_timescale",
            "90000",
        ],
        Source::H264 => &[
            "testsrc=size=320x240:rate=10",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            "-video_track_timescale",
            "90000",
        ],
        Source::Long => &["testsrc=size=1280x720:rate=30", "-t", "30", "-c:v", "mpeg4"],
        Source::Thumbnail => &["testsrc=size=320x240:rate=10", "-t", "2"],
    };
    let status = Command::new(&tools.ffmpeg)
        .args(["-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
        .args(spec)
        .args(["-pix_fmt", "yuv420p", "-fflags", "+bitexact", "-flags:v"])
        .args(["+bitexact", "-map_metadata", "-1"])
        .arg(out)
        .status()
        .expect("ffmpeg runs");
    assert!(status.success(), "ffmpeg made the source");
}

fn at(text: &str) -> Utc {
    Utc::parse(text).expect("RFC 3339 timestamp")
}

fn metadata(clip_id: &str) -> ClipMetadata {
    ClipMetadata {
        clip_id: clip_id.to_owned(),
        camera_id: CAMERA.to_owned(),
        facility_id: "facility-1".to_owned(),
        domain: "fall".to_owned(),
        event_type: "fall".to_owned(),
        event_refs: vec![EVENT_REF.to_owned()],
        detected_at: at("2026-08-20T17:20:58.197192+00:00"),
        started_at: at("2026-08-20T17:20:57.197192+00:00"),
        clip_start_at: at("2026-08-20T17:20:57.197192+00:00"),
        clip_end_at: at("2026-08-20T17:20:59.197192+00:00"),
        finalized_at: at("2026-08-20T17:21:00+00:00"),
        duration_ms: 2000,
        encoder: "ffmpeg".to_owned(),
        truncation_reasons: Vec::new(),
        extension: None,
    }
}

/// A READY clip published into `root` from a fresh synthetic source.
struct Clip {
    dir: PathBuf,
    manifest: PathBuf,
    sha256: String,
}

fn publish(tools: &Tools, root: &Path, clip_id: &str, source: Source) -> Clip {
    fs::create_dir_all(root).expect("store root");
    let store = ClipStore::new(root);
    let queue = DeliveryQueue::open(&root.join("delivery-queue"), true).expect("queue");
    let reservation = store.reserve(CAMERA, clip_id).expect("reserve");
    let artifact = reservation.artifact_path();
    make_source(tools, &artifact, source);
    let sha256 = sha256_hex(&artifact);
    let size = fs::metadata(&artifact).expect("artifact").len();
    let codec = if matches!(source, Source::H264) {
        "h264"
    } else {
        "mpeg4"
    };
    let facts = MediaFacts {
        sha256: sha256.clone(),
        size_bytes: i64::try_from(size).expect("size"),
        codec: codec.to_owned(),
        duration_ms: 2000,
    };
    let published = Publisher::new(&queue)
        .publish_ready(&reservation, &metadata(clip_id), &facts)
        .expect("publish READY");
    Clip {
        dir: store.clip_dir(clip_id),
        manifest: published.manifest_path,
        sha256,
    }
}

/// The first video stream and its packet timestamps, read by ffprobe.
#[derive(Debug, PartialEq)]
struct Stream {
    codec: String,
    pix_fmt: String,
    time_base: String,
    pts: Vec<i64>,
}

fn probe(tools: &Tools, file: &Path) -> Stream {
    let output = Command::new(&tools.ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .args([
            "stream=codec_name,pix_fmt,time_base:packet=pts",
            "-of",
            "json",
        ])
        .arg(file)
        .output()
        .expect("ffprobe runs");
    assert!(output.status.success(), "ffprobe reads {}", file.display());
    let report: Value = serde_json::from_slice(&output.stdout).expect("ffprobe JSON");
    let stream = &report["streams"][0];
    let text = |key: &str| stream[key].as_str().expect("stream field").to_owned();
    let packets = report["packets"].as_array().expect("packets");
    let mut pts: Vec<i64> = packets
        .iter()
        .map(|packet| packet["pts"].as_i64().expect("packet pts"))
        .collect();
    pts.sort_unstable();
    Stream {
        codec: text("codec_name"),
        pix_fmt: text("pix_fmt"),
        time_base: text("time_base"),
        pts,
    }
}

/// Top-level MP4 box types in file order.
fn top_level_boxes(file: &Path) -> Vec<String> {
    let bytes = fs::read(file).expect("MP4 bytes");
    let mut boxes = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let word = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().expect("u32"));
        let size = match word(offset) {
            0 => bytes.len() - offset,
            1 => {
                let wide = bytes[offset + 8..offset + 16]
                    .try_into()
                    .expect("largesize");
                usize::try_from(u64::from_be_bytes(wide)).expect("box size")
            }
            size => usize::try_from(size).expect("box size"),
        };
        assert!(size >= 8, "box at {offset} has a valid size");
        boxes.push(String::from_utf8_lossy(&bytes[offset + 4..offset + 8]).into_owned());
        offset += size;
    }
    assert_eq!(offset, bytes.len(), "boxes cover the file exactly");
    boxes
}

fn backend(mode: &str, root: &Path, clip_id: &str, extra: &[&Path]) -> Value {
    let output = Command::new(required("SEEON_TEST_PYTHON"))
        .arg("-c")
        .arg(ORACLE)
        .arg(mode)
        .arg(root)
        .arg(clip_id)
        .args(extra)
        .output()
        .expect("backend oracle runs");
    eprintln!("{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "backend oracle succeeds");
    serde_json::from_slice(&output.stdout).expect("oracle JSON")
}

fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .expect("clip dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .into_string()
                .expect("utf-8")
        })
        .collect();
    names.sort();
    names
}

fn assert_no_rendition(dir: &Path) {
    let names = names(dir);
    assert!(!names.iter().any(|name| name == TEMP_FILE), "no temp left");
    assert!(
        !names.iter().any(|name| name == ATTESTATION_FILE),
        "no attestation"
    );
    assert!(
        !names.iter().any(|name| name.starts_with(RENDITION_PREFIX)),
        "no rendition"
    );
}

#[test]
#[ignore = "requires SEEON_TEST_FFMPEG, SEEON_TEST_FFPROBE and SEEON_TEST_PYTHON"]
fn mpeg4_clip_gets_a_faststart_h264_rendition_with_identical_timestamps() {
    let tools = tools();
    let root = work("rendition").join("store");
    let id = clip_id(1);
    let clip = publish(&tools, &root, &id, Source::Mpeg4);

    let attestation = write_playback_rendition(&tools, &clip.dir)
        .expect("rendition published")
        .expect("an MPEG-4 source gets a rendition");

    let rendition = clip.dir.join(&attestation.rendition);
    let source = probe(&tools, &clip.dir.join(MEDIA_FILE));
    let output = probe(&tools, &rendition);
    assert_eq!(source.codec, "mpeg4");
    assert_eq!(
        (output.codec.as_str(), output.pix_fmt.as_str()),
        ("h264", "yuv420p")
    );
    assert_eq!(output.time_base, source.time_base);
    assert_eq!(output.pts, source.pts, "same frame count and timestamps");
    let boxes = top_level_boxes(&rendition);
    let moov = boxes.iter().position(|kind| kind == "moov").expect("moov");
    let mdat = boxes.iter().position(|kind| kind == "mdat").expect("mdat");
    assert!(moov < mdat, "faststart: moov before mdat in {boxes:?}");

    let written = read_json(&clip.dir.join(ATTESTATION_FILE));
    let golden = read_json(&wire().join("d/clip-playback/clip.playback-h264.json"));
    let stable = |value: &Value| {
        let mut object = value.as_object().expect("attestation object").clone();
        for key in ["rendition", "rendition_sha256", "source_sha256"] {
            assert!(object.remove(key).is_some(), "{key} present");
        }
        object
    };
    assert_eq!(stable(&written), stable(&golden));
    let rendition_sha256 = sha256_hex(&rendition);
    assert_eq!(written["source_sha256"], clip.sha256.as_str());
    assert_eq!(written["rendition_sha256"], rendition_sha256.as_str());
    let expected_name = format!("{RENDITION_PREFIX}{}.mp4", &rendition_sha256[..16]);
    assert_eq!(written["rendition"], expected_name.as_str());
    assert_eq!(attestation.rendition, expected_name);
    assert_eq!(attestation.rendition_sha256, rendition_sha256);
    assert!(!clip.dir.join(TEMP_FILE).exists(), "temp renamed away");

    let served = backend("playback", &root, &id, &[]);
    assert_eq!(served["served_kind"], "rendition");
    assert_eq!(served["served_media_sha256"], rendition_sha256.as_str());
    assert_eq!(served["original_sha256"], clip.sha256.as_str());
    assert_eq!(served["served_pts_identical"], true);
    assert_eq!(served["playback_codec"], "h264");
}

#[test]
#[ignore = "requires SEEON_TEST_FFMPEG and SEEON_TEST_FFPROBE"]
fn h264_clip_gets_no_rendition() {
    let tools = tools();
    let root = work("h264").join("store");
    let clip = publish(&tools, &root, &clip_id(2), Source::H264);
    let before = names(&clip.dir);

    assert_eq!(write_playback_rendition(&tools, &clip.dir), Ok(None));

    assert_no_rendition(&clip.dir);
    assert_eq!(names(&clip.dir), before, "nothing written");
    assert_eq!(sha256_hex(&clip.dir.join(MEDIA_FILE)), clip.sha256);
}

#[test]
#[ignore = "requires SEEON_TEST_FFMPEG and SEEON_TEST_FFPROBE"]
fn transcode_past_its_deadline_is_killed_and_leaves_no_temp() {
    let mut tools = tools();
    tools.transcode_deadline = Duration::from_millis(500);
    let root = work("deadline").join("store");
    let clip = publish(&tools, &root, &clip_id(3), Source::Long);
    let before = names(&clip.dir);

    assert_eq!(
        write_playback_rendition(&tools, &clip.dir),
        Err(RenditionError::Deadline)
    );

    assert_no_rendition(&clip.dir);
    assert_eq!(names(&clip.dir), before, "nothing left behind");
}

/// JPEG `(width, height)` from the baseline SOF0 segment.
fn sof0_size(bytes: &[u8]) -> (u16, u16) {
    assert_eq!(bytes.get(..2), Some(&[0xFF, 0xD8][..]), "JPEG SOI");
    let mut offset = 2usize;
    while offset + 9 <= bytes.len() {
        assert_eq!(bytes[offset], 0xFF, "segment marker at {offset}");
        let marker = bytes[offset + 1];
        if marker == 0xC0 {
            let height = u16::from_be_bytes([bytes[offset + 5], bytes[offset + 6]]);
            let width = u16::from_be_bytes([bytes[offset + 7], bytes[offset + 8]]);
            return (width, height);
        }
        let length = u16::from_be_bytes([bytes[offset + 2], bytes[offset + 3]]);
        offset += 2 + usize::from(length);
    }
    panic!("no SOF0 segment");
}

#[test]
#[ignore = "requires SEEON_TEST_FFMPEG, SEEON_TEST_FFPROBE and SEEON_TEST_PYTHON"]
fn thumbnail_is_a_640_wide_jpeg_the_backend_serves() {
    let tools = tools();
    let root = work("thumbnail").join("store");
    let id = clip_id(4);
    let clip = publish(&tools, &root, &id, Source::Thumbnail);
    assert_eq!(seek_offset(2000), "1.5");

    write_thumbnail(&tools, &clip.dir, 2000).expect("thumbnail written");

    let thumbnail = clip.dir.join(THUMBNAIL_FILE);
    let bytes = fs::read(&thumbnail).expect("thumbnail bytes");
    assert_eq!(sof0_size(&bytes), (640, 480));
    let state = backend("thumbnail", &root, &id, &[&clip.manifest]);
    assert_eq!(state["available"], true);
    assert_eq!(state["path"], thumbnail.to_str().expect("utf-8 path"));
}

/// The writes T30 traces, run in a child process under strace.
#[test]
#[ignore = "requires SEEON_TEST_FFMPEG and SEEON_TEST_FFPROBE; traced by t30_durable_writes_fsync_rename_then_fsync_dir"]
fn t30_inner_writes() {
    let tools = tools();
    let root = env::var_os("SEEON_T30_ROOT")
        .map_or_else(|| work("t30-inner").join("store"), PathBuf::from);
    let clip = publish(&tools, &root, T30_CLIP, Source::Mpeg4);
    write_playback_rendition(&tools, &clip.dir)
        .expect("rendition published")
        .expect("an MPEG-4 source gets a rendition");
    let sealed = SealedClip {
        clip_id: T30_CLIP.to_owned(),
        path: clip
            .dir
            .join(MEDIA_FILE)
            .to_str()
            .expect("utf-8")
            .to_owned(),
        duration_ms: 2000,
        boundary: "hard".to_owned(),
        contributors: vec![SealedContributor {
            event_ref: EVENT_REF.to_owned(),
            detected_at: "2026-08-20T17:20:58.197192+00:00".to_owned(),
        }],
    };
    let event = SealedEvent {
        domain: "fall".to_owned(),
        event_type: "fall".to_owned(),
        identity: "7".to_owned(),
        camera_id: CAMERA.to_owned(),
        facility_id: "facility-1".to_owned(),
        time_sec: 1.0,
        probability: Some(0.9),
    };
    let events = BTreeMap::from([(EVENT_REF.to_owned(), event)]);
    let sidecar = SealedSidecars::new(root.join(SIDECAR_DIR))
        .persist(&sealed, &events)
        .expect("sidecar persisted");
    for path in [clip.manifest, clip.dir.join(ATTESTATION_FILE), sidecar] {
        assert!(path.is_file(), "{} written", path.display());
    }
}

/// One traced system call.
struct Call {
    name: String,
    args: String,
    paths: Vec<String>,
    ret: i64,
}

fn quoted(args: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut chars = args.chars();
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        let mut text = String::new();
        while let Some(c) = chars.next() {
            match c {
                '"' => break,
                '\\' => text.extend(chars.next()),
                c => text.push(c),
            }
        }
        found.push(text);
    }
    found
}

fn parse_call(text: &str) -> Option<Call> {
    // strace pads short calls before the result: `fsync(3)        = 0`.
    let (call, ret) = text.rsplit_once(" = ")?;
    let call = call.trim_end().strip_suffix(')')?;
    let (name, args) = call.split_once('(')?;
    Some(Call {
        name: name.to_owned(),
        args: args.to_owned(),
        paths: quoted(args),
        ret: ret.split_whitespace().next()?.parse().ok()?,
    })
}

/// `strace -f` output as each thread's calls in order, with interrupted
/// calls (`<unfinished ...>` / `<... resumed>`) joined.
fn per_thread(log: &str) -> BTreeMap<u64, Vec<Call>> {
    let mut unfinished: BTreeMap<u64, String> = BTreeMap::new();
    let mut threads: BTreeMap<u64, Vec<Call>> = BTreeMap::new();
    for line in log.lines() {
        let Some((tid, rest)) = line.split_once(' ') else {
            continue;
        };
        let Ok(tid) = tid.parse::<u64>() else {
            continue;
        };
        let rest = rest.trim_start();
        let text = if let Some(head) = rest.strip_suffix(" <unfinished ...>") {
            unfinished.insert(tid, head.to_owned());
            continue;
        } else if let Some(resumed) = rest.strip_prefix("<... ") {
            let Some((_, tail)) = resumed.split_once(" resumed>") else {
                continue;
            };
            format!("{}{tail}", unfinished.remove(&tid).unwrap_or_default())
        } else {
            rest.to_owned()
        };
        if let Some(call) = parse_call(&text) {
            threads.entry(tid).or_default().push(call);
        }
    }
    threads
}

fn opened(call: Option<&Call>, path: &str, flag: &str) -> Option<i64> {
    call.filter(|call| {
        matches!(call.name.as_str(), "open" | "openat")
            && call.paths.first().map(String::as_str) == Some(path)
            && call.args.contains(flag)
            && call.ret >= 0
    })
    .map(|call| call.ret)
}

fn synced(call: Option<&Call>, fd: i64) -> bool {
    call.is_some_and(|call| {
        matches!(call.name.as_str(), "fsync" | "fdatasync")
            && call.args.trim().parse() == Ok(fd)
            && call.ret == 0
    })
}

/// The writer's own thread did, with nothing traced in between: open the
/// temp, fsync it, rename it over `target`, open the directory, fsync it.
fn assert_durable(threads: &BTreeMap<u64, Vec<Call>>, target: &Path) {
    let dir = target.parent().and_then(Path::to_str).expect("utf-8 dir");
    let name = target.file_name().and_then(|n| n.to_str()).expect("name");
    let temp = format!("{dir}/.{name}.tmp");
    let target = target.to_str().expect("utf-8 target");
    let renames: Vec<(&Vec<Call>, usize)> = threads
        .values()
        .flat_map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(move |(i, call)| (calls, i, call))
        })
        .filter(|(_, _, call)| {
            matches!(call.name.as_str(), "rename" | "renameat" | "renameat2")
                && call.paths == [temp.as_str(), target]
                && call.ret == 0
        })
        .map(|(calls, i, _)| (calls, i))
        .collect();
    assert_eq!(renames.len(), 1, "{target} is renamed into place once");
    let (calls, i) = renames[0];
    let near = |back: usize, ahead: usize| calls.get((i + ahead).checked_sub(back)?);
    let file = opened(near(2, 0), &temp, "O_CREAT")
        .unwrap_or_else(|| panic!("{target}: temp opened, then fsync, then rename"));
    assert!(
        synced(near(1, 0), file),
        "{target}: temp fsync before rename"
    );
    let dir_fd = opened(near(0, 1), dir, "O_DIRECTORY")
        .unwrap_or_else(|| panic!("{target}: directory opened right after rename"));
    assert!(synced(near(0, 2), dir_fd), "{target}: directory fsync");
}

#[test]
#[ignore = "requires SEEON_TEST_STRACE, SEEON_TEST_FFMPEG and SEEON_TEST_FFPROBE"]
fn t30_durable_writes_fsync_rename_then_fsync_dir() {
    let strace = required("SEEON_TEST_STRACE");
    let work = work("t30");
    let root = work.join("store");
    let log = work.join("strace.log");
    let status = Command::new(strace)
        .args(["-f", "-qq", "-s", "4096", "-e", "signal=none", "-e"])
        .arg("trace=open,openat,fsync,fdatasync,rename,renameat,renameat2")
        .arg("-o")
        .arg(&log)
        .arg(env::current_exe().expect("test binary"))
        .args(["--exact", "t30_inner_writes", "--include-ignored"])
        .arg("--test-threads=1")
        .env("SEEON_T30_ROOT", &root)
        .status()
        .expect("strace runs");
    assert!(status.success(), "traced writes succeed");

    let threads = per_thread(&fs::read_to_string(&log).expect("strace log"));
    let clip_dir = ClipStore::new(&root).clip_dir(T30_CLIP);
    assert_durable(&threads, &clip_dir.join(MANIFEST_FILE));
    assert_durable(&threads, &clip_dir.join(ATTESTATION_FILE));
    assert_durable(
        &threads,
        &root.join(SIDECAR_DIR).join(format!("{T30_CLIP}.json")),
    );
}
