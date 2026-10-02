//! Private tests of actual writer bytes and state, with literal input expectations.
//! Nested under camera so tests need no production-only accessors or constructors.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use seeon_deepstream_native::{FrameIdentity, MEDIA_MAX_OBJECTS, MEDIA_MAX_SOURCES, MediaBinding};
use seeon_worker::detection_window::DetectionWindow;
use serde_json::{Value, json};

use super::super::ReplayCapture;
use crate::config::pull::{ConfigSource, PulledConfig};
use crate::config::windows::AdmittedWindow;
use crate::json::Json;
use crate::policy::ingest::{Frame, ObservedPose};
use crate::relay::cameras::policies::resolve_detection_policies;
use crate::relay::cameras::{
    BedZoneRegion, DetectionWindow as WindowDefinition, RuntimeCamera, WorkerConfigPayload,
};
use crate::seam::{Clock, IdSource, RandomIds};
use crate::trace_out::{HEADER_LINE, ReplayTraceWriter};

struct FixedClock(u64);
impl Clock for FixedClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }
    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(self.0)
    }
    fn pause(&self, _: Duration) {}
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "seeon-replay-{name}-{}",
        RandomIds.uuid4().unwrap()
    ));
    fs::create_dir(&path).unwrap();
    path
}

fn camera(id: &str) -> RuntimeCamera {
    RuntimeCamera {
        camera_id: id.to_owned(),
        facility_id: "facility-1".to_owned(),
        rtsp_url: "rtsp://fixture.invalid/live".to_owned(),
        fps: 30.0,
        frame_stride: 1,
        decode_backend: None,
        label: None,
        bed_zone_regions: Vec::new(),
        bed_zone_image_width: None,
        bed_zone_image_height: None,
    }
}

fn config(cameras: Vec<RuntimeCamera>, domains: Option<Value>) -> PulledConfig {
    let roster: Vec<_> = cameras.iter().map(|camera| json!({
        "camera_id": camera.camera_id, "facility_id": camera.facility_id, "rtsp_url": camera.rtsp_url,
    })).collect();
    let mut document = json!({"config_version":1, "cameras":roster});
    if let Some(domains) = domains {
        document["domains"] = domains;
    }
    let payload = Json::from(&document);
    let config = WorkerConfigPayload::parse(&payload).unwrap();
    let ids: Vec<_> = cameras
        .iter()
        .map(|camera| camera.camera_id.clone())
        .collect();
    PulledConfig {
        directive: config.directive(),
        policies: resolve_detection_policies(config.detection_policies(), &ids).unwrap(),
        payload,
        config,
        cameras,
        windows: BTreeMap::new(),
        source: ConfigSource::Pulled,
        stale: false,
    }
}

fn window() -> AdmittedWindow {
    AdmittedWindow {
        definition: WindowDefinition {
            start: "22:00".into(),
            end: "06:00".into(),
            tz: "UTC".into(),
        },
        window: DetectionWindow::from_zoneinfo_dir(
            "22:00",
            "06:00",
            "UTC",
            Path::new("/usr/share/zoneinfo"),
        )
        .unwrap(),
    }
}

fn owner(root: &Path, count: usize) -> ReplayCapture {
    let cameras = (0..count).map(|id| camera(&format!("cam-{id}"))).collect();
    ReplayCapture::new(root, &config(cameras, None)).unwrap()
}

fn frame(source: u32, epoch: u64, generation: u64, pts: u64, live: &[u64]) -> Frame {
    Frame {
        identity: FrameIdentity {
            binding: MediaBinding {
                token: u64::from(source) + 1,
                generation,
                epoch,
            },
            source_id: source,
            sequence: 1,
            pts_ns: pts,
            pts_valid: 1,
            source_width: 100,
            source_height: 50,
            ..FrameIdentity::default()
        },
        width: 100,
        height: 50,
        live_track_ids: live.to_vec(),
        rows: BTreeMap::new(),
        time_sec: Some(pts as f64 / 1e9),
        frame_index: 1,
    }
}

fn pose(id: u64, confidence: f64) -> ObservedPose {
    let mut keypoints = vec![[0.0; 3]; 17];
    keypoints[0] = [10.0, 20.0, confidence];
    ObservedPose {
        track_id: id,
        bbox: [10.0, 5.0, 40.0, 25.0],
        confidence,
        keypoints,
    }
}

fn track(id: u64, lifecycle: &str, confidence: f64) -> Value {
    let mut points = [[0.0; 3]; 17];
    points[0] = [0.1, 0.4, confidence];
    json!({"track_id":id,"lifecycle":lifecycle,"bbox":[0.1,0.1,0.4,0.5,confidence],"keypoints":points})
}

fn expected(seq: u64, pts: u64, epoch: u64, event: &str, tracks: Vec<Value>) -> Value {
    json!({"camera_id":"cam-0","seq":seq,"pts_ns":pts,"epoch":epoch,"source_event":event,
        "source":"nvdcf","tracks":tracks,"bed_polygon_id":null,"bed_polygon":null,
        "bed_polygon_image_size":null,"night_window_active":false,"frame_width":100,"frame_height":50})
}

fn rows(capture: &ReplayCapture, index: usize) -> Vec<Value> {
    let text = fs::read_to_string(capture.cameras[index].writer.path()).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(HEADER_LINE.trim_end()));
    lines
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn actual_generation_and_epoch_changes_reconnect_without_clearing_history() {
    let root = scratch("lifecycle");
    let mut capture = owner(&root, 1);
    for (generation, epoch, pts) in [(7, 4, 100), (8, 4, 200), (8, 5, 300)] {
        capture.capture(
            &frame(0, epoch, generation, pts, &[3]),
            &[pose(3, 0.5)],
            &FixedClock(0),
        );
    }
    assert_eq!(
        rows(&capture, 0),
        vec![
            expected(0, 100, 4, "open", vec![]),
            expected(1, 100, 4, "frame", vec![track(3, "new", 0.5)]),
            expected(2, 200, 4, "reconnect", vec![]),
            expected(3, 200, 4, "frame", vec![track(3, "tracked", 0.5)]),
            expected(4, 300, 5, "reconnect", vec![]),
            expected(5, 300, 5, "frame", vec![track(3, "tracked", 0.5)]),
        ]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_order_duplicate_last_value_shadow_and_lost_use_complete_live_ids() {
    let root = scratch("tracks");
    let mut capture = owner(&root, 1);
    capture.capture(
        &frame(0, 1, 1, 100, &[9, 2, u64::MAX]),
        &[pose(9, 0.5), pose(2, 0.5), pose(u64::MAX, 0.5)],
        &FixedClock(0),
    );
    assert_eq!(
        rows(&capture, 0)[1],
        expected(
            1,
            100,
            1,
            "frame",
            vec![
                track(9, "new", 0.5),
                track(2, "new", 0.5),
                track(u64::MAX, "new", 0.5)
            ]
        )
    );
    capture.capture(
        &frame(0, 1, 1, 200, &[9, u64::MAX, 7]),
        &[pose(9, 0.25), pose(9, 0.75)],
        &FixedClock(0),
    );
    assert_eq!(
        rows(&capture, 0)[2],
        expected(
            2,
            200,
            1,
            "frame",
            vec![
                track(9, "tracked", 0.75),
                track(2, "lost", 0.5),
                track(u64::MAX, "shadow", 0.5)
            ]
        )
    );
    capture.capture(&frame(0, 1, 1, 300, &[u64::MAX]), &[], &FixedClock(0));
    assert_eq!(
        rows(&capture, 0)[3],
        expected(
            3,
            300,
            1,
            "frame",
            vec![track(9, "lost", 0.75), track(u64::MAX, "shadow", 0.5)]
        )
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_observed_geometry_cannot_poison_previous_shadow() {
    let root = scratch("invalid");
    let mut capture = owner(&root, 1);
    capture.capture(&frame(0, 1, 1, 100, &[1]), &[pose(1, 0.5)], &FixedClock(0));
    capture.capture(&frame(0, 1, 1, 200, &[1]), &[pose(1, 1.5)], &FixedClock(0));
    assert_eq!(rows(&capture, 0).len(), 2);
    capture.capture(&frame(0, 1, 1, 300, &[1]), &[], &FixedClock(0));
    assert_eq!(
        rows(&capture, 0)[2],
        expected(2, 300, 1, "frame", vec![track(1, "shadow", 0.5)])
    );
    assert_eq!(capture.cameras[0].failures, 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn source_loss_requires_saved_metadata_and_rearms_only_after_capture() {
    let root = scratch("loss");
    let mut capture = owner(&root, 2);
    capture.processed(1, Duration::ZERO);
    capture.wait_empty(Duration::from_secs(5));
    assert!(!capture.cameras[1].writer.path().exists());
    capture.capture(&frame(0, 2, 1, 500, &[]), &[], &FixedClock(0));
    capture.processed(0, Duration::from_millis(1000));
    capture.processed(1, Duration::from_millis(1400));
    capture.wait_empty(Duration::from_millis(1499));
    assert_eq!(rows(&capture, 0).len(), 2);
    capture.wait_empty(Duration::from_millis(1500));
    assert_eq!(rows(&capture, 0)[2], expected(2, 500, 2, "lost", vec![]));
    capture.wait_empty(Duration::from_secs(9));
    assert_eq!(rows(&capture, 0).len(), 3);
    capture.capture(&frame(0, 2, 1, 800, &[]), &[], &FixedClock(0));
    capture.processed(0, Duration::from_millis(9000));
    capture.wait_empty(Duration::from_millis(9499));
    assert_eq!(rows(&capture, 0).len(), 4);
    capture.wait_empty(Duration::from_millis(9500));
    assert_eq!(rows(&capture, 0)[4], expected(4, 800, 2, "lost", vec![]));
    assert!(!capture.cameras[1].writer.path().exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn persisted_first_polygon_keeps_its_dimensions_or_real_source_fallback() {
    for dimensions in [Some((200, 100)), None, Some((0, 0))] {
        let root = scratch("polygon");
        let mut cam = camera("cam-0");
        cam.bed_zone_regions = vec![
            BedZoneRegion {
                id: "first".into(),
                origin: "manual".into(),
                polygon: vec![(10, 5), (40, 5), (40, 25)],
            },
            BedZoneRegion {
                id: "second".into(),
                origin: "manual".into(),
                polygon: vec![(0, 0), (1, 0), (1, 1)],
            },
        ];
        cam.bed_zone_image_width = dimensions.map(|size| size.0);
        cam.bed_zone_image_height = dimensions.map(|size| size.1);
        let mut capture = ReplayCapture::new(&root, &config(vec![cam], None)).unwrap();
        capture.capture(&frame(0, 1, 1, 10, &[]), &[], &FixedClock(0));
        let mut row = expected(1, 10, 1, "frame", vec![]);
        row["bed_polygon_id"] = json!("persisted");
        if dimensions == Some((200, 100)) {
            row["bed_polygon"] = json!([[0.05, 0.05], [0.2, 0.05], [0.2, 0.25]]);
            row["bed_polygon_image_size"] = json!([200, 100]);
        } else {
            row["bed_polygon"] = json!([[0.1, 0.1], [0.4, 0.1], [0.4, 0.5]]);
            row["bed_polygon_image_size"] = json!([100, 50]);
        }
        assert_eq!(rows(&capture, 0)[1], row);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn malformed_existing_polygon_is_not_silently_omitted() {
    for polygon in [vec![], vec![(0, 0), (1, 1)]] {
        let root = scratch("bad-polygon");
        let mut cam = camera("cam-0");
        cam.bed_zone_regions.push(BedZoneRegion {
            id: "bad".into(),
            origin: "manual".into(),
            polygon,
        });
        let mut capture = ReplayCapture::new(&root, &config(vec![cam], None)).unwrap();
        capture.capture(&frame(0, 1, 1, 10, &[]), &[], &FixedClock(0));
        assert_eq!(rows(&capture, 0), vec![expected(0, 10, 1, "open", vec![])]);
        assert_eq!(capture.cameras[0].failures, 1);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn only_enabled_admitted_bed_window_controls_trace_flag() {
    for (domains, key, expected_night) in [
        (None, None, false),
        (None, Some("fall"), false),
        (
            Some(json!({"bed_exit":{"enabled":false}})),
            Some("bed_exit"),
            false,
        ),
        (None, Some("bed_exit"), true),
    ] {
        let root = scratch("window");
        let mut cfg = config(vec![camera("cam-0")], domains);
        if let Some(key) = key {
            cfg.windows.insert(key.into(), window());
        }
        let mut capture = ReplayCapture::new(&root, &cfg).unwrap();
        capture.capture(&frame(0, 1, 1, 10, &[]), &[], &FixedClock(23 * 3600));
        assert_eq!(rows(&capture, 0)[1]["night_window_active"], expected_night);
        capture.capture(&frame(0, 1, 1, 20, &[]), &[], &FixedClock(12 * 3600));
        assert_eq!(rows(&capture, 0)[2]["night_window_active"], false);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn clock_refusal_consumes_sequence_and_keeps_history_but_never_writes_false() {
    let root = scratch("clock");
    let mut cfg = config(vec![camera("cam-0")], None);
    cfg.windows.insert("bed_exit".into(), window());
    let mut capture = ReplayCapture::new(&root, &cfg).unwrap();
    capture.capture(
        &frame(0, 1, 1, 10, &[1]),
        &[pose(1, 0.5)],
        &FixedClock(253_402_300_800),
    );
    assert_eq!(rows(&capture, 0), vec![expected(0, 10, 1, "open", vec![])]);
    assert_eq!(capture.cameras[0].failures, 1);
    capture.capture(
        &frame(0, 1, 1, 20, &[1]),
        &[pose(1, 0.5)],
        &FixedClock(12 * 3600),
    );
    assert_eq!(
        rows(&capture, 0)[1],
        expected(2, 20, 1, "frame", vec![track(1, "tracked", 0.5)])
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn actual_io_refusal_recovers_without_repeating_open_or_sequence() {
    let root = scratch("io");
    let mut capture = owner(&root, 1);
    let path = capture.cameras[0].writer.path().to_owned();
    fs::create_dir(&path).unwrap();
    capture.capture(&frame(0, 1, 1, 10, &[1]), &[pose(1, 0.5)], &FixedClock(0));
    assert_eq!(capture.cameras[0].failures, 1);
    capture.processed(0, Duration::ZERO);
    capture.wait_empty(Duration::from_secs(1));
    assert!(!capture.cameras[0].source_lost); // No invented initial 1x1 metadata.
    fs::remove_dir(path).unwrap();
    capture.capture(&frame(0, 1, 1, 20, &[1]), &[pose(1, 0.5)], &FixedClock(0));
    assert_eq!(
        rows(&capture, 0),
        vec![expected(1, 20, 1, "frame", vec![track(1, "new", 0.5)])]
    );
    assert_eq!(capture.cameras[0].failures, 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn writer_drop_continues_real_metadata_bookkeeping_without_capture_error() {
    let root = scratch("drop");
    let mut capture = owner(&root, 1);
    capture.cameras[0].writer = ReplayTraceWriter::new(&root, "cam-0", 32, 3).unwrap();
    capture.capture(&frame(0, 1, 1, 10, &[]), &[], &FixedClock(0));
    assert_eq!(capture.cameras[0].writer.dropped_rows_total(), 2);
    assert_eq!(capture.cameras[0].failures, 0);
    assert_eq!(capture.cameras[0].seq, Some(2));
    assert_eq!(capture.cameras[0].dims, Some((100, 50)));
    assert_eq!(capture.cameras[0].last_pts_ns, 10);
    assert!(!capture.cameras[0].writer.path().exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn maximum_sequence_is_emitted_once_without_wrap_or_counter_panic() {
    let root = scratch("seq");
    let mut capture = owner(&root, 1);
    capture.cameras[0].seq = Some(u64::MAX);
    capture.capture(&frame(0, 1, 1, 10, &[]), &[], &FixedClock(0));
    assert_eq!(
        rows(&capture, 0),
        vec![expected(u64::MAX, 10, 1, "open", vec![])]
    );
    assert_eq!(capture.cameras[0].seq, None);
    assert_eq!(capture.cameras[0].failures, 1);
    capture.cameras[0].failures = u64::MAX;
    capture.capture(&frame(0, 1, 1, 20, &[]), &[], &FixedClock(0));
    assert_eq!(capture.cameras[0].failures, u64::MAX);
    assert_eq!(rows(&capture, 0).len(), 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_pts_is_not_zero_and_absent_pts_is_zero() {
    let root = scratch("pts");
    let mut capture = owner(&root, 1);
    let mut invalid = frame(0, 1, 1, 99, &[]);
    invalid.identity.pts_valid = 2;
    capture.capture(&invalid, &[], &FixedClock(0));
    assert!(!capture.cameras[0].writer.path().exists());
    invalid.identity.pts_valid = 0;
    capture.capture(&invalid, &[], &FixedClock(0));
    assert_eq!(
        rows(&capture, 0),
        vec![
            expected(0, 0, 1, "open", vec![]),
            expected(1, 0, 1, "frame", vec![])
        ]
    );
    assert_eq!(capture.cameras[0].failures, 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_domains_and_excessive_roster_refuse_before_creating_writer_root() {
    let root = scratch("admission");
    let mut invalid = config(vec![camera("cam-0")], None);
    // Replacement lists belong to camera entries, not the global override object.
    invalid.payload = Json::from(&json!({
        "config_version": 1,
        "cameras": [{
            "camera_id": "cam-0", "facility_id": "facility-1",
            "rtsp_url": "rtsp://fixture.invalid/live",
            "domains": ["unknown-replacement"]
        }]
    }));
    invalid.config = WorkerConfigPayload::parse(&invalid.payload).unwrap();
    assert!(invalid.config.domain_selection().resolve().is_err());
    let path = root.join("invalid-domains");
    assert!(ReplayCapture::new(&path, &invalid).is_err());
    assert!(!path.exists());
    let excessive = config(
        (0..=MEDIA_MAX_SOURCES)
            .map(|id| camera(&format!("cam-{id}")))
            .collect(),
        None,
    );
    let path = root.join("excessive-roster");
    assert!(ReplayCapture::new(&path, &excessive).is_err());
    assert!(!path.exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn observations_cannot_exceed_or_invent_the_accepted_live_roster() {
    let root = scratch("capacity");
    let mut capture = owner(&root, 1);
    capture.capture(&frame(0, 1, 1, 10, &[]), &[pose(1, 0.5)], &FixedClock(0));
    let oversized = vec![pose(1, 0.5); MEDIA_MAX_OBJECTS + 1];
    capture.capture(&frame(0, 1, 1, 20, &[1]), &oversized, &FixedClock(0));
    assert!(capture.cameras[0].tracks.is_empty());
    assert_eq!(capture.cameras[0].failures, 2);
    assert_eq!(rows(&capture, 0), vec![expected(0, 10, 1, "open", vec![])]);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn integer_box_zero_is_canonical_without_changing_float_confidence_sign() {
    let root = scratch("integer-zero");
    let mut capture = owner(&root, 1);
    let mut observed = pose(1, -0.0);
    observed.bbox = [-0.0, -0.0, 40.0, 25.0];
    capture.capture(&frame(0, 1, 1, 10, &[1]), &[observed], &FixedClock(0));
    let text = fs::read_to_string(capture.cameras[0].writer.path()).unwrap();
    assert!(text.contains("\"bbox\":[0.0,0.0,0.4,0.5,-0.0]"));
    assert!(text.contains("\"keypoints\":[[0.1,0.4,-0.0],"));
    fs::remove_dir_all(root).unwrap();
}
