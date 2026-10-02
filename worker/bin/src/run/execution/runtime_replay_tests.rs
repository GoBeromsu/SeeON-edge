//! CPU queue integration only: these tests do not prove SDK or shutdown timing.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::{Duration, SystemTime};

use serde_json::{Value, json};

use super::{LiveSink, RuntimeError, budget, camera, packet, pump_queue_turn};
use crate::config::pull::{ConfigSource, PulledConfig};
use crate::json::Json;
use crate::msg::{FallRequest, FallResponse, PosePacket};
use crate::records::id::sha256_hex;
use crate::relay::cameras::WorkerConfigPayload;
use crate::relay::cameras::policies::resolve_detection_policies;
use crate::run::pump::{CameraPolicy, PolicyPump};
use crate::run::replay::ReplayCapture;
use crate::seam::{Clock, IdSource, RandomIds};

#[derive(Default)]
struct TestClock(AtomicU64);

impl TestClock {
    fn at(&self, millis: u64) {
        self.0.store(millis, Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn monotonic(&self) -> Duration {
        Duration::from_millis(self.0.load(Ordering::SeqCst))
    }

    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    fn pause(&self, limit: Duration) {
        self.0
            .fetch_add(limit.as_millis().try_into().unwrap(), Ordering::SeqCst);
    }
}

struct Queue {
    pose_tx: Option<SyncSender<PosePacket>>,
    poses: Receiver<PosePacket>,
    requests: SyncSender<FallRequest>,
    request_rx: Receiver<FallRequest>,
    _response_tx: SyncSender<FallResponse>,
    responses: Receiver<FallResponse>,
    stop: AtomicBool,
    sink: LiveSink,
}

impl Queue {
    fn new() -> Self {
        let (pose_tx, poses) = mpsc::sync_channel(64);
        let (requests, request_rx) = mpsc::sync_channel(64);
        let (response_tx, responses) = mpsc::sync_channel(64);
        Self {
            pose_tx: Some(pose_tx),
            poses,
            requests,
            request_rx,
            _response_tx: response_tx,
            responses,
            stop: AtomicBool::new(false),
            sink: LiveSink::new(),
        }
    }

    fn send(&self, packet: PosePacket) {
        self.pose_tx.as_ref().unwrap().send(packet).unwrap();
    }

    fn turn(
        &mut self,
        pump: &mut PolicyPump,
        poses: usize,
        responses: usize,
    ) -> Result<(), RuntimeError> {
        pump_queue_turn(
            pump,
            &self.poses,
            &self.requests,
            &self.responses,
            &self.stop,
            &mut self.sink,
            budget(poses, responses),
        )
    }
}

fn config(count: u32, fall_enabled: bool) -> PulledConfig {
    let cameras: Vec<Value> = (0..count)
        .map(|source| {
            json!({
                "camera_id": format!("camera-{source}"),
                "facility_id": "facility-1",
                "rtsp_url": "rtsp://fixture.invalid/live"
            })
        })
        .collect();
    let payload = Json::from(&json!({
        "config_version": 1, "cameras": cameras,
        "domains": {"fall": {"enabled": fall_enabled}, "bed_exit": {"enabled": false}}
    }));
    let config = WorkerConfigPayload::parse(&payload).unwrap();
    let ids: Vec<String> = config
        .cameras()
        .into_iter()
        .map(|camera| camera.camera_id)
        .collect();
    PulledConfig {
        directive: config.directive(),
        cameras: config.runtime_cameras().unwrap(),
        policies: resolve_detection_policies(config.detection_policies(), &ids).unwrap(),
        config,
        payload,
        windows: Default::default(),
        source: ConfigSource::Pulled,
        stale: false,
    }
}

fn directory() -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("seeon-replay-queue-{}", RandomIds.uuid4().unwrap()));
    fs::create_dir(&path).unwrap();
    eprintln!("REPLAY_QUEUE_FIXTURE={}", path.display());
    path
}

fn trace_path(root: &Path, source: u32) -> PathBuf {
    root.join(format!(
        "{}.jsonl",
        &sha256_hex(format!("camera-{source}").as_bytes())[..16]
    ))
}

fn rows(root: &Path, source: u32) -> Vec<Value> {
    let text = fs::read_to_string(trace_path(root, source)).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(crate::trace_out::HEADER_LINE.trim_end()));
    lines
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn disabled(count: u32, clock: Arc<TestClock>) -> PolicyPump {
    PolicyPump::new(
        (0..count)
            .map(|source_id| CameraPolicy {
                source_id,
                stage: None,
            })
            .collect(),
        None,
        clock,
    )
    .unwrap()
}

#[test]
fn capture_precedes_disabled_stage_and_outside_fall_window() {
    for enabled in [false, true] {
        let root = directory();
        let clock = Arc::new(TestClock::default());
        let mut pump = if enabled {
            let window = seeon_worker::detection_window::DetectionWindow::from_zoneinfo_dir(
                "12:00",
                "13:00",
                "UTC",
                Path::new("/usr/share/zoneinfo"),
            )
            .unwrap();
            PolicyPump::new(vec![camera(0)], Some(window), clock.clone()).unwrap()
        } else {
            disabled(1, clock.clone())
        };
        pump.set_replay(ReplayCapture::new(&root, &config(1, enabled)).unwrap());
        let mut queue = Queue::new();
        queue.send(packet(0, 1, &[7]));
        queue.turn(&mut pump, 2, 2).unwrap();
        let actual = rows(&root, 0);
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0]["source_event"], "open");
        assert_eq!(actual[1]["source_event"], "frame");
        assert_eq!(actual[1]["tracks"][0]["track_id"], 7);
        assert_eq!(actual[1]["night_window_active"], false);
        assert!(queue.request_rx.try_iter().next().is_none());
        assert_eq!(queue.sink.ready.len(), usize::from(enabled));
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn only_observed_empty_input_can_report_per_camera_availability_loss() {
    let root = directory();
    let clock = Arc::new(TestClock::default());
    let mut pump = disabled(3, clock.clone());
    pump.set_replay(ReplayCapture::new(&root, &config(3, false)).unwrap());
    let mut queue = Queue::new();
    queue.send(packet(0, 1, &[7]));
    queue.send(packet(1, 1, &[8]));
    queue.turn(&mut pump, 2, 4).unwrap();
    clock.at(1000);
    queue.send(packet(1, 2, &[8]));
    queue.turn(&mut pump, 0, 4).unwrap();
    assert_eq!(rows(&root, 0).len(), 2);
    assert_eq!(rows(&root, 1).len(), 2);
    queue.turn(&mut pump, 1, 4).unwrap(); // Consumed budget, not an empty receive.
    assert_eq!(rows(&root, 0).len(), 2);
    assert_eq!(rows(&root, 1).len(), 3);
    queue.turn(&mut pump, 1, 0).unwrap(); // Response budget exhaustion is not absence.
    assert_eq!(rows(&root, 0).len(), 2);
    queue.turn(&mut pump, 1, 4).unwrap();
    let missing = rows(&root, 0);
    assert_eq!(missing.len(), 3);
    assert_eq!(missing[2]["source_event"], "lost");
    assert_eq!(missing[2]["pts_ns"], 100_000_000);
    assert_eq!(missing[2]["tracks"], json!([]));
    assert_eq!(rows(&root, 1).len(), 3); // Its queued packet won over elapsed time.
    queue.turn(&mut pump, 1, 4).unwrap();
    assert_eq!(rows(&root, 0).len(), 3); // One control row per gap.
    clock.at(2000);
    queue.stop.store(true, Ordering::SeqCst);
    queue.turn(&mut pump, 1, 4).unwrap();
    // This seam sees classifier stop, not the process SIGTERM deadline.
    assert_eq!(rows(&root, 1).len(), 3);
    assert!(!trace_path(&root, 2).exists()); // No accepted epoch for this camera.
    drop(queue.pose_tx.take());
    assert!(matches!(
        queue.turn(&mut pump, 1, 4),
        Err(RuntimeError::MediaFatal)
    ));
    assert_eq!(rows(&root, 1).len(), 3);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejected_metadata_restarts_wait_without_fabricating_a_trace_frame() {
    let root = directory();
    let clock = Arc::new(TestClock::default());
    let mut pump = disabled(1, clock.clone());
    pump.set_replay(ReplayCapture::new(&root, &config(1, false)).unwrap());
    let mut queue = Queue::new();
    queue.send(packet(0, 1, &[7]));
    queue.turn(&mut pump, 2, 4).unwrap();
    clock.at(1000);
    let mut refused = packet(0, 2, &[]);
    refused.frame.source_width = 0;
    queue.send(refused);
    queue.turn(&mut pump, 2, 4).unwrap();
    assert_eq!(queue.sink.failures.get(&0), Some(&1));
    assert_eq!(queue.sink.failure_receipts.len(), 1);
    assert_eq!(rows(&root, 0).len(), 2);
    clock.at(1499);
    queue.turn(&mut pump, 1, 4).unwrap();
    assert_eq!(rows(&root, 0).len(), 2);
    clock.at(1500);
    queue.turn(&mut pump, 1, 4).unwrap();
    let actual = rows(&root, 0);
    assert_eq!(actual.len(), 3);
    assert_eq!(actual[2]["source_event"], "lost");
    assert_eq!(actual[2]["frame_width"], 640);
    assert_eq!(actual[2]["pts_ns"], 100_000_000);
    queue.send(packet(0, 3, &[]));
    queue.turn(&mut pump, 2, 4).unwrap();
    let actual = rows(&root, 0);
    assert_eq!(actual.len(), 4);
    assert_eq!(actual[3]["source_event"], "frame");
    assert_eq!(actual[3]["tracks"][0]["track_id"], 7);
    assert_eq!(actual[3]["tracks"][0]["lifecycle"], "lost");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn real_trace_append_failure_does_not_change_fall_requests_or_policy_counters() {
    let root = directory();
    let mut results = Vec::new();
    for traced in [false, true] {
        let clock = Arc::new(TestClock::default());
        let mut pump = PolicyPump::new(vec![camera(0)], None, clock).unwrap();
        if traced {
            pump.set_replay(ReplayCapture::new(&root, &config(1, true)).unwrap());
            fs::create_dir(trace_path(&root, 0)).unwrap(); // Genuine append IO refusal.
        }
        let mut queue = Queue::new();
        for sequence in 1..=40 {
            queue.send(packet(0, sequence, &[7]));
        }
        queue.turn(&mut pump, 64, 64).unwrap();
        let requests: Vec<_> = queue
            .request_rx
            .try_iter()
            .map(|request| (request.frame, request.track_id, *request.window))
            .collect();
        assert!(
            !requests.is_empty(),
            "exercise actual model request windows"
        );
        results.push((requests, pump.exit_observations().collect::<Vec<_>>()));
    }
    assert_eq!(results[0], results[1]);
    assert!(trace_path(&root, 0).is_dir());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn unknown_source_remains_fatal_with_replay_enabled() {
    let root = directory();
    let mut pump = disabled(1, Arc::new(TestClock::default()));
    pump.set_replay(ReplayCapture::new(&root, &config(1, false)).unwrap());
    let mut queue = Queue::new();
    queue.send(packet(99, 1, &[7]));
    assert!(matches!(
        queue.turn(&mut pump, 2, 4),
        Err(RuntimeError::Pump(
            crate::run::pump::PumpError::UnknownSource(99)
        ))
    ));
    assert!(!trace_path(&root, 0).exists());
    assert!(queue.sink.failure_receipts.is_empty());
    fs::remove_dir_all(root).unwrap();
}
