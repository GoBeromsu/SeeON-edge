//! Constructed receipts exercise only the private Rust handoff, not SDK acceptance
//! or native callback/slot retirement. No media owner is opened by these tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, SystemTime};

use seeon_deepstream_native::{MediaBinding, MediaConfig, MediaResult, MediaState, RecordTicket};

use super::{MediaParams, RecordReceipt, ShutdownControl, deliver_record, record_failure};
use crate::exit::Exit;
use crate::media::diagnostics::{Diagnostics, Snapshot};
use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;

const NOW: Duration = Duration::from_secs(2);
const DEADLINE: Duration = Duration::from_secs(12);

struct UntouchedClock;

impl Clock for UntouchedClock {
    fn monotonic(&self) -> Duration {
        panic!("receipt handoff must not sample or reset time")
    }

    fn wall(&self) -> SystemTime {
        panic!("receipt handoff must not sample wall time")
    }

    fn pause(&self, _: Duration) {
        panic!("receipt handoff must not wait or retry")
    }
}

fn fixture(permitted: bool) -> (MediaParams, Receiver<RecordReceipt>) {
    let (pose_tx, _pose_rx) = mpsc::sync_channel(1);
    let (preview_tx, _preview_rx) = mpsc::sync_channel(1);
    let (record_tx, record_rx) = mpsc::sync_channel(1);
    let (_command_tx, commands) = mpsc::sync_channel(1);
    let shutdown = Arc::new(ShutdownControl::new(Arc::new(
        ShutdownDeadline::new(Duration::from_secs(10)).unwrap(),
    )));
    assert_eq!(shutdown.begin(NOW).unwrap(), DEADLINE);
    if permitted {
        shutdown.permit_close(NOW).unwrap();
    }
    let diagnostics = Arc::new(Diagnostics::new(1));
    diagnostics.update(|snapshot| {
        snapshot.state = Some(MediaState::Stopped);
        snapshot.callbacks_active = 3;
        snapshot.records_reserved = 2;
        snapshot.receipts_dropped = 5;
        snapshot.finalization_started = true;
        snapshot.stopped = true;
        snapshot.close_withheld = true;
    });
    let params = MediaParams {
        // Unused by this handoff-only unit fixture; never passed to native open.
        config: MediaConfig {
            sources: Vec::new(),
            infer_config_path: "infer.txt".into(),
            tracker_config_path: "tracker.yml".into(),
            tracker_library_path: "tracker.so".into(),
            record_directory: "records".into(),
            record_cache_seconds: 30,
            record_capacity: 4,
            mux_width: 640,
            mux_height: 360,
            mux_batch_timeout_us: 40000,
            mux_live_source: false,
            tracker_width: 960,
            tracker_height: 544,
            queue_max_buffers: 4,
            preview_enabled: false,
            max_preview_bytes: 0,
            allow_file_uris: true,
            rtsp_reconnect_interval_sec: 0,
        },
        open_budget_ms: 1,
        shutdown,
        stop: Arc::new(AtomicBool::new(true)),
        clock: Arc::new(UntouchedClock),
        pose_tx,
        preview_tx,
        record_tx,
        commands,
        diagnostics,
    };
    (params, record_rx)
}

fn receipt(request_id: u64) -> RecordReceipt {
    RecordReceipt {
        ticket: RecordTicket {
            binding: MediaBinding {
                token: 7,
                generation: 3,
                epoch: 11,
            },
            request_id,
            source_id: 0,
            session_id: 19,
            session_valid: 1,
            coalesced: 0,
        },
        result: MediaResult::Ok,
        error: 0,
        duration_ms: 30_000,
        width: 1280,
        height: 720,
        contains_video: true,
        contains_audio: false,
        directory: "unit-records".into(),
        filename: "unit-record.mp4".into(),
    }
}

fn assert_receipt_unchanged(actual: RecordReceipt, request_id: u64) {
    let expected = receipt(request_id);
    assert_eq!(actual.ticket, expected.ticket);
    assert_eq!(actual.result, expected.result);
    assert_eq!(actual.error, expected.error);
    assert_eq!(actual.duration_ms, expected.duration_ms);
    assert_eq!(actual.width, expected.width);
    assert_eq!(actual.height, expected.height);
    assert_eq!(actual.contains_video, expected.contains_video);
    assert_eq!(actual.contains_audio, expected.contains_audio);
    assert_eq!(actual.directory, expected.directory);
    assert_eq!(actual.filename, expected.filename);
}

fn assert_state(params: &MediaParams, expected: &Snapshot, permitted: bool) {
    assert_eq!(&params.diagnostics.snapshot(), expected);
    assert_eq!(params.shutdown.deadline(), Some(DEADLINE));
    assert_eq!(params.shutdown.close_permitted(NOW), permitted);
    assert!(params.stop.load(Ordering::SeqCst));
}

#[test]
fn successful_delivery_preserves_receipt_and_all_state() {
    let (params, records) = fixture(false);
    let before = params.diagnostics.snapshot();
    deliver_record(&params, receipt(1));
    assert_receipt_unchanged(records.try_recv().unwrap(), 1);
    assert!(matches!(records.try_recv(), Err(TryRecvError::Empty)));
    assert_state(&params, &before, false);
}

#[test]
fn full_delivery_counts_each_loss_without_eviction_and_failure_stays_sticky() {
    let (params, records) = fixture(true);
    let mut expected = params.diagnostics.snapshot();
    deliver_record(&params, receipt(1));
    deliver_record(&params, receipt(2));
    deliver_record(&params, receipt(3));
    expected.receipts_dropped += 2;
    expected.failure = Some(Exit::Runtime);
    assert_state(&params, &expected, true);
    assert_receipt_unchanged(records.try_recv().unwrap(), 1);
    assert!(matches!(records.try_recv(), Err(TryRecvError::Empty)));

    deliver_record(&params, receipt(4));
    assert_receipt_unchanged(records.try_recv().unwrap(), 4);
    record_failure(&params, Exit::Config);
    assert_state(&params, &expected, true);

    record_failure(&params, Exit::FatalAccelerator);
    expected.failure = Some(Exit::FatalAccelerator);
    assert_state(&params, &expected, true);
}

#[test]
fn disconnected_delivery_counts_each_loss_and_records_runtime() {
    let (params, records) = fixture(false);
    let mut expected = params.diagnostics.snapshot();
    drop(records);
    deliver_record(&params, receipt(1));
    deliver_record(&params, receipt(2));
    expected.receipts_dropped += 2;
    expected.failure = Some(Exit::Runtime);
    assert_state(&params, &expected, false);
}

#[test]
fn both_delivery_failures_preserve_existing_config_or_accelerator_failure() {
    for failure in [Exit::Config, Exit::FatalAccelerator] {
        for disconnected in [false, true] {
            let (params, records) = fixture(false);
            let _records = if disconnected {
                drop(records);
                None
            } else {
                assert!(params.record_tx.try_send(receipt(1)).is_ok());
                Some(records)
            };
            record_failure(&params, failure);
            let mut expected = params.diagnostics.snapshot();
            deliver_record(&params, receipt(2));
            deliver_record(&params, receipt(3));
            expected.receipts_dropped += 2;
            assert_state(&params, &expected, false);
        }
    }
}
