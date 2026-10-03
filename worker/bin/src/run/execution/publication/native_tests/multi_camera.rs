//! Two actual sources share one production media owner, command queue, and
//! publication queue. The approved infer config must have batch size two.

use std::process;
use std::sync::PoisonError;
use std::sync::mpsc::{self, RecvTimeoutError, TryRecvError};
use std::time::{Duration, Instant};

use super::assertions;
use super::published;
use super::receipt::{self, Captured};
use super::support::{ACTUAL_MEDIA, CAMERA_ONE, CAMERA_ZERO, SECOND_EVENT_IDENTITY, Started};
use super::wait::poll_until;
use crate::msg::RecordReceipt;

/// Readiness (15s) + both linked sources (15s) + file growth (15s) + shared
/// shutdown/finalization (25s), with slack for real SDK scheduling.
const OVERALL: Duration = Duration::from_secs(90);

#[test]
#[ignore = "requires actual GPU/SDK, a batch-2 infer config, isolated RTSP and a new owned output directory"]
fn two_camera_native_recording_publishes_actual_clips_and_closes() {
    let _turn = ACTUAL_MEDIA.lock().unwrap_or_else(PoisonError::into_inner);
    let overall = Instant::now() + OVERALL;
    std::thread::scope(|scope| {
        let (finished, completion) = mpsc::channel();
        scope.spawn(move || {
            match completion.recv_timeout(overall.saturating_duration_since(Instant::now())) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
                Err(RecvTimeoutError::Timeout) => {
                    eprintln!("two-camera native publication overall deadline");
                    process::exit(1);
                }
            }
        });
        publish_two_cameras(overall);
        finished
            .send(())
            .expect("watchdog exited before publication finished");
    });
}

fn publish_two_cameras(overall: Instant) {
    assert_distinct_inputs();
    let mut started = Started::open_two(overall);
    let first = assertions::admit_for_source(
        &mut started,
        0,
        &CAMERA_ZERO,
        super::support::EVENT_IDENTITY,
    );
    let second = assertions::admit_for_source(&mut started, 1, &CAMERA_ONE, SECOND_EVENT_IDENTITY);
    assert!(first.ticket.request_id != 0 && second.ticket.request_id != 0);
    assert!(first.ticket.request_id < second.ticket.request_id);
    assert_eq!(first.ticket.coalesced, 0);
    assert_eq!(second.ticket.coalesced, 0);

    let growth_deadline = overall.min(Instant::now() + Duration::from_secs(15));
    let grown = [
        assertions::growing_started_mp4_for(&started, first.record_prefix, growth_deadline),
        assertions::growing_started_mp4_for(&started, second.record_prefix, growth_deadline),
    ];
    assert_ne!((grown[0].dev, grown[0].ino), (grown[1].dev, grown[1].ino));
    assert_no_receipts(&started, "before shared shutdown");

    // One quiesce and one shared 25s stop. No per-camera stop or retry is used.
    started.request_stop();
    let cutoff = started.shutdown_deadline();
    let receipts = capture_receipts(&mut started, cutoff);
    poll_until(cutoff, "two-source finalization deadline", || {
        started.drain_pose_packets();
        let status = started.diagnostics().snapshot();
        assert!(!status.fatal && status.failure.is_none(), "{status:?}");
        status.finalization_complete.then_some(())
    });
    assert_no_receipts(&started, "after finalization");

    let admitted = [first, second];
    let mut by_source: Vec<Option<Captured>> = (0..admitted.len()).map(|_| None).collect();
    for observed in receipts {
        let source_index = usize::try_from(observed.ticket.source_id).expect("source id");
        let expected = admitted
            .get(source_index)
            .expect("receipt source is admitted");
        assert_eq!(expected.source_index, source_index);
        let captured = receipt::check_source_receipt(
            &started.record_dir,
            expected,
            observed,
            &grown[source_index],
        );
        assert!(
            by_source[source_index].replace(captured).is_none(),
            "duplicate source receipt"
        );
    }
    let mut captured: Vec<_> = by_source
        .into_iter()
        .map(|receipt| receipt.expect("one native receipt per source"))
        .collect();
    assert_eq!(captured.len(), 2);
    assertions::assert_exact_native_files(
        &started.record_dir,
        &[
            captured[0].receipt.filename.to_str().expect("filename"),
            captured[1].receipt.filename.to_str().expect("filename"),
        ],
    );

    let first_capture = captured.remove(0);
    let second_capture = captured.remove(0);
    let first_ticket = first_capture.receipt.ticket;
    let second_ticket = second_capture.receipt.ticket;
    forward_once(&mut started, first_capture.receipt, 0);
    let first_state = published::assert_clip(
        &started.publications,
        &started.record_dir,
        &admitted[0],
        &first_ticket,
        first_capture.duration_ms,
        &first_capture.sealed,
    );
    assert_eq!(first_state.sha256, first_capture.sha256);
    published::assert_exact_clips(&started.publications, &[first_ticket]);

    forward_once(&mut started, second_capture.receipt, 1);
    let second_state = published::assert_clip(
        &started.publications,
        &started.record_dir,
        &admitted[1],
        &second_ticket,
        second_capture.duration_ms,
        &second_capture.sealed,
    );
    assert_eq!(second_state.sha256, second_capture.sha256);
    let first_after_second = published::assert_clip(
        &started.publications,
        &started.record_dir,
        &admitted[0],
        &first_ticket,
        first_capture.duration_ms,
        &first_capture.sealed,
    );
    assert_eq!(
        first_state, first_after_second,
        "first READY clip was mutated"
    );
    let second_after_publication = published::assert_clip(
        &started.publications,
        &started.record_dir,
        &admitted[1],
        &second_ticket,
        second_capture.duration_ms,
        &second_capture.sealed,
    );
    assert_eq!(second_state, second_after_publication);
    published::assert_exact_clips(&started.publications, &[first_ticket, second_ticket]);
    published::assert_no_mp4(&started.record_dir);
    for recorder in &started.publications.recorders {
        assert_eq!(recorder.state(), crate::clips::recorder::State::Idle);
        assert_eq!(recorder.pending(), 0);
    }

    started.finish(overall);
    let status = started.diagnostics().snapshot();
    assert!(
        status.finalization_started && status.finalization_complete,
        "{status:?}"
    );
    assert!(status.stopped && status.closed, "{status:?}");
    assert!(
        !status.close_withheld && status.failure.is_none(),
        "{status:?}"
    );
    assert_eq!(status.cameras.len(), 2, "{status:?}");
    // Readiness required both links live; native teardown clears them.
    assert!(
        status.cameras.iter().all(|camera| camera.video_linked == 0
            && camera.published_frames > 0
            && camera.handoff_dropped_frames == 0),
        "{status:?}"
    );
    assert_eq!(
        (
            status.previews_dropped,
            status.receipts_dropped,
            status.replies_dropped,
            status.records_reserved
        ),
        (0, 0, 0, 0),
        "{status:?}"
    );
    assert!(Instant::now() < overall, "overall deadline");
}

fn assert_distinct_inputs() {
    assert_eq!((CAMERA_ZERO.source_id, CAMERA_ONE.source_id), (0, 1));
    assert_ne!(CAMERA_ZERO.binding, CAMERA_ONE.binding);
    assert_ne!(CAMERA_ZERO.camera_id, CAMERA_ONE.camera_id);
    assert_ne!(CAMERA_ZERO.facility_id, CAMERA_ONE.facility_id);
    assert_ne!(CAMERA_ZERO.record_prefix, CAMERA_ONE.record_prefix);
    assert_ne!(super::support::EVENT_IDENTITY, SECOND_EVENT_IDENTITY);
}

fn capture_receipts(started: &mut Started, deadline: Instant) -> Vec<RecordReceipt> {
    let mut receipts = Vec::with_capacity(2);
    poll_until(deadline, "two native receipt deadline", || {
        started.drain_pose_packets();
        let status = started.diagnostics().snapshot();
        assert!(!status.fatal && status.failure.is_none(), "{status:?}");
        loop {
            match started.publications.records.try_recv() {
                Ok(receipt) => {
                    receipts.push(receipt);
                    assert!(receipts.len() <= 2, "extra native receipt");
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => panic!("native receipt channel disconnected"),
            }
        }
        (receipts.len() == 2).then(|| std::mem::take(&mut receipts))
    })
}

fn forward_once(started: &mut Started, observed: RecordReceipt, source_index: usize) {
    assert_no_receipts(started, "before production drain");
    started
        .receipt_tx
        .send(observed)
        .expect("forward actual receipt unchanged");
    assert!(
        started
            .publications
            .drain_records(started.clock.as_ref())
            .expect("production drain refused native receipt"),
        "production drain published nothing"
    );
    assert_no_receipts(started, "after production drain");
    assert_eq!(
        started.publications.recorders[source_index].state(),
        crate::clips::recorder::State::Idle
    );
}

fn assert_no_receipts(started: &Started, phase: &str) {
    assert!(
        matches!(
            started.publications.records.try_recv(),
            Err(TryRecvError::Empty)
        ),
        "unexpected receipt {phase}"
    );
}
