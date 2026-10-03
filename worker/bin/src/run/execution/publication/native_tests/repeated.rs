//! Two actual recordings: the first ends through the unchanged 45s recorder
//! schedule; the second ends through the shared shutdown path. No fake clock,
//! shortened production cap, or synthesized native receipt is used.

use std::collections::BTreeSet;
use std::fs;
use std::sync::PoisonError;
use std::sync::mpsc::{self, RecvTimeoutError, TryRecvError};
use std::time::{Duration, Instant};

use seeon_deepstream_native::{MediaResult, RecordTicket};
use serde_json::Value;

use super::assertions::{self, Admitted, GrownFile};
use super::published;
use super::receipt;
use super::support::{ACTUAL_MEDIA, BOOT_ID, Started};
use super::wait::poll_until;
use crate::msg::RecordReceipt;

const SECOND_EVENT: &str = "00000000-0000-4000-8000-000000000072";

#[test]
#[ignore = "requires actual GPU/SDK and RTSP; waits the unchanged production 45s stop schedule"]
fn reused_native_session_publishes_two_distinct_requests() {
    let _turn = ACTUAL_MEDIA.lock().unwrap_or_else(PoisonError::into_inner);
    let overall = Instant::now() + Duration::from_secs(150);
    std::thread::scope(|scope| {
        let (done, completion) = mpsc::channel();
        scope.spawn(move || {
            if let Err(RecvTimeoutError::Timeout) =
                completion.recv_timeout(overall.saturating_duration_since(Instant::now()))
            {
                eprintln!("repeated native publication overall deadline");
                std::process::exit(1);
            }
        });
        publish_twice(overall);
        done.send(()).expect("watchdog exited early");
    });
}

fn publish_twice(overall: Instant) {
    let mut started = Started::open(overall);
    let first = assertions::admit(&mut started);
    let grown = assertions::growing_mp4(
        &started.record_dir,
        Instant::now() + Duration::from_secs(15),
    );
    let observed = running_receipt(&mut started, Instant::now() + Duration::from_secs(60));
    let first_ticket = observed.ticket;
    let sealed = checked_receipt(&first, &observed, &grown);
    let first_duration = observed.duration_ms;
    forward_running(&mut started, observed);
    published::assert_published(
        &started.publications,
        &started.record_dir,
        &first,
        &first_ticket,
        first_duration,
        &sealed,
    );

    let second = assertions::admit_event(&mut started, SECOND_EVENT);
    assert!(second.ticket.request_id > first.ticket.request_id);
    let grown_second = assertions::growing_mp4(
        &started.record_dir,
        Instant::now() + Duration::from_secs(15),
    );
    assert!(matches!(
        started.publications.records.try_recv(),
        Err(TryRecvError::Empty)
    ));
    started.request_stop();
    let cutoff = started.shutdown_deadline();
    let observed_second = receipt::capture(&mut started, cutoff, &grown_second);
    let second_ticket = observed_second.ticket;
    let second_duration = observed_second.duration_ms;
    let sealed_second = checked_receipt(&second, &observed_second, &grown_second);
    assert_eq!(
        first_ticket.session_id, second_ticket.session_id,
        "this regression must exercise actual SDK session reuse"
    );
    receipt::forward_once(&mut started, observed_second);
    published::assert_clip(
        &started.publications,
        &started.record_dir,
        &second,
        &second_ticket,
        second_duration,
        &sealed_second,
    );
    // The second publication must not replace or mutate the first one.
    published::assert_clip(
        &started.publications,
        &started.record_dir,
        &first,
        &first_ticket,
        first_duration,
        &sealed,
    );
    assert_two_ids(&started, &first_ticket, &second_ticket);
    started.finish(overall);
    let status = started.diagnostics().snapshot();
    assert!(
        status.finalization_complete && status.stopped && status.closed,
        "{status:?}"
    );
    assert!(
        !status.close_withheld && status.failure.is_none(),
        "{status:?}"
    );
    assert_eq!(
        (
            status.receipts_dropped,
            status.replies_dropped,
            status.records_reserved
        ),
        (0, 0, 0),
        "{status:?}"
    );
}

fn running_receipt(started: &mut Started, deadline: Instant) -> RecordReceipt {
    poll_until(deadline, "scheduled native receipt deadline", || {
        started.publications.tick_recorders();
        let status = started.diagnostics().snapshot();
        assert!(!status.fatal && status.failure.is_none(), "{status:?}");
        match started.publications.records.try_recv() {
            Ok(receipt) => Some(receipt),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => panic!("recording channel disconnected"),
        }
    })
}

fn checked_receipt(admitted: &Admitted, observed: &RecordReceipt, grown: &GrownFile) -> GrownFile {
    assert_eq!(
        (observed.result, observed.error, observed.contains_video),
        (MediaResult::Ok, 0, true),
        "native receipt grade"
    );
    receipt::assert_ticket(&admitted.ticket, &observed.ticket);
    let sealed = receipt::file_identity(observed);
    assert_eq!((sealed.dev, sealed.ino), (grown.dev, grown.ino));
    assert!(sealed.len >= grown.len);
    eprintln!(
        "REPEATED_NATIVE_RECEIPT ticket={:?} duration_ms={} size={} filename={:?}",
        observed.ticket, observed.duration_ms, sealed.len, observed.filename
    );
    sealed
}

fn forward_running(started: &mut Started, observed: RecordReceipt) {
    assert!(matches!(
        started.publications.records.try_recv(),
        Err(TryRecvError::Empty)
    ));
    started
        .receipt_tx
        .send(observed)
        .expect("forward actual receipt unchanged");
    assert!(
        started
            .publications
            .drain_records(started.clock.as_ref())
            .expect("production publication refused actual receipt")
    );
    assert_eq!(
        started.publications.recorders[0].state(),
        crate::clips::recorder::State::Idle
    );
}

fn assert_two_ids(started: &Started, first: &RecordTicket, second: &RecordTicket) {
    let expected: BTreeSet<_> = [first, second]
        .map(|ticket| format!("{BOOT_ID}-{}-{}", ticket.source_id, ticket.request_id))
        .into_iter()
        .collect();
    assert_eq!(expected.len(), 2);
    let actual: BTreeSet<_> = fs::read_dir(started.publications.store.root().join("clips"))
        .expect("clips")
        .map(|entry| {
            entry
                .expect("clip entry")
                .file_name()
                .into_string()
                .expect("clip name")
        })
        .filter(|name| !name.starts_with('.'))
        .collect();
    assert_eq!(actual, expected);
    let entries = started.publications.queue.entries().expect("queue");
    let clip_ids: Vec<_> = entries
        .iter()
        .filter(|entry| entry.get("kind").and_then(Value::as_str) == Some("CLIP"))
        .map(|entry| {
            entry
                .get("clip_id")
                .and_then(Value::as_str)
                .expect("clip ID")
                .to_owned()
        })
        .collect();
    assert_eq!(clip_ids.len(), 2);
    assert_eq!(clip_ids.into_iter().collect::<BTreeSet<_>>(), expected);
}
