//! Resource-selected genuine SDK-withheld child. This covers typed lifecycle
//! return, not execute/main/SIGTERM or a universal process-exit budget.

#[cfg(test)]
#[path = "lifecycle_n2_child.rs"]
mod child;
#[cfg(test)]
#[path = "lifecycle_n2_control.rs"]
mod control;
#[cfg(test)]
#[path = "lifecycle_n2_supervision.rs"]
mod supervision;

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use crate::media::Command;
use crate::media::diagnostics::Diagnostics;
use crate::seam::{Clock, SystemClock};
use crate::shutdown::{MAX_SHUTDOWN_BUDGET, ShutdownDeadline};
use control::{Directory, nonce, ns};
use seeon_deepstream_native::{MediaPoll, RecordTicket};
use supervision::Receipt;

const NAME: &str =
    "run::execution::lifecycle::n2_tests::genuine_sdk_withheld_lifecycle_retains_gpu_lease";

fn required(name: &str) -> OsString {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("missing explicit N2 fixture resource: {name}"))
}

#[test]
#[ignore = "requires private N2 native build, real stock models, isolated ml-api/RTSP, and fresh owned directories"]
fn genuine_sdk_withheld_lifecycle_retains_gpu_lease() {
    let state_path = PathBuf::from(required("SEEON_TEST_N2_STATE_DIR"));
    let record_path = PathBuf::from(required("ML_WORKER_FLOW_RECORD_DIR"));
    let auxiliary = required("SEEON_TEST_N2_AUXILIARY_RUNTIME");
    assert_eq!(
        auxiliary.to_str(),
        Some("onnxruntime-cpu"),
        "explicit approved CPU auxiliary provider required"
    );
    let state = Directory::open(&state_path, true);
    let record = Directory::open(&record_path, true);
    assert_ne!(
        state.identity(),
        record.identity(),
        "distinct fresh owned directories"
    );
    match std::env::var_os("SEEON_TEST_N2_CHILD_NONCE") {
        None => {
            let run = nonce();
            supervision::supervise(NAME, &state_path, &state, &record, run, |receipt| {
                verify(&record, run, receipt);
            });
        }
        Some(raw) => {
            let text = raw.into_string().expect("child nonce UTF-8");
            assert!(text.len() == 32 && text.bytes().all(|byte| byte.is_ascii_hexdigit()));
            let run = [
                u64::from_str_radix(&text[..16], 16).unwrap(),
                u64::from_str_radix(&text[16..], 16).unwrap(),
            ];
            assert_ne!(run, [0, 0]);
            child::child(state_path, auxiliary, state, record, run);
        }
    }
}

struct Entered {
    ticket: RecordTicket,
    entry: [u64; 13],
    observed_ns: u64,
}

fn controller(
    record: Directory,
    run: [u64; 2],
    source: seeon_deepstream_native::SourceConfig,
    diagnostics: Arc<Diagnostics>,
    commands: mpsc::SyncSender<Command>,
    clock: Arc<SystemClock>,
    deadline: Arc<ShutdownDeadline>,
) -> Entered {
    crate::poll::poll_until(
        clock.as_ref(),
        clock.monotonic() + Duration::from_secs(20),
        "actual N2 frames",
        || {
            let snapshot = diagnostics.snapshot();
            assert!(!snapshot.open_refused && !snapshot.fatal && snapshot.failure.is_none());
            snapshot
                .cameras
                .first()
                .is_some_and(|camera| camera.video_linked == 1 && camera.published_frames >= 10)
        },
    )
    .expect("actual frames from the real media owner");
    let (reply, ack) = mpsc::sync_channel(crate::msg::ONESHOT_CAPACITY);
    commands
        .try_send(Command::RecordStart {
            source_id: source.source_id,
            binding: source.binding,
            lookback_seconds: 1,
            forward_seconds: 3,
            reply,
        })
        .unwrap_or_else(|_| panic!("actual public RecordStart command refused"));
    let ticket = match ack
        .recv_timeout(Duration::from_secs(5))
        .expect("actual RecordStart ACK")
        .expect("SDK start")
    {
        MediaPoll::Ready(ticket) => ticket,
        MediaPoll::Status(status) => {
            panic!("RecordStart did not allocate a real accepted ticket: {status:?}")
        }
    };
    assert_eq!(
        (ticket.source_id, ticket.binding),
        (source.source_id, source.binding)
    );
    assert!(ticket.request_id > 0 && ticket.session_valid <= 1 && ticket.coalesced == 0);
    let arm = [
        run[0],
        run[1],
        u64::from(ticket.source_id),
        ticket.binding.token,
        ticket.binding.generation,
        ticket.binding.epoch,
        ticket.request_id,
    ];
    record.publish(".n2-arm", b"N2ARM001", &arm);
    let entry: [u64; 13] = record.wait(".n2-entry", b"N2ENT001", Duration::from_secs(20));
    assert_eq!(&entry[..7], &arm);
    sdk_facts(&entry);
    if ticket.session_valid == 1 {
        assert_eq!(
            entry[7],
            u64::from(ticket.session_id),
            "actual ACK/session correlation"
        );
    }
    let observed_ns = ns(clock.monotonic());
    deadline
        .request_at(clock.monotonic())
        .expect("real shared shutdown request after entry");
    Entered {
        ticket,
        entry,
        observed_ns,
    }
}

fn sdk_facts(entry: &[u64; 13]) {
    assert!(entry[2] <= u64::from(u32::MAX) && entry[7] <= u64::from(u32::MAX));
    assert!(
        entry[3..7].iter().all(|value| *value > 0),
        "actual allocated ticket identity"
    );
    assert!(entry[8] > 0 && entry[9] > 0 && entry[10] > 0);
    assert!(entry[9] <= u64::from(u32::MAX) && entry[10] <= u64::from(u32::MAX));
    assert_eq!(entry[11], 1, "genuine successful SDK video info");
    assert!(entry[12] <= 1, "actual SDK audio boolean");
}

fn verify(record: &Directory, run: [u64; 2], receipt: &Receipt) {
    let arm: [u64; 7] = record
        .read(".n2-arm", b"N2ARM001")
        .expect("actual accepted-ticket arm");
    let entry: [u64; 13] = record
        .read(".n2-entry", b"N2ENT001")
        .expect("actual native entry");
    let query: [u64; 4] = record
        .read(".n2-query", b"N2QRY001")
        .expect("fresh post-return query");
    let witness: [u64; 25] = record
        .read(".n2-witness", b"N2WIT002")
        .expect("live native witness");
    assert_eq!(&arm[..2], &run);
    assert_eq!(&entry[..7], &arm);
    sdk_facts(&entry);
    assert_eq!(&query[..2], &run);
    assert_ne!(&query[2..], &[0, 0]);
    assert_ne!(&query[2..], &run);
    assert_eq!(&witness[..13], &entry);
    assert_eq!(&witness[13..15], &query[2..]);
    assert_eq!(receipt.witness, witness);
    assert_eq!(
        receipt.model_failures, [0; 3],
        "no unrelated model-owner failure"
    );
    assert!(
        witness[15] > 0
            && witness[15] <= crate::msg::RECORD_CAPACITY as u64
            && witness[16] == 0
            && witness[17..20]
                .iter()
                .all(|count| *count > 0 && *count < (1u64 << 63)),
        "actual held reservation and all three masked actual lease gates"
    );
    assert!(receipt.ack[0] <= u64::from(u32::MAX) && receipt.ack[1] <= 1 && receipt.ack[2] == 0);
    if receipt.ack[1] == 1 {
        assert_eq!(entry[7], receipt.ack[0]);
    }
    let t = receipt.times;
    assert!(t[0] <= t[2] && t[2] <= t[3] && t[3] <= t[1] && t[1] <= t[5] && t[5] <= t[6]);
    assert_eq!(
        t[4].checked_sub(t[3]),
        Some(ns(MAX_SHUTDOWN_BUDGET)),
        "unchanged shared 25s budget"
    );
    assert!(t[7] >= t[6] && t[7] >= t[4] && t[8] >= t[7] && t[9] >= t[8]);
    assert!(
        control::expected_withheld_timeout(&witness),
        "fresh native first failure must be the completed held-owner STOP_TIMEOUT"
    );
    for snapshot in [receipt.returned, receipt.collected] {
        assert!(
            [0, 3, 9, 10, 11, 12, 13]
                .into_iter()
                .all(|index| snapshot[index] <= 1),
            "diagnostic booleans"
        );
        assert!(
            snapshot[0] == 0 && snapshot[1] == 0,
            "no refused open or unrelated Rust owner failure"
        );
        assert!(
            snapshot[14] == 1
                && snapshot[15] >= 10
                && snapshot[19] > 0
                && snapshot[19] <= seeon_deepstream_native::MEDIA_MAX_SOURCES as u64
        );
    }
    let snapshot = receipt.collected;
    assert!(
        snapshot[9] == 1 && snapshot[10] == 1 && snapshot[13] == 1 && snapshot[12] == 0,
        "completed actual finalization attempt with native close withheld"
    );
}
