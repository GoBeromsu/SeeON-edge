//! Typed post-return evidence and parent supervision, private to the test root.

use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use super::super::RunExit;
use super::control::Directory;
use crate::media::diagnostics::Snapshot;
use rustix::fs;

// All fields are individual LE u64s, not a native struct layout. Diagnostics
// are the actual global snapshot plus the selected first camera, never SDK proof.
pub(super) struct Receipt {
    pub(super) actual: RunExit,
    pub(super) mapping: u8,
    // Runtime start/return, entry observed, request, deadline, shutdown call/return,
    // query publication, witness collection, final diagnostic collection (ns).
    pub(super) times: [u64; 10],
    pub(super) lease: [u64; 2],
    // Actual accepted ticket's session id, validity, and coalesced flag.
    pub(super) ack: [u64; 3],
    pub(super) returned: [u64; 20],
    pub(super) collected: [u64; 20],
    pub(super) witness: [u64; 20],
    pub(super) model_failures: [u64; 3],
}

impl Receipt {
    pub(super) fn fields(&self) -> [u64; 80] {
        let mut fields = [0; 80];
        fields[0] = match self.actual {
            RunExit::Clean => 0,
            RunExit::Runtime => 1,
            RunExit::Config => 2,
            RunExit::Fatal => 3,
        };
        fields[1] = u64::from(self.mapping);
        fields[2..12].copy_from_slice(&self.times);
        fields[12..14].copy_from_slice(&self.lease);
        fields[14..17].copy_from_slice(&self.ack);
        fields[17..37].copy_from_slice(&self.returned);
        fields[37..57].copy_from_slice(&self.collected);
        fields[57..77].copy_from_slice(&self.witness);
        fields[77..80].copy_from_slice(&self.model_failures);
        fields
    }

    fn decode(fields: [u64; 80]) -> Self {
        Self {
            actual: match fields[0] {
                0 => RunExit::Clean,
                1 => RunExit::Runtime,
                2 => RunExit::Config,
                3 => RunExit::Fatal,
                _ => panic!("unknown typed RunExit"),
            },
            mapping: u8::try_from(fields[1]).expect("OS mapping"),
            times: fields[2..12].try_into().unwrap(),
            lease: fields[12..14].try_into().unwrap(),
            ack: fields[14..17].try_into().unwrap(),
            returned: fields[17..37].try_into().unwrap(),
            collected: fields[37..57].try_into().unwrap(),
            witness: fields[57..77].try_into().unwrap(),
            model_failures: fields[77..80].try_into().unwrap(),
        }
    }
}

pub(super) fn diagnostics(snapshot: &Snapshot) -> [u64; 20] {
    use seeon_deepstream_native::MediaState;
    let camera = snapshot
        .cameras
        .first()
        .expect("actual nonempty media roster");
    let state = match snapshot.state {
        None => 0,
        Some(MediaState::Open) => 1,
        Some(MediaState::Starting) => 2,
        Some(MediaState::Running) => 3,
        Some(MediaState::Stopping) => 4,
        Some(MediaState::Stopped) => 5,
        Some(MediaState::Unknown(value)) => 6 + u64::from(value as u32),
    };
    [
        snapshot.open_refused.into(),
        snapshot
            .failure
            .map_or(0, |exit| u64::from(exit.code()) + 1),
        state,
        snapshot.fatal.into(),
        snapshot.callbacks_active.into(),
        snapshot.records_reserved.into(),
        snapshot.previews_dropped,
        snapshot.receipts_dropped,
        snapshot.replies_dropped,
        snapshot.finalization_started.into(),
        snapshot.finalization_complete.into(),
        snapshot.stopped.into(),
        snapshot.closed.into(),
        snapshot.close_withheld.into(),
        camera.video_linked.into(),
        camera.published_frames,
        camera.objects_observed,
        camera.frames_without_pose_tensor,
        camera.handoff_dropped_frames,
        snapshot.cameras.len() as u64,
    ]
}

struct ContainedChild(Child);
impl Drop for ContainedChild {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

pub(super) fn supervise(
    name: &str,
    state_path: &Path,
    state: &Directory,
    record: &Directory,
    run: [u64; 2],
    inspect: impl FnOnce(&Receipt),
) {
    let mut child = ContainedChild(
        Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                name,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(
                "SEEON_TEST_N2_CHILD_NONCE",
                format!("{:016x}{:016x}", run[0], run[1]),
            )
            .spawn()
            .expect("supervised genuine lifecycle child"),
    );
    let limit = Instant::now() + Duration::from_secs(180);
    let receipt = loop {
        assert!(
            child.0.try_wait().expect("child status").is_none(),
            "child exited without live post-return evidence"
        );
        if let Some(fields) = record.read(".n2-return", b"N2RET001") {
            break Receipt::decode(fields);
        }
        assert!(
            Instant::now() < limit,
            "containment timeout is a failure, not lifecycle proof"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    inspect(&receipt);
    assert_eq!(
        Directory::open(state_path, false).identity(),
        state.identity()
    );
    let original_inode = state
        .regular(".gpu.lease")
        .expect("retain original boot lease inode");
    let original_stat = fs::fstat(&original_inode).expect("original boot lease identity");
    assert_eq!([original_stat.st_dev, original_stat.st_ino], receipt.lease);
    assert_eq!(
        state.lease_identity(),
        receipt.lease,
        "same boot lease inode"
    );
    assert!(
        matches!(
            crate::gpu::lease::acquire(state_path),
            Err(crate::gpu::lease::LeaseError::Unavailable { .. })
        ),
        "live child must still hold the GPU lease"
    );
    assert_eq!(
        state.lease_identity(),
        receipt.lease,
        "no lease replacement during contention"
    );
    assert!(child.0.try_wait().expect("child still alive").is_none());
    let witness = receipt.witness;
    record.publish(
        ".n2-parent",
        b"N2ACK001",
        &[
            run[0],
            run[1],
            witness[13],
            witness[14],
            receipt.lease[0],
            receipt.lease[1],
        ],
    );
    let exit_limit = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.0.try_wait().expect("child exit") {
            break status;
        }
        assert!(
            Instant::now() < exit_limit,
            "parent handshake did not reach actual process exit"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    let lease =
        crate::gpu::lease::acquire(state_path).expect("lease released by actual child exit");
    assert_eq!(
        state.lease_identity(),
        receipt.lease,
        "same inode after process exit"
    );
    drop(lease);
    eprintln!(
        "N2 actual_return={:?} mapping={} OS={:?} times_ns={:?} return_diagnostics={:?} collected_diagnostics={:?} ticket_ack={:?} witness={:?} lease={:?} model_failures={:?}",
        receipt.actual,
        receipt.mapping,
        status.code(),
        receipt.times,
        receipt.returned,
        receipt.collected,
        receipt.ack,
        receipt.witness,
        receipt.lease,
        receipt.model_failures
    );
    // Expected outcome is exclusively a parent assertion. A causal Clean mutant
    // completes the handshake and visibly exits OS0 before it is rejected here.
    assert_eq!(receipt.actual, RunExit::Runtime);
    assert_eq!(receipt.mapping, receipt.actual.exit().code());
    assert_eq!(
        status.code(),
        Some(1),
        "panic/watchdog/signal/kill is not the returned lifecycle exit"
    );
}
