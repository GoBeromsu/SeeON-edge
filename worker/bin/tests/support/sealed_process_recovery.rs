//! Actual process death at completed publication boundaries, not power loss.
//! Media bytes are the existing synthetic T35 fixture, not native recordings.

use std::os::unix::process::ExitStatusExt;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rustix::process::{Signal, getpid, kill_process};

use super::*;

const ROOT: &str = "SEEON_TEST_SEALED_PROCESS_ROOT";
const POINT: &str = "SEEON_TEST_SEALED_PROCESS_POINT";
const TEST: &str = "process_recovery::sigkill_after_durable_boundaries_replays_once";

struct OwnedChild {
    child: Child,
    reaped: bool,
}

impl OwnedChild {
    fn wait(&mut self) -> ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().expect("observe owned child") {
                self.reaped = true;
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "sealed producer exceeded deadline"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn die() -> ! {
    kill_process(getpid(), Signal::KILL).expect("SIGKILL owned producer");
    panic!("producer survived SIGKILL");
}

fn bench_at(work: PathBuf) -> Bench {
    Bench {
        sidecars: SealedSidecars::new(work.join("state/flow-sealed")),
        work,
    }
}

fn producer(work: PathBuf, point: &str) -> ! {
    assert!(["persisted", "before-put", "after-put"].contains(&point));
    let bench = bench_at(work);
    let cases = cases();
    bench.persist_with_media(&cases, "non-ascii", true);
    if point == "persisted" {
        die();
    }
    let store = ClipStore::new(bench.work.join("store"));
    let queue_dir = bench.work.join("store/delivery-queue");
    fs::create_dir_all(&queue_dir).expect("queue directory");
    let queue = DeliveryQueue::open(&queue_dir, true).expect("producer queue");
    bench
        .sidecars
        .replay(
            &bench.camera(&cases),
            |recovery| -> ReplayOutcome<Published, PublishError> {
                if point == "before-put" {
                    die();
                }
                let published = publish_recovery(&store, &queue, recovery);
                assert!(
                    matches!(published, ReplayOutcome::Published(_)),
                    "{published:?}"
                );
                // The real publisher has returned, but replay has not retired its
                // sidecar. SIGKILL cannot run destructors or unwind this callback.
                die();
            },
        )
        .expect("producer replay");
    panic!("producer did not reach the selected death boundary");
}

#[test]
fn sigkill_after_durable_boundaries_replays_once() {
    if let Some(work) = std::env::var_os(ROOT) {
        producer(
            PathBuf::from(work),
            &std::env::var(POINT).expect("child point"),
        );
    }
    let run = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let scratch = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&scratch).expect("test scratch parent");
    for point in ["persisted", "before-put", "after-put"] {
        let work = scratch.join(format!(
            "flow-sealed-process-{}-{run}-{point}",
            std::process::id()
        ));
        fs::create_dir(&work).expect("exclusive owned work directory");
        let mut child = OwnedChild {
            child: Command::new(std::env::current_exe().expect("test executable"))
                .args(["--exact", TEST, "--test-threads=1", "--nocapture"])
                .env(ROOT, &work)
                .env(POINT, point)
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("spawn owned producer"),
            reaped: false,
        };
        assert_eq!(child.wait().signal(), Some(9), "{point}: actual SIGKILL");
        let bench = bench_at(work);
        let cases = cases();
        let camera = bench.camera(&cases);
        let sidecar = bench.sidecars.directory().join(format!("{A}.json"));
        let bytes = fs::read(&sidecar).expect("attribution survives process death");
        let mut expected: Value = serde_json::from_slice(&golden_sidecar(A)).expect("golden");
        expected["path"] = Value::from(
            bench
                .work
                .join("media")
                .join(format!("{A}.mp4"))
                .to_str()
                .expect("media path"),
        );
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), expected);
        let pending = bench.sidecars.pending(&camera).expect("fresh pending read");
        assert_eq!(pending.recoveries.len(), 1);
        assert!(pending.malformed.is_empty());
        assert_eq!(
            fs::read(&sidecar).unwrap(),
            bytes,
            "inspection preserves attribution"
        );

        let store = ClipStore::new(bench.work.join("store"));
        let queue_dir = bench.work.join("store/delivery-queue");
        fs::create_dir_all(&queue_dir).expect("recovery queue directory");
        let queue = DeliveryQueue::open(&queue_dir, true).expect("fresh recovery queue");
        let before = queue.entries().expect("queue before recovery");
        assert_eq!(before.len(), usize::from(point == "after-put"));
        let report = bench
            .sidecars
            .replay(&camera, |recovery| {
                publish_recovery(&store, &queue, recovery)
            })
            .expect("replay after process death");
        assert_eq!(
            report,
            ReplayReport {
                published: 1,
                ..ReplayReport::default()
            }
        );
        let after = queue.entries().expect("published queue");
        assert_eq!(after.len(), 1, "{point}: one immutable clip entry");
        assert_eq!(after[0]["clip_id"], A);
        if point == "after-put" {
            assert_eq!(after, before, "existing publication is not replaced");
        }
        assert!(bench.sidecar_names().is_empty());
        let repeated = bench
            .sidecars
            .replay(&camera, |_| -> ReplayOutcome<(), ()> {
                panic!("retired attribution must not be replayed");
            })
            .expect("second replay");
        assert_eq!(repeated, ReplayReport::default());
        assert_eq!(queue.entries().unwrap(), after);
        eprintln!("T35_PROCESS_BOUNDARY={point} signal=9 recovered=1 repeated=0");
    }
}
