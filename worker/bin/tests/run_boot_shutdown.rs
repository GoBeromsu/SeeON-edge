//! Public cancellation proof without native startup. Intermediate owner readiness,
//! drain ordering and genuine GPU-failure reports require real GPU qualification.

use std::error::Error;
use std::io;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use seeon_ml_worker::cli::Flags;
use seeon_ml_worker::config::env::Env;
use seeon_ml_worker::exit::Exit;
use seeon_ml_worker::gpu::lease;
use seeon_ml_worker::run::boot::{BootError, BootStopped, boot};
use seeon_ml_worker::run::models::CleanupError;
use seeon_ml_worker::run::{BootPolicy, BootStatusContext, ReportIdentity, Settings};
use seeon_ml_worker::seam::Clock;
use seeon_ml_worker::shutdown::{DeadlineError, ShutdownDeadline};
use seeon_ml_worker::telemetry::status::ClipExportStatus;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct TestClock(Mutex<Duration>);

impl TestClock {
    fn at(now: Duration) -> Self {
        Self(Mutex::new(now))
    }

    fn time(&self) -> MutexGuard<'_, Duration> {
        match self.0.lock() {
            Ok(time) => time,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl Clock for TestClock {
    fn monotonic(&self) -> Duration {
        *self.time()
    }

    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    fn pause(&self, limit: Duration) {
        assert!(!limit.is_zero());
        let mut now = self.time();
        *now = now.saturating_add(limit);
    }
}

struct Fixture {
    state_dir: PathBuf,
    listener: TcpListener,
}

impl Fixture {
    fn new(label: &str) -> TestResult<Self> {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        std::fs::create_dir_all(&root)?;
        let state_dir = root.join(format!("boot-shutdown-{}-{label}", std::process::id()));
        std::fs::create_dir(&state_dir)?;
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            state_dir,
            listener,
        })
    }

    fn run(&self, clock: &dyn Clock, shutdown: Arc<ShutdownDeadline>) -> TestResult<BootError> {
        let mut env = Env::new();
        env.insert("RELAY_TOKEN".to_owned(), "boot-test-token".to_owned());
        for key in [
            "ML_WORKER_FALL_ENGINE_PATH",
            "ML_WORKER_BED_ENGINE_PATH",
            "ML_WORKER_STORED_POSE_ENGINE_PATH",
        ] {
            env.insert(
                key.to_owned(),
                self.state_dir.join("absent-engine").display().to_string(),
            );
        }
        let settings = Settings::from_flags(
            env,
            Flags {
                heartbeat_on_start: false,
                state_dir: Some(self.state_dir.clone()),
                auxiliary_runtime:
                    seeon_ml_worker::config::model_bundle::identity::AuxiliaryRuntime::TensorRt,
            },
            BootPolicy {
                device_ordinal: 0,
                stored_pose_threshold: 0.5,
                deployed_batch: None,
                readiness_budget: Duration::from_secs(1),
            },
        )
        .map_err(|error| io::Error::other(format!("settings: {error:?}")))?;
        let report = BootStatusContext::new(
            &format!("http://{}", self.listener.local_addr()?),
            "boot-test-token",
            ReportIdentity {
                facility_id: "boot-test-facility".to_owned(),
                seq: 1,
                generation: None,
                clip_export: ClipExportStatus {
                    enabled: false,
                    version: 1,
                },
                started_at_sec: 1.0,
            },
        )
        .map_err(|error| io::Error::other(format!("report context: {error:?}")))?;
        match boot(settings, report, clock, shutdown) {
            Err(error) => Ok(error),
            Ok(_) => Err(io::Error::other("cancelled boot returned warmed owners").into()),
        }
    }

    fn no_status_attempt(&self) -> TestResult {
        match self.listener.accept() {
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(()),
            Err(error) => Err(error.into()),
            Ok(_) => Err(io::Error::other("cancellation contacted the status endpoint").into()),
        }
    }

    fn lease_available(&self) -> TestResult {
        let lease = lease::acquire(&self.state_dir)
            .map_err(|error| io::Error::other(format!("lease retained: {error:?}")))?;
        drop(lease);
        Ok(())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.state_dir);
    }
}

fn stopped(error: &BootError) -> TestResult<&BootStopped> {
    match error {
        BootError::Stopped(stopped) => Ok(stopped),
        BootError::Failed(failure) => {
            Err(io::Error::other(format!("cancellation became a boot failure: {failure:?}")).into())
        }
    }
}

#[test]
fn requested_shutdown_preempts_even_a_contended_lease_without_reporting() -> TestResult {
    let fixture = Fixture::new("requested")?;
    let held = lease::acquire(&fixture.state_dir)
        .map_err(|error| io::Error::other(format!("fixture lease: {error:?}")))?;
    let clock = TestClock::at(Duration::from_secs(10));
    let shared = Arc::new(ShutdownDeadline::new(Duration::from_secs(25))?);
    let original = shared.request_at(clock.monotonic())?;

    let outcome = fixture.run(&clock, Arc::clone(&shared))?;
    assert!(stopped(&outcome)?.cleanup.is_empty());
    assert_eq!(stopped(&outcome)?.exit(&clock), Exit::CleanShutdown);
    assert_eq!(outcome.exit(&clock), Exit::CleanShutdown);
    assert_eq!(shared.deadline(), Some(original));
    fixture.no_status_attempt()?;
    drop(held);
    fixture.lease_available()
}

#[test]
fn expired_shutdown_is_stopped_but_never_clean_or_reported() -> TestResult {
    let fixture = Fixture::new("expired")?;
    let shared = Arc::new(ShutdownDeadline::new(Duration::from_secs(25))?);
    let original = shared.request_at(Duration::from_secs(1))?;
    let clock = TestClock::at(original);

    let outcome = fixture.run(&clock, Arc::clone(&shared))?;
    assert_eq!(stopped(&outcome)?.cleanup, vec![CleanupError::Expired]);
    assert_eq!(outcome.exit(&clock), Exit::Runtime);
    assert_eq!(shared.deadline(), Some(original));
    fixture.no_status_attempt()?;
    fixture.lease_available()
}

#[test]
fn earlier_publication_revokes_clean_exit_of_the_same_stopped_outcome() -> TestResult {
    let fixture = Fixture::new("earlier")?;
    let shared = Arc::new(ShutdownDeadline::new(Duration::from_secs(25))?);
    shared.request_at(Duration::from_secs(10))?;
    let clock = TestClock::at(Duration::from_secs(30));
    let outcome = fixture.run(&clock, Arc::clone(&shared))?;
    assert_eq!(outcome.exit(&clock), Exit::CleanShutdown);

    shared.request_at(Duration::from_secs(5))?;
    assert_eq!(stopped(&outcome)?.exit(&clock), Exit::Runtime);
    assert_eq!(outcome.exit(&clock), Exit::Runtime);
    fixture.no_status_attempt()?;
    fixture.lease_available()
}

#[test]
fn cancellation_observation_overflow_is_a_control_error_not_clean_exit() -> TestResult {
    let fixture = Fixture::new("overflow")?;
    let shared = Arc::new(ShutdownDeadline::new(Duration::from_secs(25))?);
    let original = shared.request_at(Duration::from_secs(2))?;
    let clock = TestClock::at(Duration::MAX);

    let outcome = fixture.run(&clock, Arc::clone(&shared))?;
    assert_eq!(
        stopped(&outcome)?.cleanup,
        vec![CleanupError::Control(DeadlineError::TimestampOverflow)],
    );
    assert_eq!(outcome.exit(&clock), Exit::Runtime);
    assert_eq!(shared.deadline(), Some(original));
    fixture.no_status_attempt()?;
    fixture.lease_available()
}

#[test]
fn genuine_lease_failure_preserves_reason_but_rechecks_shutdown_deadline() -> TestResult {
    let fixture = Fixture::new("lease-failure")?;
    let held = lease::acquire(&fixture.state_dir)
        .map_err(|error| io::Error::other(format!("fixture lease: {error:?}")))?;
    let clock = TestClock::at(Duration::from_secs(10));
    let shared = Arc::new(ShutdownDeadline::new(Duration::from_secs(25))?);

    let outcome = fixture.run(&clock, Arc::clone(&shared))?;
    let BootError::Failed(failure) = &outcome else {
        return Err(io::Error::other("real lease refusal became cancellation").into());
    };
    assert_eq!(
        failure.reason,
        seeon_ml_worker::run::status::BootReason::GpuLease
    );
    assert_eq!(outcome.exit(&clock), Exit::RefuseToStart);
    // The real bounded report attempted one connection, even without a responder.
    let (connection, _) = fixture.listener.accept()?;
    drop(connection);
    fixture.no_status_attempt()?;

    *clock.time() = Duration::from_secs(30);
    shared.request_at(Duration::from_secs(5))?;
    assert_eq!(failure.exit(&clock), Exit::Runtime);
    assert_eq!(outcome.exit(&clock), Exit::Runtime);
    drop(held);
    fixture.lease_available()
}
