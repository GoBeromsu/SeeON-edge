//! Bounded polls for the native publication harness.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(super) fn poll_until<T>(
    deadline: Instant,
    what: &str,
    mut attempt: impl FnMut() -> Option<T>,
) -> T {
    loop {
        assert!(Instant::now() < deadline, "{what}");
        if let Some(value) = attempt() {
            assert!(Instant::now() < deadline, "{what}: result arrived late");
            return value;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub(super) fn wall_seconds(time: SystemTime) -> f64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => after.as_secs_f64(),
        Err(before) => -before.duration().as_secs_f64(),
    }
}
