//! Approved Stage4 shutdown order: only root may authorize media close, and
//! every phase shares one absolute deadline. These are control-unit proofs,
//! not substitutes for native finalization/order/lease GPU scenarios.

use std::sync::Arc;
use std::time::Duration;

use seeon_ml_worker::media::shutdown::{ShutdownControl, ShutdownError};
use seeon_ml_worker::shutdown::{DeadlineError, ShutdownDeadline};

fn shared_deadline(budget: Duration) -> Arc<ShutdownDeadline> {
    Arc::new(ShutdownDeadline::new(budget).expect("valid shared shutdown budget"))
}

#[test]
fn close_requires_explicit_unexpired_root_permission() {
    let shared = shared_deadline(Duration::from_secs(25));
    let control = ShutdownControl::new(Arc::clone(&shared));
    let start = Duration::from_secs(7);
    assert_eq!(control.permit_close(start), Err(ShutdownError::NotStarted));
    assert!(!control.close_permitted(start));
    let deadline = control.begin(start).expect("representable observation");
    assert_eq!(shared.requested_at(), Some(start));
    assert_eq!(shared.deadline(), Some(deadline));
    assert!(!control.close_permitted(start + Duration::from_secs(1)));
    control
        .permit_close(start + Duration::from_secs(2))
        .expect("root completed prior phases");
    assert!(control.close_permitted(start + Duration::from_secs(2)));
    assert!(!control.close_permitted(deadline));
    assert_eq!(control.permit_close(deadline), Err(ShutdownError::Expired));
    assert!(!control.close_permitted(deadline + Duration::from_secs(1)));
}

#[test]
fn repeated_stop_requests_cannot_extend_or_reopen_the_deadline() {
    let control = ShutdownControl::new(shared_deadline(Duration::from_secs(25)));
    let deadline = control
        .begin(Duration::from_secs(13))
        .expect("representable observation");
    assert_eq!(deadline, Duration::from_secs(38));
    assert_eq!(control.begin(Duration::from_secs(14)), Ok(deadline));
    control
        .permit_close(Duration::from_secs(14))
        .expect("timely permission");
    assert_eq!(control.begin(Duration::from_secs(100)), Ok(deadline));
    assert_eq!(control.deadline(), Some(deadline));
    assert_eq!(
        control.permit_close(Duration::from_secs(100)),
        Err(ShutdownError::Expired)
    );
    assert!(!control.close_permitted(Duration::from_secs(100)));
}

#[test]
fn native_stop_budget_is_positive_and_never_rounds_past_the_deadline() {
    let control = ShutdownControl::new(shared_deadline(Duration::from_millis(25)));
    let start = Duration::from_secs(3);
    assert_eq!(control.remaining_ms(start), None);
    let deadline = control.begin(start).expect("representable observation");
    assert_eq!(
        control.remaining_ms(start + Duration::from_millis(7)),
        Some(18)
    );
    assert_eq!(
        control.remaining_ms(deadline - Duration::from_millis(1)),
        Some(1)
    );
    assert_eq!(
        control.remaining_ms(deadline - Duration::from_micros(999)),
        None
    );
    assert_eq!(control.remaining_ms(deadline), None);
    assert_eq!(
        control.remaining_ms(deadline + Duration::from_millis(1)),
        None
    );
}

#[test]
fn signal_deadline_precedes_control_construction_and_late_media_begin() {
    let shared = shared_deadline(Duration::from_secs(25));
    assert_eq!(
        shared.request_at(Duration::from_secs(3)),
        Ok(Duration::from_secs(28))
    );
    let control = ShutdownControl::new(Arc::clone(&shared));
    let late = Duration::from_secs(27);
    assert_eq!(control.deadline(), Some(Duration::from_secs(28)));
    assert_eq!(control.remaining_ms(late), Some(1_000));
    assert!(!control.close_permitted(late));
    control
        .permit_close(late)
        .expect("shared shutdown is active");
    assert_eq!(control.begin(late), Ok(Duration::from_secs(28)));
    assert_eq!(shared.requested_at(), Some(Duration::from_secs(3)));
    assert!(control.close_permitted(late));
    assert_eq!(
        control.begin(Duration::from_secs(30)),
        Ok(Duration::from_secs(28))
    );
    assert!(!control.close_permitted(Duration::from_secs(30)));
    assert_eq!(control.remaining_ms(Duration::from_secs(30)), None);
}

#[test]
fn earlier_signal_publication_revokes_existing_permission_at_current_expiry() {
    let shared = shared_deadline(Duration::from_secs(25));
    let control = ShutdownControl::new(Arc::clone(&shared));
    assert_eq!(
        control.begin(Duration::from_secs(13)),
        Ok(Duration::from_secs(38))
    );
    let now = Duration::from_secs(30);
    control.permit_close(now).expect("initial deadline is live");
    assert!(control.close_permitted(now));
    assert_eq!(control.remaining_ms(now), Some(8_000));

    assert_eq!(
        shared.request_at(Duration::from_secs(5)),
        Ok(Duration::from_secs(30))
    );
    assert_eq!(control.deadline(), Some(now));
    assert!(!control.close_permitted(now));
    assert_eq!(control.permit_close(now), Err(ShutdownError::Expired));
    assert_eq!(control.remaining_ms(now), None);
    assert_eq!(control.begin(now), Ok(now));
    assert!(!control.close_permitted(now));
}

#[test]
fn begin_overflow_does_not_fabricate_a_deadline_or_permission() {
    let shared = shared_deadline(Duration::from_millis(25));
    let control = ShutdownControl::new(Arc::clone(&shared));
    assert_eq!(
        control.begin(Duration::from_nanos(u64::MAX - 1)),
        Err(DeadlineError::DeadlineOverflow)
    );
    assert_eq!(shared.deadline(), None);
    assert_eq!(control.deadline(), None);
    assert_eq!(shared.requested_at(), None);
    assert_eq!(
        control.permit_close(Duration::ZERO),
        Err(ShutdownError::NotStarted)
    );
    assert!(!control.close_permitted(Duration::ZERO));
    assert_eq!(control.remaining_ms(Duration::ZERO), None);
}
