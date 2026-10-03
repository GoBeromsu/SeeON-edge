//! Private lifecycle tests; synthetic failures are not native execution evidence.
use super::*;
use crate::seam::SystemClock;
use std::sync::mpsc;

fn empty() -> ModelOwners {
    ModelOwners {
        fall: None,
        bed: None,
        stored_pose: None,
        fall_responses: None,
        stop: Arc::new(AtomicBool::new(false)),
        shutdown: Arc::new(ShutdownDeadline::new(Duration::from_secs(2)).unwrap()),
        lease: None,
        states: [None, None, None],
    }
}

#[test]
fn late_failure_survives_removing_a_joined_owner() {
    let mut models = empty();
    let state = Arc::new(State::default());
    let actor_state = Arc::clone(&state);
    let stop = Arc::clone(&models.stop);
    let (requests, _queue) = mpsc::sync_channel(1);
    let (_ready, readiness) = mpsc::sync_channel(1);
    let thread = std::thread::spawn(move || {
        while !stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        actor_state.fail(Exit::Runtime);
    });
    models.states[0] = Some(Arc::clone(&state));
    models.fall = Some(Owner {
        thread,
        readiness,
        requests,
        state,
    });
    assert_eq!(models.failure(), None);
    assert!(models.close(&SystemClock::new()).is_empty());
    assert!(
        models.fall.is_none(),
        "finished handles were joined and removed"
    );
    assert_eq!(models.failure(), Some(Exit::Runtime));
    assert_eq!(
        models.failure(),
        Some(Exit::Runtime),
        "inspection is non-consuming"
    );
}

#[test]
fn accelerator_failure_outranks_cpu_failure_regardless_of_role_order() {
    for accelerator in 0..3 {
        let mut models = empty();
        for index in 0..3 {
            let state = Arc::new(State::default());
            state.fail(if index == accelerator {
                Exit::FatalAccelerator
            } else {
                Exit::Runtime
            });
            models.states[index] = Some(state);
        }
        assert_eq!(models.failure(), Some(Exit::FatalAccelerator));
    }
}
