use std::io;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::super::identity::{IdentityError, Limits};
use super::IncidentError;
use super::test_support::{assert_preserved, assert_uuid4, attempts, charge, event, setup};
use crate::json::JsonError;

#[path = "capacity_tests.rs"]
mod capacity_tests;

#[test]
fn genuine_opaque_event_becomes_uuid_without_mutating_source_or_other_fields() {
    let source = event();
    let original = source.clone();
    assert_eq!(source.identity, "boot:epoch:fall:none:7:7:11:1");
    let (mut manager, clock, ids) = setup(Limits::default().max_bytes);
    let admission = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    assert_uuid4(&admission.event.identity);
    assert_eq!(source, original);
    assert_preserved(&source, &admission);
    assert_eq!(manager.last_audit_snapshot(), Some(&admission.audit));
    assert_eq!(attempts(&ids), 1);
    assert_eq!(clock.monotonic_samples.load(Ordering::SeqCst), 0);
}

#[test]
fn source_key_is_exact_python_ascii_json_with_escaping_and_float_forms() {
    let mut source = event();
    source.facility_id = "병원𐐀".into();
    source.camera_id = "c\"\\\n".into();
    source.domain = "낙상".into();
    let prefix = r#"["\ubcd1\uc6d0\ud801\udc00","c\"\\\n","\ub099\uc0c1","fall","boot:epoch:fall:none:7:7:11:1","#;
    let record =
        r#"["c\"\\\n","\ub099\uc0c1","fall","boot:epoch:fall:none:7:7:11:1",4294967296,999999999]"#;
    assert_eq!(
        charge(&source, Duration::new(4_294_967_296, 999_999_999)),
        record.len()
    );
    let (mut manager, _, _) = setup(Limits::default().max_bytes);
    for (time, rendered) in [
        (-0.0, "-0.0"),
        (1.0, "1.0"),
        (0.0001, "0.0001"),
        (1e-5, "1e-05"),
        (1e-7, "1e-07"),
        (1e16, "1e+16"),
    ] {
        source.time_sec = time;
        let admission = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
        assert_eq!(admission.source_key, format!("{prefix}{rendered}]"));
        assert!(admission.source_key.is_ascii());
        assert_eq!(admission.event.time_sec.to_bits(), time.to_bits());
        assert_eq!(admission.audit.time_sec.to_bits(), time.to_bits());
        manager.release(admission.release);
    }
}

#[test]
fn cooldown_expires_only_at_30_monotonic_seconds_despite_source_and_wall_time() {
    let mut source = event();
    let (mut manager, clock, ids) = setup(Limits::default().max_bytes);
    let first = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    source.time_sec = -1_000.0;
    clock.wall_seconds.store(-3_600, Ordering::SeqCst);
    assert_eq!(
        manager.admit(&source, Duration::from_secs(29)).unwrap(),
        None
    );
    clock.wall_seconds.store(1_000_000_000, Ordering::SeqCst);
    assert_eq!(
        manager
            .admit(&source, Duration::new(29, 999_999_999))
            .unwrap(),
        None
    );
    assert_eq!(manager.last_audit_snapshot(), Some(&first.audit));
    assert_eq!(attempts(&ids), 1);
    assert_eq!(manager.cooldown_suppressed_total(), 2);
    assert!(
        manager
            .admit(&source, Duration::from_secs(30))
            .unwrap()
            .is_some()
    );
    assert_eq!(attempts(&ids), 2);
}

#[test]
fn only_camera_domain_event_type_and_original_identity_define_cooldown() {
    let source = event();
    let (mut manager, _, ids) = setup(Limits::default().max_bytes);
    manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    for field in 0..4 {
        let mut other = source.clone();
        match field {
            0 => other.camera_id.push('x'),
            1 => other.domain.push('x'),
            2 => other.event_type.push('x'),
            _ => other.identity.push('x'),
        }
        assert!(manager.admit(&other, Duration::ZERO).unwrap().is_some());
    }
    let mut same_key = source.clone();
    same_key.facility_id.push('x');
    same_key.time_sec = -50.0;
    same_key.probability = None;
    same_key.person_id = None;
    same_key.bed_id = Some(12);
    assert_eq!(manager.admit(&same_key, Duration::ZERO).unwrap(), None);
    assert_eq!(attempts(&ids), 5);
}

#[test]
fn regressing_sample_is_fatal_without_changing_cooldown_counter_or_audit() {
    let source = event();
    let (mut manager, _, ids) = setup(Limits::default().max_bytes);
    let admitted = manager
        .admit(&source, Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let bytes = manager.accounted_bytes();
    let error = IncidentError::ClockRegression {
        previous: Duration::from_secs(10),
        now: Duration::from_secs(9),
    };
    assert_eq!(manager.admit(&source, Duration::from_secs(9)), Err(error));
    assert_eq!(manager.last_audit_snapshot(), Some(&admitted.audit));
    assert_eq!(manager.accounted_bytes(), bytes);
    assert_eq!(manager.cooldown_suppressed_total(), 0);
    assert_eq!(attempts(&ids), 1);
    manager.reset();
    assert_eq!(manager.admit(&source, Duration::from_secs(9)), Err(error));
}

#[test]
fn reset_clears_cooldown_and_audit_but_reuses_identity_and_preserves_counter() {
    let source = event();
    let (mut manager, _, ids) = setup(Limits::default().max_bytes);
    let first = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    assert_eq!(manager.admit(&source, Duration::ZERO).unwrap(), None);
    manager.reset();
    manager.reset();
    assert_eq!(manager.accounted_bytes(), 0);
    assert_eq!(manager.last_audit_snapshot(), None);
    assert_eq!(manager.cooldown_suppressed_total(), 1);
    let again = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    assert_eq!(again.event.identity, first.event.identity);
    assert_eq!(attempts(&ids), 1);
    assert_eq!(manager.accounted_bytes(), charge(&source, Duration::ZERO));
}
