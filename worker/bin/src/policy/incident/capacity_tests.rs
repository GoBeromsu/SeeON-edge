use super::{Duration, IdentityError, IncidentError, JsonError, Limits, Ordering, io};
use super::{assert_uuid4, attempts, charge, event, setup};

#[test]
fn exact_encoded_fit_duplicate_at_capacity_and_oversize_do_not_evict_or_mint() {
    let source = event();
    let encoded = r#"["camera","fall","fall","boot:epoch:fall:none:7:7:11:1",0,0]"#;
    assert_eq!(charge(&source, Duration::ZERO), encoded.len());
    let (mut manager, _, ids) = setup(encoded.len());
    let first = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    assert_eq!(manager.accounted_bytes(), encoded.len());
    let mut other = source.clone();
    other.identity.push('x');
    assert_eq!(
        manager.admit(&other, Duration::ZERO),
        Err(IncidentError::Capacity)
    );
    assert_eq!(
        manager
            .admit(&source, Duration::new(29, 999_999_999))
            .unwrap(),
        None
    );
    assert_eq!(manager.last_audit_snapshot(), Some(&first.audit));
    assert_eq!(manager.cooldown_suppressed_total(), 1);
    assert_eq!(attempts(&ids), 1);
    assert_eq!(manager.accounted_bytes(), encoded.len());
    for budget in [0, encoded.len() - 1] {
        let (mut too_small, _, unused) = setup(budget);
        assert_eq!(
            too_small.admit(&source, Duration::ZERO),
            Err(IncidentError::Capacity)
        );
        assert_eq!(attempts(&unused), 0);
        assert_eq!(too_small.accounted_bytes(), 0);
        assert_eq!(too_small.last_audit_snapshot(), None);
    }
}

#[test]
fn expiry_and_repeated_expired_release_debit_only_the_original_record() {
    let source = event();
    let exact = Duration::from_secs(30);
    let budget = charge(&source, exact);
    let (mut manager, _, ids) = setup(budget);
    let first = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    let mut other = source.clone();
    other.identity = "boot:epoch:fall:none:7:7:11:2".into();
    assert_eq!(
        manager.admit(&other, Duration::from_secs(29)),
        Err(IncidentError::Capacity)
    );
    let next = manager.admit(&other, exact).unwrap().unwrap();
    assert_eq!(manager.accounted_bytes(), budget);
    manager.release(first.release.clone());
    manager.release(first.release);
    assert_eq!(manager.accounted_bytes(), budget);
    manager.release(next.release.clone());
    manager.release(next.release);
    assert_eq!(manager.accounted_bytes(), 0);
    assert_eq!(manager.last_audit_snapshot(), Some(&next.audit));
    assert_eq!(attempts(&ids), 2);
}

#[test]
fn identity_allocation_errors_commit_neither_cooldown_nor_audit() {
    let source = event();
    for invalid in [false, true] {
        let (mut manager, _, ids) = setup(Limits::default().max_bytes);
        let first = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
        let bytes = manager.accounted_bytes();
        let mut other = source.clone();
        other.identity.push('x');
        ids.fail.store(!invalid, Ordering::SeqCst);
        ids.invalid.store(invalid, Ordering::SeqCst);
        let kind = if invalid {
            io::ErrorKind::InvalidData
        } else {
            io::ErrorKind::Other
        };
        assert_eq!(
            manager.admit(&other, Duration::ZERO),
            Err(IncidentError::Identity(IdentityError::Io(kind)))
        );
        assert_eq!(manager.accounted_bytes(), bytes);
        assert_eq!(manager.last_audit_snapshot(), Some(&first.audit));
        assert_eq!(manager.cooldown_suppressed_total(), 0);
        ids.fail.store(false, Ordering::SeqCst);
        ids.invalid.store(false, Ordering::SeqCst);
        assert!(manager.admit(&other, Duration::ZERO).unwrap().is_some());
        assert_eq!(attempts(&ids), 3);
    }
}

#[test]
fn nonfinite_source_time_is_typed_serialization_failure_without_commit() {
    let (mut manager, _, ids) = setup(Limits::default().max_bytes);
    let source = event();
    let first = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    let bytes = manager.accounted_bytes();
    let mut other = source.clone();
    other.identity.push('x');
    for time in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        other.time_sec = time;
        assert_eq!(
            manager.admit(&other, Duration::ZERO),
            Err(IncidentError::Serialization(JsonError::NonFinite))
        );
        assert_eq!(manager.accounted_bytes(), bytes);
        assert_eq!(manager.last_audit_snapshot(), Some(&first.audit));
        assert_eq!(attempts(&ids), 1);
    }
    other.time_sec = 10.0;
    assert!(manager.admit(&other, Duration::ZERO).unwrap().is_some());
}

#[test]
fn suppression_and_capacity_arithmetic_cannot_wrap() {
    let source = event();
    let (mut manager, _, ids) = setup(usize::MAX);
    let first = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    let bytes = manager.accounted_bytes();
    manager.cooldown.suppressed_total = u64::MAX;
    assert_eq!(
        manager.admit(&source, Duration::ZERO),
        Err(IncidentError::CounterOverflow)
    );
    assert_eq!(manager.cooldown_suppressed_total(), u64::MAX);
    assert_eq!(manager.accounted_bytes(), bytes);
    assert_eq!(manager.last_audit_snapshot(), Some(&first.audit));
    assert_eq!(attempts(&ids), 1);
    manager.cooldown.accounted_bytes = usize::MAX;
    let mut other = source.clone();
    other.identity.push('x');
    assert_eq!(
        manager.admit(&other, Duration::ZERO),
        Err(IncidentError::Capacity)
    );
    assert_eq!(manager.accounted_bytes(), usize::MAX);
    assert_eq!(attempts(&ids), 1);
}

#[test]
fn returned_admission_owns_uuid_even_when_identity_cache_cannot_retain_it() {
    let source = event();
    let (mut manager, _, ids) = setup(charge(&source, Duration::ZERO));
    let pending = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    let retained = pending.clone();
    manager.reset();
    let later = manager.admit(&source, Duration::ZERO).unwrap().unwrap();
    assert_ne!(pending.event.identity, later.event.identity);
    assert_eq!(attempts(&ids), 2);
    assert_eq!(pending, retained);
    assert_uuid4(&pending.event.identity);
    assert_eq!(pending.audit.edge_event_id, pending.event.identity);
}
