//! Decision handoff retention, failure and overflow regressions.

use seeon_worker::trace::{DecisionTraceValueName, NumericTraceValue};

use crate::run::pump::PolicySink;

use super::*;

#[test]
fn disabled_exporter_retires_snapshots_and_scores_without_disabling_events() {
    let mut fixture = Fixture::new(64);
    fixture.session.records = None;
    fixture.session.decision_policies.clear();
    let mut stage = stage(19);
    let mut sink = LiveSink::new(fixture.clock.clone());
    let mut sequence = 0;
    let requests = next_due(&mut stage, 19, &mut sequence, &[7]);
    consume(&mut stage, &requests, 2.0, &mut sink);
    stage
        .skip_outside_window(&input(19, sequence, &[7]), &mut |update| {
            sink.decision(update)
        })
        .unwrap();
    assert_eq!(sink.ready.len(), 2);
    assert_eq!(sink.triggered.len(), 1);
    fixture.deliver(&mut sink).unwrap();
    assert!(sink.ready.is_empty() && sink.scores.is_empty() && sink.triggered.is_empty());
    assert!(!fixture.lanes.has_work());
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        1
    );
    assert_eq!(fixture.session.publications.recorders[0].pending(), 1);
}

#[test]
fn conversion_and_attribution_failures_retire_only_the_emitted_prefix() {
    let mut fixture = Fixture::new(64);
    let mut stage = stage(19);
    let mut sink = LiveSink::new(fixture.clock.clone());
    let mut sequence = 0;
    let requests = next_due(&mut stage, 19, &mut sequence, &[9, 7, 2]);
    let snapshots = consume(&mut stage, &requests, 2.0, &mut sink);
    assert_eq!(
        snapshots
            .iter()
            .map(|snapshot| snapshot.track_id)
            .collect::<Vec<_>>(),
        vec![Some(2), Some(7), Some(9)]
    );
    let outside = input(19, sequence, &[]);
    stage
        .skip_outside_window(&outside, &mut |update| sink.decision(update))
        .unwrap();
    let events = sink.triggered.clone();
    assert_eq!(events.len(), 3);
    let original = snapshots[1].clone();
    let mut values = original.values().clone();
    values.insert(
        DecisionTraceValueName::TransitionVotes,
        NumericTraceValue::Integer(usize::MAX),
    );
    let invalid = DecisionTraceSnapshot::new(
        original.reason,
        (original.previous_state, original.current_state),
        original.triggered,
        original.track_id,
        original.bed_id,
        values,
        original.missing_values().clone(),
    )
    .unwrap();
    assert!(matches!(
        crate::run::decision::adapt_decision(&invalid),
        Err(crate::run::decision::DecisionError::IntegerOutOfRange(_))
    ));
    sink.ready[0].decisions[1].snapshot = invalid;
    sink.ready[1].frame.source_id = 99;
    assert!(matches!(
        fixture.deliver(&mut sink),
        Err(RuntimeError::Identity)
    ));
    assert!(sink.ready[0].scores.is_empty());
    assert_eq!(
        sink.ready[0]
            .decisions
            .iter()
            .map(|item| item.snapshot.track_id)
            .collect::<Vec<_>>(),
        vec![Some(7), Some(9)]
    );
    assert_eq!(sink.triggered, events);
    let prefix = fixture.drain(19);
    assert_eq!(of_kind(&prefix, RecordKind::ModelScore).len(), 3);
    let first = of_kind(&prefix, RecordKind::PolicyDecision);
    assert_eq!(first.len(), 1);
    assert_snapshot(
        &fixture,
        first[0],
        requests[0].frame,
        &snapshots[0],
        Some(0),
    );
    assert!(matches!(
        fixture.deliver(&mut sink),
        Err(RuntimeError::Identity)
    ));
    assert!(fixture.drain(19).is_empty());
    sink.ready[0].decisions[0].snapshot = original;
    assert!(matches!(
        fixture.deliver(&mut sink),
        Err(RuntimeError::Identity)
    ));
    assert_eq!(sink.ready.len(), 1);
    assert_eq!(sink.ready[0].frame.source_id, 99);
    assert_eq!(sink.triggered, events);
    let middle = fixture.drain(19);
    let decisions = of_kind(&middle, RecordKind::PolicyDecision);
    assert_eq!(decisions.len(), 2);
    for (index, record) in decisions.iter().enumerate() {
        assert_snapshot(
            &fixture,
            record,
            requests[0].frame,
            &snapshots[index + 1],
            Some(0),
        );
        assert_eq!(record.body().producer_sequence, index as u64 + 1);
    }
    assert!(of_kind(&middle, RecordKind::ModelScore).is_empty());
    assert!(matches!(
        fixture.deliver(&mut sink),
        Err(RuntimeError::Identity)
    ));
    assert!(fixture.drain(19).is_empty());
    sink.ready[0].frame.source_id = 19;
    fixture.deliver(&mut sink).unwrap();
    let suffix = fixture.drain(19);
    let decisions = of_kind(&suffix, RecordKind::PolicyDecision);
    assert_eq!(decisions.len(), 1);
    assert_snapshot(
        &fixture,
        decisions[0],
        outside.identity,
        &DecisionTraceSnapshot::outside_detection_window(),
        None,
    );
    assert_eq!(decisions[0].body().producer_sequence, 3);
    assert_eq!(of_kind(&suffix, RecordKind::EventDelivery).len(), 3);
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        3
    );
    assert!(sink.ready.is_empty() && sink.triggered.is_empty());
    fixture.deliver(&mut sink).unwrap();
    assert!(fixture.drain(19).is_empty());
}

#[test]
fn full_policy_lane_drops_incoming_records_without_replaying_them() {
    let mut fixture = Fixture::new(1);
    let mut stage = stage(19);
    let mut sink = LiveSink::new(fixture.clock.clone());
    let mut sequence = 0;
    let requests = next_due(&mut stage, 19, &mut sequence, &[9, 7, 2]);
    consume(&mut stage, &requests, -2.0, &mut sink);
    fixture.deliver(&mut sink).unwrap();
    assert!(sink.ready.is_empty());
    let drained = fixture
        .lanes
        .drain_for("camera-19", BOOT, 256)
        .unwrap()
        .unwrap();
    let decisions = of_kind(&drained.records, RecordKind::PolicyDecision);
    assert_eq!(decisions.len(), 1);
    assert_eq!(wire(decisions[0])["payload"]["track_id"], 2);
    assert_eq!(
        drained
            .gaps
            .iter()
            .filter(|gap| gap.producer == decisions[0].body().producer
                && gap.cause == crate::records::lanes::LANE_OVERFLOW)
            .map(|gap| gap.record_count)
            .sum::<u64>(),
        2
    );
    fixture.deliver(&mut sink).unwrap();
    assert!(!fixture.lanes.has_work());
    stage
        .skip_outside_window(&input(19, sequence, &[]), &mut |update| {
            sink.decision(update)
        })
        .unwrap();
    fixture.deliver(&mut sink).unwrap();
    let records = fixture.drain(19);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].body().producer_sequence, 3);
}
