//! Decision receipt identity, ordering and generation regressions.

use std::sync::mpsc;

use crate::run::pump::PolicySink;

use super::*;

#[test]
fn genuine_normal_triggered_and_outside_snapshots_keep_scores_and_event_handoff() {
    let mut fixture = Fixture::new(64);
    let mut stage = stage(19);
    let mut sink = LiveSink::new(fixture.clock.clone());
    let mut sequence = 0;
    let normal_requests = next_due(&mut stage, 19, &mut sequence, &[7]);
    let normal = consume(&mut stage, &normal_requests, -2.0, &mut sink);
    assert_eq!(normal[0].reason.as_str(), "below-threshold");
    let triggered_requests = next_due(&mut stage, 19, &mut sequence, &[7]);
    let triggered = consume(&mut stage, &triggered_requests, 2.0, &mut sink);
    assert!(triggered[0].triggered);
    assert_eq!(sink.triggered.len(), 1);
    let event = sink.triggered[0].event.clone();
    let outside = input(19, sequence, &[7]);
    stage
        .skip_outside_window(&outside, &mut |update| sink.decision(update))
        .unwrap();
    fixture.deliver(&mut sink).unwrap();
    let records = fixture.drain(19);
    let decisions = of_kind(&records, RecordKind::PolicyDecision);
    assert_eq!(decisions.len(), 3);
    assert_snapshot(
        &fixture,
        decisions[0],
        normal_requests[0].frame,
        &normal[0],
        Some(0),
    );
    assert_snapshot(
        &fixture,
        decisions[1],
        triggered_requests[0].frame,
        &triggered[0],
        Some(0),
    );
    assert_snapshot(
        &fixture,
        decisions[2],
        outside.identity,
        &DecisionTraceSnapshot::outside_detection_window(),
        None,
    );
    let scores = of_kind(&records, RecordKind::ModelScore);
    assert_eq!(scores.len(), 2);
    assert_eq!(
        scores
            .iter()
            .map(|record| record.body().frame_seq)
            .collect::<Vec<_>>(),
        vec![
            Some(normal_requests[0].frame.sequence),
            Some(triggered_requests[0].frame.sequence)
        ]
    );
    assert_eq!(wire(scores[0])["payload"]["raw_logit"], -2.0);
    assert_eq!(wire(scores[1])["payload"]["raw_logit"], 2.0);
    assert_eq!(
        scores[0].body().causal_unit_id,
        decisions[0].body().causal_unit_id
    );
    assert_eq!(
        scores[1].body().causal_unit_id,
        decisions[1].body().causal_unit_id
    );
    assert_eq!(of_kind(&records, RecordKind::EventDelivery).len(), 1);
    assert_eq!(event.person_id, Some(7));
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        1
    );
    assert_eq!(fixture.session.publications.recorders[0].pending(), 1);
    assert!(sink.ready.is_empty() && sink.triggered.is_empty());
    fixture.deliver(&mut sink).unwrap();
    assert!(fixture.drain(19).is_empty());
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        1
    );
}

#[test]
fn snapshot_order_is_independent_of_score_response_order() {
    let mut fixture = Fixture::new(64);
    let mut stage = stage(19);
    let mut sink = LiveSink::new(fixture.clock.clone());
    let mut sequence = 0;
    let mut requests = next_due(&mut stage, 19, &mut sequence, &[9, 2]);
    requests.sort_by_key(|request| std::cmp::Reverse(request.track_id));
    let snapshots = consume(&mut stage, &requests, -2.0, &mut sink);
    assert_eq!(
        snapshots
            .iter()
            .map(|snapshot| snapshot.track_id)
            .collect::<Vec<_>>(),
        vec![Some(2), Some(9)]
    );
    fixture.deliver(&mut sink).unwrap();
    let records = fixture.drain(19);
    let scores = of_kind(&records, RecordKind::ModelScore);
    let decisions = of_kind(&records, RecordKind::PolicyDecision);
    assert_eq!(
        scores
            .iter()
            .map(|record| wire(record)["payload"]["track_id"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![9, 2]
    );
    assert_eq!(decisions.len(), 2);
    for (record, snapshot) in decisions.iter().zip(&snapshots) {
        assert_snapshot(&fixture, record, requests[0].frame, snapshot, Some(0));
    }
    assert_eq!(
        decisions
            .iter()
            .map(|record| record.body().producer_sequence)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        scores[0].body().causal_unit_id,
        decisions[1].body().causal_unit_id
    );
    assert_eq!(
        scores[1].body().causal_unit_id,
        decisions[0].body().causal_unit_id
    );
}

#[test]
fn captured_generation_survives_eviction_and_track_reentry_before_delivery() {
    let mut fixture = Fixture::new(64);
    let mut stage = stage(19);
    let mut sink = LiveSink::new(fixture.clock.clone());
    let mut sequence = 0;
    let first_requests = next_due(&mut stage, 19, &mut sequence, &[7]);
    let first = consume(&mut stage, &first_requests, -2.0, &mut sink);
    let old_generation = stage.decider().generation_for(7).unwrap();
    let (sender, _receiver) = mpsc::sync_channel(64);
    for _ in 0..46 {
        stage
            .observe(&input(19, sequence, &[]), &sender, &mut |_| {})
            .unwrap();
        sequence += 1;
    }
    assert_eq!(stage.decider().generation_for(7), None);
    let second_requests = next_due(&mut stage, 19, &mut sequence, &[7]);
    let second = consume(&mut stage, &second_requests, -2.0, &mut sink);
    let new_generation = stage.decider().generation_for(7).unwrap();
    assert_ne!(old_generation, new_generation);
    assert_eq!(sink.ready[0].decisions[0].generation, Some(old_generation));
    fixture.deliver(&mut sink).unwrap();
    let records = fixture.drain(19);
    let decisions = of_kind(&records, RecordKind::PolicyDecision);
    assert_eq!(decisions.len(), 2);
    assert_snapshot(
        &fixture,
        decisions[0],
        first_requests[0].frame,
        &first[0],
        Some(old_generation),
    );
    assert_snapshot(
        &fixture,
        decisions[1],
        second_requests[0].frame,
        &second[0],
        Some(new_generation),
    );
    assert_ne!(
        decisions[0].body().causal_unit_id,
        decisions[1].body().causal_unit_id
    );
    let scores = of_kind(&records, RecordKind::ModelScore);
    assert_eq!(wire(scores[0])["payload"]["generation"], old_generation);
    assert_eq!(wire(scores[1])["payload"]["generation"], new_generation);
}

#[test]
fn camera_override_and_default_fallback_produce_distinct_canonical_trace_ids() {
    let mut fixture = Fixture::new(64);
    let mut sink = LiveSink::new(fixture.clock.clone());
    for source_id in [19, 20] {
        stage(source_id)
            .skip_outside_window(&input(source_id, 3, &[7]), &mut |update| {
                sink.decision(update)
            })
            .unwrap();
    }
    fixture.deliver(&mut sink).unwrap();
    let records = [fixture.drain(19), fixture.drain(20)];
    for (index, camera_records) in records.iter().enumerate() {
        assert_eq!(camera_records.len(), 1);
        assert_snapshot(
            &fixture,
            &camera_records[0],
            input(19 + index as u32, 3, &[7]).identity,
            &DecisionTraceSnapshot::outside_detection_window(),
            None,
        );
        let payload = wire(&camera_records[0]);
        assert_eq!(payload["payload"]["module_qualified_id"], "fall.v2");
        assert_eq!(payload["payload"]["authority_role"], "authoritative");
        assert_eq!(
            payload["payload"]["missing_values"]["decision_state"],
            "outside-detection-window"
        );
        assert_eq!(payload["payload"]["values"], serde_json::json!({}));
        assert!(payload["payload"]["track_id"].is_null());
    }
    assert_ne!(
        wire(&records[0][0])["payload"]["decision_trace_id"],
        wire(&records[1][0])["payload"]["decision_trace_id"]
    );
}
