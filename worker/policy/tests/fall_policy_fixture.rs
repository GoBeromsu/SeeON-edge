//! Port of `tests/test_rust_fall_policy_parity.py`. The oracle is the Python
//! `worker.domains.fall.policy.FallPolicyDecider` driven by that test's own
//! `_exercise`: every FALLPROBE v1 row it rendered (events, traces with every
//! numeric bit, missing maps, generations and fallen state) is replayed here
//! through the public `FallPolicyDecider` and compared row for row.

#[path = "fixtures/support.rs"]
mod support;

#[path = "fixtures/policy.rs"]
mod policy;

use policy::{Fields, MAX_ITEMS, optional};
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{
    FallCapacities, FallPolicy, FallPolicyDecider, FallPolicyParameters, FallProbabilities,
};
use seeon_worker::trace::DecisionTraceMissingReason;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::sync::OnceLock;

const FIXTURE: &str = "fall_policy/fall_policy.json";
const ORACLE_SOURCES: [(&str, &str); 7] = [
    (
        "tests/test_rust_fall_policy_parity.py",
        "3a5497b7541e6d3126ac7dd9acd2d1ce21b0128d8b2afab177013f9668ae8322",
    ),
    (
        "worker/domains/fall/policy.py",
        "4c59f13d64bf540639c49794abcc0c74107d0642a82805e4aae74f854e8b1522",
    ),
    (
        "worker/domains/episode/authority.py",
        "e29a9fcbb1a5823d0630378e0dd0a84477f42fa60b33528a75e603df0acd065d",
    ),
    (
        "shared/detection_policies.py",
        "6fa6a49832e0aa63da7905bd6d4ebe512ff152999335af32433403649470389c",
    ),
    (
        "worker/interfaces/fall_model.py",
        "ff3acae1e409f95a0f44c8176a57a547e82775aa82b5e9a2280d71efad4214a1",
    ),
    (
        "worker/types/business_event.py",
        "c59eda20a1dbb4fd9aa35c1f41afd9b1d6ec05be6d1905c5a18fb74f3f778b3d",
    ),
    (
        "worker/types/trace.py",
        "ac569c2dfb1a8fa1e057f9e5555b971fce76f606e225f61f1e282601cb2a7b46",
    ),
];
const HEADER: &str = "FALLPROBE\t1";

fn fixture() -> &'static Value {
    static FIXTURE_VALUE: OnceLock<Value> = OnceLock::new();
    FIXTURE_VALUE.get_or_init(|| support::load_sources(FIXTURE, &ORACLE_SOURCES))
}

fn construct(fields: &mut Fields<'_>) -> Result<(FallPolicyDecider, Vec<u64>), String> {
    assert_eq!(fields.next(), "N", "construction row");
    let (camera, facility, boot, epoch) =
        (fields.text(), fields.text(), fields.text(), fields.text());
    let source_generation = fields.number();
    let policy = FallPolicy::new(FallPolicyParameters {
        transition_threshold: fields.float(),
        transition_votes: fields.number(),
        transition_window: fields.number(),
        fallen_threshold: fields.float(),
        fallen_consecutive: fields.number(),
        recovery_transition_max: fields.float(),
        recovery_fallen_max: fields.float(),
        recovery_consecutive: fields.number(),
        track_ttl_frames: fields.number(),
        cooldown_frames: fields.number(),
    })
    .map_err(|error| format!("policy: {error:?}"))?;
    let watch: Vec<u64> = (0..fields.count()).map(|_| fields.number()).collect();
    assert!(
        watch.iter().collect::<BTreeSet<_>>().len() == watch.len(),
        "duplicate watch track"
    );
    fields.end();
    let capacities = FallCapacities {
        retained_tracks: MAX_ITEMS,
        generation_identities: MAX_ITEMS,
        episodes: MAX_ITEMS,
        vote_window: MAX_ITEMS,
    };
    let decider = FallPolicyDecider::new(
        camera,
        facility,
        boot,
        epoch,
        source_generation,
        policy,
        capacities,
    )
    .map_err(|error| format!("decider: {error:?}"))?;
    Ok((decider, watch))
}

fn update(
    decider: &mut FallPolicyDecider,
    fields: &mut Fields<'_>,
) -> Result<Vec<BusinessEvent>, String> {
    let frame = fields.number();
    let time = fields.float();
    let live: Vec<u64> = (0..fields.count()).map(|_| fields.number()).collect();
    let mut scores = BTreeMap::new();
    for _ in 0..fields.count() {
        let id = fields.number();
        let score = FallProbabilities::new(fields.float(), fields.float(), fields.float())
            .map_err(|error| format!("probabilities: {error:?}"))?;
        assert!(scores.insert(id, score).is_none(), "duplicate score {id}");
    }
    let missing = match fields.next() {
        "-" => None,
        count => {
            let mut reasons = BTreeMap::new();
            for _ in 0..Fields::new(count).count() {
                let id = fields.number();
                let token = fields.text();
                let reason = DecisionTraceMissingReason::from_token(&token)
                    .ok_or_else(|| format!("missing reason outside vocabulary: {token}"))?;
                assert!(
                    reasons.insert(id, reason).is_none(),
                    "duplicate reason {id}"
                );
            }
            Some(reasons)
        }
    };
    fields.end();
    decider
        .update(frame, time, &scores, live, missing.as_ref())
        .map_err(|error| format!("update: {error:?}"))
}

fn observe(
    out: &mut String,
    index: usize,
    decider: &FallPolicyDecider,
    events: &[BusinessEvent],
    watch: &[u64],
) {
    writeln!(
        out,
        "O\t{index}\t{}\t{}\t{}\t{}",
        u8::from(decider.last_update_evaluated()),
        decider.track_id_switch_absorbed_total(),
        events.len(),
        decider.last_trace_snapshots().len()
    )
    .expect("writing to String");
    policy::events(out, events);
    policy::traces(out, decider.last_trace_snapshots());
    for &track in watch {
        writeln!(
            out,
            "S\t{track}\t{}\t{}",
            optional(decider.generation_for(track)),
            u8::from(decider.is_fallen(track))
        )
        .expect("writing to String");
    }
}

/// The request replayed through the public API, rendered as FALLPROBE v1 rows.
fn replay(request: &[&str]) -> Result<String, String> {
    assert_eq!(request.first(), Some(&HEADER), "request header");
    let (mut decider, watch) = construct(&mut Fields::new(request[1]))?;
    let mut out = format!("{HEADER}\n");
    observe(&mut out, 0, &decider, &[], &watch);
    let mut calls = 1;
    for line in &request[2..] {
        let mut fields = Fields::new(line);
        let events = match fields.next() {
            "U" => update(&mut decider, &mut fields)?,
            "C" => {
                fields.end();
                decider
                    .coast()
                    .map_err(|error| format!("coast: {error:?}"))?
            }
            "R" => {
                let event = policy::event(&mut fields);
                fields.end();
                // Python's release returns None; only its later effects are compared.
                decider
                    .release_onset(&event)
                    .map_err(|error| format!("release: {error:?}"))?;
                Vec::new()
            }
            verb => panic!("unknown request verb {verb:?}"),
        };
        observe(&mut out, calls, &decider, &events, &watch);
        calls += 1;
    }
    writeln!(out, "END\t{calls}").expect("writing to String");
    Ok(out)
}

fn assert_parity(test: &str, cases: usize, exchanges: usize) {
    for transcript in policy::transcripts(fixture(), test, cases, exchanges) {
        let replayed = replay(&transcript.request());
        policy::assert_transcript(&transcript, replayed);
    }
}

#[test]
fn d1_every_missing_reason_on_unknown_track() {
    assert_parity("test_d1_every_missing_reason_on_unknown_track", 15, 15);
}

#[test]
fn d1_unknown_reason_is_rejected_by_both_real_owners() {
    let recorded = &fixture()["unknown_missing_reason"];
    let unknown = support::text(&recorded["reason"]);
    assert_eq!(unknown, "synthetic-unknown-reason", "recorded input");
    assert_eq!(support::text(&recorded["exception"]), "ValueError");
    assert!(
        support::text(&recorded["message"]).contains("missing reason must use compiled vocabulary"),
        "Python refused for the vocabulary reason"
    );
    let vocabulary = support::array(&fixture()["missing_reason_vocabulary"]);
    assert_eq!(
        vocabulary.len(),
        15,
        "Python missing-reason vocabulary size"
    );
    for token in vocabulary.iter().map(support::text) {
        assert_eq!(
            DecisionTraceMissingReason::from_token(token).map(DecisionTraceMissingReason::as_str),
            Some(token),
            "Python vocabulary token {token}"
        );
    }
    assert_eq!(DecisionTraceMissingReason::from_token(unknown), None);
    assert_parity(
        "test_d1_unknown_reason_is_rejected_by_both_real_owners",
        1,
        1,
    );
}

#[test]
fn d3_threshold_uses_unrounded_score() {
    assert_parity("test_d3_threshold_uses_unrounded_score", 3, 3);
}

#[test]
fn d3_canonical_trace_numbers_keep_unrounded_event_bits() {
    assert_parity(
        "test_d3_canonical_trace_numbers_keep_unrounded_event_bits",
        5,
        5,
    );
}

#[test]
fn d4_missing_coast_initial_fallen_reassociation_and_exact_release() {
    assert_parity(
        "test_d4_missing_coast_initial_fallen_reassociation_and_exact_release",
        1,
        1,
    );
}

#[test]
fn release_is_identity_exact_metadata_independent_and_never_rewinds() {
    assert_parity(
        "test_release_is_identity_exact_metadata_independent_and_never_rewinds",
        1,
        1,
    );
}

#[test]
fn d5_strict_recovery_boundaries_rearm_without_cooldown() {
    assert_parity(
        "test_d5_strict_recovery_boundaries_rearm_without_cooldown",
        1,
        1,
    );
}

#[test]
fn d5_initial_fallen_requires_two_recovery_scores() {
    assert_parity("test_d5_initial_fallen_requires_two_recovery_scores", 1, 1);
}

#[test]
fn d6_loss_insertion_order_reassociation_and_release_identity() {
    assert_parity(
        "test_d6_loss_insertion_order_reassociation_and_release_identity",
        1,
        1,
    );
}

#[test]
fn d6_ttl_44_retained_45_evicted_46_generation_one() {
    assert_parity("test_d6_ttl_44_retained_45_evicted_46_generation_one", 1, 1);
}

#[test]
fn d6_scoreless_live_refreshes_liveness_without_advancing_streaks() {
    assert_parity(
        "test_d6_scoreless_live_refreshes_liveness_without_advancing_streaks",
        1,
        1,
    );
}

#[test]
fn d7_six_votes_use_authority_window_not_five_entry_trace_history() {
    assert_parity(
        "test_d7_six_votes_use_authority_window_not_five_entry_trace_history",
        1,
        1,
    );
}

#[test]
fn d7_trace_history_expires_after_five_scores_not_policy_window() {
    assert_parity(
        "test_d7_trace_history_expires_after_five_scores_not_policy_window",
        1,
        1,
    );
}

#[test]
fn d7_sorted_events_traces_unknown_live_and_ignored_nonlive_score() {
    assert_parity(
        "test_d7_sorted_events_traces_unknown_live_and_ignored_nonlive_score",
        1,
        1,
    );
}

#[test]
fn per_call_missing_reasons_do_not_inherit_or_reset_votes() {
    assert_parity(
        "test_per_call_missing_reasons_do_not_inherit_or_reset_votes",
        2,
        2,
    );
}

#[test]
fn utf8_hex_transport_preserves_identity_and_all_event_strings() {
    assert_parity(
        "test_utf8_hex_transport_preserves_identity_and_all_event_strings",
        1,
        1,
    );
}
