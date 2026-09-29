//! Port of `tests/test_rust_bed_policy_parity.py`. The oracle is the Python
//! `worker.domains.bed_exit.detector.BedExitMonitor` driven by that test's own
//! `_exercise`: every BEDPROBE v1 row it rendered (events, traces with every
//! numeric bit, assignments, recovery, debug snapshot, scoring and watched
//! episode states) is replayed through the public `BedExitMonitor` and compared
//! row for row. The two Rust-only failure tests replay the requests that test
//! built and check its structural assertions, which name the refusal causes.

#[path = "fixtures/support.rs"]
mod support;

#[path = "fixtures/policy.rs"]
mod policy;

#[path = "fixtures/bed.rs"]
mod bed;

use policy::hex;
use serde_json::Value;
use std::sync::OnceLock;

const FIXTURE: &str = "bed_policy/bed_policy.json";
const ORACLE_SOURCES: [(&str, &str); 14] = [
    (
        "tests/test_rust_bed_policy_parity.py",
        "ffa548537788de88baa0540966b8d57c5806d7349377a2de013401a95f04b041",
    ),
    (
        "worker/domains/bed_exit/detector.py",
        "8cda28e29330e7061b4562840a836dd91b4c4c5772275e189bfb26fff1303cf1",
    ),
    (
        "worker/domains/bed_exit/geometry.py",
        "b6f160a131d95476e48b5b3288b9ac50d7c266d060d0f447933971c9b075a46d",
    ),
    (
        "worker/domains/bed_exit/latch.py",
        "fd3d63afe85941cd77238b336814cb17f463a5cc2eb4bc47c6d9b4754ea9479a",
    ),
    (
        "worker/domains/bed_exit/night_window.py",
        "8b48f6d3d1dc99863ce02a64522f9dea485f32cb9d22ccd17bc5d3a094ca3b56",
    ),
    (
        "worker/domains/bed_exit/schema.py",
        "3802cfaa6209f787524fe2dd4ee8037fad11b43c4f1ab3668a073d4a276e52a3",
    ),
    (
        "worker/domains/detection_window.py",
        "0618af9c8f7df60b502131d214305f98be1adc47b977b2ab2db75d6b903d5b44",
    ),
    (
        "worker/domains/episode/authority.py",
        "e29a9fcbb1a5823d0630378e0dd0a84477f42fa60b33528a75e603df0acd065d",
    ),
    (
        "worker/domains/staleness.py",
        "a204973339fbcb573a7afd023f36d20e6b5ae98e4392d3d7f0491430d5f8019f",
    ),
    (
        "shared/detection_policies.py",
        "6fa6a49832e0aa63da7905bd6d4ebe512ff152999335af32433403649470389c",
    ),
    (
        "contracts/observation.py",
        "75c06ff9ddef6cc93b23d5c3ff515312e1c00e91ca725284fbca40efb53b6823",
    ),
    (
        "worker/types/bed_pose_features.py",
        "9254e6bd498077ecbdce2e9be339d8b874da47dbbee1345bc10fd876df70d2bf",
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

fn fixture() -> &'static Value {
    static FIXTURE_VALUE: OnceLock<Value> = OnceLock::new();
    FIXTURE_VALUE.get_or_init(|| support::load_sources(FIXTURE, &ORACLE_SOURCES))
}

fn assert_parity(test: &str, cases: usize, exchanges: usize) {
    for transcript in policy::transcripts(fixture(), test, cases, exchanges) {
        let replayed = bed::replay(&transcript.request(), None);
        policy::assert_transcript(&transcript, replayed);
    }
}

/// Replayed rows of a request the Python test requires to end in a domain failure.
fn failed_replay(request: &[String], context: &str) -> String {
    let request: Vec<&str> = request.iter().map(String::as_str).collect();
    let (out, success) = bed::replay_monitor(&request, None)
        .unwrap_or_else(|cause| panic!("{context}: construction refused: {cause}"));
    assert!(!success, "{context}: every call succeeded");
    out
}

#[test]
fn inclusive_posture_thresholds() {
    assert_parity("test_inclusive_posture_thresholds", 5, 5);
}

#[test]
fn inclusive_geometry_gate() {
    assert_parity("test_inclusive_geometry_gate", 3, 3);
}

#[test]
fn hold_candidate_dwell_reset_armed_retention_and_other_bed() {
    assert_parity(
        "test_hold_candidate_dwell_reset_armed_retention_and_other_bed",
        1,
        1,
    );
}

#[test]
fn duplicate_observations_features_and_tie_order() {
    assert_parity("test_duplicate_observations_features_and_tie_order", 2, 2);
}

#[test]
fn positional_ids_ignore_explicit_live_list_and_missing_pose() {
    assert_parity(
        "test_positional_ids_ignore_explicit_live_list_and_missing_pose",
        1,
        1,
    );
}

#[test]
fn cache_early_returns_coast_freshness_and_reversed_clock() {
    assert_parity(
        "test_cache_early_returns_coast_freshness_and_reversed_clock",
        1,
        1,
    );
}

#[test]
fn missing_signed_reversed_pts_and_coast_do_not_move_anchor() {
    assert_parity(
        "test_missing_signed_reversed_pts_and_coast_do_not_move_anchor",
        1,
        1,
    );
}

#[test]
fn live_unobserved_track_and_coast_retain_the_actual_pts_anchor() {
    assert_parity(
        "test_live_unobserved_track_and_coast_retain_the_actual_pts_anchor",
        1,
        1,
    );
}

#[test]
fn unrounded_dwell_edges_and_outside_interruption() {
    assert_parity("test_unrounded_dwell_edges_and_outside_interruption", 3, 6);
}

#[test]
fn integer_boundaries_and_unused_pose_scalars_are_preserved() {
    assert_parity(
        "test_integer_boundaries_and_unused_pose_scalars_are_preserved",
        2,
        2,
    );
}

#[test]
fn inside_handoff_ties_choose_last_observation() {
    assert_parity("test_inside_handoff_ties_choose_last_observation", 3, 3);
}

#[test]
fn outside_handoff_requires_unique_overlap_and_no_reoccupancy() {
    assert_parity(
        "test_outside_handoff_requires_unique_overlap_and_no_reoccupancy",
        6,
        6,
    );
}

#[test]
fn recovery_reassociation_release_identity_and_sequence() {
    assert_parity(
        "test_recovery_reassociation_release_identity_and_sequence",
        1,
        1,
    );
}

#[test]
fn reassociation_frame_pts_boundaries_and_recovery() {
    assert_parity("test_reassociation_frame_pts_boundaries_and_recovery", 8, 8);
}

#[test]
fn stale_retirement_order_and_removed_own_bed() {
    assert_parity("test_stale_retirement_order_and_removed_own_bed", 1, 1);
}

#[test]
fn real_polygon_scores_and_full_box_metadata_transport() {
    assert_parity(
        "test_real_polygon_scores_and_full_box_metadata_transport",
        4,
        4,
    );
}

#[test]
fn prior_box_polygon_uses_real_geometry_not_an_aabb_substitute() {
    assert_parity(
        "test_prior_box_polygon_uses_real_geometry_not_an_aabb_substitute",
        1,
        1,
    );
}

#[test]
fn sixty_four_events_conserved_and_ordered_at_collection_bound() {
    assert_parity(
        "test_sixty_four_events_conserved_and_ordered_at_collection_bound",
        1,
        1,
    );
}

#[test]
fn missing_geometry_is_explicit_rejection_not_empty_success() {
    let cases = support::array(&fixture()["rust_only"]["missing_geometry"]);
    assert_eq!(cases.len(), 2, "missing_overlap cases");
    for (case, missing_overlap) in cases.iter().zip([false, true]) {
        assert_eq!(case["missing_overlap"].as_bool(), Some(missing_overlap));
        let context = format!("missing_geometry missing_overlap={missing_overlap}");
        let out = failed_replay(&policy::request(case, &context), &context);
        assert!(out.contains("X\t4\trejected\t"), "{context}: {out}");
        let cause = if missing_overlap {
            "MissingPriorOverlap(7)"
        } else {
            "InvalidShape(\"containments\")"
        };
        assert!(out.contains(&hex(cause)), "{context}: cause {cause}");
        assert!(out.ends_with("END\t5\n"), "{context}: call count");
        let mut blocks: Vec<Vec<&str>> = Vec::new();
        for row in out.lines() {
            match row.split('\t').next() {
                Some("O") => {
                    let index: usize = row
                        .split('\t')
                        .nth(1)
                        .and_then(|i| i.parse().ok())
                        .expect("O index");
                    assert_eq!(index, blocks.len(), "{context}: O rows in order");
                    blocks.push(Vec::new());
                }
                Some("Q" | "X" | "END") => {}
                // Like Python, the header before the first O row belongs to no block.
                _ => {
                    if let Some(block) = blocks.last_mut() {
                        block.push(row);
                    }
                }
            }
        }
        assert_eq!(
            blocks[3], blocks[4],
            "{context}: admission rejection must conserve the entire previous snapshot"
        );
    }
}

#[test]
fn fatal_recovery_capacity_poison_never_retries() {
    let context = "fatal_recovery_capacity";
    let request = policy::request(&fixture()["rust_only"]["fatal_recovery_capacity"], context);
    let out = failed_replay(&request, context);
    assert!(out.contains("X\t2\tfatal\t"), "{context}: {out}");
    assert!(out.contains("X\t3\tpoisoned\t"), "{context}: {out}");
    // No accepted prefix at recovery, not fake events.
    assert!(!out.contains("\nE\t"), "{context}: events emitted");
    assert!(out.ends_with("END\t4\n"), "{context}: call count");
}
