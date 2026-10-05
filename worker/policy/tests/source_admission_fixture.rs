//! Port of `tests/test_rust_source_admission_parity.py`. The oracle is the
//! Python `worker.runtime.flow.metadata_slot.LatestMetadataSlot` driven by
//! that test's own `_pair` reference loop: every registration and frame it
//! built, the raw reason it expected, and the Python snapshot after each
//! operation (result, expected binding, readiness, high water and retained
//! latest) are replayed here through the public `SourceAdmission`.

#[path = "fixtures/support.rs"]
mod support;

use seeon_worker::source_admission::{
    AcceptedWork, AdmissionError, FrameMetadata, HighWater, SourceAdmission, SourceBinding,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::OnceLock;

const FIXTURE: &str = "source_admission/source_admission.json";
const ORACLE_SOURCES: [(&str, &str); 4] = [
    (
        "tests/test_rust_source_admission_parity.py",
        "fbf54dce901bcf814b27177fbd2a002b8ca055ae35889bba524be4b9e8901e5e",
    ),
    (
        "worker/runtime/flow/metadata_slot.py",
        "a5d704c502ba5c04e78a09863482a13d7fdf55e4fc4d43f825014c43298ae5c0",
    ),
    (
        "worker/types/metadata.py",
        "94edd4502b2e2a5d224485d2da5e35995a10cceca444914626bb787a1b52e64e",
    ),
    (
        "worker/types/perception_frame.py",
        "fe70153a22348b4cd044372afb888b7dd104322f86d3356fde351a539536edf6",
    ),
];

/// The test's `_STATUS` order: two success names, then one name per
/// `AdmissionError` variant in declaration order.
const STATUS: [&str; 19] = [
    "registered",
    "accepted",
    "malformed",
    "invalid_child_instance_id",
    "unknown_source",
    "boot_mismatch",
    "child_mismatch",
    "generation_mismatch",
    "epoch_mismatch",
    "transform_mismatch",
    "transport_mismatch",
    "pts_missing",
    "discarded_publication",
    "non_increasing_pts",
    "non_increasing_canonical_sequence",
    "non_increasing_native_publication_sequence",
    "regressing_fence",
    "generation_exhausted",
    "epoch_exhausted",
];

fn fixture() -> &'static Value {
    static FIXTURE_VALUE: OnceLock<Value> = OnceLock::new();
    FIXTURE_VALUE.get_or_init(|| support::load_sources(FIXTURE, &ORACLE_SOURCES))
}

fn reason(error: AdmissionError) -> &'static str {
    let index = match error {
        AdmissionError::MalformedCamera => 2,
        AdmissionError::InvalidChildInstanceId => 3,
        AdmissionError::UnknownSource => 4,
        AdmissionError::BootMismatch => 5,
        AdmissionError::ChildMismatch => 6,
        AdmissionError::GenerationMismatch => 7,
        AdmissionError::EpochMismatch => 8,
        AdmissionError::TransformMismatch => 9,
        AdmissionError::TransportMismatch => 10,
        AdmissionError::PtsMissing => 11,
        AdmissionError::DiscardedPublication => 12,
        AdmissionError::NonIncreasingPts => 13,
        AdmissionError::NonIncreasingCanonicalSequence => 14,
        AdmissionError::NonIncreasingNativePublicationSequence => 15,
        AdmissionError::RegressingFence => 16,
        AdmissionError::GenerationExhausted => 17,
        AdmissionError::EpochExhausted => 18,
    };
    STATUS[index]
}

/// Plain text, or the recorder's exact run-length parts with their UTF-8
/// digest and length.
fn decode(value: &Value) -> String {
    if let Value::String(text) = value {
        return text.clone();
    }
    let mut out = String::new();
    for part in support::array(&value["parts"]) {
        out.push_str(&support::text(&part["text"]).repeat(support::usize_of(&part["count"])));
    }
    assert_eq!(out.len(), support::usize_of(&value["bytes"]));
    assert_eq!(
        support::sha256(out.as_bytes()),
        support::text(&value["sha256"])
    );
    out
}

fn binding(value: &Value) -> SourceBinding {
    let fields = support::array(value);
    assert_eq!(fields.len(), 6);
    SourceBinding {
        worker_boot_id: decode(&fields[0]),
        child_instance_id: decode(&fields[1]),
        camera_id: decode(&fields[2]),
        source_generation: support::uint(&fields[3]),
        stream_epoch: support::uint(&fields[4]),
        transform_id: decode(&fields[5]),
    }
}

fn water(value: &Value) -> Option<HighWater> {
    if value.is_null() {
        return None;
    }
    let ordinals = support::array(value);
    assert_eq!(ordinals.len(), 3);
    Some(HighWater {
        source_pts: support::int(&ordinals[0]),
        canonical_sequence: support::uint(&ordinals[1]),
        native_publish_sequence: support::uint(&ordinals[2]),
    })
}

fn transport() -> String {
    decode(&fixture()["transport"])
}

fn observable() -> BTreeMap<&'static str, &'static str> {
    fixture()["python_observable"]
        .as_object()
        .expect("python_observable")
        .iter()
        .map(|(name, shown)| (name.as_str(), support::text(shown)))
        .collect()
}

fn admit(
    owner: &mut SourceAdmission,
    transport: &str,
    operation: &Value,
) -> Result<AcceptedWork, AdmissionError> {
    let frame = binding(&operation["frame"]);
    let pts = &operation["pts"];
    owner.admit(FrameMetadata {
        transport_id: transport,
        worker_boot_id: &frame.worker_boot_id,
        child_instance_id: &frame.child_instance_id,
        camera_id: &frame.camera_id,
        source_generation: frame.source_generation,
        stream_epoch: frame.stream_epoch,
        transform_id: &frame.transform_id,
        source_pts: (!pts.is_null()).then(|| support::int(pts)),
        canonical_sequence: support::uint(&operation["seq"]),
        native_publish_sequence: support::uint(&operation["native"]),
    })
}

/// The Python snapshot after one operation: expected binding, readiness,
/// high water and the retained latest's binding and ordinals.
fn check_row(
    context: &str,
    owner: &SourceAdmission,
    retained: Option<&AcceptedWork>,
    transport: &str,
    row: &Value,
) {
    assert_eq!(owner.transport_id(), transport, "{context}: transport");
    assert_eq!(
        owner.binding(),
        &binding(&row["binding"]),
        "{context}: binding"
    );
    assert_eq!(
        owner.is_ready(),
        row["ready"].as_bool().expect("ready"),
        "{context}: ready"
    );
    assert_eq!(owner.high_water(), water(&row["water"]), "{context}: water");
    let expected = &row["retained"];
    let expected = (!expected.is_null()).then(|| {
        (
            binding(&expected["binding"]),
            water(&expected["water"]).expect("retained water"),
        )
    });
    let actual = retained.map(|work| (work.binding().clone(), work.high_water()));
    assert_eq!(actual, expected, "{context}: retained");
}

fn replay(context: &str, pair: &Value) {
    let transport = transport();
    let observable = observable();
    let operations = support::array(&pair["operations"]);
    let reference = support::array(&pair["reference"]);
    assert_eq!(reference.len(), operations.len() + 1, "{context}");
    let expected = (!pair["expected"].is_null()).then(|| support::array(&pair["expected"]));
    if let Some(expected) = expected {
        assert_eq!(expected.len(), reference.len(), "{context}");
        assert_eq!(support::text(&expected[0]), "registered", "{context}");
    }
    let mut owner = SourceAdmission::new(transport.clone(), binding(&pair["initial"]))
        .unwrap_or_else(|error| panic!("{context}: initial registration refused: {error:?}"));
    let mut retained: Option<AcceptedWork> = None;
    assert_eq!(support::text(&reference[0]["result"]), "registered");
    check_row(
        &format!("{context} registration"),
        &owner,
        None,
        &transport,
        &reference[0],
    );
    for (index, operation) in operations.iter().enumerate() {
        let step = format!("{context} operation {index}");
        let raw = if let Some(registration) = operation.get("register") {
            match owner.reregister(binding(registration)) {
                Ok(()) => {
                    retained = None;
                    "registered"
                }
                Err(error) => reason(error),
            }
        } else {
            match admit(&mut owner, &transport, operation) {
                Ok(work) => {
                    retained = Some(work);
                    "accepted"
                }
                Err(error) => reason(error),
            }
        };
        let row = &reference[index + 1];
        let shown = observable.get(raw).copied().unwrap_or(raw);
        assert_eq!(shown, support::text(&row["result"]), "{step}: result");
        if let Some(expected) = expected {
            assert_eq!(raw, support::text(&expected[index + 1]), "{step}: raw");
        }
        check_row(&step, &owner, retained.as_ref(), &transport, row);
    }
}

fn run(name: &str, cases: usize, pairs: usize) {
    let recorded = support::array(&fixture()["tests"][name]);
    assert_eq!(recorded.len(), cases, "{name}");
    for case in recorded {
        let label = format!("{name}[{}]", support::text(&case["case"]));
        let recorded_pairs = support::array(&case["pairs"]);
        assert_eq!(recorded_pairs.len(), pairs, "{label}");
        for (index, pair) in recorded_pairs.iter().enumerate() {
            replay(&format!("{label} pair {index}"), pair);
        }
    }
}

#[test]
fn status_table_is_the_oracle_table() {
    let recorded: Vec<&str> = support::array(&fixture()["status"])
        .iter()
        .map(support::text)
        .collect();
    assert_eq!(recorded, STATUS);
    let observable = observable();
    assert_eq!(observable.len(), 3);
    for name in STATUS {
        let shown = observable.get(name).copied().unwrap_or(name);
        assert_eq!(
            shown == "late",
            name.starts_with("non_increasing_"),
            "{name}"
        );
    }
    assert_eq!(binding(&fixture()["base"]).camera_id, "camera-a");
}

#[test]
fn first_missing_negative_pts_and_exact_integer_limits() {
    run(
        "test_first_missing_negative_pts_and_exact_integer_limits",
        4,
        1,
    );
}

#[test]
fn independent_highwaters_and_combined_faults_are_atomic() {
    run(
        "test_independent_highwaters_and_combined_faults_are_atomic",
        9,
        1,
    );
}

#[test]
fn each_identity_refusal_preserves_unready_and_ready_state() {
    run(
        "test_each_identity_refusal_preserves_unready_and_ready_state",
        14,
        1,
    );
}

#[test]
fn combined_identity_fault_precedence_before_missing_pts_and_lateness() {
    run(
        "test_combined_identity_fault_precedence_before_missing_pts_and_lateness",
        2,
        1,
    );
}

#[test]
fn identity_strings_are_exact_utf8_not_trimmed_normalized_or_truncated() {
    run(
        "test_identity_strings_are_exact_utf8_not_trimmed_normalized_or_truncated",
        4,
        1,
    );
}

#[test]
fn frame_uuid_spellings_use_actual_uuid_normalization() {
    run(
        "test_frame_uuid_spellings_use_actual_uuid_normalization",
        1,
        1,
    );
}

#[test]
fn equal_binding_reregistration_resets_readiness_and_all_ordinals() {
    run(
        "test_equal_binding_reregistration_resets_readiness_and_all_ordinals",
        1,
        1,
    );
}

#[test]
fn changed_binding_invalidates_old_headers_but_not_transport_identity() {
    run(
        "test_changed_binding_invalidates_old_headers_but_not_transport_identity",
        6,
        1,
    );
}

#[test]
fn seeded_trace_is_generated_without_observed_decisions() {
    run(
        "test_seeded_trace_is_generated_without_observed_decisions",
        1,
        1,
    );
}

#[test]
fn transport_accepts_exact_input_and_operation_bounds() {
    run(
        "test_transport_accepts_exact_input_and_operation_bounds",
        1,
        2,
    );
}

/// Registration validation Python does not have: Python's slot registers the
/// invalid binding verbatim, the Rust owner refuses it atomically, both on
/// creation and on re-registration between two accepted frames.
#[test]
fn rust_only_registration_validation_is_not_python_parity() {
    let transport = transport();
    let base = binding(&fixture()["base"]);
    let cases = support::array(&fixture()["rust_only"]);
    assert_eq!(cases.len(), 9);
    for case in cases {
        let label = support::text(&case["case"]);
        let invalid = binding(&case["invalid"]);
        let expected_reason = support::text(&case["reason"]);
        assert_eq!(
            binding(&case["python_expected_binding"]),
            invalid,
            "{label}"
        );
        let refused = SourceAdmission::new(transport.clone(), invalid.clone())
            .expect_err("invalid binding must be refused on creation");
        assert_eq!(reason(refused), expected_reason, "{label}: creation");

        let operations = support::array(&case["operations"]);
        let reference = support::array(&case["reference"]);
        assert_eq!((operations.len(), reference.len()), (2, 3), "{label}");
        let mut owner = SourceAdmission::new(transport.clone(), base.clone())
            .unwrap_or_else(|error| panic!("{label}: base refused: {error:?}"));
        assert_eq!(owner.binding(), &base, "{label}");
        check_row(label, &owner, None, &transport, &reference[0]);
        let first = admit(&mut owner, &transport, &operations[0])
            .unwrap_or_else(|error| panic!("{label}: first frame refused: {error:?}"));
        check_row(label, &owner, Some(&first), &transport, &reference[1]);
        let refused = owner
            .reregister(invalid)
            .expect_err("invalid binding must be refused on re-registration");
        assert_eq!(reason(refused), expected_reason, "{label}: re-registration");
        check_row(label, &owner, Some(&first), &transport, &reference[1]);
        let second = admit(&mut owner, &transport, &operations[1])
            .unwrap_or_else(|error| panic!("{label}: second frame refused: {error:?}"));
        check_row(label, &owner, Some(&second), &transport, &reference[2]);
        assert_eq!(
            support::text(&reference[2]["result"]),
            "accepted",
            "{label}"
        );
    }
}
