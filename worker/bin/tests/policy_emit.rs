//! T23: `policy::emit` turns one policy event into the delivery `EventEntry`
//! that Python `DurableEvidenceStager.stage` hands to `try_admit`. Oracles:
//! the reviewed golden `d/delivery-queue/event-*.json` (bytes) and, in the
//! xlang lane, the Python stager and `emit_policy.model_score_record` run
//! as children through `$SEEON_TEST_PYTHON`.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use seeon_ml_worker::delivery::DeliveryEntry;
use seeon_ml_worker::json::Json;
use seeon_ml_worker::policy::emit::{EmitError, ModelEvidence, ModelScore, Stager};
use serde_json::Value;

const CAMERA_ID: &str = "cmsnw6rjc01vhlh01oswn99yq";
const FACILITY_ID: &str = "facility-1";
const CONFIG_VERSION: i64 = 7;
const EVENT_GOLDEN: &str = "d/delivery-queue/event-a5e15ff2-90fd-4764-be74-a7da4f573cc9.json";

fn worker_wire() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/worker-wire")
}

fn manifest_sha() -> String {
    format!("6f6f793e244baf48449e6024{}", "a".repeat(40))
}

fn text(value: &str) -> Json {
    Json::Str(value.to_owned())
}

fn object(members: Vec<(&str, Json)>) -> Vec<(String, Json)> {
    members
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
}

/// The staged EVENT fixture behind the golden (manifest.json row 10).
fn golden_event() -> Vec<(String, Json)> {
    object(vec![
        (
            "edge_event_id",
            text("a5e15ff2-90fd-4764-be74-a7da4f573cc9"),
        ),
        ("event_type", text("bed-exit")),
        ("detected_at", text("2026-08-20T17:20:58.197192Z")),
        ("probability", Json::Float(0.87)),
        (
            "audit",
            Json::Object(object(vec![
                ("clock_source", text("monotonic")),
                ("detector_version", text("fall-2026.08")),
                ("model_version", text("lstm-v3")),
                ("operating_threshold", Json::Float(0.62)),
            ])),
        ),
    ])
}

#[test]
fn staged_event_bytes_equal_the_golden() {
    let sha = manifest_sha();
    let stager = Stager::new(CAMERA_ID, FACILITY_ID, CONFIG_VERSION, Some(&sha)).expect("stager");
    let entry = stager.event_entry(&golden_event()).expect("event entry");
    let bytes = DeliveryEntry::from(entry).to_bytes().expect("entry bytes");
    let golden = fs::read(worker_wire().join(EVENT_GOLDEN)).expect("event golden");
    assert_eq!(bytes, golden);
}

/// Order-preserving JSON text, so Python sees each event's insertion order.
fn json_text(value: &Json) -> String {
    let quote = |raw: &str| serde_json::to_string(raw).expect("string encodes");
    match value {
        Json::Null => "null".to_owned(),
        Json::Bool(flag) => flag.to_string(),
        Json::Int(number) => number.to_string(),
        Json::Float(number) => format!("{number:?}"),
        Json::Str(raw) => quote(raw),
        Json::Array(items) => {
            let items: Vec<String> = items.iter().map(json_text).collect();
            format!("[{}]", items.join(","))
        }
        Json::Object(members) => {
            let members: Vec<String> = members
                .iter()
                .map(|(key, member)| format!("{}:{}", quote(key), json_text(member)))
                .collect();
            format!("{{{}}}", members.join(","))
        }
    }
}

fn run_python(snippet: &str, input: &str) -> Value {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON is set");
    let mut child = Command::new(python)
        .arg("-c")
        .arg(snippet)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("python spawns");
    child
        .stdin
        .take()
        .expect("python stdin")
        .write_all(input.as_bytes())
        .expect("cases written");
    let output = child.wait_with_output().expect("python exits");
    assert!(output.status.success(), "python oracle exited non-zero");
    serde_json::from_slice(&output.stdout).expect("python prints JSON")
}

const STAGER_SNIPPET: &str = r#"
import json, sys, tempfile
from pathlib import Path
from shared.events.delivery_queue import EventEntry
from worker.pipeline.output.evidence.evidence_stager import (
    DurableEvidenceStager, _required_text)
cases = json.loads(sys.stdin.read())
out = []
with tempfile.TemporaryDirectory() as tmp:
    for index, case in enumerate(cases):
        config, event = case["config"], case["event"]
        try:
            stager = DurableEvidenceStager(
                Path(tmp) / str(index), camera_id=config["camera_id"],
                facility_id=config["facility_id"], resident_id=config["resident_id"],
                config_version=config["config_version"],
                runtime_manifest_sha256=config["sha"])
            texts = [_required_text(event, key)
                     for key in ("edge_event_id", "detected_at", "event_type")]
            values, trace, shed = stager._envelope(event)
            entry = EventEntry(edge_event_id=texts[0], event_type=texts[2],
                               detected_at=texts[1], camera_id=stager.camera_id,
                               facility_id=stager.facility_id, decision_trace=trace,
                               values=values, shed_detail_keys=shed)
            out.append({"edge_event_id": entry.edge_event_id,
                        "detected_at": entry.detected_at, "event_type": entry.event_type,
                        "entry_id": entry.entry_id, "values": values.decode("ascii"),
                        "trace": trace.decode("ascii"), "shed": list(shed)})
        except Exception as error:
            out.append({"error": type(error).__name__})
print(json.dumps(out))
"#;

/// The one reviewed Python-exception-to-Rust-refusal table for T23. Python
/// raises `ValueError` from `_required_text`, `validate_runtime_manifest_sha256`,
/// `_shed_to_limit` and `EventEntry.__post_init__`; the Rust refusals below are
/// the only ones with a Python counterpart the stager can reach.
fn python_class(error: &EmitError) -> Option<&'static str> {
    match error {
        EmitError::BlankField(_)
        | EmitError::ManifestSha
        | EmitError::TooLarge { .. }
        | EmitError::Entry(_) => Some("ValueError"),
        // Python `str()` accepts containers and `json.dumps` allows NaN; both
        // are recorded deviations, so no case here reaches them.
        EmitError::FieldType(_) | EmitError::Json(_) => None,
    }
}

type Refusal = fn(&EmitError) -> bool;

struct Case {
    name: &'static str,
    sha: Option<String>,
    resident_id: Option<&'static str>,
    event: Vec<(String, Json)>,
    refusal: Option<Refusal>,
}

fn case(name: &'static str, event: Vec<(String, Json)>) -> Case {
    Case {
        name,
        sha: Some(manifest_sha()),
        resident_id: None,
        event,
        refusal: None,
    }
}

fn identity(edge_event_id: Json, detected_at: Json, event_type: Json) -> Vec<(&'static str, Json)> {
    vec![
        ("edge_event_id", edge_event_id),
        ("detected_at", detected_at),
        ("event_type", event_type),
    ]
}

fn plain(extra: Vec<(&'static str, Json)>) -> Vec<(String, Json)> {
    let mut members = identity(
        text("0b6c1d2e-3f40-4a5b-8c6d-7e8f90a1b2c3"),
        text("2026-08-20T17:20:58Z"),
        text("fall"),
    );
    members.push(("probability", Json::Float(0.5)));
    members.extend(extra);
    object(members)
}

fn stager_cases() -> Vec<Case> {
    let big = |fill: &str| text(&fill.repeat(20_000));
    vec![
        case("golden fixture", golden_event()),
        Case {
            sha: None,
            ..case(
                "equal-size detail sheds in insertion order",
                plain(vec![
                    ("zeta_trace", big("z")),
                    ("alpha_trace", big("a")),
                    ("note", text("낙상 감지 \u{1f6a8}")),
                    ("count", Json::Int(3)),
                ]),
            )
        },
        Case {
            resident_id: Some("resident-9"),
            ..case(
                "audit extras are listed and trace bulk sheds silently",
                plain(vec![(
                    "audit",
                    Json::Object(object(vec![
                        ("clock_source", text("monotonic")),
                        ("debug_blob", text("x")),
                        ("decision_trace_id", text(&"d".repeat(17_000))),
                        ("model_version", text("lstm-v3")),
                    ])),
                )]),
            )
        },
        case(
            "event identity fields are overridden and snapshots popped",
            plain(vec![
                ("camera_id", text("spoofed")),
                ("snapshot_jpeg", text("AAAA")),
                ("snapshot", Json::Object(Vec::new())),
            ]),
        ),
        case(
            "non-object audit still carries the configured sha",
            plain(vec![("audit", text("not-a-mapping"))]),
        ),
        Case {
            sha: None,
            ..case(
                "non-object audit without sha gives an empty trace",
                plain(vec![("audit", Json::Array(vec![Json::Int(1)]))]),
            )
        },
        case(
            "numeric id and padded timestamp become trimmed text",
            object(identity(
                Json::Int(12345),
                text(" 2026-08-20T17:20:58Z \n"),
                text("fall"),
            )),
        ),
        case(
            "small floats keep Python repr",
            plain(vec![
                ("score", Json::Float(1e-7)),
                ("ratio", Json::Float(0.1)),
            ]),
        ),
        Case {
            refusal: Some(|error| matches!(error, EmitError::BlankField("edge_event_id"))),
            ..case(
                "blank edge_event_id",
                object(identity(
                    text(" \t "),
                    text("2026-08-20T17:20:58Z"),
                    text("fall"),
                )),
            )
        },
        Case {
            refusal: Some(|error| matches!(error, EmitError::BlankField("detected_at"))),
            ..case(
                "null detected_at",
                object(identity(text("id-1"), Json::Null, text("fall"))),
            )
        },
        Case {
            refusal: Some(|error| matches!(error, EmitError::ManifestSha)),
            ..case(
                "uppercase audit sha",
                plain(vec![(
                    "audit",
                    Json::Object(object(vec![(
                        "runtime_manifest_sha256",
                        text(&"A".repeat(64)),
                    )])),
                )]),
            )
        },
        Case {
            sha: Some("abc".to_owned()),
            refusal: Some(|error| matches!(error, EmitError::ManifestSha)),
            ..case("short configured sha", plain(Vec::new()))
        },
        Case {
            refusal: Some(
                |error| matches!(error, EmitError::TooLarge { limit, .. } if *limit == 32 * 1024),
            ),
            ..case(
                "protected core over the values limit",
                object(vec![
                    ("edge_event_id", text("id-2")),
                    ("detected_at", text("2026-08-20T17:20:58Z")),
                    ("event_type", text("fall")),
                    ("probability", text(&"p".repeat(40_000))),
                ]),
            )
        },
        Case {
            refusal: Some(
                |error| matches!(error, EmitError::Entry(entry) if entry.field == "event_type"),
            ),
            ..case(
                "event_type over its envelope limit",
                object(identity(
                    text("id-3"),
                    text("2026-08-20T17:20:58Z"),
                    text(&"e".repeat(40)),
                )),
            )
        },
    ]
}

fn case_json(case: &Case) -> String {
    let optional = |value: Option<&str>| value.map_or(Json::Null, text);
    let config = Json::Object(object(vec![
        ("camera_id", text(CAMERA_ID)),
        ("facility_id", text(FACILITY_ID)),
        ("resident_id", optional(case.resident_id)),
        ("config_version", Json::Int(i128::from(CONFIG_VERSION))),
        ("sha", optional(case.sha.as_deref())),
    ]));
    let event = Json::Object(case.event.clone());
    json_text(&Json::Object(object(vec![
        ("config", config),
        ("event", event),
    ])))
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn envelope_matches_the_python_stager() {
    let cases = stager_cases();
    let input = format!(
        "[{}]",
        cases.iter().map(case_json).collect::<Vec<_>>().join(",")
    );
    let Value::Array(expected) = run_python(STAGER_SNIPPET, &input) else {
        panic!("python prints one result per case");
    };
    assert_eq!(expected.len(), cases.len());
    for (case, python) in cases.iter().zip(&expected) {
        let stager = Stager::new(CAMERA_ID, FACILITY_ID, CONFIG_VERSION, case.sha.as_deref()).map(
            |stager| match case.resident_id {
                Some(resident) => stager.with_resident_id(resident),
                None => stager,
            },
        );
        match (
            stager.and_then(|stager| stager.event_entry(&case.event)),
            case.refusal,
        ) {
            (Ok(entry), None) => {
                let fields = entry.fields();
                let rust = serde_json::json!({
                    "edge_event_id": fields.edge_event_id,
                    "detected_at": fields.detected_at,
                    "event_type": fields.event_type,
                    "entry_id": fields.entry_id,
                    "values": String::from_utf8(fields.values.clone()).expect("ascii values"),
                    "trace": String::from_utf8(fields.decision_trace.clone()).expect("ascii trace"),
                    "shed": fields.shed_detail_keys,
                });
                assert_eq!(&rust, python, "case: {}", case.name);
            }
            (Err(error), Some(refusal)) => {
                assert!(refusal(&error), "case {}: unexpected {error:?}", case.name);
                let class = python_class(&error).expect("refusal has a Python counterpart");
                assert_eq!(python["error"], class, "case: {}", case.name);
            }
            (outcome, _) => panic!("case {}: unexpected outcome {:?}", case.name, outcome.err()),
        }
    }
}

const SCORE_SNIPPET: &str = r#"
import json, sys
from worker.interfaces.fall_model import BinaryFallScoreEvidence, FallProbabilities
from worker.pipeline.diagnostics.emit_policy import model_score_record
out = []
for case in json.loads(sys.stdin.read()):
    probability = None
    if case["probability"] is not None:
        p = case["probability"]
        evidence = None
        if p["evidence"] is not None:
            evidence = BinaryFallScoreEvidence(**p["evidence"])
        probability = FallProbabilities(background=p["background"],
            fall_transition=p["fall_transition"], fallen=p["fallen"],
            model_evidence=evidence)
    record = model_score_record(camera_id="camera-1", worker_boot_id="boot-1",
        source_generation=1, stream_epoch=2, frame_seq=3, source_pts_ns=None,
        track_id=case["track_id"], generation=case["generation"],
        probability=probability, observed_at_ns=5)
    out.append(list(record.payload.items()))
print(json.dumps(out))
"#;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn model_score_payload_matches_python() {
    let scores = [
        ModelScore {
            track_id: 7,
            generation: Some(3),
            fall_transition: Some(0.75),
            background: Some(0.25),
            fallen: Some(0.0),
            evidence: Some(ModelEvidence {
                raw_logit: 1.0986122886681098,
                applied_temperature: 1.0,
            }),
        },
        ModelScore {
            track_id: 0,
            generation: None,
            fall_transition: None,
            background: None,
            fallen: None,
            evidence: None,
        },
    ];
    let input = r#"[
        {"track_id": 7, "generation": 3, "probability": {"background": 0.25,
         "fall_transition": 0.75, "fallen": 0.0, "evidence": {
         "raw_logit": 1.0986122886681098, "applied_temperature": 1.0}}},
        {"track_id": 0, "generation": null, "probability": null}
    ]"#;
    let Value::Array(expected) = run_python(SCORE_SNIPPET, input) else {
        panic!("python prints one payload per case");
    };
    assert_eq!(expected.len(), scores.len());
    for (score, python) in scores.iter().zip(&expected) {
        let Value::Array(pairs) = python else {
            panic!("payload is a list of pairs");
        };
        let python: Vec<(String, Json)> = pairs
            .iter()
            .map(|pair| {
                let key = pair[0].as_str().expect("payload key").to_owned();
                (key, Json::from(&pair[1]))
            })
            .collect();
        assert_eq!(score.payload(), Json::Object(python));
    }
}
