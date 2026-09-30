//! `contracts/model_selection.py:parse_model_selection` over a `Json` value,
//! in the Python validation order. A refusal names the same dotted field as
//! the Python message; the canonical bytes come from `json.rs`.

use std::collections::BTreeSet;

use super::{is_hex, is_segment, lookup};
use crate::json::{Json, JsonError, model_selection_digest};

const WHERE: &str = "model-selection";
const KEYS: &str = "schema_version model_publication bundle_members_digest dataset_publication \
    evaluation_receipt_digest field_evaluation_receipt_digest calibration_digest \
    conformance_digest input_observation_schema output_class_count \
    output_class_semantics_digest policy_digest runtime_format bundle_format \
    preprocessing_identity transition_threshold threshold_source";

/// Which Python check refused the document.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionKind {
    NotObject,
    Keys,
    SchemaVersion,
    NotString,
    Locator,
    Revision,
    Digest,
    PositiveInt,
    Probability,
    Range,
    ThresholdSource,
}

/// A refusal: the dotted field Python names and the failed check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionError {
    pub field: String,
    pub kind: SelectionKind,
}

/// A publication; `content` is `bundle_sha256` for the model and
/// `payload_digest` for the dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct Publication {
    pub source_locator: String,
    pub revision: String,
    pub content: String,
}

/// `ModelSelection`, field for field.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSelection {
    pub model_publication: Publication,
    pub bundle_members_digest: String,
    pub dataset_publication: Publication,
    pub evaluation_receipt_digest: String,
    pub field_evaluation_receipt_digest: String,
    pub calibration_digest: String,
    pub conformance_digest: String,
    pub input_observation_schema: String,
    pub output_class_count: i128,
    pub output_class_semantics_digest: String,
    pub policy_digest: String,
    pub runtime_format: String,
    pub bundle_format: String,
    pub preprocessing_identity: String,
    pub transition_threshold: f64,
    pub threshold_source: String,
}

fn text_members(pairs: &[(&str, &String)]) -> Vec<(String, Json)> {
    let member = |(key, value): &(&str, &String)| ((*key).to_owned(), Json::Str((*value).clone()));
    pairs.iter().map(member).collect()
}

fn publication_json(value: &Publication, content_key: &str) -> Json {
    Json::Object(text_members(&[
        ("source_locator", &value.source_locator),
        ("revision", &value.revision),
        (content_key, &value.content),
    ]))
}

impl ModelSelection {
    /// `ModelSelection.as_dict()`.
    pub fn as_json(&self) -> Json {
        let mut members = text_members(&[
            ("bundle_members_digest", &self.bundle_members_digest),
            ("evaluation_receipt_digest", &self.evaluation_receipt_digest),
            (
                "field_evaluation_receipt_digest",
                &self.field_evaluation_receipt_digest,
            ),
            ("calibration_digest", &self.calibration_digest),
            ("conformance_digest", &self.conformance_digest),
            ("input_observation_schema", &self.input_observation_schema),
            (
                "output_class_semantics_digest",
                &self.output_class_semantics_digest,
            ),
            ("policy_digest", &self.policy_digest),
            ("runtime_format", &self.runtime_format),
            ("bundle_format", &self.bundle_format),
            ("preprocessing_identity", &self.preprocessing_identity),
            ("threshold_source", &self.threshold_source),
        ]);
        let model = publication_json(&self.model_publication, "bundle_sha256");
        let dataset = publication_json(&self.dataset_publication, "payload_digest");
        members.extend([
            ("schema_version".to_owned(), Json::Int(2)),
            ("model_publication".to_owned(), model),
            ("dataset_publication".to_owned(), dataset),
            (
                "output_class_count".to_owned(),
                Json::Int(self.output_class_count),
            ),
            (
                "transition_threshold".to_owned(),
                Json::Float(self.transition_threshold),
            ),
        ]);
        Json::Object(members)
    }

    /// `ModelSelection.digest()`.
    pub fn digest(&self) -> Result<String, JsonError> {
        model_selection_digest(&self.as_json())
    }
}

type Members = [(String, Json)];
type Refusal<T> = Result<T, SelectionError>;

fn refuse<T>(place: &str, key: &str, kind: SelectionKind) -> Refusal<T> {
    let dot = if key.is_empty() { "" } else { "." };
    let field = format!("{place}{dot}{key}");
    Err(SelectionError { field, kind })
}

pub(crate) fn object<'a>(raw: &'a Json, place: &str) -> Refusal<&'a Members> {
    match raw {
        Json::Object(members) => Ok(members),
        _ => refuse(place, "", SelectionKind::NotObject),
    }
}

pub(crate) fn exact_keys<'k>(
    members: &Members,
    expected: impl IntoIterator<Item = &'k str>,
    place: &str,
) -> Refusal<()> {
    let present: BTreeSet<&str> = members.iter().map(|(key, _)| key.as_str()).collect();
    if present != expected.into_iter().collect() {
        return refuse(place, "", SelectionKind::Keys);
    }
    Ok(())
}

pub(crate) fn string(members: &Members, key: &str, place: &str) -> Refusal<String> {
    match lookup(members, key) {
        Some(Json::Str(value)) if !value.is_empty() => Ok(value.clone()),
        _ => refuse(place, key, SelectionKind::NotString),
    }
}

pub(crate) fn digest(members: &Members, key: &str, place: &str) -> Refusal<String> {
    let value = string(members, key, place)?;
    if !is_hex(&value, 64) {
        return refuse(place, key, SelectionKind::Digest);
    }
    Ok(value)
}

pub(crate) fn positive_integer(m: &Members, key: &str, place: &str) -> Refusal<i128> {
    match lookup(m, key) {
        Some(Json::Int(value)) if *value >= 1 => Ok(*value),
        _ => refuse(place, key, SelectionKind::PositiveInt),
    }
}

fn publication(m: &Members, key: &str, content_key: &str) -> Refusal<Publication> {
    let place = format!("{WHERE}.{key}");
    let inner = object(lookup(m, key).unwrap_or(&Json::Null), &place)?;
    exact_keys(inner, ["source_locator", "revision", content_key], &place)?;
    let source_locator = string(inner, "source_locator", &place)?;
    let split = source_locator.split_once('/');
    if !split.is_some_and(|(owner, repository)| is_segment(owner) && is_segment(repository)) {
        return refuse(&place, "source_locator", SelectionKind::Locator);
    }
    let revision = string(inner, "revision", &place)?;
    if !is_hex(&revision, 40) {
        return refuse(&place, "revision", SelectionKind::Revision);
    }
    let content = digest(inner, content_key, &place)?;
    Ok(Publication {
        source_locator,
        revision,
        content,
    })
}

fn probability(members: &Members, key: &str) -> Refusal<f64> {
    let parsed = match lookup(members, key) {
        Some(Json::Int(value)) => *value as f64,
        Some(Json::Float(value)) => *value,
        _ => return refuse(WHERE, key, SelectionKind::Probability),
    };
    if !(0.0..=1.0).contains(&parsed) {
        return refuse(WHERE, key, SelectionKind::Range);
    }
    Ok(parsed)
}

fn threshold_source(members: &Members, key: &str) -> Refusal<String> {
    let value = string(members, key, WHERE)?;
    if !matches!(value.as_str(), "default" | "receipt") {
        return refuse(WHERE, key, SelectionKind::ThresholdSource);
    }
    Ok(value)
}

/// `parse_model_selection(raw)`. Struct fields are evaluated in the order
/// written, which is the Python `_desired_fields` order.
pub fn parse_model_selection(raw: &Json) -> Refusal<ModelSelection> {
    let m = object(raw, WHERE)?;
    exact_keys(m, KEYS.split_whitespace(), WHERE)?;
    let version = lookup(m, "schema_version");
    if version != Some(&Json::Int(2)) && version != Some(&Json::Float(2.0)) {
        return refuse(WHERE, "schema_version", SelectionKind::SchemaVersion);
    }
    let text = |key: &str| string(m, key, WHERE);
    let hex = |key: &str| digest(m, key, WHERE);
    Ok(ModelSelection {
        model_publication: publication(m, "model_publication", "bundle_sha256")?,
        bundle_members_digest: hex("bundle_members_digest")?,
        dataset_publication: publication(m, "dataset_publication", "payload_digest")?,
        evaluation_receipt_digest: hex("evaluation_receipt_digest")?,
        field_evaluation_receipt_digest: hex("field_evaluation_receipt_digest")?,
        calibration_digest: hex("calibration_digest")?,
        conformance_digest: hex("conformance_digest")?,
        input_observation_schema: text("input_observation_schema")?,
        output_class_count: positive_integer(m, "output_class_count", WHERE)?,
        output_class_semantics_digest: hex("output_class_semantics_digest")?,
        policy_digest: hex("policy_digest")?,
        runtime_format: text("runtime_format")?,
        bundle_format: text("bundle_format")?,
        preprocessing_identity: text("preprocessing_identity")?,
        transition_threshold: probability(m, "transition_threshold")?,
        threshold_source: threshold_source(m, "threshold_source")?,
    })
}
