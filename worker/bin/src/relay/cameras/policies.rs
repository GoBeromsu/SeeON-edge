//! Python `shared/detection_policies.py`: the closed parser for the
//! `detection_policies` bundle of the pulled worker config. Module identity,
//! policy schema, field set, numeric type and range, cross-field rules and
//! content identity are checked in the Python order, so the first refusal is
//! the one Python raises. Every refusal is a typed `Err`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sha2::{Digest, Sha256};

use crate::config::lookup;
use crate::json::{Json, JsonError, Serialiser};

/// The only bundle schema version `parse_policy_bundle` accepts.
pub const BUNDLE_SCHEMA_VERSION: i128 = 1;

const FALL: &str = "fall";
const BED_EXIT: &str = "bed_exit";
const MAX_FRAMES: i128 = 300;

/// `PolicyDefinition` without units or image default.
struct Definition {
    module_id: &'static str,
    module_version: i128,
    schema_id: &'static str,
    schema_version: i128,
}

/// `_POLICY_DEFINITIONS`; each module has exactly one (latest) version.
const DEFINITIONS: [Definition; 2] = [
    Definition {
        module_id: FALL,
        module_version: 2,
        schema_id: "fall.policy",
        schema_version: 2,
    },
    Definition {
        module_id: BED_EXIT,
        module_version: 1,
        schema_id: "bed_exit.policy",
        schema_version: 1,
    },
];

const EFFECTIVE_FIELDS: [&str; 9] = [
    "module_id",
    "module_version",
    "schema_id",
    "schema_version",
    "source",
    "facility_revision_id",
    "camera_revision_id",
    "values",
    "effective_policy_id",
];

/// `PolicySource`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicySource {
    ImageDefault,
    FacilityDefault,
    CameraOverride,
}

impl PolicySource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ImageDefault => "image-default",
            Self::FacilityDefault => "facility-default",
            Self::CameraOverride => "camera-override",
        }
    }

    /// `_policy_source`: equality with one of the three literals.
    fn parse(value: &Json) -> Result<Self, PolicyError> {
        match value {
            Json::Str(text) if text == "image-default" => Ok(Self::ImageDefault),
            Json::Str(text) if text == "facility-default" => Ok(Self::FacilityDefault),
            Json::Str(text) if text == "camera-override" => Ok(Self::CameraOverride),
            _ => Err(PolicyError::UnknownSource),
        }
    }
}

/// `FallPolicyV2` (only its wire field) or `BedExitPolicyV1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PolicyValues {
    FallV2 {
        transition_threshold: f64,
    },
    BedExitV1 {
        min_containment: f64,
        hold_frames: i128,
        grace_frames: i128,
    },
}

impl PolicyValues {
    /// `FALL_POLICY_V2_DEFAULT`.
    pub const FALL_DEFAULT: Self = Self::FallV2 {
        transition_threshold: 0.5,
    };
    /// `BED_EXIT_POLICY_V1_DEFAULT`.
    pub const BED_EXIT_DEFAULT: Self = Self::BedExitV1 {
        min_containment: 0.35,
        hold_frames: 2,
        grace_frames: 3,
    };

    /// `policy_values_dict`.
    pub fn as_json(&self) -> Json {
        match *self {
            Self::FallV2 {
                transition_threshold,
            } => Json::Object(vec![(
                "transition_threshold".to_owned(),
                Json::Float(transition_threshold),
            )]),
            Self::BedExitV1 {
                min_containment,
                hold_frames,
                grace_frames,
            } => Json::Object(vec![
                ("min_containment".to_owned(), Json::Float(min_containment)),
                ("hold_frames".to_owned(), Json::Int(hold_frames)),
                ("grace_frames".to_owned(), Json::Int(grace_frames)),
            ]),
        }
    }
}

/// `EffectivePolicy`.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectivePolicy {
    pub module_id: String,
    pub module_version: i128,
    pub schema_id: String,
    pub schema_version: i128,
    pub source: PolicySource,
    pub facility_revision_id: Option<i128>,
    pub camera_revision_id: Option<i128>,
    pub values: PolicyValues,
    pub effective_policy_id: String,
}

impl EffectivePolicy {
    /// `EffectivePolicy.as_dict`.
    pub fn as_json(&self) -> Json {
        let mut members = identity_members(
            &self.module_id,
            self.module_version,
            &self.schema_id,
            self.schema_version,
            self.source,
            self.facility_revision_id,
            self.camera_revision_id,
            &self.values,
        );
        members.push((
            "effective_policy_id".to_owned(),
            Json::Str(self.effective_policy_id.clone()),
        ));
        Json::Object(members)
    }
}

/// `PolicyBundle`: defaults and camera maps keyed by module id.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyBundle {
    pub schema_version: i128,
    pub defaults: BTreeMap<String, EffectivePolicy>,
    pub cameras: BTreeMap<String, BTreeMap<String, EffectivePolicy>>,
}

impl PolicyBundle {
    /// `PolicyBundle.as_dict`.
    pub fn as_json(&self) -> Json {
        let modules = |policies: &BTreeMap<String, EffectivePolicy>| {
            Json::Object(
                policies
                    .iter()
                    .map(|(module_id, policy)| (module_id.clone(), policy.as_json()))
                    .collect(),
            )
        };
        Json::Object(vec![
            ("schema_version".to_owned(), Json::Int(self.schema_version)),
            ("defaults".to_owned(), modules(&self.defaults)),
            (
                "cameras".to_owned(),
                Json::Object(
                    self.cameras
                        .iter()
                        .map(|(camera_id, policies)| (camera_id.clone(), modules(policies)))
                        .collect(),
                ),
            ),
        ])
    }

    /// `PolicyBundle.content_sha256`: lowercase hex sha256 of the canonical
    /// `as_dict` text.
    pub fn content_sha256(&self) -> Result<String, JsonError> {
        sha256_hex(&self.as_json())
    }
}

/// Why a policy document was refused (`PolicyDocumentError`). The variants
/// follow the Python check order; labels and field names are the Python
/// ones, except that a camera entry is labelled without its id and a module
/// id is quoted as a Rust string rather than a Python `repr`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyError {
    /// `_mapping`: `<label> must be an object`.
    NotObject(&'static str),
    /// `_require_fields`: unknown fields, sorted; reported before missing.
    UnknownFields {
        label: &'static str,
        fields: Vec<String>,
    },
    /// `_require_fields`: missing fields, sorted.
    MissingFields {
        label: &'static str,
        fields: Vec<String>,
    },
    /// `_integer`: not a JSON integer (booleans included).
    NotInteger(&'static str),
    /// `_finite_number`: not a JSON number (booleans included).
    NotNumeric(&'static str),
    /// `_finite_number`: NaN or infinite.
    NotFinite(&'static str),
    /// `_text`: not a non-empty string.
    EmptyText(&'static str),
    /// The bundle `schema_version` is not 1.
    BundleSchemaVersion,
    /// The policy `module_id` differs from its map key.
    ModuleMismatch,
    /// The policy `module_version` differs from the latest version.
    VersionMismatch,
    /// `policy_definition`: no such module.
    UnknownModule(String),
    /// `policy_definition`: the module exists at another version.
    UnsupportedVersion {
        module_id: String,
        received: i128,
        supported: i128,
    },
    /// The policy schema differs from the module's schema; each side is
    /// qualified as `<id>.v<version>`.
    SchemaDrift {
        module: String,
        received: String,
        supported: String,
    },
    /// `_policy_source`: not one of the three sources.
    UnknownSource,
    /// `_optional_revision`: an integer below 1.
    RevisionNotPositive(&'static str),
    /// A value outside its range, written as Python writes it.
    OutOfRange {
        field: &'static str,
        range: &'static str,
    },
    /// `hold_frames + grace_frames` exceeds 300.
    FramesExceed,
    /// The source and revision ids disagree (`make_effective_policy`).
    RevisionConsistency(PolicySource),
    /// `effective_policy_id` is not the content identity.
    IdentityMismatch,
    /// The identity text could not be written. The shapes built here have
    /// unique keys, so the canonical writer does not refuse them; the case
    /// stays a typed refusal instead of a panic.
    Canonical(JsonError),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotObject(label) => write!(formatter, "{label} must be an object"),
            Self::UnknownFields { label, fields } => write!(
                formatter,
                "{label} contains unknown field(s): {}",
                fields.join(", ")
            ),
            Self::MissingFields { label, fields } => write!(
                formatter,
                "{label} is missing field(s): {}",
                fields.join(", ")
            ),
            Self::NotInteger(field) => write!(formatter, "{field} must be an integer"),
            Self::NotNumeric(field) => write!(formatter, "{field} must be numeric"),
            Self::NotFinite(field) => write!(formatter, "{field} must be finite"),
            Self::EmptyText(field) => write!(formatter, "{field} must be a non-empty string"),
            Self::BundleSchemaVersion => {
                formatter.write_str("unsupported detection policy bundle schema version")
            }
            Self::ModuleMismatch => {
                formatter.write_str("effective policy module id does not match its map key")
            }
            Self::VersionMismatch => {
                formatter.write_str("effective policy module version does not match selection")
            }
            Self::UnknownModule(module_id) => {
                write!(formatter, "unknown policy module {module_id:?}")
            }
            Self::UnsupportedVersion {
                module_id,
                received,
                supported,
            } => write!(
                formatter,
                "unsupported policy module version for {module_id:?}: \
                 received {received}, supported {supported}"
            ),
            Self::SchemaDrift {
                module,
                received,
                supported,
            } => write!(
                formatter,
                "policy schema drift for {module}: received {received}, supported {supported}"
            ),
            Self::UnknownSource => formatter.write_str("effective policy source is unknown"),
            Self::RevisionNotPositive(field) => {
                write!(formatter, "{field} must be positive or null")
            }
            Self::OutOfRange { field, range } => write!(formatter, "{field} must be in {range}"),
            Self::FramesExceed => formatter
                .write_str("combined hold_frames + grace_frames must not exceed 300 frames"),
            Self::RevisionConsistency(PolicySource::ImageDefault) => {
                formatter.write_str("image-default policy cannot carry revision ids")
            }
            Self::RevisionConsistency(PolicySource::FacilityDefault) => {
                formatter.write_str("facility-default policy requires a facility revision")
            }
            Self::RevisionConsistency(PolicySource::CameraOverride) => {
                formatter.write_str("camera-override policy requires a camera revision")
            }
            Self::IdentityMismatch => {
                formatter.write_str("effective policy identity does not match its content")
            }
            Self::Canonical(error) => write!(formatter, "effective policy identity: {error}"),
        }
    }
}

impl std::error::Error for PolicyError {}

/// `resolved_detection_policies`: an absent or null bundle is the image
/// default for every parsed camera; anything else must parse.
pub fn resolve_detection_policies(
    value: &Json,
    camera_ids: &[String],
) -> Result<PolicyBundle, PolicyError> {
    match value {
        Json::Null => default_policy_bundle(camera_ids),
        bundle => parse_policy_bundle(bundle),
    }
}

/// `default_policy_bundle`.
pub fn default_policy_bundle(camera_ids: &[String]) -> Result<PolicyBundle, PolicyError> {
    let mut defaults = BTreeMap::new();
    for (definition, values) in DEFINITIONS
        .iter()
        .zip([PolicyValues::FALL_DEFAULT, PolicyValues::BED_EXIT_DEFAULT])
    {
        let policy = make_effective_policy(
            definition.module_id,
            definition.module_version,
            &values,
            PolicySource::ImageDefault,
            None,
            None,
        )?;
        defaults.insert(definition.module_id.to_owned(), policy);
    }
    let cameras = camera_ids
        .iter()
        .map(|camera_id| (camera_id.clone(), defaults.clone()))
        .collect();
    Ok(PolicyBundle {
        schema_version: BUNDLE_SCHEMA_VERSION,
        defaults,
        cameras,
    })
}

/// `parse_policy_bundle`.
pub fn parse_policy_bundle(value: &Json) -> Result<PolicyBundle, PolicyError> {
    let members = mapping(value, "detection policy bundle")?;
    require_fields(
        members,
        &["schema_version", "defaults", "cameras"],
        "policy bundle",
    )?;
    if integer(field(members, "schema_version"), "schema_version")? != BUNDLE_SCHEMA_VERSION {
        return Err(PolicyError::BundleSchemaVersion);
    }
    let raw_defaults = mapping(field(members, "defaults"), "policy defaults")?;
    let defaults = module_policies(raw_defaults, "policy defaults")?;
    let raw_cameras = mapping(field(members, "cameras"), "camera policies")?;
    let mut cameras = BTreeMap::new();
    for (raw_camera_id, raw_policies) in raw_cameras {
        // Object keys are always strings, so `_text(raw_camera_id,
        // "camera policy id")` in Python can only refuse the empty key.
        if raw_camera_id.is_empty() {
            return Err(PolicyError::EmptyText("camera policy id"));
        }
        let label = "camera policies for camera";
        let policies = module_policies(mapping(raw_policies, label)?, label)?;
        cameras.insert(raw_camera_id.clone(), policies);
    }
    Ok(PolicyBundle {
        schema_version: BUNDLE_SCHEMA_VERSION,
        defaults,
        cameras,
    })
}

/// Exactly the known modules, each parsed at its latest version, in sorted
/// module order.
fn module_policies(
    members: &[(String, Json)],
    label: &'static str,
) -> Result<BTreeMap<String, EffectivePolicy>, PolicyError> {
    let mut modules: Vec<&Definition> = DEFINITIONS.iter().collect();
    modules.sort_by_key(|definition| definition.module_id);
    let expected: Vec<&str> = modules
        .iter()
        .map(|definition| definition.module_id)
        .collect();
    require_fields(members, &expected, label)?;
    let mut policies = BTreeMap::new();
    for definition in modules {
        let policy = parse_effective_policy(
            field(members, definition.module_id),
            Some(definition.module_id),
            Some(definition.module_version),
        )?;
        policies.insert(definition.module_id.to_owned(), policy);
    }
    Ok(policies)
}

/// `parse_effective_policy`.
pub fn parse_effective_policy(
    value: &Json,
    expected_module_id: Option<&str>,
    expected_module_version: Option<i128>,
) -> Result<EffectivePolicy, PolicyError> {
    let members = mapping(value, "effective policy")?;
    require_fields(members, &EFFECTIVE_FIELDS, "effective policy")?;
    let module_id = text(field(members, "module_id"), "module_id")?;
    let module_version = integer(field(members, "module_version"), "module_version")?;
    if expected_module_id.is_some_and(|expected| module_id != expected) {
        return Err(PolicyError::ModuleMismatch);
    }
    if expected_module_version.is_some_and(|expected| module_version != expected) {
        return Err(PolicyError::VersionMismatch);
    }
    let schema_id = text(field(members, "schema_id"), "schema_id")?;
    let schema_version = integer(field(members, "schema_version"), "schema_version")?;
    let source = PolicySource::parse(field(members, "source"))?;
    let facility_revision_id = optional_revision(
        field(members, "facility_revision_id"),
        "facility_revision_id",
    )?;
    let camera_revision_id =
        optional_revision(field(members, "camera_revision_id"), "camera_revision_id")?;
    let values = parse_policy_values(
        module_id,
        module_version,
        schema_id,
        schema_version,
        field(members, "values"),
    )?;
    let parsed = make_effective_policy(
        module_id,
        module_version,
        &values,
        source,
        facility_revision_id,
        camera_revision_id,
    )?;
    let identity = text(field(members, "effective_policy_id"), "effective_policy_id")?;
    if identity != parsed.effective_policy_id {
        return Err(PolicyError::IdentityMismatch);
    }
    Ok(parsed)
}

/// `parse_policy_values`.
pub fn parse_policy_values(
    module_id: &str,
    module_version: i128,
    schema_id: &str,
    schema_version: i128,
    values: &Json,
) -> Result<PolicyValues, PolicyError> {
    let definition = policy_definition(module_id, module_version)?;
    if (schema_id, schema_version) != (definition.schema_id, definition.schema_version) {
        return Err(PolicyError::SchemaDrift {
            module: format!("{}.v{}", definition.module_id, definition.module_version),
            received: format!("{schema_id}.v{schema_version}"),
            supported: format!("{}.v{}", definition.schema_id, definition.schema_version),
        });
    }
    let members = mapping(values, "policy values")?;
    if definition.module_id == FALL {
        require_fields(members, &["transition_threshold"], "fall policy")?;
        let threshold = finite_number(
            field(members, "transition_threshold"),
            "transition_threshold",
        )?;
        if !(0.0..=1.0).contains(&threshold) {
            return Err(PolicyError::OutOfRange {
                field: "transition_threshold",
                range: "[0, 1]",
            });
        }
        return Ok(PolicyValues::FallV2 {
            transition_threshold: threshold,
        });
    }
    require_fields(
        members,
        &["min_containment", "hold_frames", "grace_frames"],
        "bed-exit policy",
    )?;
    let containment = finite_number(field(members, "min_containment"), "min_containment")?;
    let hold_frames = integer(field(members, "hold_frames"), "hold_frames")?;
    let grace_frames = integer(field(members, "grace_frames"), "grace_frames")?;
    if !(containment > 0.0 && containment <= 1.0) {
        return Err(PolicyError::OutOfRange {
            field: "min_containment",
            range: "(0, 1]",
        });
    }
    if !(1..=MAX_FRAMES).contains(&hold_frames) {
        return Err(PolicyError::OutOfRange {
            field: "hold_frames",
            range: "[1, 300]",
        });
    }
    if !(0..=MAX_FRAMES).contains(&grace_frames) {
        return Err(PolicyError::OutOfRange {
            field: "grace_frames",
            range: "[0, 300]",
        });
    }
    if hold_frames + grace_frames > MAX_FRAMES {
        return Err(PolicyError::FramesExceed);
    }
    Ok(PolicyValues::BedExitV1 {
        min_containment: containment,
        hold_frames,
        grace_frames,
    })
}

/// `make_effective_policy`: re-checks the values, checks the source against
/// the revision ids and derives the content identity.
pub fn make_effective_policy(
    module_id: &str,
    module_version: i128,
    values: &PolicyValues,
    source: PolicySource,
    facility_revision_id: Option<i128>,
    camera_revision_id: Option<i128>,
) -> Result<EffectivePolicy, PolicyError> {
    let definition = policy_definition(module_id, module_version)?;
    let parsed = parse_policy_values(
        module_id,
        module_version,
        definition.schema_id,
        definition.schema_version,
        &values.as_json(),
    )?;
    let consistent = match source {
        PolicySource::ImageDefault => {
            facility_revision_id.is_none() && camera_revision_id.is_none()
        }
        PolicySource::FacilityDefault => facility_revision_id.is_some(),
        PolicySource::CameraOverride => camera_revision_id.is_some(),
    };
    if !consistent {
        return Err(PolicyError::RevisionConsistency(source));
    }
    let identity = sha256_hex(&Json::Object(identity_members(
        module_id,
        module_version,
        definition.schema_id,
        definition.schema_version,
        source,
        facility_revision_id,
        camera_revision_id,
        &parsed,
    )))
    .map_err(PolicyError::Canonical)?;
    Ok(EffectivePolicy {
        module_id: module_id.to_owned(),
        module_version,
        schema_id: definition.schema_id.to_owned(),
        schema_version: definition.schema_version,
        source,
        facility_revision_id,
        camera_revision_id,
        values: parsed,
        effective_policy_id: identity,
    })
}

/// `policy_definition`.
fn policy_definition(
    module_id: &str,
    module_version: i128,
) -> Result<&'static Definition, PolicyError> {
    if let Some(definition) = DEFINITIONS.iter().find(|definition| {
        definition.module_id == module_id && definition.module_version == module_version
    }) {
        return Ok(definition);
    }
    match DEFINITIONS
        .iter()
        .find(|definition| definition.module_id == module_id)
    {
        Some(known) => Err(PolicyError::UnsupportedVersion {
            module_id: module_id.to_owned(),
            received: module_version,
            supported: known.module_version,
        }),
        None => Err(PolicyError::UnknownModule(module_id.to_owned())),
    }
}

/// The `make_effective_policy` identity payload, which is also the
/// `as_dict` of an effective policy without its id.
#[allow(clippy::too_many_arguments)]
fn identity_members(
    module_id: &str,
    module_version: i128,
    schema_id: &str,
    schema_version: i128,
    source: PolicySource,
    facility_revision_id: Option<i128>,
    camera_revision_id: Option<i128>,
    values: &PolicyValues,
) -> Vec<(String, Json)> {
    let revision = |id: Option<i128>| id.map_or(Json::Null, Json::Int);
    vec![
        ("module_id".to_owned(), Json::Str(module_id.to_owned())),
        ("module_version".to_owned(), Json::Int(module_version)),
        ("schema_id".to_owned(), Json::Str(schema_id.to_owned())),
        ("schema_version".to_owned(), Json::Int(schema_version)),
        ("source".to_owned(), Json::Str(source.as_str().to_owned())),
        (
            "facility_revision_id".to_owned(),
            revision(facility_revision_id),
        ),
        (
            "camera_revision_id".to_owned(),
            revision(camera_revision_id),
        ),
        ("values".to_owned(), values.as_json()),
    ]
}

/// `hashlib.sha256(_canonical_json(value).encode()).hexdigest()`.
/// `_canonical_json` is `sort_keys`, compact, `ensure_ascii=False` and
/// `allow_nan=True`: the `ExecutionRecords` mode.
fn sha256_hex(value: &Json) -> Result<String, JsonError> {
    let text = Serialiser::ExecutionRecords.canonical(value)?;
    Ok(Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// A member the preceding `require_fields` proved present.
fn field<'a>(members: &'a [(String, Json)], key: &str) -> &'a Json {
    lookup(members, key).unwrap_or(&Json::Null)
}

/// `_mapping`.
fn mapping<'a>(value: &'a Json, label: &'static str) -> Result<&'a [(String, Json)], PolicyError> {
    match value {
        Json::Object(members) => Ok(members),
        _ => Err(PolicyError::NotObject(label)),
    }
}

/// `_require_fields`: unknown fields first, then missing ones, each sorted.
fn require_fields(
    members: &[(String, Json)],
    expected: &[&str],
    label: &'static str,
) -> Result<(), PolicyError> {
    let actual: BTreeSet<&str> = members.iter().map(|(key, _)| key.as_str()).collect();
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    let unknown: Vec<String> = actual
        .difference(&expected)
        .map(|key| (*key).to_owned())
        .collect();
    if !unknown.is_empty() {
        return Err(PolicyError::UnknownFields {
            label,
            fields: unknown,
        });
    }
    let missing: Vec<String> = expected
        .difference(&actual)
        .map(|key| (*key).to_owned())
        .collect();
    if !missing.is_empty() {
        return Err(PolicyError::MissingFields {
            label,
            fields: missing,
        });
    }
    Ok(())
}

/// `_finite_number`: an integer or float (never a boolean), as a float.
fn finite_number(value: &Json, name: &'static str) -> Result<f64, PolicyError> {
    #[allow(clippy::cast_precision_loss)]
    let parsed = match value {
        Json::Int(number) => *number as f64,
        Json::Float(number) => *number,
        _ => return Err(PolicyError::NotNumeric(name)),
    };
    if !parsed.is_finite() {
        return Err(PolicyError::NotFinite(name));
    }
    Ok(parsed)
}

/// `_integer`: a JSON integer (never a boolean).
fn integer(value: &Json, name: &'static str) -> Result<i128, PolicyError> {
    match value {
        Json::Int(number) => Ok(*number),
        _ => Err(PolicyError::NotInteger(name)),
    }
}

/// `_optional_revision`: null or an integer of at least 1.
fn optional_revision(value: &Json, name: &'static str) -> Result<Option<i128>, PolicyError> {
    match value {
        Json::Null => Ok(None),
        other => match integer(other, name)? {
            revision if revision >= 1 => Ok(Some(revision)),
            _ => Err(PolicyError::RevisionNotPositive(name)),
        },
    }
}

/// `_text`: a non-empty string.
fn text<'a>(value: &'a Json, name: &'static str) -> Result<&'a str, PolicyError> {
    match value {
        Json::Str(text) if !text.is_empty() => Ok(text),
        _ => Err(PolicyError::EmptyText(name)),
    }
}
