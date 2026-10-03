//! Pure engine-entry contract shared by the offline identity writer and the
//! runtime reader. No I/O, native build, or GPU observation belongs here.

use std::ffi::CStr;

use serde_json::{Map, Value};

use crate::config::is_hex;
use crate::run::ModelRole;

/// Why measured engine or CPU source entries violate their identity contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SchemaError {
    Image,
    Receipt,
}

/// Approved static profile for [`ModelRole`]: bed `images` `[1, 3, 1280, 1280]`,
/// stored pose `images` `[1, 3, 640, 640]`, fall `window` `[1, 30, 56]`.
pub(crate) fn native_profile(role: ModelRole) -> (&'static CStr, &'static [i32]) {
    match role {
        ModelRole::Bed => (c"images", &[1, 3, 1280, 1280]),
        ModelRole::StoredPose => (c"images", &[1, 3, 640, 640]),
        ModelRole::Fall => (c"window", &[1, 30, 56]),
    }
}

/// Validates the exact four measured entries. Documents are preserved, not rebuilt.
pub(crate) fn validate_entries(
    engines: &Value,
    image: &str,
    batch: u32,
) -> Result<(), SchemaError> {
    let entries = entries(engines, image, batch, 4)?;
    let checked = [
        entry(entries, "live_pose", Role::Live, image, batch)?,
        entry(entries, "stored_pose", Role::Stored, image, batch)?,
        entry(entries, "bed", Role::Bed, image, batch)?,
        entry(entries, "fall", Role::Fall, image, batch)?,
    ];
    same_source(checked[0], checked[1])?;
    same_hardware(&checked)
}

/// Shared hybrid contract for the offline writer and runtime reader.
pub(crate) fn validate_hybrid_entries(
    engines: &Value,
    auxiliary: &Value,
    image: &str,
    batch: u32,
) -> Result<(), SchemaError> {
    validate_live_entry(engines, image, batch)?;
    validate_cpu_models(auxiliary, &engines["live_pose"])
}

/// Hybrid mode retains the unchanged live TensorRT receipt and no other engines.
pub(crate) fn validate_live_entry(
    engines: &Value,
    image: &str,
    batch: u32,
) -> Result<(), SchemaError> {
    let entries = entries(engines, image, batch, 1)?;
    entry(entries, "live_pose", Role::Live, image, batch)?;
    Ok(())
}

pub(crate) fn validate_cpu_models(auxiliary: &Value, live: &Value) -> Result<(), SchemaError> {
    let auxiliary = exact_object(auxiliary, &["runtime", "provider", "models"])?;
    if field_str(auxiliary, "runtime")? != "onnxruntime"
        || field_str(auxiliary, "provider")? != "cpu"
    {
        return Err(SchemaError::Receipt);
    }
    let models = exact_object(&auxiliary["models"], &["stored_pose", "bed", "fall"])?;
    for model in models.values() {
        exact_object(model, &["onnx_sha256"])?;
        sha_field(model, "onnx_sha256")?;
    }
    same_source(live, &models["stored_pose"])
}

fn entries<'a>(
    engines: &'a Value,
    image: &str,
    batch: u32,
    count: usize,
) -> Result<&'a Map<String, Value>, SchemaError> {
    if super::deployment_image_digest(image) != Some(image) {
        return Err(SchemaError::Image);
    }
    if !(1..=16).contains(&batch) {
        return Err(SchemaError::Receipt);
    }
    let fields = engines.as_object().ok_or(SchemaError::Receipt)?;
    if fields.len() != count {
        return Err(SchemaError::Receipt);
    }
    Ok(fields)
}

fn exact_object<'a>(
    value: &'a Value,
    keys: &[&str],
) -> Result<&'a Map<String, Value>, SchemaError> {
    let fields = value.as_object().ok_or(SchemaError::Receipt)?;
    if fields.len() != keys.len() || !keys.iter().all(|key| fields.contains_key(*key)) {
        return Err(SchemaError::Receipt);
    }
    Ok(fields)
}

fn entry<'a>(
    entries: &'a Map<String, Value>,
    key: &str,
    role: Role,
    image: &str,
    batch: u32,
) -> Result<&'a Value, SchemaError> {
    let document = entries.get(key).ok_or(SchemaError::Receipt)?;
    let Value::Object(fields) = document else {
        return Err(SchemaError::Receipt);
    };
    common(fields, image)?;
    sha_text(fields.get("onnx_sha256"))?;
    sha_text(fields.get("engine_sha256"))?;
    geometry(fields, role, batch)?;
    Ok(document)
}

fn same_source(live: &Value, stored: &Value) -> Result<(), SchemaError> {
    let live_sha = sha_field(live, "onnx_sha256")?;
    let stored_sha = sha_field(stored, "onnx_sha256")?;
    if live_sha != stored_sha {
        return Err(SchemaError::Receipt);
    }
    Ok(())
}

fn same_hardware(documents: &[&Value]) -> Result<(), SchemaError> {
    let first = documents[0];
    for document in &documents[1..] {
        if field(document, "trt_version")? != field(first, "trt_version")?
            || field(document, "device_name")? != field(first, "device_name")?
            || field(document, "compute_capability")? != field(first, "compute_capability")?
        {
            return Err(SchemaError::Receipt);
        }
    }
    Ok(())
}

fn common(document: &Map<String, Value>, image: &str) -> Result<(), SchemaError> {
    let engine = field_str(document, "engine")?;
    let mut components = std::path::Path::new(engine).components();
    if engine.contains('\0')
        || !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(SchemaError::Receipt);
    }
    if field_str(document, "image_digest")? != image {
        return Err(SchemaError::Image);
    }
    if document.get("device") != Some(&Value::Number(0.into())) {
        return Err(SchemaError::Receipt);
    }
    let name = field_str(document, "device_name")?;
    if name.trim().is_empty() || name.contains('\0') || !positive(document.get("trt_version")) {
        return Err(SchemaError::Receipt);
    }
    let (major, minor) = field_str(document, "compute_capability")?
        .split_once('.')
        .ok_or(SchemaError::Receipt)?;
    if !digits(major, false) || !digits(minor, true) {
        return Err(SchemaError::Receipt);
    }
    Ok(())
}

fn geometry(document: &Map<String, Value>, role: Role, batch: u32) -> Result<(), SchemaError> {
    match role {
        Role::Live => live(document, batch),
        Role::Stored => native(document, ModelRole::StoredPose),
        Role::Bed => native(document, ModelRole::Bed),
        Role::Fall => native(document, ModelRole::Fall),
    }
}

fn live(document: &Map<String, Value>, batch: u32) -> Result<(), SchemaError> {
    let observer = field_str(document, "observer_library_sha256")?;
    let batch = i64::from(batch);
    let ok = field_str(document, "precision")? == "fp16"
        && field_str(document, "input")? == "images"
        && matches!(document.get("tf32_enabled"), Some(Value::Bool(_)))
        && is_hex(observer, 64)
        && ints(document.get("min_dimensions"))? == [1, 3, 640, 640]
        && ints(document.get("opt_dimensions"))? == [batch, 3, 640, 640]
        && ints(document.get("max_dimensions"))? == [batch, 3, 640, 640];
    ok.then_some(()).ok_or(SchemaError::Receipt)
}

fn native(document: &Map<String, Value>, role: ModelRole) -> Result<(), SchemaError> {
    let (input, dimensions) = native_profile(role);
    let input = input.to_str().map_err(|_| SchemaError::Receipt)?;
    let expected = dimensions
        .iter()
        .map(|item| i64::from(*item))
        .collect::<Vec<_>>();
    let ok = field_str(document, "precision")? == "fp32"
        && document.get("tf32_enabled") == Some(&Value::Bool(false))
        && field_str(document, "input")? == input
        && ints(document.get("dimensions"))? == expected;
    ok.then_some(()).ok_or(SchemaError::Receipt)
}

fn ints(value: Option<&Value>) -> Result<Vec<i64>, SchemaError> {
    value
        .and_then(Value::as_array)
        .ok_or(SchemaError::Receipt)?
        .iter()
        .map(|item| item.as_i64().ok_or(SchemaError::Receipt))
        .collect()
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, SchemaError> {
    value.get(key).ok_or(SchemaError::Receipt)
}

fn field_str<'a>(document: &'a Map<String, Value>, key: &str) -> Result<&'a str, SchemaError> {
    document
        .get(key)
        .and_then(Value::as_str)
        .ok_or(SchemaError::Receipt)
}

fn sha_field<'a>(document: &'a Value, key: &str) -> Result<&'a str, SchemaError> {
    sha_text(document.get(key))
}

fn sha_text(value: Option<&Value>) -> Result<&str, SchemaError> {
    let text = value.and_then(Value::as_str).ok_or(SchemaError::Receipt)?;
    if !is_hex(text, 64) {
        return Err(SchemaError::Receipt);
    }
    Ok(text)
}

fn positive(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_i64)
        .is_some_and(|version| version > 0)
}

fn digits(text: &str, zero_ok: bool) -> bool {
    !text.is_empty()
        && text.bytes().all(|byte| byte.is_ascii_digit())
        && (text == "0" && zero_ok || !text.starts_with('0'))
}

#[derive(Clone, Copy)]
enum Role {
    Live,
    Stored,
    Bed,
    Fall,
}
