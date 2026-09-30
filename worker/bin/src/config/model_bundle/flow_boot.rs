//! Flow cold-start boot verification (`verify_flow_boot_inputs`,
//! `worker/runtime/flow/cold_start.py` L132-173): the flow profile wiring,
//! the configured batch, the engine identity and, above batch 1, a dynamic
//! ONNX batch axis.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::identity::{FLOW_IDENTITY_FILES, IdentityKind, verify_engine_identity};
use super::onnx_shape::{Dim, OnnxShapeKind, batch_axis_is_dynamic, input_dims};
use crate::config::env::Env;

/// `FLOW_BOOT_ENV`, in the order Python names missing keys.
pub const FLOW_BOOT_ENV: [&str; 12] = [
    "ML_WORKER_FLOW_ENGINE_PATH",
    "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH",
    "ML_WORKER_FLOW_INFER_CONFIG",
    "ML_WORKER_FLOW_TRACKER_CONFIG",
    "ML_WORKER_FLOW_TRACKER_LIBRARY",
    "ML_WORKER_FLOW_ONNX_PATH",
    "ML_WORKER_FLOW_PARSER_LIBRARY",
    "ML_WORKER_FLOW_RECORD_DIR",
    "ML_WORKER_FLOW_RECORD_CACHE_SECONDS",
    "ML_WORKER_FLOW_FRAME_WIDTH",
    "ML_WORKER_FLOW_FRAME_HEIGHT",
    "ML_WORKER_FLOW_BATCH_SIZE",
];

const ENGINE_PATH: &str = "ML_WORKER_FLOW_ENGINE_PATH";
const ENGINE_IDENTITY_PATH: &str = "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH";
const ONNX_PATH: &str = "ML_WORKER_FLOW_ONNX_PATH";
const BATCH_SIZE: &str = "ML_WORKER_FLOW_BATCH_SIZE";

/// `sys.int_info.default_max_str_digits`: a longer string fails `int()`.
const MAX_STR_DIGITS: usize = 4300;

/// Which check refused the boot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlowBootKind {
    /// `flow profile wiring is missing: {keys}`; the subject lists the unset
    /// or empty keys joined by `, ` in `FLOW_BOOT_ENV` order.
    WiringMissing,
    /// `flow profile batch size must be a positive integer`.
    BatchNotPositive,
    /// A `verify_engine_identity` refusal; the subject is its subject.
    Identity(IdentityKind),
    /// `Flow engine batch {identity} does not match configured batch
    /// {configured}`; the subject is `"{identity} {configured}"`.
    BatchMismatch,
    /// `Flow ONNX has fixed batch dimension {dim}; rebuild ...`: the engine
    /// must be rebuilt from a dynamic-batch ONNX. The subject is the dim.
    FixedBatch,
    /// The `IndexError` `batch_axis_is_dynamic` raises on an input without
    /// dims (rank 0 or no declared shape).
    ShapeRankZero,
    /// An `input_dims` refusal, onnxruntime load errors included; the
    /// subject is the ONNX path.
    Shape(OnnxShapeKind),
}

/// A refusal and what the Python message names (empty when it names nothing).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlowBootError {
    pub kind: FlowBootKind,
    pub subject: String,
}

type Boot<T> = Result<T, FlowBootError>;

fn refuse<T>(kind: FlowBootKind, subject: &str) -> Boot<T> {
    Err(FlowBootError {
        kind,
        subject: subject.to_owned(),
    })
}

fn value<'a>(env: &'a Env, key: &str) -> &'a str {
    env.get(key).map_or("", String::as_str)
}

/// Python `int(text)` in base 10 followed by `> 0`: the canonical digits of
/// a positive value, or `None` for a `ValueError` or a value `<= 0`.
///
/// `int()` skips the whitespace around the digits that `char::is_whitespace`
/// names. It does not skip U+001C..U+001F, which `str.strip()` does.
/// Unicode digits (fullwidth, Arabic-Indic) are refused although Python
/// `int()` accepts them: a deliberate fail-closed deviation.
fn positive_int(text: &str) -> Option<String> {
    let trimmed = text.trim_matches(char::is_whitespace);
    let (negative, body) = match trimmed.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let mut digits = String::new();
    for group in body.split('_') {
        if group.is_empty() || !group.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        digits.push_str(group);
    }
    if digits.len() > MAX_STR_DIGITS {
        return None;
    }
    let canonical = digits.trim_start_matches('0');
    (!negative && !canonical.is_empty()).then(|| canonical.to_owned())
}

/// Python `str(dims[0])` for the fixed-batch message.
fn printed(dim: &Dim) -> String {
    match dim {
        Dim::Value(value) => value.to_string(),
        Dim::Param(name) => name.clone(),
        Dim::Unknown => "None".to_owned(),
    }
}

/// `verify_flow_boot_inputs(env, deployed_batch=...)`: the verified engine
/// identity map, or the first refusal in Python's order.
pub fn verify_flow_boot_inputs(
    env: &Env,
    deployed_batch: Option<i128>,
) -> Boot<BTreeMap<String, String>> {
    let missing: Vec<&str> = FLOW_BOOT_ENV
        .into_iter()
        .filter(|key| value(env, key).is_empty())
        .collect();
    if !missing.is_empty() {
        return refuse(FlowBootKind::WiringMissing, &missing.join(", "));
    }
    let Some(configured) = positive_int(value(env, BATCH_SIZE)) else {
        return refuse(FlowBootKind::BatchNotPositive, "");
    };
    let files: Vec<(&str, PathBuf)> = FLOW_IDENTITY_FILES
        .into_iter()
        .map(|(key, name)| (key, PathBuf::from(value(env, name))))
        .collect();
    let identity = verify_engine_identity(
        &PathBuf::from(value(env, ENGINE_PATH)),
        &PathBuf::from(value(env, ENGINE_IDENTITY_PATH)),
        &files,
        deployed_batch,
    )
    .map_err(|error| FlowBootError {
        kind: FlowBootKind::Identity(error.kind),
        subject: error.subject,
    })?;
    let built = identity.get("batch_size").map_or("", String::as_str);
    if built != configured {
        return refuse(
            FlowBootKind::BatchMismatch,
            &format!("{built} {configured}"),
        );
    }
    if configured != "1" {
        let dims =
            input_dims(&PathBuf::from(value(env, ONNX_PATH))).map_err(|error| FlowBootError {
                kind: FlowBootKind::Shape(error.kind),
                subject: error.subject,
            })?;
        match (batch_axis_is_dynamic(&dims), dims.first()) {
            (Some(true), _) => {}
            (Some(false), Some(first)) => return refuse(FlowBootKind::FixedBatch, &printed(first)),
            _ => return refuse(FlowBootKind::ShapeRankZero, ""),
        }
    }
    Ok(identity)
}
