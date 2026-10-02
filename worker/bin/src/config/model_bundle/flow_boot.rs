//! Flow cold-start boot verification (`verify_flow_boot_inputs`,
//! `worker/runtime/flow/cold_start.py` L132-173): the flow profile wiring,
//! the configured batch, the engine identity and, above batch 1, a dynamic
//! ONNX batch axis.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::identity::{IdentityKind, verify_environment};
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

const ONNX_PATH: &str = "ML_WORKER_FLOW_ONNX_PATH";
const BATCH_SIZE: &str = "ML_WORKER_FLOW_BATCH_SIZE";
const RTSP_RECONNECT_INTERVAL_SEC: &str = "ML_WORKER_FLOW_RTSP_RECONNECT_INTERVAL_SEC";

/// Reconnect interval used when `RTSP_RECONNECT_INTERVAL_SEC` is unset.
const DEFAULT_RECONNECT_INTERVAL_SEC: u32 = 5;
/// Upper bound of the accepted reconnect interval, one day in seconds.
const MAX_RECONNECT_INTERVAL_SEC: u32 = 86_400;

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
    /// An aggregate identity refusal; the subject names the failed binding.
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
    /// `ML_WORKER_FLOW_RTSP_RECONNECT_INTERVAL_SEC must be a non-negative
    /// integer (0-86400, digits only)`; the subject is the raw value.
    ReconnectInterval,
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
pub(crate) fn positive_int(text: &str) -> Option<String> {
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

/// ADR: the nvurisrcbin RTSP reconnect interval contract
/// (`worker/runtime/worker.py:_flow_rtsp_reconnect_interval_sec`).
///
/// - Unset key: the default, 5 seconds.
/// - Set key: accepted iff the value is a non-empty string of ASCII digits
///   only (no sign, no whitespace, no `_`, no Unicode digits) whose integer
///   value is `<= 86400`; leading zeros are digits (`"007"` is 7).
/// - Everything else is refused, the empty string included: an empty value is
///   a set key with no usable content (a blank compose substitution), not a
///   missing key, so it must not silently take the default.
///
/// Python `int()` is deliberately not mirrored (it accepts `" 5"`, `"+5"`,
/// `"5_0"` and Unicode digits), and the check compares digit counts first so
/// a string beyond `int()`'s 4300-digit limit is refused by the bound rather
/// than by an exception. Boot wiring of this check belongs to the stage that
/// owns `verify_flow_boot_inputs` integration; it is not called from there.
pub fn rtsp_reconnect_interval_sec(env: &Env) -> Boot<u32> {
    let Some(raw) = env.get(RTSP_RECONNECT_INTERVAL_SEC) else {
        return Ok(DEFAULT_RECONNECT_INTERVAL_SEC);
    };
    let significant = raw.trim_start_matches('0');
    let ascii_digits = !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit());
    let within_width = significant.len() <= MAX_RECONNECT_INTERVAL_SEC.to_string().len();
    let interval = match (ascii_digits && within_width, significant) {
        (false, _) => None,
        (true, "") => Some(0),
        (true, digits) => digits.parse::<u32>().ok(),
    };
    match interval {
        Some(interval) if interval <= MAX_RECONNECT_INTERVAL_SEC => Ok(interval),
        _ => refuse(FlowBootKind::ReconnectInterval, raw),
    }
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
    verify_with(env, deployed_batch, || {
        let selection = crate::config::admit_selected_bundle(env).map_err(|_| FlowBootError {
            kind: FlowBootKind::Identity(IdentityKind::Schema),
            subject: "fall.bundle".to_owned(),
        })?;
        verify_environment(env, selection.as_ref(), None).map_err(|error| FlowBootError {
            kind: FlowBootKind::Identity(error.kind),
            subject: error.subject,
        })
    })
}

/// Runtime reuses the exact identity and selection admitted by check_config.
/// It must not reload a potentially different selection between these gates.
pub(crate) fn verify_admitted_flow_inputs(
    env: &Env,
    deployed_batch: Option<i128>,
    identity: BTreeMap<String, String>,
) -> Boot<BTreeMap<String, String>> {
    verify_with(env, deployed_batch, || Ok(identity))
}

fn verify_with(
    env: &Env,
    deployed_batch: Option<i128>,
    load: impl FnOnce() -> Boot<BTreeMap<String, String>>,
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
    let identity = load()?;
    let built = identity.get("batch_size").map_or("", String::as_str);
    if let Some(deployed) = deployed_batch {
        if deployed < 0 {
            return refuse(
                FlowBootKind::Identity(IdentityKind::NegativeDeployedBatch),
                "",
            );
        }
        if built
            .parse::<i128>()
            .ok()
            .is_none_or(|batch| batch < deployed)
        {
            return refuse(FlowBootKind::Identity(IdentityKind::BatchNotCovering), "");
        }
    }
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

#[cfg(test)]
#[path = "flow_boot/policy_tests.rs"]
mod policy_tests;
