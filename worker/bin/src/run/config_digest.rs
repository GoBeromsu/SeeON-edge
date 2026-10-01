//! ADR0009: bind execution records to canonical, secret-redacted effective config.

use crate::json::{Json, JsonError, Serialiser};
use crate::records::id::sha256_hex;

const SECRET_FIELDS: [&str; 8] = [
    "token",
    "relay_token",
    "access_token",
    "refresh_token",
    "password",
    "client_secret",
    "authorization",
    "rtsp_url",
];

/// The caller supplies resolved runtime configuration, including applied defaults.
/// Never pass the process environment or an unvalidated relay payload here.
pub fn config_digest(effective_configuration: &Json) -> Result<String, JsonError> {
    let redacted = redact(effective_configuration);
    // This existing mode is strict UTF-8 canonical JSON, matching the ADR recipe.
    let canonical = Serialiser::FetchModelsManifest.canonical(&redacted)?;
    Ok(sha256_hex(canonical.as_bytes()))
}

fn redact(value: &Json) -> Json {
    match value {
        Json::Object(members) => Json::Object(
            members
                .iter()
                .map(|(key, value)| {
                    let value = if SECRET_FIELDS
                        .iter()
                        .any(|field| key.eq_ignore_ascii_case(field))
                    {
                        Json::Str("[redacted]".to_owned())
                    } else {
                        redact(value)
                    };
                    (key.clone(), value)
                })
                .collect(),
        ),
        Json::Array(values) => Json::Array(values.iter().map(redact).collect()),
        value => value.clone(),
    }
}
