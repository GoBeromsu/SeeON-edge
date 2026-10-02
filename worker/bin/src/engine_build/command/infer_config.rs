//! Serving text names an already-built engine; ONNX belongs only to the
//! separately rendered offline child configuration.
use super::{CommandError, files};
use crate::config::model_bundle::identity::engine_only_config;
use std::path::Path;

pub(super) fn render_served(
    template: &str,
    engine: &str,
    batch: u32,
) -> Result<String, CommandError> {
    refuse(engine)?;
    let mut property = false;
    let mut sections = 0;
    let mut engines = 0;
    let mut batches = 0;
    let mut onnx = 0;
    let mut rendered = String::new();
    for line in template.split_inclusive('\n') {
        let bare = line.trim_end_matches(['\n', '\r']);
        let trimmed = bare.trim_ascii();
        if let Some(section) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            property = section.trim_ascii() == "property";
            sections += usize::from(property);
            rendered.push_str(line);
            continue;
        }
        let replacement = if property {
            match bare
                .split_once('=')
                .map(|(key, value)| (key.trim_ascii(), value))
            {
                Some(("onnx-file", value)) => {
                    refuse(value.trim_ascii())?;
                    onnx += 1;
                    continue;
                }
                Some(("model-engine-file", value)) => {
                    refuse(value.trim_ascii())?;
                    engines += 1;
                    Some(format!("model-engine-file={engine}"))
                }
                Some(("batch-size", value)) => {
                    refuse(value.trim_ascii())?;
                    batches += 1;
                    Some(format!("batch-size={batch}"))
                }
                _ => None,
            }
        } else {
            None
        };
        if let Some(value) = replacement {
            rendered.push_str(&value);
            if line.ends_with("\r\n") {
                rendered.push_str("\r\n");
            } else if line.ends_with('\n') {
                rendered.push('\n');
            }
        } else {
            rendered.push_str(line);
        }
    }
    if sections != 1 || engines != 1 || batches != 1 || onnx > 1 || !engine_only_config(&rendered) {
        return Err(CommandError::Config);
    }
    Ok(rendered)
}

fn refuse(value: &str) -> Result<(), CommandError> {
    if value.is_empty() || value.bytes().any(|byte| matches!(byte, b'\n' | b'\r' | 0)) {
        return Err(CommandError::Config);
    }
    Ok(())
}

pub(super) fn verify_parser(text: &str, parser: &Path) -> Result<(), CommandError> {
    let expected = files::absolute_text(parser)?;
    let mut found = None;
    for line in text.lines() {
        let Some(value) = line.strip_prefix("custom-lib-path=") else {
            continue;
        };
        refuse(value)?;
        if found.replace(value).is_some() {
            return Err(CommandError::Parser);
        }
    }
    (files::absolute_text(Path::new(found.ok_or(CommandError::Parser)?))? == expected)
        .then_some(())
        .ok_or(CommandError::Parser)
}
