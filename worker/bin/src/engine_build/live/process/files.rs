//! Exclusive scratch staging and no-follow engine copy. Paths are absolute
//! before any private directory is created. A refused source is never followed
//! and an existing target is never opened.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use rustix::fs::{self, FileType, Mode, OFlags};

use crate::seam::IdSource;

use super::super::super::MAX_ENGINE_BYTES;
use super::super::{LiveBuildError, Prepared};
use super::{Scratch, Staged};

const FILE_MODE: Mode = Mode::from_bits_truncate(0o600);
const PRIVATE_DIR: Mode = Mode::RWXU;

pub(in crate::engine_build::live::process) fn create_scratch(
    parent: &Path,
) -> Result<Scratch, LiveBuildError> {
    let root = absolute_parent(parent)?;
    let name = crate::seam::RandomIds
        .uuid4()
        .map_err(|_| LiveBuildError::Process)?;
    let directory = root.join(format!(".live-pose-{name}"));
    fs::mkdir(&directory, PRIVATE_DIR).map_err(|_| LiveBuildError::Process)?;
    Ok(Scratch { directory })
}

fn absolute_parent(parent: &Path) -> Result<PathBuf, LiveBuildError> {
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    parent.canonicalize().map_err(|_| LiveBuildError::Process)
}

pub(in crate::engine_build::live::process) fn record(
    scratch: &Scratch,
    bytes: &[u8],
) -> Result<(), LiveBuildError> {
    let path = scratch.directory.join("build.log");
    write_exclusive(&path, bytes)
}

pub(in crate::engine_build::live::process) fn stage_files(
    scratch: &Scratch,
    prepared: &Prepared<'_>,
) -> Result<Staged, LiveBuildError> {
    let onnx = scratch.directory.join("model.onnx");
    write_exclusive(&onnx, prepared.request.onnx)?;
    let child_engine = scratch.directory.join("child.engine");
    let config = scratch.directory.join("nvinfer-build.txt");
    let text = render_config(
        prepared.request.infer_config,
        &text_of(&onnx)?,
        &text_of(&child_engine)?,
        prepared.request.batch_size,
    )?;
    write_exclusive(&config, text.as_bytes())?;
    let batch = prepared.request.batch_size;
    Ok(Staged {
        argv: super::gst_argv(&text_of(&config)?, batch),
        child_engine,
        generated: onnx.with_file_name(format!("model.onnx_b{batch}_gpu0_fp16.engine")),
    })
}

pub(in crate::engine_build::live::process) fn render_config(
    template: &str,
    onnx: &str,
    engine: &str,
    batch: u32,
) -> Result<String, LiveBuildError> {
    refuse_value(onnx)?;
    refuse_value(engine)?;
    let mut engine_count = 0_u8;
    let mut batch_count = 0_u8;
    let mut rendered = String::new();
    for line in template.split_inclusive('\n') {
        let bare = line.trim_end_matches(['\n', '\r']);
        let Some(value) = replaced_value(bare, engine, batch)? else {
            rendered.push_str(line);
            continue;
        };
        if value.starts_with("model-engine-file=") {
            engine_count = engine_count.checked_add(1).ok_or(LiveBuildError::Config)?;
        } else if value.starts_with("batch-size=") {
            batch_count = batch_count.checked_add(1).ok_or(LiveBuildError::Config)?;
        }
        rendered.push_str(&value);
        if line.ends_with('\n') {
            rendered.push('\n');
        }
    }
    if engine_count != 1 || batch_count != 1 {
        return Err(LiveBuildError::Config);
    }
    insert_captured_onnx(&rendered, onnx)
}

fn replaced_value(line: &str, engine: &str, batch: u32) -> Result<Option<String>, LiveBuildError> {
    if let Some(value) = line.strip_prefix("model-engine-file=") {
        refuse_value(value)?;
        return Ok(Some(format!("model-engine-file={engine}")));
    }
    if let Some(value) = line.strip_prefix("batch-size=") {
        refuse_value(value)?;
        return Ok(Some(format!("batch-size={batch}")));
    }
    Ok(None)
}

fn insert_captured_onnx(template: &str, onnx: &str) -> Result<String, LiveBuildError> {
    let mut property = false;
    let mut seen_property = false;
    let mut onnx_at = None;
    let mut insert_at = None;
    for (index, line) in template.split_inclusive('\n').enumerate() {
        let bare = line.trim_end_matches(['\n', '\r']);
        if let Some(name) = property_section(bare) {
            if property && onnx_at.is_none() {
                insert_at = Some(index);
            }
            if name == "property" {
                if seen_property {
                    return Err(LiveBuildError::Config);
                }
                seen_property = true;
            }
            property = name == "property";
            continue;
        }
        let Some((key, value)) = bare.split_once('=') else {
            continue;
        };
        if !property || key.trim_matches(child_space) != "onnx-file" {
            continue;
        }
        refuse_value(value.trim_matches(child_space))?;
        if onnx_at.replace(index).is_some() {
            return Err(LiveBuildError::Config);
        }
    }
    if property && onnx_at.is_none() {
        insert_at = Some(template.split_inclusive('\n').count());
    }
    if !seen_property || (onnx_at.is_some() && insert_at.is_some()) {
        return Err(LiveBuildError::Config);
    }
    let serving: String = template
        .split_inclusive('\n')
        .enumerate()
        .filter_map(|(index, line)| (onnx_at != Some(index)).then_some(line))
        .collect();
    if !crate::config::model_bundle::identity::engine_only_config(&serving) {
        return Err(LiveBuildError::Config);
    }
    let replacement = format!("onnx-file={onnx}");
    let mut rendered = String::new();
    for (index, line) in template.split_inclusive('\n').enumerate() {
        if onnx_at == Some(index) {
            rendered.push_str(&replacement);
            if line.ends_with('\n') {
                rendered.push('\n');
            }
            continue;
        }
        if insert_at == Some(index) {
            rendered.push_str(&replacement);
            rendered.push('\n');
        }
        rendered.push_str(line);
    }
    if insert_at == Some(template.split_inclusive('\n').count()) {
        if !rendered.is_empty() && !rendered.ends_with('\n') {
            rendered.push('\n');
        }
        rendered.push_str(&replacement);
        rendered.push('\n');
    }
    Ok(rendered)
}

fn property_section(line: &str) -> Option<&str> {
    let body = line
        .trim_matches(child_space)
        .strip_prefix('[')?
        .strip_suffix(']')?;
    let name = body.trim_matches(child_space);
    (!name.is_empty()
        && !name.contains('[')
        && !name.contains(']')
        && name.bytes().all(|byte| byte.is_ascii_graphic()))
    .then_some(name)
}

fn child_space(byte: char) -> bool {
    matches!(byte, ' ' | '\t' | '\u{000B}' | '\u{000C}')
}
fn refuse_value(value: &str) -> Result<(), LiveBuildError> {
    if value.is_empty() || value.bytes().any(|byte| matches!(byte, b'\n' | b'\r' | 0)) {
        return Err(LiveBuildError::Config);
    }
    Ok(())
}

fn text_of(path: &Path) -> Result<String, LiveBuildError> {
    path.to_str()
        .filter(|text| !text.contains('\0'))
        .map(str::to_owned)
        .ok_or(LiveBuildError::Path)
}

fn write_exclusive(path: &Path, bytes: &[u8]) -> Result<(), LiveBuildError> {
    let flags = OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL | OFlags::WRONLY | OFlags::NOFOLLOW;
    let mut file =
        File::from(fs::open(path, flags, FILE_MODE).map_err(|_| LiveBuildError::Process)?);
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|_| LiveBuildError::Process)?;
    sync_parent(path)
}

fn sync_parent(path: &Path) -> Result<(), LiveBuildError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW;
    let directory = fs::open(parent, flags, Mode::empty()).map_err(|_| LiveBuildError::Process)?;
    fs::fsync(directory).map_err(|_| LiveBuildError::Process)
}

pub(in crate::engine_build::live::process) fn regular_engine(path: &Path) -> bool {
    fs::lstat(path).is_ok_and(|info| {
        FileType::from_raw_mode(info.st_mode).is_file()
            && (1..=i64::try_from(MAX_ENGINE_BYTES).unwrap_or(i64::MAX)).contains(&info.st_size)
    })
}

pub(in crate::engine_build::live::process) fn copy_exclusive(
    source: &Path,
    target: &Path,
) -> Result<(), LiveBuildError> {
    let input = open_source(source)?;
    let declared = bounded_length(&fs::fstat(&input).map_err(|_| LiveBuildError::Output)?)?;
    if fs::lstat(target).is_ok() {
        return Err(LiveBuildError::Output);
    }
    let flags = OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL | OFlags::WRONLY | OFlags::NOFOLLOW;
    let mut output =
        File::from(fs::open(target, flags, FILE_MODE).map_err(|_| LiveBuildError::Output)?);
    if let Err(error) = copy_declared(&input, &mut output, declared) {
        drop(output);
        return Err(error);
    }
    output.sync_all().map_err(|_| LiveBuildError::Output)?;
    sync_parent(target).map_err(|_| LiveBuildError::Output)
}

fn open_source(source: &Path) -> Result<File, LiveBuildError> {
    let flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    Ok(File::from(
        fs::open(source, flags, Mode::empty()).map_err(|_| LiveBuildError::Output)?,
    ))
}

fn bounded_length(info: &fs::Stat) -> Result<u64, LiveBuildError> {
    let declared = u64::try_from(info.st_size).map_err(|_| LiveBuildError::Output)?;
    if !FileType::from_raw_mode(info.st_mode).is_file()
        || !(1..=MAX_ENGINE_BYTES).contains(&declared)
    {
        return Err(LiveBuildError::Output);
    }
    Ok(declared)
}

fn copy_declared(input: &File, output: &mut File, declared: u64) -> Result<(), LiveBuildError> {
    let mut reader = input
        .try_clone()
        .map_err(|_| LiveBuildError::Output)?
        .take(declared + 1);
    let mut copied = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|_| LiveBuildError::Output)?;
        if count == 0 {
            break;
        }
        copied = copied
            .checked_add(u64::try_from(count).map_err(|_| LiveBuildError::Output)?)
            .ok_or(LiveBuildError::Output)?;
        if copied > declared {
            return Err(LiveBuildError::Output);
        }
        output
            .write_all(&buffer[..count])
            .map_err(|_| LiveBuildError::Output)?;
    }
    (copied == declared)
        .then_some(())
        .ok_or(LiveBuildError::Output)
}

#[cfg(test)]
mod tests {
    use rustix::fs::{MemfdFlags, memfd_create};
    use std::fs::File;
    use std::io::{Seek, Write};

    fn anonymous() -> File {
        File::from(memfd_create(c"live-engine-copy-test", MemfdFlags::CLOEXEC).unwrap())
    }

    #[test]
    fn truncation_after_descriptor_measurement_refuses_partial_copy() {
        let mut input = anonymous();
        input.write_all(b"engine").unwrap();
        let declared = input.metadata().unwrap().len();
        input.set_len(3).unwrap();
        input.rewind().unwrap();
        let mut output = anonymous();
        assert!(super::copy_declared(&input, &mut output, declared).is_err());
        assert_eq!(output.metadata().unwrap().len(), 3);
    }

    #[test]
    fn growth_after_descriptor_measurement_never_writes_past_declared_size() {
        let mut input = anonymous();
        input.write_all(b"eng").unwrap();
        let declared = input.metadata().unwrap().len();
        input.write_all(b"ine").unwrap();
        input.rewind().unwrap();
        let mut output = anonymous();
        assert!(super::copy_declared(&input, &mut output, declared).is_err());
        assert_eq!(output.metadata().unwrap().len(), 0);
    }
}
