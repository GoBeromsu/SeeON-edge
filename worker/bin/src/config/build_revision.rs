//! Build attribution is a caller declaration captured by Cargo, not runtime Git
//! discovery or a measurement of the running container. A trusted builder must
//! bind this declaration to its frozen source inputs.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use rustix::fs::{Mode, OFlags};
use rustix::io::Errno;

use super::env::Env;

const KEY: &str = "ML_WORKER_BUILD_REVISION";
const MARKER: &str = "/opt/seeon/ml-worker-image-revision";
const COMPILED: Option<&str> = option_env!("ML_WORKER_BUILD_REVISION");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BuildRevisionError {
    Invalid,
    Mismatch,
}

pub(crate) type ResolvedRevision = Result<Option<String>, BuildRevisionError>;

pub(crate) fn current(env: &Env) -> ResolvedRevision {
    // An environment/marker pair cannot attribute an undeclared binary.
    if COMPILED.is_none_or(|value| value.trim().is_empty()) {
        return Ok(None);
    }
    resolve(COMPILED, env, read_marker(Path::new(MARKER)))
}

pub(crate) fn resolve(
    compiled: Option<&str>,
    env: &Env,
    marker: ResolvedRevision,
) -> ResolvedRevision {
    let Some(compiled) = compiled.filter(|value| !value.trim().is_empty()) else {
        return Ok(None);
    };
    if !super::is_hex(compiled, 40) || compiled.bytes().all(|byte| byte == b'0') {
        return Err(BuildRevisionError::Invalid);
    }
    let runtime = env.get(KEY).map(String::as_str);
    if runtime.is_some_and(|value| value != compiled) {
        return Err(BuildRevisionError::Mismatch);
    }
    if let Some(marker) = marker?
        && (marker != compiled || runtime != Some(compiled))
    {
        return Err(BuildRevisionError::Mismatch);
    }
    Ok(Some(compiled.to_owned()))
}

fn read_marker(path: &Path) -> ResolvedRevision {
    let fd = match rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NONBLOCK | OFlags::NOFOLLOW,
        Mode::empty(),
    ) {
        Ok(fd) => fd,
        Err(Errno::NOENT) => return Ok(None),
        Err(_) => return Err(BuildRevisionError::Mismatch),
    };
    let file = File::from(fd);
    let metadata = file.metadata().map_err(|_| BuildRevisionError::Mismatch)?;
    if !metadata.is_file() || metadata.len() > 41 {
        return Err(BuildRevisionError::Mismatch);
    }
    let mut bytes = Vec::with_capacity(42);
    file.take(42)
        .read_to_end(&mut bytes)
        .map_err(|_| BuildRevisionError::Mismatch)?;
    if bytes.len() as u64 != metadata.len() {
        return Err(BuildRevisionError::Mismatch);
    }
    let mut value = String::from_utf8(bytes).map_err(|_| BuildRevisionError::Mismatch)?;
    if value.ends_with('\n') {
        value.pop();
    }
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::{IdSource, RandomIds};

    const REVISION: &str = "1111111111111111111111111111111111111111";
    const OTHER: &str = "2222222222222222222222222222222222222222";

    #[test]
    fn runtime_values_cannot_attribute_an_undeclared_binary() {
        let env = Env::from([(KEY.to_owned(), REVISION.to_owned())]);
        for compiled in [None, Some(""), Some("   ")] {
            assert_eq!(
                resolve(compiled, &env, Ok(Some(REVISION.to_owned()))),
                Ok(None)
            );
        }
    }

    #[test]
    fn compiled_declaration_is_checked_against_every_present_authority() {
        let mut env = Env::new();
        assert_eq!(
            resolve(Some(REVISION), &env, Ok(None)),
            Ok(Some(REVISION.to_owned()))
        );
        assert_eq!(
            resolve(Some(REVISION), &env, Ok(Some(REVISION.to_owned()))),
            Err(BuildRevisionError::Mismatch)
        );
        env.insert(KEY.to_owned(), REVISION.to_owned());
        assert_eq!(
            resolve(Some(REVISION), &env, Ok(Some(REVISION.to_owned()))),
            Ok(Some(REVISION.to_owned()))
        );
        for marker in [
            OTHER,
            "",
            " 1111111111111111111111111111111111111111",
            "1111111111111111111111111111111111111111\n",
        ] {
            assert_eq!(
                resolve(Some(REVISION), &env, Ok(Some(marker.to_owned()))),
                Err(BuildRevisionError::Mismatch)
            );
        }
        assert_eq!(
            resolve(Some(REVISION), &env, Err(BuildRevisionError::Mismatch)),
            Err(BuildRevisionError::Mismatch)
        );
        for runtime in [OTHER, "", "1111111111111111111111111111111111111111\n"] {
            env.insert(KEY.to_owned(), runtime.to_owned());
            assert_eq!(
                resolve(Some(REVISION), &env, Ok(None)),
                Err(BuildRevisionError::Mismatch)
            );
        }
    }

    #[test]
    fn malformed_compiled_declarations_are_not_repaired() {
        for revision in [
            "abc".to_owned(),
            "0".repeat(40),
            "A".repeat(40),
            "g".repeat(40),
            format!("{REVISION}\n"),
        ] {
            assert_eq!(
                resolve(Some(&revision), &Env::new(), Ok(None)),
                Err(BuildRevisionError::Invalid)
            );
        }
    }

    #[test]
    fn marker_capture_is_bounded_and_refuses_nonregular_aliases() {
        let root = std::env::temp_dir().join(format!(
            "seeon-revision-{}",
            RandomIds.uuid4().expect("test directory identity")
        ));
        std::fs::create_dir(&root).unwrap();
        let marker = root.join("revision");
        assert_eq!(read_marker(&marker), Ok(None));
        for contents in [REVISION.to_owned(), format!("{REVISION}\n")] {
            std::fs::write(&marker, contents).unwrap();
            assert_eq!(read_marker(&marker), Ok(Some(REVISION.to_owned())));
        }
        let alias = root.join("alias");
        std::os::unix::fs::symlink(&marker, &alias).unwrap();
        assert_eq!(read_marker(&alias), Err(BuildRevisionError::Mismatch));
        assert_eq!(read_marker(&root), Err(BuildRevisionError::Mismatch));
        std::fs::write(&marker, format!("{REVISION}\n\n")).unwrap();
        assert_eq!(read_marker(&marker), Err(BuildRevisionError::Mismatch));
        std::fs::write(&marker, [0xff; 40]).unwrap();
        assert_eq!(read_marker(&marker), Err(BuildRevisionError::Mismatch));
        std::fs::remove_dir_all(root).unwrap();
    }
}
