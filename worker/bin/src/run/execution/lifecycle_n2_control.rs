//! Private binary controls and nonces. No production registration.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path};
use std::time::{Duration, Instant};

use crate::seam::{IdSource, RandomIds};
use rustix::fs::{self, FileType, Mode, OFlags, RenameFlags};

pub(super) struct Directory(File);

impl Directory {
    pub(super) fn open(path: &Path, fresh: bool) -> Self {
        assert!(
            path.is_absolute() && path.parent().is_some(),
            "absolute owned directory required"
        );
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let mut fd = fs::open("/", flags, Mode::empty()).expect("root directory");
        for component in path.components() {
            match component {
                Component::RootDir => {}
                Component::Normal(name) => {
                    fd = fs::openat(&fd, name, flags, Mode::empty()).expect("no-follow directory");
                }
                _ => panic!("directory path must not contain traversal"),
            }
        }
        let stat = fs::fstat(&fd).expect("directory stat");
        assert_eq!(
            stat.st_uid,
            rustix::process::geteuid().as_raw(),
            "directory owner"
        );
        assert_eq!(
            stat.st_mode & 0o077,
            0,
            "directory must be exclusively owned"
        );
        if fresh {
            for entry in fs::Dir::read_from(&fd).expect("owned directory entries") {
                let entry = entry.expect("directory entry");
                assert!(
                    matches!(entry.file_name().to_bytes(), b"." | b".."),
                    "fresh empty directory required"
                );
            }
        }
        Self(File::from(fd))
    }

    pub(super) fn copy(&self) -> Self {
        Self(self.0.try_clone().expect("retain directory descriptor"))
    }

    pub(super) fn identity(&self) -> [u64; 2] {
        let stat = fs::fstat(&self.0).expect("directory identity");
        [stat.st_dev, stat.st_ino]
    }

    pub(super) fn regular(&self, name: &str) -> Option<File> {
        assert!(
            !name.contains('/') && name.starts_with('.'),
            "fixed basename required"
        );
        let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
        let fd = match fs::openat(&self.0, name, flags, Mode::empty()) {
            Ok(fd) => fd,
            Err(rustix::io::Errno::NOENT) => return None,
            Err(error) => panic!("control open {name}: {error}"),
        };
        let stat = fs::fstat(&fd).expect("regular file stat");
        assert_eq!(FileType::from_raw_mode(stat.st_mode), FileType::RegularFile);
        assert_eq!(stat.st_nlink, 1, "immutable singly linked evidence");
        Some(File::from(fd))
    }

    pub(super) fn lease_identity(&self) -> [u64; 2] {
        let file = self.regular(".gpu.lease").expect("actual boot lease inode");
        let stat = fs::fstat(&file).expect("lease identity");
        [stat.st_dev, stat.st_ino]
    }

    pub(super) fn read<const N: usize>(&self, stem: &str, magic: &[u8; 8]) -> Option<[u64; N]> {
        assert!(N <= 80, "bounded binary control");
        let file = self.regular(&format!("{stem}.bin"))?;
        let length = 8 + N * 8;
        assert_eq!(
            fs::fstat(&file).expect("control size").st_size,
            length as i64
        );
        let mut bytes = Vec::with_capacity(length + 1);
        file.take((length + 1) as u64)
            .read_to_end(&mut bytes)
            .expect("bounded control read");
        assert_eq!(bytes.len(), length, "exact binary control size");
        assert_eq!(&bytes[..8], magic, "binary control magic");
        Some(std::array::from_fn(|index| {
            u64::from_le_bytes(bytes[8 + index * 8..16 + index * 8].try_into().unwrap())
        }))
    }

    pub(super) fn publish(&self, stem: &str, magic: &[u8; 8], fields: &[u64]) {
        assert!(fields.len() <= 80 && !stem.contains('/') && stem.starts_with(".n2-"));
        let temporary = format!("{stem}.tmp");
        let flags = OFlags::WRONLY
            | OFlags::CREATE
            | OFlags::EXCL
            | OFlags::NOFOLLOW
            | OFlags::NONBLOCK
            | OFlags::CLOEXEC;
        let fd = fs::openat(&self.0, &temporary, flags, Mode::RUSR | Mode::WUSR)
            .expect("exclusive control temporary");
        let mut file = File::from(fd);
        file.write_all(magic).expect("control magic write");
        for field in fields {
            file.write_all(&field.to_le_bytes())
                .expect("control field write");
        }
        file.sync_all().expect("complete control write");
        fs::renameat_with(
            &self.0,
            &temporary,
            &self.0,
            format!("{stem}.bin"),
            RenameFlags::NOREPLACE,
        )
        .expect("immutable atomic control publication");
    }

    pub(super) fn wait<const N: usize>(
        &self,
        stem: &str,
        magic: &[u8; 8],
        budget: Duration,
    ) -> [u64; N] {
        let limit = Instant::now() + budget;
        loop {
            if let Some(fields) = self.read(stem, magic) {
                return fields;
            }
            assert!(
                Instant::now() < limit,
                "missing native/parent control: {stem}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

pub(super) fn nonce() -> [u64; 2] {
    let text = RandomIds
        .uuid4()
        .expect("fresh kernel-random nonce")
        .replace('-', "");
    let value = u128::from_str_radix(&text, 16).expect("nonce encoding");
    let nonce = [value as u64, (value >> 64) as u64];
    assert_ne!(nonce, [0, 0]);
    nonce
}

pub(super) fn ns(at: Duration) -> u64 {
    u64::try_from(at.as_nanos()).expect("representable real monotonic time")
}
