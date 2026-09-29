//! Time and id seams (design §7.1). Production passes `SystemClock` and
//! `RandomIds`; tests pass fixed ones. Frozen at the end of stage 0.

use std::io;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use rustix::rand::{GetRandomFlags, getrandom};

pub trait Clock: Send + Sync {
    /// Time since an arbitrary fixed origin; never goes backwards.
    fn monotonic(&self) -> Duration;
    fn wall(&self) -> SystemTime;
    /// Lets up to `limit` of monotonic time pass. Callers re-check their
    /// condition afterwards; a pause never decides an outcome (D6).
    fn pause(&self, limit: Duration);
}

pub trait IdSource: Send + Sync {
    /// A random (version 4) UUID in lowercase hyphenated form, as Python
    /// `str(uuid.uuid4())` writes it.
    fn uuid4(&self) -> io::Result<String>;
}

pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn monotonic(&self) -> Duration {
        self.origin.elapsed()
    }

    fn wall(&self) -> SystemTime {
        SystemTime::now()
    }

    fn pause(&self, limit: Duration) {
        thread::sleep(limit);
    }
}

/// Kernel `getrandom` bytes, like Python `os.urandom`.
#[derive(Default)]
pub struct RandomIds;

impl IdSource for RandomIds {
    fn uuid4(&self) -> io::Result<String> {
        let mut bytes = [0u8; 16];
        let mut filled = 0;
        while filled < bytes.len() {
            match getrandom(&mut bytes[filled..], GetRandomFlags::empty()) {
                Ok(count) => filled += count,
                Err(rustix::io::Errno::INTR) => {}
                Err(error) => return Err(error.into()),
            }
        }
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        let mut text = String::with_capacity(36);
        for (index, byte) in bytes.iter().enumerate() {
            if matches!(index, 4 | 6 | 8 | 10) {
                text.push('-');
            }
            text.push(char::from(HEX[usize::from(byte >> 4)]));
            text.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        Ok(text)
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";
