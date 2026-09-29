//! SIGTERM and SIGINT set one shared flag (design §2.4); the shutdown
//! sequence polls it.

use std::io;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::flag;

/// Registers both signals on a fresh flag and returns it.
pub fn register() -> io::Result<Arc<AtomicBool>> {
    let requested = Arc::new(AtomicBool::new(false));
    flag::register(SIGTERM, Arc::clone(&requested))?;
    flag::register(SIGINT, Arc::clone(&requested))?;
    Ok(requested)
}
