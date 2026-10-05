//! Concrete worker composition over an explicitly supplied native model.
//!
//! Not a process runtime, CLI, engine authenticator, or numeric-admission gate.
#![forbid(unsafe_code)]
pub mod bed_gpu;
pub mod cpu;
pub mod evidence;
pub mod fall_gpu;
pub mod stored_pose;
