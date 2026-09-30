//! SeeON edge ML worker process: one binary that owns media, GPU models,
//! policy, clips, delivery and telemetry threads over the Pass 1 crates.
#![forbid(unsafe_code)]

pub mod b64;
pub mod cli;
pub mod config;
pub mod delivery;
pub mod exit;
pub mod gpu;
pub mod json;
pub mod media;
pub mod msg;
pub mod policy;
pub mod poll;
pub mod records;
pub mod relay;
pub mod seam;
pub mod shutdown;
pub mod trace_out;
