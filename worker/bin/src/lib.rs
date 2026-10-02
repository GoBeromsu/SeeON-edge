//! SeeON edge ML worker process: one binary that owns media, GPU models,
//! policy, clips, delivery and telemetry threads over the Pass 1 crates.
// The shutdown module owns the reviewed OS signal-registration boundary.
// Every other module retains an unoverrideable unsafe-code prohibition.
#![deny(unsafe_code)]

#[forbid(unsafe_code)]
pub mod b64;
#[forbid(unsafe_code)]
pub mod cli;
#[forbid(unsafe_code)]
pub mod clips;
#[forbid(unsafe_code)]
pub mod config;
#[forbid(unsafe_code)]
pub mod delivery;
#[forbid(unsafe_code)]
pub mod engine_build;
#[forbid(unsafe_code)]
pub mod exit;
pub mod gpu;
#[forbid(unsafe_code)]
pub mod json;
#[forbid(unsafe_code)]
pub mod media;
#[forbid(unsafe_code)]
pub mod msg;
#[forbid(unsafe_code)]
pub mod policy;
#[forbid(unsafe_code)]
pub mod poll;
#[forbid(unsafe_code)]
pub mod records;
#[forbid(unsafe_code)]
pub mod relay;
#[forbid(unsafe_code)]
pub mod run;
#[forbid(unsafe_code)]
pub mod seam;
pub mod shutdown;
#[forbid(unsafe_code)]
pub mod telemetry;
#[forbid(unsafe_code)]
pub mod trace_out;
