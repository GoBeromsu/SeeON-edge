//! SeeON edge ML worker process: one binary that owns media, GPU models,
//! policy, clips, delivery and telemetry threads over the Pass 1 crates.
#![forbid(unsafe_code)]

pub mod b64;
pub mod exit;
pub mod json;
pub mod msg;
pub mod poll;
pub mod seam;
pub mod shutdown;
