//! Original CPU inference around the existing pure preprocessing and decoding.
//! Model admission and explicit provider selection remain composition duties.

pub use seeon_onnxruntime_native::{Info, Model, Threads};

pub mod bed;
pub mod fall;
pub mod stored_pose;
