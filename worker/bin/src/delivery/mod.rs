//! Durable delivery state shared with the Python producer: the on-disk
//! queue of accepted entries and its dead-letter sibling directory.

pub mod queue;
pub mod sender;
pub mod snapshot;

pub use queue::*;
