//! Replay trace output. `row` is the replay-trace-v2 row contract
//! (`contracts/replay_trace.py`), `replay` the opt-in rotating JSONL writer
//! (`worker/pipeline/trace/replay_trace_writer.py`) and `wire` the canonical
//! replay trace body (`shared/events/replay_wire.py`). Serving GET `/replay`
//! is Pass 3; this module only encodes and writes.

pub mod jsonl;
pub mod replay;
pub mod row;
pub mod vocab;
pub mod wire;

pub use replay::{DEFAULT_MAX_BYTES, DEFAULT_ROTATION_COUNT, ReplayTraceWriter, TraceError};
pub use row::{
    BedPolygon, HEADER_LINE, KEYPOINTS, REPLAY_TRACE_VERSION, ReplayRow, ReplayTrack, RowError,
};
pub use vocab::{Lifecycle, Source, SourceEvent};
pub use wire::{FrameKey, ReplayWire, Truncation, WireError};
