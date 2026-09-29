//! Numeric worker primitives and lifecycle policy; no inference or runtime composition.
//! Detection-window construction reads an explicitly supplied zoneinfo directory.
//! Source admission is a single-handle primitive; it does not capture, publish, or record.
#![forbid(unsafe_code)]

pub mod bed_contour;
pub mod bed_exit;
pub mod bed_input;
pub mod bed_polygon;
pub mod bed_sigmoid;
pub mod detection_window;
pub mod episode;
pub mod fall;
pub mod pose_bbox56;
pub mod source_admission;
pub mod stored_pose;
pub mod temporal;
pub mod trace;
