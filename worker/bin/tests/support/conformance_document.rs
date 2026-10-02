use serde_json::{Value, json};

pub fn document() -> Value {
    json!({
        "preprocessing_identity": "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        "vector": {"length": 56, "tail_indices": {"x1":51,"y1":52,"x2":53,"y2":54,"valid":55}},
        "keypoint_order": ["nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist",
            "right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
        "confidence": {"gate":0.5},
        "temporal": {"window_frames":30,"stride_frames":5,"fps":15.0},
        "coordinate_system": {
            "origin":"top_left",
            "xy_normalization_denominators":{"x":"frame_width","y":"frame_height"},
            "xy_normalization_rule":"clip finite raw coordinates to inclusive raw bounds, then divide x by frame_width and y by frame_height"
        }
    })
}
