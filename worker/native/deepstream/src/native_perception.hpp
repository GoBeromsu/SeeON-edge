#pragma once

// Native ports of the C3 parity references (worker/native/deepstream/parity/
// geometry.py, parse.py) and the C4 active association strategy
// (worker/native/deepstream/association/legacy_greedy_iou.py). Behavior is
// pinned by the cross-language parity driver (native_perception_driver.cpp)
// against the Python references; any divergence is a parity break, not an
// improvement.

#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace seeon::perception {

// -- geometry ---------------------------------------------------------------
// Mirrors parity/geometry.py: the shipped ultralytics LetterBox(auto=True,
// stride=32, center=True) forward mapping and its TWO inverse rules. Boxes use
// the rounded pad (ops.scale_boxes); keypoints/mask contours use the unrounded
// pad (ops.scale_coords). Both are load-bearing.
struct AffineMetadata {
  int source_height;
  int source_width;
  int tensor_height;
  int tensor_width;
  int content_height;
  int content_width;
  double gain;
  int box_pad_x;
  int box_pad_y;
  double keypoint_pad_x;
  double keypoint_pad_y;

  struct Box {
    int x1;
    int y1;
    int x2;
    int y2;
  };
  [[nodiscard]] Box invert_box(double x1, double y1, double x2, double y2) const;
  [[nodiscard]] std::pair<int, int> invert_keypoint(double x, double y) const;
  [[nodiscard]] int pad_top() const { return static_cast<int>(keypoint_pad_y); }
  [[nodiscard]] int pad_bottom() const {
    return tensor_height - content_height - pad_top();
  }
  [[nodiscard]] int pad_left() const { return static_cast<int>(keypoint_pad_x); }
  [[nodiscard]] int pad_right() const {
    return tensor_width - content_width - pad_left();
  }
};

inline constexpr int kLetterboxSize = 640;
inline constexpr int kLetterboxStride = 32;
inline constexpr int kLetterboxPadValue = 114;

// Throws std::invalid_argument on non-positive geometry.
[[nodiscard]] AffineMetadata letterbox_affine(int source_height, int source_width,
                                              int size = kLetterboxSize,
                                              int stride = kLetterboxStride);

// -- parsers ----------------------------------------------------------------
// Row strides fixed by the export profile (parity/parse.py).
inline constexpr int kPoseRowStride = 57;
inline constexpr int kPersonRowStride = 6;
inline constexpr int kBedRowStride = 38;
inline constexpr int kMaxRows = 300;
inline constexpr double kPoseScoreThreshold = 0.05;  // strict >
inline constexpr int kCocoKeypointCount = 17;
inline constexpr int kCocoPersonClassId = 0;
inline constexpr int kCocoBedClassId = 59;
inline constexpr int kBedMaskMaxPoints = 48;
inline constexpr int kBedPrototypeChannels = 32;
inline constexpr int kBedPrototypeHeight = 160;
inline constexpr int kBedPrototypeWidth = 160;

struct ParsedBox {
  int x1;
  int y1;
  int x2;
  int y2;
  double confidence;
};

struct ParsedKeypoint {
  int x;
  int y;
  double score;
};

struct ParsedPose {
  std::vector<ParsedBox> boxes;
  std::vector<std::vector<ParsedKeypoint>> poses;
};

struct ParsedBedRegion {
  ParsedBox bounds;
  std::vector<std::pair<int, int>> polygon;  // empty == no polygon
};

// rows: row-major [count, stride] planes straight off the engine.
// Source order preserved; strict score > 0.05; no second NMS (parse.py).
[[nodiscard]] ParsedPose parse_pose_rows(const std::vector<double>& rows,
                                         const AffineMetadata& affine);
[[nodiscard]] std::vector<ParsedBox> parse_person_rows(const std::vector<double>& rows,
                                                       const AffineMetadata& affine,
                                                       double confidence);
// prototypes: row-major [32, 160, 160].
[[nodiscard]] std::vector<ParsedBedRegion> parse_bed_rows(
    const std::vector<double>& rows, const std::vector<double>& prototypes,
    const AffineMetadata& affine, double confidence,
    int max_points = kBedMaskMaxPoints);

// -- association ------------------------------------------------------------
// Port of association/legacy_greedy_iou.py: greedy descending-IoU match,
// stable tie order (lower existing-track index wins), empty observation counts
// a miss, coast() never does, eviction on misses > max_misses, durable ids
// minted from zero per reset().
inline constexpr double kDefaultMinIou = 0.3;
inline constexpr int kDefaultMaxMisses = 30;

struct AssociationOutput {
  std::vector<std::int64_t> track_ids;        // parallel to cue indexes 0..n-1
  std::vector<std::uint16_t> selected_cue_indexes;
};

class LegacyGreedyBboxIou {
 public:
  explicit LegacyGreedyBboxIou(double min_iou = kDefaultMinIou,
                               int max_misses = kDefaultMaxMisses)
      : min_iou_(min_iou), max_misses_(max_misses) {}

  [[nodiscard]] AssociationOutput observe(const std::vector<ParsedBox>& boxes);
  void coast() {}
  void reset();
  [[nodiscard]] std::set<std::int64_t> live_ids() const;

 private:
  struct Track {
    std::int64_t track_id;
    ParsedBox last_box;
    int misses;
  };
  double min_iou_;
  int max_misses_;
  std::vector<Track> tracks_;
  std::int64_t next_id_ = 0;
};

}  // namespace seeon::perception
