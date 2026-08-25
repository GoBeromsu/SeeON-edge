#include "native_perception.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace seeon::perception {
namespace {

double clamp_to(double value, int upper) {
  if (value < 0.0) return 0.0;
  const auto bound = static_cast<double>(upper);
  return value > bound ? bound : value;
}

// Python round() is banker's rounding (round-half-even); std::round is
// half-away-from-zero. letterbox_affine and box_pad use Python round(), so the
// same rule is reproduced here for exact parity on *.5 boundaries.
double python_round(double value) {
  const double floor_value = std::floor(value);
  const double difference = value - floor_value;
  if (difference > 0.5) return floor_value + 1.0;
  if (difference < 0.5) return floor_value;
  // exact .5: to even
  return std::fmod(floor_value, 2.0) == 0.0 ? floor_value : floor_value + 1.0;
}

void validate_rows(const std::vector<double>& rows, int stride, const char* task) {
  if (stride <= 0 || rows.size() % static_cast<std::size_t>(stride) != 0) {
    throw std::invalid_argument{std::string{task} + " tensor rows have a partial row"};
  }
  const auto count = rows.size() / static_cast<std::size_t>(stride);
  if (count > static_cast<std::size_t>(kMaxRows)) {
    throw std::invalid_argument{std::string{task} + " tensor exceeds the export row profile"};
  }
}

}  // namespace

AffineMetadata::Box AffineMetadata::invert_box(double x1, double y1, double x2,
                                               double y2) const {
  const double sx1 = (x1 - box_pad_x) / gain;
  const double sy1 = (y1 - box_pad_y) / gain;
  const double sx2 = (x2 - box_pad_x) / gain;
  const double sy2 = (y2 - box_pad_y) / gain;
  return Box{
      static_cast<int>(clamp_to(sx1, source_width)),
      static_cast<int>(clamp_to(sy1, source_height)),
      static_cast<int>(clamp_to(sx2, source_width)),
      static_cast<int>(clamp_to(sy2, source_height)),
  };
}

std::pair<int, int> AffineMetadata::invert_keypoint(double x, double y) const {
  const double sx = (x - keypoint_pad_x) / gain;
  const double sy = (y - keypoint_pad_y) / gain;
  return {static_cast<int>(clamp_to(sx, source_width)),
          static_cast<int>(clamp_to(sy, source_height))};
}

AffineMetadata letterbox_affine(int source_height, int source_width, int size,
                                int stride) {
  if (source_height <= 0 || source_width <= 0) {
    throw std::invalid_argument{"source geometry must be positive"};
  }
  const double gain = std::min(static_cast<double>(size) / source_height,
                               static_cast<double>(size) / source_width);
  const int content_width = static_cast<int>(python_round(source_width * gain));
  const int content_height = static_cast<int>(python_round(source_height * gain));
  const int pad_width = ((size - content_width) % stride + stride) % stride;
  const int pad_height = ((size - content_height) % stride + stride) % stride;
  return AffineMetadata{
      source_height,
      source_width,
      content_height + pad_height,
      content_width + pad_width,
      content_height,
      content_width,
      gain,
      static_cast<int>(python_round(pad_width / 2.0 - 0.1)),
      static_cast<int>(python_round(pad_height / 2.0 - 0.1)),
      pad_width / 2.0,
      pad_height / 2.0,
  };
}

ParsedPose parse_pose_rows(const std::vector<double>& rows, const AffineMetadata& affine) {
  validate_rows(rows, kPoseRowStride, "pose");
  ParsedPose parsed;
  for (std::size_t offset = 0; offset < rows.size();
       offset += static_cast<std::size_t>(kPoseRowStride)) {
    const double* row = rows.data() + offset;
    const double score = row[4];
    if (!(score > kPoseScoreThreshold)) continue;
    const auto box = affine.invert_box(row[0], row[1], row[2], row[3]);
    parsed.boxes.push_back(ParsedBox{box.x1, box.y1, box.x2, box.y2, score});
    std::vector<ParsedKeypoint> keypoints;
    keypoints.reserve(kCocoKeypointCount);
    for (int index = 0; index < kCocoKeypointCount; ++index) {
      const int base = kPersonRowStride + index * 3;
      const auto [x, y] = affine.invert_keypoint(row[base], row[base + 1]);
      keypoints.push_back(ParsedKeypoint{x, y, row[base + 2]});
    }
    parsed.poses.push_back(std::move(keypoints));
  }
  return parsed;
}

std::vector<ParsedBox> parse_person_rows(const std::vector<double>& rows,
                                         const AffineMetadata& affine,
                                         double confidence) {
  validate_rows(rows, kPersonRowStride, "person");
  std::vector<ParsedBox> boxes;
  for (std::size_t offset = 0; offset < rows.size();
       offset += static_cast<std::size_t>(kPersonRowStride)) {
    const double* row = rows.data() + offset;
    const double score = row[4];
    if (static_cast<int>(row[5]) != kCocoPersonClassId || score < confidence) continue;
    const auto box = affine.invert_box(row[0], row[1], row[2], row[3]);
    boxes.push_back(ParsedBox{box.x1, box.y1, box.x2, box.y2, score});
  }
  return boxes;
}

namespace {

// Mirrors parse.py::_trace_contour: deterministic boundary walk ordered by
// (angle around the centroid, y, x) with numpy lexsort's stable semantics.
std::vector<std::pair<double, double>> trace_contour(
    const std::vector<std::pair<int, int>>& points_yx) {
  double center_y = 0.0;
  double center_x = 0.0;
  for (const auto& [y, x] : points_yx) {
    center_y += y;
    center_x += x;
  }
  center_y /= static_cast<double>(points_yx.size());
  center_x /= static_cast<double>(points_yx.size());
  struct Item {
    double angle;
    int y;
    int x;
    std::size_t index;
  };
  std::vector<Item> items;
  items.reserve(points_yx.size());
  for (std::size_t index = 0; index < points_yx.size(); ++index) {
    const auto& [y, x] = points_yx[index];
    items.push_back(Item{std::atan2(y - center_y, x - center_x), y, x, index});
  }
  std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    if (a.angle != b.angle) return a.angle < b.angle;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
  });
  std::vector<std::pair<double, double>> contour;
  contour.reserve(items.size());
  for (const Item& item : items) {
    contour.emplace_back(static_cast<double>(item.x), static_cast<double>(item.y));
  }
  return contour;
}

// Mirrors np.linspace(0, n-1, max_points).astype(int64): truncation, not round.
std::vector<std::size_t> linspace_indexes(std::size_t count, int max_points) {
  std::vector<std::size_t> indexes;
  indexes.reserve(static_cast<std::size_t>(max_points));
  const double last = static_cast<double>(count - 1);
  for (int index = 0; index < max_points; ++index) {
    const double position = last * static_cast<double>(index) /
                            static_cast<double>(max_points - 1);
    indexes.push_back(static_cast<std::size_t>(position));
  }
  return indexes;
}

}  // namespace

std::vector<ParsedBedRegion> parse_bed_rows(const std::vector<double>& rows,
                                            const std::vector<double>& prototypes,
                                            const AffineMetadata& affine,
                                            double confidence, int max_points) {
  validate_rows(rows, kBedRowStride, "bed");
  constexpr int kChannels = kBedPrototypeChannels;
  constexpr int kMaskHeight = kBedPrototypeHeight;
  constexpr int kMaskWidth = kBedPrototypeWidth;
  if (prototypes.size() != static_cast<std::size_t>(kChannels * kMaskHeight * kMaskWidth)) {
    throw std::invalid_argument{"bed prototype tensor has the wrong shape"};
  }
  const double width_ratio = static_cast<double>(kMaskWidth) / affine.tensor_width;
  const double height_ratio = static_cast<double>(kMaskHeight) / affine.tensor_height;
  std::vector<ParsedBedRegion> regions;
  for (std::size_t offset = 0; offset < rows.size();
       offset += static_cast<std::size_t>(kBedRowStride)) {
    const double* row = rows.data() + offset;
    const double score = row[4];
    if (static_cast<int>(row[5]) != kCocoBedClassId || score < confidence) continue;
    const double tx1 = row[0];
    const double ty1 = row[1];
    const double tx2 = row[2];
    const double ty2 = row[3];
    const auto box = affine.invert_box(tx1, ty1, tx2, ty2);
    // Crop bounds in prototype space (parse.py::_mask_polygon).
    const int left = std::max(0, static_cast<int>(std::floor(tx1 * width_ratio)));
    const int top = std::max(0, static_cast<int>(std::floor(ty1 * height_ratio)));
    const int right = std::min(kMaskWidth, static_cast<int>(std::ceil(tx2 * width_ratio)));
    const int bottom = std::min(kMaskHeight, static_cast<int>(std::ceil(ty2 * height_ratio)));
    std::vector<std::pair<int, int>> active_yx;  // np.argwhere order: row-major
    if (right > left && bottom > top) {
      for (int y = top; y < bottom; ++y) {
        for (int x = left; x < right; ++x) {
          double value = 0.0;
          for (int channel = 0; channel < kChannels; ++channel) {
            value += row[kPersonRowStride + channel] *
                     prototypes[static_cast<std::size_t>(channel) * kMaskHeight * kMaskWidth +
                                static_cast<std::size_t>(y) * kMaskWidth +
                                static_cast<std::size_t>(x)];
          }
          if (value > 0.0) active_yx.emplace_back(y, x);
        }
      }
    }
    ParsedBedRegion region{ParsedBox{box.x1, box.y1, box.x2, box.y2, score}, {}};
    if (!active_yx.empty()) {
      auto contour = trace_contour(active_yx);
      if (contour.size() > static_cast<std::size_t>(max_points)) {
        std::vector<std::pair<double, double>> sampled;
        sampled.reserve(static_cast<std::size_t>(max_points));
        for (const std::size_t index : linspace_indexes(contour.size(), max_points)) {
          sampled.push_back(contour[index]);
        }
        contour = std::move(sampled);
      }
      region.polygon.reserve(contour.size());
      for (const auto& [x, y] : contour) {
        region.polygon.push_back(
            affine.invert_keypoint(x / width_ratio, y / height_ratio));
      }
    }
    regions.push_back(std::move(region));
  }
  return regions;
}

namespace {

double intersection_over_union(const ParsedBox& a, const ParsedBox& b) {
  const int inter_x1 = std::max(a.x1, b.x1);
  const int inter_y1 = std::max(a.y1, b.y1);
  const int inter_x2 = std::min(a.x2, b.x2);
  const int inter_y2 = std::min(a.y2, b.y2);
  const int inter_width = inter_x2 - inter_x1;
  const int inter_height = inter_y2 - inter_y1;
  if (inter_width <= 0 || inter_height <= 0) return 0.0;
  const double inter_area = static_cast<double>(inter_width) * inter_height;
  const double area_a = static_cast<double>(a.x2 - a.x1) * (a.y2 - a.y1);
  const double area_b = static_cast<double>(b.x2 - b.x1) * (b.y2 - b.y1);
  const double union_area = area_a + area_b - inter_area;
  if (union_area <= 0.0) return 0.0;
  return inter_area / union_area;
}

}  // namespace

AssociationOutput LegacyGreedyBboxIou::observe(const std::vector<ParsedBox>& boxes) {
  struct Candidate {
    double score;
    std::size_t track_index;
    std::size_t box_index;
  };
  std::vector<Candidate> candidates;
  for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
    for (std::size_t box_index = 0; box_index < boxes.size(); ++box_index) {
      const double score = intersection_over_union(tracks_[track_index].last_box,
                                                   boxes[box_index]);
      if (score > 0.0) candidates.push_back(Candidate{score, track_index, box_index});
    }
  }
  // Python list.sort(key=score, reverse=True) is stable: equal scores keep
  // discovery order (lower track index first, then lower box index).
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
  std::set<std::size_t> taken_tracks;
  std::set<std::size_t> taken_boxes;
  std::vector<std::pair<std::size_t, std::size_t>> matches;
  for (const Candidate& candidate : candidates) {
    if (candidate.score < min_iou_) break;
    if (taken_tracks.contains(candidate.track_index) ||
        taken_boxes.contains(candidate.box_index)) {
      continue;
    }
    taken_tracks.insert(candidate.track_index);
    taken_boxes.insert(candidate.box_index);
    matches.emplace_back(candidate.track_index, candidate.box_index);
  }
  std::vector<std::int64_t> box_track_ids(boxes.size(), -1);
  std::set<std::size_t> matched_tracks;
  for (const auto& [track_index, box_index] : matches) {
    matched_tracks.insert(track_index);
    Track& track = tracks_[track_index];
    track.last_box = boxes[box_index];
    track.misses = 0;
    box_track_ids[box_index] = track.track_id;
  }
  std::vector<Track> new_tracks;
  for (std::size_t box_index = 0; box_index < boxes.size(); ++box_index) {
    if (box_track_ids[box_index] >= 0) continue;
    box_track_ids[box_index] = next_id_;
    new_tracks.push_back(Track{next_id_, boxes[box_index], 0});
    ++next_id_;
  }
  std::vector<Track> surviving;
  for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
    Track& track = tracks_[track_index];
    if (matched_tracks.contains(track_index)) {
      surviving.push_back(track);
      continue;
    }
    ++track.misses;
    if (track.misses <= max_misses_) surviving.push_back(track);
  }
  for (Track& track : new_tracks) surviving.push_back(track);
  tracks_ = std::move(surviving);
  AssociationOutput output;
  output.selected_cue_indexes.reserve(boxes.size());
  output.track_ids.reserve(boxes.size());
  for (std::size_t box_index = 0; box_index < boxes.size(); ++box_index) {
    output.selected_cue_indexes.push_back(static_cast<std::uint16_t>(box_index));
    output.track_ids.push_back(box_track_ids[box_index]);
  }
  return output;
}

void LegacyGreedyBboxIou::reset() {
  tracks_.clear();
  next_id_ = 0;
}

std::set<std::int64_t> LegacyGreedyBboxIou::live_ids() const {
  std::set<std::int64_t> ids;
  for (const Track& track : tracks_) ids.insert(track.track_id);
  return ids;
}

}  // namespace seeon::perception
